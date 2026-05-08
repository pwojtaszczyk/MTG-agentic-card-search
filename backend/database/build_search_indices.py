#!/usr/bin/env python3
"""Rebuild FTS5 and sqlite-vec tables from ``cards`` / ``card_embeddings``."""

from __future__ import annotations

import argparse
import array
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

_PKG_PARENT = Path(__file__).resolve().parents[2]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

import sqlite_vec  # type: ignore[import-untyped]
from backend.database.embedding_text import build_card_embedding_text

EMBEDDING_MODEL_DEFAULT = "text-embedding-3-small"
EMBEDDING_DIM_DEFAULT = 1536


def _default_database_path() -> Path:
    return Path(__file__).resolve().parent / "all-cards.db"


def _try_enable_sqlite_vec(conn: sqlite3.Connection) -> bool:
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return True
    except Exception:
        return False


def _floats_to_blob(values: list[float]) -> bytes:
    return array.array("f", values).tobytes()


def _create_embeddings_with_retry(
    client: OpenAI,
    model: str,
    inputs: list[str],
    dimensions: int,
    max_attempts: int = 5,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.embeddings.create(
                model=model,
                input=inputs,
                dimensions=dimensions,
            )
            if not getattr(resp, "data", None):
                raise ValueError("No embedding data received")
            return resp
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            sleep_s = min(8.0, 0.75 * (2 ** (attempt - 1)))
            print(
                f"Embeddings batch failed (attempt {attempt}/{max_attempts}): {exc}. "
                f"Retrying in {sleep_s:.1f}s..."
            )
            time.sleep(sleep_s)
    raise RuntimeError(
        f"Embeddings API failed after {max_attempts} attempts for model={model}."
    ) from last_error


def _existing_embedding_signature(conn: sqlite3.Connection) -> tuple[list[str], list[int], int]:
    model_rows = conn.execute(
        "SELECT DISTINCT model FROM card_embeddings WHERE model IS NOT NULL AND model != '' ORDER BY model"
    ).fetchall()
    dim_rows = conn.execute("SELECT DISTINCT dimensions FROM card_embeddings ORDER BY dimensions").fetchall()
    count_row = conn.execute("SELECT COUNT(*) FROM card_embeddings").fetchone()
    models = [str(r[0]) for r in model_rows]
    dims = [int(r[0]) for r in dim_rows if r[0] is not None]
    count = int(count_row[0]) if count_row else 0
    return models, dims, count


def _rebuild_embeddings_if_needed(
    conn: sqlite3.Connection,
    model: str = EMBEDDING_MODEL_DEFAULT,
    dimensions: int = EMBEDDING_DIM_DEFAULT,
    batch_size: int = 64,
    force: bool = False,
) -> None:
    cards_count_row = conn.execute("SELECT COUNT(*) FROM cards").fetchone()
    cards_count = int(cards_count_row[0]) if cards_count_row else 0
    if cards_count == 0:
        print("No cards in database; skipping embeddings refresh.")
        return

    models, dims, emb_count = _existing_embedding_signature(conn)
    up_to_date = (models == [model]) and (dims == [dimensions]) and (emb_count == cards_count)
    if up_to_date and not force:
        print(
            f"Embeddings already match requested config: model={model}, dimensions={dimensions}, rows={emb_count}"
        )
        return

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is required to refresh embeddings. "
            "Set the key or pass --skip-embedding-refresh."
        )

    print(
        "Refreshing embeddings table: "
        f"old_models={models or ['<none>']}, old_dims={dims or ['<none>']}, old_rows={emb_count}"
    )
    print(f"Target embeddings: model={model}, dimensions={dimensions}, cards={cards_count}")

    client = OpenAI(
        api_key=openrouter_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )

    rows = conn.execute("SELECT oracle_id, raw_json FROM cards ORDER BY rowid").fetchall()
    texts: list[tuple[str, str]] = []
    for row in rows:
        card_id = str(row["oracle_id"])
        raw_json = row["raw_json"] or "{}"
        try:
            card: dict[str, Any] = json.loads(raw_json)
        except Exception:
            card = {}
        text = build_card_embedding_text(card).strip() or "(empty)"
        texts.append((card_id, text))

    conn.execute("DELETE FROM card_embeddings")
    conn.execute("DELETE FROM card_embedding_texts")
    conn.executemany(
        "INSERT INTO card_embedding_texts (card_id, embedding_text) VALUES (?, ?)",
        texts,
    )

    batch_size = max(1, batch_size)
    progress = tqdm(range(0, len(texts), batch_size), desc="Embeddings API", unit="batch")
    to_insert: list[tuple[str, str, int, bytes]] = []
    for start in progress:
        batch = texts[start : start + batch_size]
        inputs = [t for _, t in batch]
        resp = _create_embeddings_with_retry(
            client,
            model=model,
            inputs=inputs,
            dimensions=dimensions,
        )
        for (card_id, _), item in zip(batch, resp.data, strict=True):
            vec = item.embedding
            if len(vec) != dimensions:
                raise ValueError(
                    f"Expected {dimensions} dims from {model}, got {len(vec)} for card {card_id}"
                )
            to_insert.append((card_id, model, dimensions, _floats_to_blob(vec)))
    progress.close()

    conn.executemany(
        "INSERT INTO card_embeddings (card_id, model, dimensions, embedding) VALUES (?,?,?,?)",
        to_insert,
    )
    print(f"Embeddings refresh complete: model={model}, dimensions={dimensions}, rows={len(to_insert)}")


def _rebuild_fts5_with_progress(conn: sqlite3.Connection) -> None:
    print("[1/3] Rebuilding FTS5 index (cards_fts)...")
    conn.execute("DROP TABLE IF EXISTS cards_fts")
    conn.executescript(
        """
        CREATE VIRTUAL TABLE cards_fts USING fts5(
            name,
            oracle_text,
            type_line,
            content='cards',
            content_rowid='rowid'
        );
        INSERT INTO cards_fts(cards_fts) VALUES('rebuild');
        """
    )
    print("[1/3] FTS5 rebuild complete.")


def _rebuild_vec_with_progress(conn: sqlite3.Connection, *, batch_size: int = 5000) -> None:
    print("[2/3] Rebuilding vector index (cards_vec)...")
    try:
        conn.execute("SELECT vec_version()").fetchone()
    except sqlite3.Error:
        print("[2/3] sqlite-vec extension not available; skipping cards_vec rebuild.")
        return

    dim_row = conn.execute("SELECT dimensions FROM card_embeddings LIMIT 1").fetchone()
    if not dim_row:
        print("[2/3] No rows in card_embeddings; skipping cards_vec rebuild.")
        return
    dim = int(dim_row[0])
    model_rows = conn.execute(
        "SELECT DISTINCT model FROM card_embeddings WHERE model IS NOT NULL AND model != '' ORDER BY model"
    ).fetchall()
    models = [str(r[0]) for r in model_rows]
    if models:
        print(f"[2/3] Embedding source: model={', '.join(models)}, dimensions={dim}")
    else:
        print(f"[2/3] Embedding source: dimensions={dim}")

    total_row = conn.execute(
        """
        SELECT COUNT(*)
        FROM cards c
        INNER JOIN card_embeddings e ON e.card_id = c.oracle_id
        """
    ).fetchone()
    total = int(total_row[0]) if total_row else 0

    conn.execute("DROP TABLE IF EXISTS cards_vec")
    conn.execute(f"CREATE VIRTUAL TABLE cards_vec USING vec0(embedding float[{dim}])")

    cursor = conn.execute(
        """
        SELECT c.rowid, e.embedding
        FROM cards c
        INNER JOIN card_embeddings e ON e.card_id = c.oracle_id
        """
    )

    progress = tqdm(total=total, unit="rows", desc="Building cards_vec")
    try:
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            conn.executemany("INSERT INTO cards_vec(rowid, embedding) VALUES (?, ?)", rows)
            progress.update(len(rows))
    finally:
        progress.close()
    print("[2/3] Vector index rebuild complete.")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Refresh embeddings (optional) and rebuild cards_fts/cards_vec search indices."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite path (default: DATABASE_PATH or backend/database/all-cards.db)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help=f"Embedding model for card_embeddings refresh (default: {EMBEDDING_MODEL_DEFAULT}).",
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=None,
        help=f"Embedding vector dimensions (default: {EMBEDDING_DIM_DEFAULT}).",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Inputs per embeddings API request when refreshing embeddings.",
    )
    parser.add_argument(
        "--skip-embedding-refresh",
        action="store_true",
        help="Do not refresh card_embeddings/card_embedding_texts; rebuild indices only.",
    )
    parser.add_argument(
        "--force-embedding-refresh",
        action="store_true",
        help="Refresh embeddings even if model/dimensions/row count already match.",
    )
    args = parser.parse_args()
    embedding_model = (
        args.embedding_model if args.embedding_model is not None else EMBEDDING_MODEL_DEFAULT
    )
    embedding_dimensions = (
        args.embedding_dimensions
        if args.embedding_dimensions is not None
        else EMBEDDING_DIM_DEFAULT
    )
    path = args.db if args.db is not None else _default_database_path()
    if not path.is_file():
        raise SystemExit(f"Database not found: {path}")
    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 10000")
    _try_enable_sqlite_vec(conn)
    try:
        if not args.skip_embedding_refresh:
            print("[0/3] Checking/updating embedding tables...")
            _rebuild_embeddings_if_needed(
                conn,
                model=embedding_model,
                dimensions=embedding_dimensions,
                batch_size=args.embedding_batch_size,
                force=args.force_embedding_refresh,
            )
            print("[0/3] Embedding table step complete.")
        _rebuild_fts5_with_progress(conn)
        _rebuild_vec_with_progress(conn)
        conn.commit()
    finally:
        conn.close()
    print(f"Search indices rebuilt for {path}")


if __name__ == "__main__":
    main()
