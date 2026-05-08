from __future__ import annotations

import argparse
import array
import io
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterator, TextIO

_PKG_PARENT = Path(__file__).resolve().parents[2]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import sqlite_vec  # type: ignore[import-untyped]

EMBEDDING_MODEL_DEFAULT = "text-embedding-3-small"
EMBEDDING_DIM_DEFAULT = 1536


DB_PATH = Path(__file__).resolve().parent / "all-cards.db"


from backend.database.index_builder import ensure_search_indices
from backend.database.embedding_text import build_card_embedding_text


def _try_enable_sqlite_vec(conn: sqlite3.Connection) -> bool:
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return True
    except Exception:
        return False


class _TrackingBinaryReader(io.RawIOBase):
    """
    Wraps an unbuffered binary stream and forwards read/readinto while updating tqdm.
    tqdm.wrapattr(..., "read") is unreliable here because TextIOWrapper/BufferedReader
    often use readinto, not read.
    """

    def __init__(self, raw: io.RawIOBase, pbar: Any) -> None:
        self._raw = raw
        self._pbar = pbar

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        data = self._raw.read(size)
        if data:
            self._pbar.update(len(data))
        return data

    def readinto(self, b: Any) -> int | None:
        if hasattr(self._raw, "readinto"):
            n = self._raw.readinto(b)
            if n is not None and n > 0:
                self._pbar.update(n)
            return n
        chunk = self._raw.read(len(b))
        if not chunk:
            return 0
        nbytes = len(chunk)
        b[:nbytes] = chunk
        self._pbar.update(nbytes)
        return nbytes

    def close(self) -> None:
        self._raw.close()

    @property
    def closed(self) -> bool:
        return self._raw.closed


def iter_scryfall_bulk_cards(
    file_obj,
    chunk_size: int = 1048576,
) -> Iterator[dict[str, Any]]:
    """Stream objects from a Scryfall bulk JSON array without loading the file into memory."""
    decoder = json.JSONDecoder()
    if file_obj.read(1) != "[":
        raise ValueError("Expected JSON array opening '['")
    buf = ""
    while True:
        while True:
            buf = buf.lstrip()
            if buf.startswith("]"):
                return
            if buf.startswith(","):
                buf = buf[1:].lstrip()
                continue
            if buf:
                break
            chunk = file_obj.read(chunk_size)
            if not chunk:
                raise ValueError("Unexpected EOF before end of array")
            buf += chunk
        while True:
            try:
                obj, idx = decoder.raw_decode(buf)
                buf = buf[idx:].lstrip()
                if not isinstance(obj, dict):
                    raise TypeError(f"Expected card object, got {type(obj)}")
                yield obj
                break
            except json.JSONDecodeError:
                # Incomplete object in buffer (e.g. "Unterminated string") often sets
                # e.pos mid-buffer; always try to read more before failing.
                chunk = file_obj.read(chunk_size)
                if not chunk:
                    raise
                buf += chunk


def take_first_cards(
    stream: Iterator[dict[str, Any]],
    n: int | None,
) -> list[dict[str, Any]]:
    """Collect card objects from a stream. If ``n`` is set, stop after ``n``; if ``None``, read until EOF."""
    out: list[dict[str, Any]] = []
    for card in stream:
        out.append(card)
        if n is not None and len(out) >= n:
            break
    return out


def _bool_to_int(v: Any) -> int | None:
    if v is None:
        return None
    return 1 if v else 0


def _release_sort_key(card: dict[str, Any]) -> tuple[str, str]:
    released_at = str(card.get("released_at") or "")
    scryfall_id = str(card.get("id") or "")
    return (released_at, scryfall_id)


def select_representative_prints(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Keep only one representative print per oracle_id:
    - consider English cards only (lang='en')
    - if multiple prints exist, sort by release date and pick the middle one
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        if str(card.get("lang") or "").lower() != "en":
            continue
        oracle_id = str(card.get("oracle_id") or "").strip()
        if not oracle_id:
            continue
        grouped.setdefault(oracle_id, []).append(card)

    selected: list[dict[str, Any]] = []
    for oracle_id in sorted(grouped.keys()):
        prints = sorted(grouped[oracle_id], key=_release_sort_key)
        selected.append(prints[len(prints) // 2])
    return selected


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;

        DROP TABLE IF EXISTS card_embeddings;
        DROP TABLE IF EXISTS card_embedding_texts;
        DROP TABLE IF EXISTS card_legalities;
        DROP TABLE IF EXISTS card_faces;
        DROP TABLE IF EXISTS cards;

        CREATE TABLE cards (
            oracle_id TEXT PRIMARY KEY,
            print_id TEXT,
            name TEXT NOT NULL,
            lang TEXT,
            layout TEXT,
            released_at TEXT,
            mana_cost TEXT,
            cmc REAL,
            type_line TEXT,
            oracle_text TEXT,
            power TEXT,
            toughness TEXT,
            loyalty TEXT,
            set_id TEXT,
            set_code TEXT,
            set_name TEXT,
            set_type TEXT,
            collector_number TEXT,
            rarity TEXT,
            digital INTEGER,
            foil INTEGER,
            nonfoil INTEGER,
            reprint INTEGER,
            variation INTEGER,
            promo INTEGER,
            scryfall_uri TEXT,
            uri TEXT,
            rulings_uri TEXT,
            prints_search_uri TEXT,
            image_uris_json TEXT,
            prices_json TEXT,
            related_uris_json TEXT,
            purchase_uris_json TEXT,
            raw_json TEXT
        );

        CREATE INDEX idx_cards_name ON cards(name);
        CREATE INDEX idx_cards_set_code ON cards(set_code);

        CREATE TABLE card_faces (
            card_id TEXT NOT NULL REFERENCES cards(oracle_id) ON DELETE CASCADE,
            face_index INTEGER NOT NULL,
            name TEXT,
            printed_name TEXT,
            mana_cost TEXT,
            type_line TEXT,
            oracle_text TEXT,
            flavor_text TEXT,
            power TEXT,
            toughness TEXT,
            loyalty TEXT,
            artist TEXT,
            image_uris_json TEXT,
            PRIMARY KEY (card_id, face_index)
        );

        CREATE TABLE card_legalities (
            card_id TEXT NOT NULL REFERENCES cards(oracle_id) ON DELETE CASCADE,
            format TEXT NOT NULL,
            status TEXT NOT NULL,
            PRIMARY KEY (card_id, format)
        );

        CREATE INDEX idx_legalities_format_status ON card_legalities(format, status);

        CREATE TABLE card_embeddings (
            card_id TEXT PRIMARY KEY REFERENCES cards(oracle_id) ON DELETE CASCADE,
            model TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            embedding BLOB NOT NULL
        );

        CREATE TABLE card_embedding_texts (
            card_id TEXT PRIMARY KEY REFERENCES cards(oracle_id) ON DELETE CASCADE,
            embedding_text TEXT NOT NULL
        );
        """
    )


def insert_sample(conn: sqlite3.Connection, cards: list[dict[str, Any]]) -> None:
    card_rows: list[tuple[Any, ...]] = []
    face_rows: list[tuple[Any, ...]] = []
    leg_rows: list[tuple[Any, ...]] = []

    for c in cards:
        cid = str(c["oracle_id"])
        raw = json.dumps(c, separators=(",", ":"), ensure_ascii=False)

        card_rows.append(
            (
                cid,
                c.get("id"),
                c.get("name") or "",
                c.get("lang"),
                c.get("layout"),
                c.get("released_at"),
                c.get("mana_cost"),
                c.get("cmc"),
                c.get("type_line"),
                c.get("oracle_text"),
                c.get("power"),
                c.get("toughness"),
                c.get("loyalty"),
                c.get("set_id"),
                c.get("set"),
                c.get("set_name"),
                c.get("set_type"),
                c.get("collector_number"),
                c.get("rarity"),
                _bool_to_int(c.get("digital")),
                _bool_to_int(c.get("foil")),
                _bool_to_int(c.get("nonfoil")),
                _bool_to_int(c.get("reprint")),
                _bool_to_int(c.get("variation")),
                _bool_to_int(c.get("promo")),
                c.get("scryfall_uri"),
                c.get("uri"),
                c.get("rulings_uri"),
                c.get("prints_search_uri"),
                json.dumps(c["image_uris"], ensure_ascii=False) if c.get("image_uris") else None,
                json.dumps(c["prices"], ensure_ascii=False) if c.get("prices") else None,
                json.dumps(c["related_uris"], ensure_ascii=False) if c.get("related_uris") else None,
                json.dumps(c["purchase_uris"], ensure_ascii=False) if c.get("purchase_uris") else None,
                raw,
            )
        )

        legalities = c.get("legalities")
        if isinstance(legalities, dict):
            for fmt, status in legalities.items():
                leg_rows.append((cid, fmt, status))

        faces = c.get("card_faces")
        if isinstance(faces, list):
            for idx, face in enumerate(faces):
                if not isinstance(face, dict):
                    continue
                face_imgs = face.get("image_uris")
                face_rows.append(
                    (
                        cid,
                        idx,
                        face.get("name"),
                        face.get("printed_name"),
                        face.get("mana_cost"),
                        face.get("type_line"),
                        face.get("oracle_text"),
                        face.get("flavor_text"),
                        face.get("power"),
                        face.get("toughness"),
                        face.get("loyalty"),
                        face.get("artist"),
                        json.dumps(face_imgs, ensure_ascii=False) if face_imgs else None,
                    )
                )

    conn.executemany(
        """
        INSERT INTO cards (
            oracle_id, print_id, name, lang, layout, released_at,
            mana_cost, cmc, type_line, oracle_text, power, toughness, loyalty,
            set_id, set_code, set_name, set_type, collector_number, rarity,
            digital, foil, nonfoil, reprint, variation, promo,
            scryfall_uri, uri, rulings_uri, prints_search_uri,
            image_uris_json, prices_json, related_uris_json, purchase_uris_json,
            raw_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        card_rows,
    )

    if face_rows:
        conn.executemany(
            """
            INSERT INTO card_faces (
                card_id, face_index, name, printed_name, mana_cost, type_line,
                oracle_text, flavor_text, power, toughness, loyalty, artist, image_uris_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            face_rows,
        )

    if leg_rows:
        conn.executemany(
            """
            INSERT INTO card_legalities (card_id, format, status)
            VALUES (?,?,?)
            """,
            leg_rows,
        )


def _floats_to_blob(values: list[float]) -> bytes:
    return array.array("f", values).tobytes()


def insert_embeddings_openai(
    conn: sqlite3.Connection,
    cards: list[dict[str, Any]],
    client: OpenAI,
    progress: bool,
    model: str = EMBEDDING_MODEL_DEFAULT,
    dimensions: int = EMBEDDING_DIM_DEFAULT,
    batch_size: int = 64,
) -> None:
    """Call OpenAI embeddings API and store float32 BLOBs (for sqlite-vec / manual similarity)."""
    batch_size = max(1, batch_size)
    print(
        f"Starting embedding build: model={model}, dimensions={dimensions}, "
        f"batch_size={batch_size}, cards={len(cards)}"
    )
    rows: list[tuple[str, str]] = []
    for c in cards:
        cid = c.get("oracle_id")
        if not cid:
            continue
        text = build_card_embedding_text(c).strip() or "(empty)"
        rows.append((str(cid), text))

    if not rows:
        return

    conn.executemany(
        """
        INSERT INTO card_embedding_texts (card_id, embedding_text)
        VALUES (?, ?)
        """,
        rows,
    )

    to_insert: list[tuple[str, str, int, bytes]] = []
    batch_size = max(1, batch_size)

    n_batches = (len(rows) + batch_size - 1) // batch_size
    batch_indices = range(0, len(rows), batch_size)
    if progress:
        batch_indices = tqdm(
            batch_indices,
            desc="Embeddings API",
            unit="batch",
            total=n_batches,
            leave=True,
        )

    for start in batch_indices:
        batch = rows[start : start + batch_size]
        inputs = [t for _, t in batch]
        resp = client.embeddings.create(
            model=model,
            input=inputs,
            dimensions=dimensions,
        )
        # API returns data in request order
        for (card_id, _), item in zip(batch, resp.data, strict=True):
            vec = item.embedding
            if len(vec) != dimensions:
                raise ValueError(
                    f"Expected {dimensions} dims from {model}, got {len(vec)} for card {card_id}"
                )
            to_insert.append((card_id, model, dimensions, _floats_to_blob(vec)))

    conn.executemany(
        """
        INSERT INTO card_embeddings (card_id, model, dimensions, embedding)
        VALUES (?,?,?,?)
        """,
        to_insert,
    )


def bootstrap(
    json_path: Path,
    db_path: Path = DB_PATH,
    limit: int | None = None,
    progress: bool = True,
    skip_embeddings: bool = False,
    embedding_batch_size: int = 64,
    embedding_model: str = EMBEDDING_MODEL_DEFAULT,
    embedding_dimensions: int = EMBEDDING_DIM_DEFAULT,
) -> None:
    """
    Create all-cards.db and populate from the bulk JSON in file order.
    If ``limit`` is set, stop after that many cards; if ``None``, import the entire array.
    Optionally stores OpenRouter embeddings in ``card_embeddings`` (text-embedding-3-small).
    """
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1 when provided")

    db_path.parent.mkdir(parents=True, exist_ok=True)

    total_bytes = json_path.stat().st_size

    def load_cards_from_text_stream(text_f: TextIO) -> list[dict[str, Any]]:
        return take_first_cards(iter_scryfall_bulk_cards(text_f), limit)

    if progress:
        # Full import: total=file size so the bar reaches 100%. With --limit, we stop after a
        # tiny prefix; using the same total would stall around ~1% (misleading).
        scan_total: int | None = total_bytes if limit is None else None
        scan_desc = (
            "Scanning bulk JSON"
            if limit is None
            else f"Scanning bulk JSON (stop after {limit} cards)"
        )
        with tqdm(
            total=scan_total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=scan_desc,
            miniters=1,
            mininterval=0.05,
        ) as pbar, json_path.open("rb", buffering=0) as raw_fp:
            tracked = _TrackingBinaryReader(raw_fp, pbar)
            buffered = io.BufferedReader(tracked, buffer_size=256 * 1024)
            text_f = io.TextIOWrapper(buffered, encoding="utf-8")
            try:
                cards = load_cards_from_text_stream(text_f)
            finally:
                text_f.close()
    else:
        with json_path.open("r", encoding="utf-8") as f:
            cards = load_cards_from_text_stream(f)

    cards = select_representative_prints(cards)

    if limit is not None:
        cards = cards[:limit]

    with sqlite3.connect(db_path) as conn:
        create_schema(conn)
        insert_sample(conn, cards)
        if not skip_embeddings:
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            if not openrouter_key:
                raise ValueError(
                    "OPENROUTER_API_KEY is required for embeddings (or pass --skip-embeddings)."
                )
            client = OpenAI(
                api_key=openrouter_key,
                base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            )
            insert_embeddings_openai(
                conn,
                cards,
                client=client,
                model=embedding_model,
                dimensions=embedding_dimensions,
                batch_size=embedding_batch_size,
                progress=progress,
            )
        _try_enable_sqlite_vec(conn)
        ensure_search_indices(conn)
        conn.commit()

    print(f"Database ready: {db_path} ({len(cards)} cards imported)")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Create all-cards.db from a Scryfall bulk JSON (all cards, or first N with --limit)."
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to Scryfall bulk all-cards JSON (top-level array).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help=f"SQLite output path (default: {DB_PATH})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Import only the first N cards. If omitted, import the full bulk file.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar (e.g. for CI or logs).",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Do not call OpenRouter embeddings API or write card_embeddings (no OPENROUTER_API_KEY needed).",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Inputs per embeddings API request (default: 64).",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help=f"Embedding model (default: {EMBEDDING_MODEL_DEFAULT}).",
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=None,
        help=f"Embedding dimensions (default: {EMBEDDING_DIM_DEFAULT}).",
    )
    args = parser.parse_args()
    if not args.json_path.is_file():
        raise SystemExit(f"Not a file: {args.json_path}")
    bootstrap(
        json_path=args.json_path,
        db_path=args.db,
        limit=args.limit,
        progress=not args.no_progress,
        skip_embeddings=args.skip_embeddings,
        embedding_batch_size=(
            args.embedding_batch_size if args.embedding_batch_size is not None else 64
        ),
        embedding_model=(
            args.embedding_model if args.embedding_model is not None else EMBEDDING_MODEL_DEFAULT
        ),
        embedding_dimensions=(
            args.embedding_dimensions
            if args.embedding_dimensions is not None
            else EMBEDDING_DIM_DEFAULT
        ),
    )


if __name__ == "__main__":
    main()
