from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from agents import function_tool
import sqlite_vec

from backend.utils.agents_sdk import openrouter_async_client, openrouter_async_model
from backend.utils.db_paths import get_database_path
from backend.utils.profiling import (
    emit_db_aggregate_summary,
    emit_event,
    record_db_bucket_time,
)

# This system prompt is used to expand the user query into a more precise query for the search tool.
# It returns two strings: fts_keywords and semantic_query.
# fts_keywords is a concise space-separated English tokens for full-text search over card names, type lines, and rules text. No Scryfall syntax.
# semantic_query is a one clear English sentence describing requested cards.
EXPAND_MODEL_DEFAULT = "gpt-4o-mini"
QUERY_EXPAND_SYSTEM_PROMPT = """
# You help search Magic: The Gathering cards.

## Output format
Reply with JSON only: {"fts_keywords": string, "semantic_query": string}.
- `fts_keywords`: concise space-separated English tokens for full-text search over card names, type lines, and rules text. No Scryfall syntax.
- `semantic_query`: one clear English sentence describing requested cards.
"""

# This tool description is used to describe the search tool to the agent.
# Using this description, the agent can call the search tool with the correct parameters
# and learns how to interpret the output of the search tool.
SEARCH_MTG_CARDS_TOOL_DESCRIPTION = """
Search "Magic: The Gathering" (MTG) cards by user query, selected formats, and result limit.

Parameters:
- query: Natural-language search request from the user.
- formats: Format slugs to constrain legality (for example: ["standard", "modern"]).
  Use an empty list to search without format filtering.
- limit: Maximum number of cards to return.
- use_prefilter: Controls SQL prefilter behavior:
  - True: use for constrained mechanic/type/color requests
    (for example: "red creatures", "blue instants", "white lifegain enchantments").
  - False: use for broad or fuzzy queries where strict SQL narrowing may hide relevant results.
  Choose this flag intentionally per request intent.

Returns:
- A list of card dictionaries. Each item contains:
  - name: Card name.
  - mana_cost: Printed mana cost string (for example "{1}{R}").
  - cmc: Mana value (formerly "converted mana cost"): total numeric cost of the spell/card,
    independent of color symbols (for example, "{1}{R}" -> 2, "{X}{U}" -> usually treated as 1 on stackless lookups).
  - type_line: Full type line (types + subtypes/supertypes).
  - oracle_text: Rules text.
  - colors: Card colors (list of mana color symbols like ["R"]).
  - color_identity: Commander color identity symbols (all colors appearing in mana cost and rules text).
    This can differ from `colors`: for example, a colorless card can have non-empty color identity
    if its text contains colored mana symbols.
  - keywords: Parsed keyword abilities.
  - produced_mana: Mana symbols this card can produce.
  - power: Creature power (string; may be empty or "*" variants).
  - toughness: Creature toughness (string; may be empty or "*" variants).
  - loyalty: Planeswalker loyalty (string, empty for non-planeswalkers).
  - defense: Battle defense value (string, empty for non-battles).
  - scryfall_url: Link to Scryfall card page.
  - format: Display format selected from requested legalities.
"""

_MEM_DB_URI = "file:mtg_cards_shared_mem?mode=memory&cache=shared"
_MEM_DB_KEEPER: sqlite3.Connection | None = None
_MEM_DB_LOCK = threading.Lock()
_EMBED_CONFIG_CACHE: tuple[str, int] | None = None


def _extract_mechanics_fields(raw_json: str | None) -> dict[str, Any]:
    if not raw_json:
        return {
            "colors": [],
            "color_identity": [],
            "keywords": [],
            "produced_mana": [],
            "defense": "",
        }
    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError, ValueError):
        data = None
    if not isinstance(data, dict):
        return {
            "colors": [],
            "color_identity": [],
            "keywords": [],
            "produced_mana": [],
            "defense": "",
        }
    return {
        "colors": data.get("colors") or [],
        "color_identity": data.get("color_identity") or [],
        "keywords": data.get("keywords") or [],
        "produced_mana": data.get("produced_mana") or [],
        "defense": data.get("defense") or "",
    }


def try_enable_sqlite_vec(conn: sqlite3.Connection) -> bool:
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return True
    except Exception:
        return False


def _validate_loaded_database(conn: sqlite3.Connection) -> None:
    """Validate required extensions/tables once after lazy load."""
    if not try_enable_sqlite_vec(conn):
        raise RuntimeError(
            "Database validation failed: sqlite-vec extension is unavailable. "
            "Install/load sqlite-vec and ensure loadable extensions are enabled."
        )
    try:
        conn.execute("SELECT vec_version()").fetchone()
    except sqlite3.Error as exc:
        raise RuntimeError(
            "Database validation failed: sqlite-vec extension is unavailable. "
            "Install/load sqlite-vec and ensure loadable extensions are enabled."
        ) from exc
    required_tables = (
        "cards",
        "cards_fts",
        "cards_vec",
        "card_embeddings",
        "card_legalities",
    )
    missing = [
        name
        for name in required_tables
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table','virtual') AND name=?",
            (name,),
        ).fetchone()
        is None
    ]
    if missing:
        raise RuntimeError(
            "Database validation failed: missing required tables: " + ", ".join(missing)
        )


def initialize_in_memory_database(db_path: str | Path) -> None:
    """Load the on-disk SQLite DB into a shared in-memory database."""
    global _MEM_DB_KEEPER, _EMBED_CONFIG_CACHE
    path = Path(db_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Database file not found: {path}")
    if _MEM_DB_KEEPER is not None:
        return
    with _MEM_DB_LOCK:
        if _MEM_DB_KEEPER is not None:
            return
        src = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10.0)
        src.execute("PRAGMA busy_timeout = 10000")
        keeper = sqlite3.connect(_MEM_DB_URI, uri=True, timeout=10.0)
        keeper.execute("PRAGMA busy_timeout = 10000")
        try:
            src.backup(keeper)
            keeper.commit()
            _validate_loaded_database(keeper)
            _EMBED_CONFIG_CACHE = _read_embedding_config_from_conn(keeper)
            if _EMBED_CONFIG_CACHE is None:
                msg = (
                    "Embedding configuration missing at DB init: "
                    "table `card_embeddings` must contain valid model and dimensions."
                )
                print(msg)
                emit_event(msg)
            _MEM_DB_KEEPER = keeper
        finally:
            src.close()


def close_in_memory_database() -> None:
    """Release the keeper connection that keeps shared-memory DB alive."""
    global _MEM_DB_KEEPER, _EMBED_CONFIG_CACHE
    if _MEM_DB_KEEPER is not None:
        _MEM_DB_KEEPER.close()
        _MEM_DB_KEEPER = None
    _EMBED_CONFIG_CACHE = None


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    initialize_in_memory_database(db_path)
    conn = sqlite3.connect(_MEM_DB_URI, uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 10000")
    # Load sqlite-vec on each connection (the extension state is per-connection).
    try_enable_sqlite_vec(conn)
    return conn


async def expand_user_query(
    user_text: str = "",
    model: str = EXPAND_MODEL_DEFAULT,
) -> tuple[str, str]:
    text = user_text.strip()
    if not text:
        return "", ""

    create = openrouter_async_model(model)
    try:
        resp = await create(
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": QUERY_EXPAND_SYSTEM_PROMPT,
                },
                {"role": "user", "content": text},
            ],
            temperature=0.2,
        )
    except Exception:
        return text, text
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return text, text
    if not isinstance(data, dict):
        return text, text
    fts = str(data.get("fts_keywords", "")).strip() or text
    sem = str(data.get("semantic_query", "")).strip() or text
    return fts, sem


def phrase_to_fts_match(phrase: str) -> str | None:
    tokens = re.findall(r"[\w']+", phrase.lower())
    if not tokens:
        return None
    return " AND ".join(tokens)


async def embed_text(
    text: str = "",
) -> list[float]:
    model, dimensions = _get_embedding_config()
    payload = text.strip()
    if not payload:
        payload = "(empty)"
    embedding_client = openrouter_async_client()
    resp = await embedding_client.embeddings.create(
        model=model,
        input=payload,
        dimensions=dimensions,
    )
    return list(resp.data[0].embedding)


def _execute_fetchall(
    conn: sqlite3.Connection,
    sql: str,
    params: tuple[Any, ...] = (),
    bucket: str = "other",
) -> list[sqlite3.Row]:
    started = time.perf_counter()
    rows = conn.execute(sql, params).fetchall()
    record_db_bucket_time(bucket, (time.perf_counter() - started) * 1000.0)
    return rows


def _execute_fetchone(
    conn: sqlite3.Connection,
    sql: str,
    params: tuple[Any, ...] = (),
    bucket: str = "other",
) -> sqlite3.Row | None:
    started = time.perf_counter()
    row = conn.execute(sql, params).fetchone()
    record_db_bucket_time(bucket, (time.perf_counter() - started) * 1000.0)
    return row


def _format_filter_sql(formats: list[str]) -> tuple[str, list[Any]]:
    if not formats:
        return "", []
    placeholders = ",".join("?" * len(formats))
    clause = (
        f"c.oracle_id IN (SELECT card_id FROM card_legalities WHERE format IN ({placeholders}) "
        f"AND status IN ('legal', 'restricted'))"
    )
    return clause, list(formats)


def _read_embedding_config_from_conn(conn: sqlite3.Connection) -> tuple[str, int] | None:
    try:
        row = conn.execute(
            "SELECT model, dimensions FROM card_embeddings ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    model = str(row[0] or "").strip()
    if not model:
        return None
    try:
        dimensions = int(row[1])
    except (TypeError, ValueError):
        return None
    if dimensions <= 0:
        return None
    return model, dimensions


def _get_embedding_config() -> tuple[str, int]:
    if _EMBED_CONFIG_CACHE is None:
        raise RuntimeError(
            "Embedding configuration missing: table `card_embeddings` must contain valid model and dimensions."
        )
    return _EMBED_CONFIG_CACHE


def _semantic_prefilter_sql(user_query: str, fts_phrase: str, semantic_phrase: str) -> tuple[str, list[Any]]:
    """Build a coarse SQL prefilter from common MTG intent hints.

    This trims candidate rows before FTS/vector ranking.
    """
    text = " ".join([user_query or "", fts_phrase or "", semantic_phrase or ""]).lower()
    clauses: list[str] = []
    args: list[Any] = []

    # Card type hints.
    type_tokens = (
        "creature",
        "instant",
        "sorcery",
        "enchantment",
        "artifact",
        "planeswalker",
        "land",
        "battle",
    )
    for t in type_tokens:
        if re.search(rf"\b{re.escape(t)}s?\b", text):
            clauses.append("LOWER(c.type_line) LIKE ?")
            args.append(f"%{t}%")
            break

    # Color hints (single-color intents only for now to keep it simple/safe).
    color_map = {
        "white": "W",
        "blue": "U",
        "black": "B",
        "red": "R",
        "green": "G",
    }
    colors = [symbol for name, symbol in color_map.items() if re.search(rf"\b{name}\b", text)]
    if len(colors) == 1:
        # raw_json contains: "colors":["R"] and "color_identity":["R"]
        symbol = colors[0]
        clauses.append(
            "("
            "c.raw_json LIKE ? OR c.raw_json LIKE ?"
            ")"
        )
        args.extend(
            [
                f'%"colors":["{symbol}"%',
                f'%"color_identity":["{symbol}"%',
            ]
        )

    if not clauses:
        return "", []
    return " AND ".join(f"({c})" for c in clauses), args


def _pick_display_format(conn: sqlite3.Connection, card_id: str, requested_formats: list[str]) -> str:
    if not requested_formats:
        row = _execute_fetchone(
            conn,
            "SELECT format FROM card_legalities WHERE card_id=? ORDER BY format LIMIT 1",
            (card_id,),
        )
        return str(row[0]) if row else ""
    for fmt in requested_formats:
        row = _execute_fetchone(
            conn,
            """
            SELECT 1 FROM card_legalities
            WHERE card_id=? AND format=? AND status IN ('legal', 'restricted')
            """,
            (card_id, fmt),
        )
        if row:
            return fmt
    return requested_formats[0]


def _rrf_fuse(a: list[str], b: list[str], *, k: int = 60, limit: int = 20) -> list[str]:
    scores: dict[str, float] = {}
    for rank, cid in enumerate(a, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, cid in enumerate(b, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    merged = sorted(scores.keys(), key=lambda cid: -scores[cid])
    return merged[:limit]


async def hybrid_search_cards(
    user_query: str,
    formats: list[str],
    limit: int,
    fts_limit: int = 100,
    vec_limit: int = 100,
    rrf_k: int = 60,
    use_prefilter: bool = True,
) -> list[dict[str, Any]]:
    path = get_database_path()
    if not path.is_file():
        return []

    fts_phrase, semantic_phrase = await expand_user_query(user_query)
    emit_event("llm created hybrid queries")
    fts_match = phrase_to_fts_match(fts_phrase) or phrase_to_fts_match(user_query)
    semantic_clause, semantic_args = (
        _semantic_prefilter_sql(user_query, fts_phrase, semantic_phrase) if use_prefilter else ("", [])
    )

    conn = get_connection(path)
    try:
        fmt_clause, fmt_args = _format_filter_sql(formats)

        fts_ids: list[str] = []
        if fts_match:
            where = "cards_fts MATCH ?"
            args: list[Any] = [fts_match]
            if fmt_clause:
                where = f"({where}) AND {fmt_clause}"
                args.extend(fmt_args)
            if semantic_clause:
                where = f"({where}) AND {semantic_clause}"
                args.extend(semantic_args)
            try:
                rows = _execute_fetchall(
                    conn,
                    f"""
                    SELECT c.oracle_id
                    FROM cards_fts
                    INNER JOIN cards c ON c.rowid = cards_fts.rowid
                    WHERE {where}
                    ORDER BY bm25(cards_fts)
                    LIMIT ?
                    """,
                    (*args, fts_limit),
                    bucket="fts",
                )
                fts_ids = [str(r[0]) for r in rows]
            except sqlite3.Error:
                fts_ids = []

        vec_ids: list[str] = []
        try:
            qvec = await embed_text(semantic_phrase)
            qblob = sqlite_vec.serialize_float32(qvec)
            # vec0 KNN: use `k = ?` (required with JOIN); plain LIMIT is rejected by sqlite-vec.
            vwhere = "cards_vec.embedding MATCH ? AND k = ?"
            vargs: list[Any] = [qblob, vec_limit]
            if fmt_clause:
                vwhere = f"({vwhere}) AND {fmt_clause}"
                vargs.extend(fmt_args)
            if semantic_clause:
                vwhere = f"({vwhere}) AND {semantic_clause}"
                vargs.extend(semantic_args)
            rows = _execute_fetchall(
                conn,
                f"""
                SELECT c.oracle_id
                FROM cards_vec
                INNER JOIN cards c ON c.rowid = cards_vec.rowid
                WHERE {vwhere}
                ORDER BY distance
                """,
                tuple(vargs),
                bucket="vector",
            )
            vec_ids = [str(r[0]) for r in rows]
        except Exception as exc:
            msg = f"Vector search failed: {exc}"
            print(msg)
            emit_event(msg)
            vec_ids = []

        if fts_ids and vec_ids:
            merged_ids = _rrf_fuse(fts_ids, vec_ids, k=rrf_k, limit=limit * 2)
        elif fts_ids:
            merged_ids = fts_ids[: limit * 2]
        elif vec_ids:
            merged_ids = vec_ids[: limit * 2]
        else:
            merged_ids = []

        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for cid in merged_ids:
            if cid in seen:
                continue
            seen.add(cid)
            row = _execute_fetchone(
                conn,
                """
                SELECT
                    name,
                    mana_cost,
                    cmc,
                    type_line,
                    oracle_text,
                    power,
                    toughness,
                    loyalty,
                    raw_json,
                    scryfall_uri
                FROM cards
                WHERE oracle_id=?
                """,
                (cid,),
            )
            if not row:
                continue
            mechanics = _extract_mechanics_fields(row["raw_json"])
            out.append(
                {
                    "name": row["name"] or "",
                    "mana_cost": row["mana_cost"] or "",
                    "cmc": row["cmc"],
                    "type_line": row["type_line"] or "",
                    "oracle_text": row["oracle_text"] or "",
                    "colors": mechanics["colors"],
                    "color_identity": mechanics["color_identity"],
                    "keywords": mechanics["keywords"],
                    "produced_mana": mechanics["produced_mana"],
                    "power": row["power"] or "",
                    "toughness": row["toughness"] or "",
                    "loyalty": row["loyalty"] or "",
                    "defense": mechanics["defense"],
                    "scryfall_url": row["scryfall_uri"] or "",
                    "format": _pick_display_format(conn, cid, formats),
                }
            )
            if len(out) >= limit:
                break
        return out
    finally:
        emit_db_aggregate_summary()
        conn.close()


@function_tool(description_override=SEARCH_MTG_CARDS_TOOL_DESCRIPTION)
async def search_mtg_cards(
    query: str,
    formats: list[str],
    limit: int,
    use_prefilter: bool = True,
) -> list[dict[str, Any]]:
    emit_event("agent called search tool")
    try:
        return await hybrid_search_cards(
            user_query=query,
            formats=formats,
            limit=limit,
            use_prefilter=use_prefilter,
        )
    except Exception as exc:
        msg = f"search_mtg_cards failed gracefully: {exc}"
        print(msg)
        emit_event(msg)
        return []

