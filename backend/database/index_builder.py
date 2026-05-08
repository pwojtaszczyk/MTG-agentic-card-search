from __future__ import annotations

import sqlite3


def _sqlite_vec_loaded(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("SELECT vec_version()").fetchone()
        return True
    except sqlite3.Error:
        return False


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','virtual') AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def ensure_fts5(conn: sqlite3.Connection) -> None:
    if _table_exists(conn, "cards_fts"):
        return
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


def rebuild_fts5(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS cards_fts")
    ensure_fts5(conn)


def embedding_dimensions(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT dimensions FROM card_embeddings LIMIT 1").fetchone()
    return int(row[0]) if row else None


def ensure_vec0(conn: sqlite3.Connection) -> bool:
    if not _sqlite_vec_loaded(conn):
        return False
    dim = embedding_dimensions(conn)
    if not dim:
        return False
    if _table_exists(conn, "cards_vec"):
        return True
    conn.execute(f"CREATE VIRTUAL TABLE cards_vec USING vec0(embedding float[{dim}])")
    rows = conn.execute(
        """
        SELECT c.rowid, e.embedding
        FROM cards c
        INNER JOIN card_embeddings e ON e.card_id = c.oracle_id
        """
    ).fetchall()
    conn.executemany(
        "INSERT INTO cards_vec(rowid, embedding) VALUES (?, ?)",
        rows,
    )
    return True


def rebuild_vec0(conn: sqlite3.Connection) -> bool:
    if not _sqlite_vec_loaded(conn):
        return False
    dim = embedding_dimensions(conn)
    if not dim:
        return False
    conn.execute("DROP TABLE IF EXISTS cards_vec")
    conn.execute(f"CREATE VIRTUAL TABLE cards_vec USING vec0(embedding float[{dim}])")
    rows = conn.execute(
        """
        SELECT c.rowid, e.embedding
        FROM cards c
        INNER JOIN card_embeddings e ON e.card_id = c.oracle_id
        """
    ).fetchall()
    conn.executemany(
        "INSERT INTO cards_vec(rowid, embedding) VALUES (?, ?)",
        rows,
    )
    return True


def ensure_search_indices(conn: sqlite3.Connection) -> None:
    ensure_fts5(conn)
    ensure_vec0(conn)


def rebuild_search_indices(conn: sqlite3.Connection) -> None:
    rebuild_fts5(conn)
    rebuild_vec0(conn)
