"""
vector_index.py — sqlite-vec vector index for artifact embeddings.

Lifecycle:
  - open_vector_db()       called once at startup; creates/opens vectors.db
  - setup_vector_tables()  creates the vec0 virtual table + rowid mapping table
  - maybe_populate()       embeds all artifacts if the index is empty; skips if already done
  - vector_search()        called per query in hybrid_search_node

Embeddings are computed over "{title} {content_text[:25000]}" so that vector similarity
is grounded in the full document body, not just the summary. content_text is truncated
to 25 000 characters (~6 250 tokens) before embedding, leaving headroom under ada-002's
8191-token limit while capturing the vast majority of even the longest transcripts.
The lazy_load_node still fetches full content when needed for answer generation.

Storage: vectors.db is a separate writable SQLite file. The main
synthetic_startup.sqlite stays read-only.

NOTE: If you change CONTENT_TRUNCATE_CHARS or the text construction formula,
delete vectors.db to force repopulation with fresh embeddings.
"""

import logging
import sqlite3
import struct

import sqlite_vec
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

from slack_bot.config import CONTENT_TRUNCATE_CHARS, EMBEDDING_BATCH_SIZE, EMBEDDING_DIM

EMBEDDING_MODEL = "text-embedding-ada-002"


def _serialize(vector: list[float]) -> bytes:
    return sqlite_vec.serialize_float32(vector)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def open_vector_db(vec_db_path: str) -> sqlite3.Connection:
    """Open (or create) the writable vector database and load the sqlite-vec extension."""
    conn = sqlite3.connect(vec_db_path, check_same_thread=False)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def setup_vector_tables(vec_conn: sqlite3.Connection) -> None:
    """Create the vec0 virtual table and artifact_id mapping table if absent."""
    vec_conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS artifact_vecs
        USING vec0(embedding float[{EMBEDDING_DIM}])
    """
    )
    # vec0 uses integer rowids internally; we map them back to artifact_id here.
    vec_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS artifact_vec_map (
            vec_rowid     INTEGER PRIMARY KEY,
            artifact_id   TEXT    NOT NULL,
            artifact_type TEXT
        )
    """
    )
    vec_conn.commit()


def is_populated(vec_conn: sqlite3.Connection) -> bool:
    row = vec_conn.execute("SELECT COUNT(*) FROM artifact_vec_map").fetchone()
    return row[0] > 0


def maybe_populate(
    vec_conn: sqlite3.Connection,
    src_conn: sqlite3.Connection,
    embedder: OpenAIEmbeddings,
) -> None:
    """
    Embed all artifacts and insert into the vector index.
    Skips entirely if the index already has rows (idempotent across restarts).
    """
    if is_populated(vec_conn):
        count = vec_conn.execute("SELECT COUNT(*) FROM artifact_vec_map").fetchone()[0]
        logger.info("Vector index already populated (%d artifacts). Skipping.", count)
        return

    rows = src_conn.execute(
        "SELECT artifact_id, artifact_type, title, content_text FROM artifacts"
    ).fetchall()

    logger.info("Building vector index for %d artifacts...", len(rows))

    for i in range(0, len(rows), EMBEDDING_BATCH_SIZE):
        batch = rows[i : i + EMBEDDING_BATCH_SIZE]
        # Embed title + truncated content_text so vector similarity reflects the
        # full document body. content_text may be None for sparse artifacts.
        texts = [f"{r[2]} {(r[3] or '')[:CONTENT_TRUNCATE_CHARS]}" for r in batch]
        vectors = embedder.embed_documents(texts)

        for row, vec in zip(batch, vectors):
            artifact_id, artifact_type = row[0], row[1]
            cursor = vec_conn.execute(
                "INSERT INTO artifact_vecs(embedding) VALUES (?)",
                [_serialize(vec)],
            )
            vec_conn.execute(
                "INSERT INTO artifact_vec_map(vec_rowid, artifact_id, artifact_type) "
                "VALUES (?, ?, ?)",
                [cursor.lastrowid, artifact_id, artifact_type],
            )

        vec_conn.commit()
        logger.info("  embedded %d / %d", min(i + EMBEDDING_BATCH_SIZE, len(rows)), len(rows))

    logger.info("Vector index ready.")


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def vector_search(
    vec_conn: sqlite3.Connection,
    query_vector: list[float],
    artifact_types: list[str] | None = None,
    top_k: int = 15,
) -> list[dict]:
    """
    kNN search against the vector index.
    Returns [{artifact_id, artifact_type, vec_distance, vec_rank}, ...].

    Fetches more than needed (top_k) then filters by artifact_type so the
    type filter doesn't starve results when only a few types are relevant.
    """
    blob = _serialize(query_vector)

    vec_rows = vec_conn.execute(
        """
        SELECT rowid, distance
        FROM artifact_vecs
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
        """,
        [blob, top_k],
    ).fetchall()

    results: list[dict] = []
    for global_rank, (rowid, distance) in enumerate(vec_rows):
        map_row = vec_conn.execute(
            "SELECT artifact_id, artifact_type FROM artifact_vec_map WHERE vec_rowid = ?",
            [rowid],
        ).fetchone()
        if not map_row:
            continue
        artifact_id, artifact_type = map_row
        if artifact_types and artifact_type not in artifact_types:
            continue
        results.append(
            {
                "artifact_id": artifact_id,
                "artifact_type": artifact_type,
                "vec_distance": distance,
                "vec_rank": global_rank,
            }
        )

    return results
