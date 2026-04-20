"""
tools/hybrid_search.py — BM25 + vector retrieval and the ``hybrid_search`` tool.

Runs BM25 (FTS5) + kNN vector search, merges with Reciprocal Rank Fusion,
and exposes a LangChain ``Command`` tool for ``create_agent``. Uses
``import slack_bot.startup as _startup`` and ``_startup.CTX`` inside functions
so this module can load during ``build_startup_context`` without touching
``CTX`` before it exists.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone
from typing import Annotated, Any

import slack_bot.startup as _startup
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from slack_bot.agent.types import ArtifactResult
from slack_bot.config import (
    FTS_PER_QUERY,
    LAZY_LOAD_MAX_ARTIFACTS,
    MAX_FTS_QUERIES,
    RECENCY_HALF_LIFE_DAYS,
    RECENCY_WEIGHT,
    RRF_FINAL_LIMIT,
    RRF_K,
    VEC_TOP_K,
)
from slack_bot.vector_index import vector_search


# ---------------------------------------------------------------------------
# BM25 + RRF helpers
# ---------------------------------------------------------------------------


def _recency_bonus(created_at_str: str | None) -> float:
    """Score how recent a document is for RRF tie-breaking.

    Applies exponential decay from ``RECENCY_HALF_LIFE_DAYS`` when
    ``recency_sensitive`` search is enabled.

    Args:
        created_at_str (str | None): ISO-8601 timestamp, or ``None``.

    Returns:
        float: Weight in ``[0, RECENCY_WEIGHT]``, or ``0.0`` if unparsable.

    """
    if not created_at_str:
        return 0.0
    try:
        created = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        days = max((datetime.now(timezone.utc) - created).days, 0)
        return RECENCY_WEIGHT * math.exp(-days / RECENCY_HALF_LIFE_DAYS)
    except (ValueError, TypeError):
        return 0.0


def _run_fts(queries: list[str], artifact_types: list[str]) -> list[ArtifactResult]:
    """Search ``artifacts_fts`` for each query phrase (BM25).

    Caps phrases at ``MAX_FTS_QUERIES`` and skips malformed MATCH inputs.

    Args:
        queries (list[str]): Keyword phrases from the agent.
        artifact_types (list[str]): Allowed ``artifact_type`` values; empty
            allows all types.

    Returns:
        list[ArtifactResult]: Deduplicated hits in encounter order.

    """
    seen: set[str] = set()
    results: list[ArtifactResult] = []

    for query in queries[:MAX_FTS_QUERIES]:
        if not query.strip():
            continue

        sql = """
            SELECT
                a.artifact_id, a.artifact_type, a.title, a.summary,
                a.created_at, a.customer_id, a.scenario_id,
                a.product_id, a.competitor_id,
                a.token_estimate, a.metadata_json, f.rank
            FROM artifacts_fts f
            JOIN artifacts a ON a.artifact_id = f.artifact_id
            WHERE artifacts_fts MATCH ?
        """
        params: list = [query]

        if artifact_types:
            placeholders = ",".join("?" * len(artifact_types))
            sql += f" AND a.artifact_type IN ({placeholders})"
            params.extend(artifact_types)

        sql += f" ORDER BY rank LIMIT {FTS_PER_QUERY}"

        try:
            cursor = _startup.CTX.conn.execute(sql, params)
            cols = [d[0] for d in cursor.description]
            for row in cursor.fetchall():
                d = dict(zip(cols, row))
                if d["artifact_id"] not in seen:
                    seen.add(d["artifact_id"])
                    results.append(ArtifactResult(**d))
        except sqlite3.OperationalError:
            continue

    return results


def _rrf_merge(
    fts_results: list[ArtifactResult],
    vec_results: list[dict],
    src_conn: sqlite3.Connection,
    recency_sensitive: bool = False,
) -> list[ArtifactResult]:
    """Merge BM25 and vector rankings with reciprocal rank fusion (RRF).

    Fetches artifact rows from ``src_conn`` for vector-only IDs when needed.

    Args:
        fts_results (list[ArtifactResult]): BM25-ranked artifacts.
        vec_results (list[dict]): sqlite-vec hits with ``artifact_id`` and
            ``vec_rank``.
        src_conn (sqlite3.Connection): Read-only main DB.
        recency_sensitive (bool, optional): Add ``_recency_bonus`` to scores.
            Defaults to False.

    Returns:
        list[ArtifactResult]: Top ``RRF_FINAL_LIMIT`` artifacts by fused score.

    """
    scores: dict[str, float] = {}

    for rank, hit in enumerate(fts_results):
        scores[hit.artifact_id] = scores.get(hit.artifact_id, 0.0) + 1.0 / (
            RRF_K + rank + 1
        )

    for hit in vec_results:
        aid = hit["artifact_id"]
        rank = hit["vec_rank"]
        scores[aid] = scores.get(aid, 0.0) + 1.0 / (RRF_K + rank + 1)

    if recency_sensitive:
        created_at_map: dict[str, str | None] = {
            r.artifact_id: r.created_at for r in fts_results
        }
        vec_only = [aid for aid in scores if aid not in created_at_map]
        if vec_only:
            placeholders = ",".join("?" * len(vec_only))
            rows = src_conn.execute(
                f"SELECT artifact_id, created_at FROM artifacts "
                f"WHERE artifact_id IN ({placeholders})",
                vec_only,
            ).fetchall()
            for artifact_id, created_at in rows:
                created_at_map[artifact_id] = created_at
        for aid in scores:
            scores[aid] += _recency_bonus(created_at_map.get(aid))

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:RRF_FINAL_LIMIT]

    fts_by_id = {r.artifact_id: r for r in fts_results}
    merged: list[ArtifactResult] = []

    for artifact_id in sorted_ids:
        if artifact_id in fts_by_id:
            merged.append(fts_by_id[artifact_id])
        else:
            row = src_conn.execute(
                """
                SELECT artifact_id, artifact_type, title, summary,
                       created_at, customer_id, scenario_id,
                       product_id, competitor_id,
                       token_estimate, metadata_json
                FROM artifacts WHERE artifact_id = ?
                """,
                (artifact_id,),
            ).fetchone()
            if row:
                cols = [
                    "artifact_id", "artifact_type", "title", "summary",
                    "created_at", "customer_id", "scenario_id",
                    "product_id", "competitor_id", "token_estimate", "metadata_json",
                ]
                merged.append(ArtifactResult(**dict(zip(cols, row))))

    return merged


# ---------------------------------------------------------------------------
# Execution function
# ---------------------------------------------------------------------------


def _execute_hybrid_search(
    queries: list[str],
    artifact_types: list[str],
    recency_sensitive: bool,
    question: str,
) -> tuple[str, dict]:
    """Run hybrid retrieval and build the tool observation plus state updates.

    Args:
        queries (list[str]): BM25 phrases (up to ``MAX_FTS_QUERIES`` used).
        artifact_types (list[str]): Doc-type filter; empty means all types.
        recency_sensitive (bool): Whether to blend in recency when merging ranks.
        question (str): User question text for the embedding query.

    Returns:
        tuple[str, dict]: ``(observation, updates)`` where ``updates`` may
            include ``fts_results``, debug lists, and ``full_artifact``.

    """
    fts_hits = _run_fts(queries, artifact_types)

    query_vector = (
        _startup.CTX.embedder.embed_query(question)
        if question
        else _startup.CTX.embedder.embed_query(" ".join(queries[:3]))
    )
    vec_hits = vector_search(
        _startup.CTX.vec_conn,
        query_vector,
        artifact_types=artifact_types or None,
        top_k=VEC_TOP_K,
    )

    merged = _rrf_merge(
        fts_hits, vec_hits, _startup.CTX.conn, recency_sensitive=recency_sensitive
    )

    if not merged:
        return "No documents found matching the search queries.", {
            "fts_results": [],
            "bm25_hits_debug": None,
            "vec_hits_debug": None,
            "full_artifact": None,
        }

    # Fetch full content_text for the top LAZY_LOAD_MAX_ARTIFACTS results.
    top_ids = [r.artifact_id for r in merged[:LAZY_LOAD_MAX_ARTIFACTS]]
    placeholders = ",".join("?" * len(top_ids))
    content_rows = _startup.CTX.conn.execute(
        f"SELECT artifact_id, content_text FROM artifacts WHERE artifact_id IN ({placeholders})",
        top_ids,
    ).fetchall()
    content_map: dict[str, str] = {aid: text for aid, text in content_rows if text}

    lines: list[str] = []
    for r in merged:
        id_parts: list[str] = []
        if r.customer_id:
            id_parts.append(f"customer_id: {r.customer_id}")
        if r.competitor_id:
            id_parts.append(f"competitor_id: {r.competitor_id}")
        if r.product_id:
            id_parts.append(f"product_id: {r.product_id}")
        if r.scenario_id:
            id_parts.append(f"scenario_id: {r.scenario_id}")
        if r.token_estimate:
            id_parts.append(f"tokens: {r.token_estimate}")
        id_str = f"\n  {', '.join(id_parts)}" if id_parts else ""
        block = f"[{r.artifact_id}] {r.artifact_type} — {r.title}{id_str}\n  {r.summary}"
        if r.artifact_id in content_map:
            block += f"\n  --- full text ---\n  {content_map[r.artifact_id]}"
        lines.append(block)

    observation = f"{len(merged)} document(s) found:\n\n" + "\n\n".join(lines)

    full_artifact = (
        "\n\n---\n\n".join(
            f"[{aid}]\n{content_map[aid]}" for aid in top_ids if aid in content_map
        )
        or None
    )

    bm25_debug = [
        {"artifact_id": r.artifact_id, "title": r.title,
         "artifact_type": r.artifact_type, "rank": r.rank}
        for r in fts_hits
    ]
    vec_debug = [
        {"artifact_id": h["artifact_id"], "artifact_type": h["artifact_type"],
         "vec_rank": h["vec_rank"], "vec_distance": h["vec_distance"]}
        for h in vec_hits
    ]

    return observation, {
        "fts_results": merged,
        "bm25_hits_debug": bm25_debug or None,
        "vec_hits_debug": vec_debug or None,
        "full_artifact": full_artifact,
    }


def _artifact_updates(updates: dict[str, Any]) -> dict[str, Any]:
    """Make hybrid_search state updates JSON-serializable for checkpointing.

    Args:
        updates (dict[str, Any]): Raw updates from ``_execute_hybrid_search``.

    Returns:
        dict[str, Any]: Same keys; ``fts_results`` entries as dicts.

    """

    out: dict[str, Any] = {}
    for key, val in updates.items():
        if key == "fts_results" and val:
            out[key] = [r.model_dump(mode="json") for r in val]
        else:
            out[key] = val
    return out


@tool
def hybrid_search(
    queries: list[str],
    artifact_types: list[str],
    recency_sensitive: bool,
    tool_call_id: Annotated[str, InjectedToolCallId],
    runtime: ToolRuntime,
) -> Command:
    """Retrieve documents via BM25 + vector search merged with RRF.

    Prefer broad synonym-heavy query lists; artifact types narrow the corpus.
    Relational IDs in the observation should feed ``run_sql`` filters.

    Args:
        queries (list[str]): Multi-phrase BM25 queries (long phrases help FTS).
        artifact_types (list[str]): Doc-type filter; empty includes all types.
        recency_sensitive (bool): Weight newer docs when comparing ranks.
        tool_call_id (str): Injected tool call id for the ``ToolMessage``.
        runtime (ToolRuntime): Provides graph ``state`` (for ``question``).

    Returns:
        Command: LangGraph command updating ``messages`` and retrieval fields.

    """
    question = (runtime.state.get("question") or "") if runtime.state else ""
    observation, updates = _execute_hybrid_search(
        queries, artifact_types, recency_sensitive, question
    )
    payload = _artifact_updates(updates)
    return Command(
        update={
            "messages": [
                ToolMessage(content=observation, tool_call_id=tool_call_id),
            ],
            **payload,
        }
    )
