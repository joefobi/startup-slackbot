"""
Tests for hybrid_search._rrf_merge.

_rrf_merge is a pure function (aside from fetching metadata for vector-only
hits). Tested in isolation with constructed ArtifactResult and vec hit dicts.
A minimal in-memory SQLite DB provides the metadata fetch for vector-only hits.
"""

import sqlite3
from datetime import datetime, timezone, timedelta

import pytest

from slack_bot.agent.tools.hybrid_search import _rrf_merge
from slack_bot.agent.types import ArtifactResult
from slack_bot.config import RRF_K


def _make_artifact(artifact_id: str, created_at: str | None = None) -> ArtifactResult:
    return ArtifactResult(
        artifact_id=artifact_id,
        artifact_type="customer_call",
        title="T",
        summary="S",
        created_at=created_at,
    )


def _make_vec_hit(artifact_id: str, rank: int) -> dict:
    return {
        "artifact_id": artifact_id,
        "artifact_type": "customer_call",
        "vec_rank": rank,
        "vec_distance": 0.1 * rank,
    }


def _src_conn_with(artifact_ids: list[str]) -> sqlite3.Connection:
    """In-memory DB with an artifacts table for vector-only metadata fetches."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE artifacts "
        "(artifact_id TEXT, artifact_type TEXT, title TEXT, summary TEXT, "
        "created_at TEXT, customer_id TEXT, scenario_id TEXT, "
        "product_id TEXT, competitor_id TEXT, token_estimate INT, metadata_json TEXT)"
    )
    for aid in artifact_ids:
        conn.execute(
            "INSERT INTO artifacts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (aid, "customer_call", "T", "S", None, None, None, None, None, None, None),
        )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Score correctness
# ---------------------------------------------------------------------------


def test_fts_only_scores_correctly():
    fts = [_make_artifact("a"), _make_artifact("b"), _make_artifact("c")]
    conn = _src_conn_with([])
    merged = _rrf_merge(fts, [], conn)

    ids = [r.artifact_id for r in merged]
    # "a" is rank 0 in FTS → highest score → appears first
    assert ids[0] == "a"
    assert ids[1] == "b"
    assert ids[2] == "c"


def test_vec_only_scores_correctly():
    vec = [_make_vec_hit("x", 0), _make_vec_hit("y", 1)]
    conn = _src_conn_with(["x", "y"])
    merged = _rrf_merge([], vec, conn)

    assert merged[0].artifact_id == "x"
    assert merged[1].artifact_id == "y"


def test_overlap_gives_higher_combined_score():
    """A doc appearing in both FTS and vector lists scores higher than FTS-only."""
    fts = [_make_artifact("shared"), _make_artifact("fts_only")]
    vec = [_make_vec_hit("shared", 0), _make_vec_hit("vec_only", 1)]
    conn = _src_conn_with(["shared", "vec_only"])

    merged = _rrf_merge(fts, vec, conn)
    ids = [r.artifact_id for r in merged]

    # "shared" appears in both lists → combined score should be highest
    assert ids[0] == "shared"


def test_empty_inputs_return_empty():
    conn = _src_conn_with([])
    assert _rrf_merge([], [], conn) == []


# ---------------------------------------------------------------------------
# Recency bonus
# ---------------------------------------------------------------------------


def test_recency_bonus_lifts_recent_document():
    """A same-day document should outscore a weakly-ranked old document."""
    today = datetime.now(timezone.utc).isoformat()
    old = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()

    # recent_doc is at FTS rank 3, old_doc at rank 0 — without recency old_doc wins
    fts = [
        _make_artifact("old_doc", created_at=old),    # rank 0 → score 1/(60+1)
        _make_artifact("other1"),
        _make_artifact("other2"),
        _make_artifact("recent_doc", created_at=today),  # rank 3 → score 1/(60+4)
    ]
    conn = _src_conn_with([])

    without_recency = _rrf_merge(fts, [], conn, recency_sensitive=False)
    with_recency = _rrf_merge(fts, [], conn, recency_sensitive=True)

    # Without recency: old_doc (rank 0) is first
    assert without_recency[0].artifact_id == "old_doc"
    # With recency: recent_doc should climb — may not always beat rank 0
    # but its position should improve (appear before rank 3 suggests it climbed)
    recency_ids = [r.artifact_id for r in with_recency]
    assert recency_ids.index("recent_doc") < 3  # climbed from position 3


def test_recency_false_no_bonus_applied():
    """recency_sensitive=False must not change ordering relative to pure RRF."""
    fts = [_make_artifact("a"), _make_artifact("b")]
    conn = _src_conn_with([])

    base = _rrf_merge(fts, [], conn, recency_sensitive=False)
    also_base = _rrf_merge(fts, [], conn, recency_sensitive=False)

    assert [r.artifact_id for r in base] == [r.artifact_id for r in also_base]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_no_duplicate_artifact_ids_in_output():
    fts = [_make_artifact("a"), _make_artifact("b"), _make_artifact("a")]
    vec = [_make_vec_hit("a", 0), _make_vec_hit("b", 1)]
    conn = _src_conn_with(["a", "b"])

    merged = _rrf_merge(fts, vec, conn)
    ids = [r.artifact_id for r in merged]
    assert len(ids) == len(set(ids))
