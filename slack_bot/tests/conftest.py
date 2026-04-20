"""
conftest.py — shared fixtures and module-level mock injection.

slack_bot.startup is injected into sys.modules BEFORE any node module
imports it, preventing build_startup_context() from running during tests.
slack_bot.vector_index is also mocked to avoid the sqlite_vec C extension
requirement in the test environment.

These injections must remain at module level (not inside fixtures) so they
fire before pytest imports any test file that does `from slack_bot... import`.
"""

import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Module-level mock injection — must happen before any slack_bot.* import
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.parent  # take_home_lc/
_DB_PATH = _PROJECT_ROOT / "synthetic_startup.sqlite"

# Real read-only connection shared across all tests that need the DB.
_real_conn = sqlite3.connect(f"file:{_DB_PATH}?mode=ro", uri=True)

KNOWN_TABLES = frozenset(
    {
        "artifacts",
        "artifacts_fts",
        "customers",
        "scenarios",
        "implementations",
        "competitors",
        "employees",
        "products",
        "company_profile",
    }
)

# CTX mock — nodes reference CTX.conn, CTX.llm, CTX.schema_by_table, etc.
mock_ctx = MagicMock()
mock_ctx.conn = _real_conn
mock_ctx.schema_by_table = {}
mock_ctx.react_agent_graph = MagicMock()
mock_ctx.react_agent_graph.invoke = MagicMock(
    return_value={
        "messages": [],
        "fts_results": [],
        "sql_results": [],
        "bm25_hits_debug": None,
        "vec_hits_debug": None,
        "full_artifact": None,
        "sql_aborted": False,
        "generated_sql": "",
    }
)

# Mock startup module
_mock_startup = MagicMock()
_mock_startup.CTX = mock_ctx
_mock_startup.KNOWN_TABLES = KNOWN_TABLES
_mock_startup.build_sql_prompt = lambda schema, few_shots: "-- mock system prompt"

# Mock vector_index to avoid requiring the sqlite_vec C extension.
_mock_vector_index = MagicMock()
_mock_vector_index.vector_search = MagicMock(return_value=[])

sys.modules["slack_bot.startup"] = _mock_startup
sys.modules["slack_bot.vector_index"] = _mock_vector_index
sys.modules["sqlite_vec"] = MagicMock()

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def real_conn():
    """Read-only connection to the real SQLite DB."""
    return _real_conn


@pytest.fixture
def ctx():
    """The shared mock CTX. Reconfigure attributes per-test as needed."""
    return mock_ctx


@pytest.fixture
def make_artifact():
    """Factory for minimal ArtifactResult instances."""
    from slack_bot.agent.types import ArtifactResult

    def _make(artifact_id, created_at=None, token_estimate=None, **kwargs):
        return ArtifactResult(
            artifact_id=artifact_id,
            artifact_type=kwargs.pop("artifact_type", "customer_call"),
            title=kwargs.pop("title", "Test title"),
            summary=kwargs.pop("summary", "Test summary"),
            created_at=created_at,
            token_estimate=token_estimate,
            **kwargs,
        )

    return _make


@pytest.fixture
def make_state():
    """Factory for minimal unified-graph state dicts."""

    def _make(**kwargs):
        base = {
            "messages": [],
            "summary": "",
            "recent_turns": [],
            "question": "",
            "fts_results": [],
            "sql_results": [],
            "final_answer": "",
            "draft_answer": "",
            "confidence": "",
        }
        base.update(kwargs)
        return base

    return _make
