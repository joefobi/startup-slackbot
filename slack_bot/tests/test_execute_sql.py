"""
Tests for ``slack_bot.agent.tools.run_sql._execute_sql``.

There is no ``execute_node`` in the graph — execution is inside the ``run_sql``
tool. ``_execute_sql`` calls the LLM to generate SQL, then validates and runs
it. The LLM is mocked to return a known SQL string so tests focus on result
shape, LIMIT injection, and error recovery.
"""

from unittest.mock import MagicMock, patch

from slack_bot.agent.tools.run_sql import _execute_sql
from slack_bot.config import SQL_MAX_ROWS


def _llm_returning(sql: str) -> MagicMock:
    m = MagicMock()
    m.content = sql
    return m


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


def test_execute_returns_list_of_dicts(real_conn):
    with patch("slack_bot.agent.tools.run_sql._startup.CTX") as ctx:
        ctx.conn = real_conn
        ctx.schema_by_table = {}
        ctx.llm.invoke.return_value = _llm_returning(
            "SELECT customer_id, name FROM customers LIMIT 3"
        )
        obs, updates = _execute_sql("get customers", ["customers"], {})

    rows = updates["sql_results"]
    assert isinstance(rows, list)
    assert len(rows) > 0
    assert "customer_id" in rows[0]
    assert "name" in rows[0]


def test_execute_returns_correct_columns(real_conn):
    with patch("slack_bot.agent.tools.run_sql._startup.CTX") as ctx:
        ctx.conn = real_conn
        ctx.schema_by_table = {}
        ctx.llm.invoke.return_value = _llm_returning(
            "SELECT name, region, account_health FROM customers LIMIT 1"
        )
        obs, updates = _execute_sql("get customer columns", ["customers"], {})

    assert set(updates["sql_results"][0].keys()) == {"name", "region", "account_health"}


def test_execute_respects_explicit_limit(real_conn):
    with patch("slack_bot.agent.tools.run_sql._startup.CTX") as ctx:
        ctx.conn = real_conn
        ctx.schema_by_table = {}
        ctx.llm.invoke.return_value = _llm_returning(
            "SELECT customer_id FROM customers LIMIT 3"
        )
        obs, updates = _execute_sql("get customers", ["customers"], {})

    assert len(updates["sql_results"]) <= 3


def test_execute_company_profile_single_row(real_conn):
    with patch("slack_bot.agent.tools.run_sql._startup.CTX") as ctx:
        ctx.conn = real_conn
        ctx.schema_by_table = {}
        ctx.llm.invoke.return_value = _llm_returning(
            "SELECT name, mission FROM company_profile LIMIT 1"
        )
        obs, updates = _execute_sql("get company profile", ["company_profile"], {})

    assert len(updates["sql_results"]) == 1


def test_execute_join_returns_rows(real_conn):
    with patch("slack_bot.agent.tools.run_sql._startup.CTX") as ctx:
        ctx.conn = real_conn
        ctx.schema_by_table = {}
        ctx.llm.invoke.return_value = _llm_returning(
            "SELECT c.name, i.status "
            "FROM implementations i "
            "JOIN customers c ON i.customer_id = c.customer_id "
            "LIMIT 5"
        )
        obs, updates = _execute_sql("join query", ["customers", "implementations"], {})

    rows = updates["sql_results"]
    assert len(rows) > 0
    assert "name" in rows[0]
    assert "status" in rows[0]


# ---------------------------------------------------------------------------
# LIMIT injection
# ---------------------------------------------------------------------------


def test_limit_injected_when_missing(real_conn):
    """SQL without LIMIT should get one appended before execution."""
    with patch("slack_bot.agent.tools.run_sql._startup.CTX") as ctx:
        ctx.conn = real_conn
        ctx.schema_by_table = {}
        ctx.llm.invoke.return_value = _llm_returning(
            "SELECT customer_id FROM customers"
        )
        obs, updates = _execute_sql("get all customers", ["customers"], {})

    assert len(updates.get("sql_results", [])) <= SQL_MAX_ROWS


# ---------------------------------------------------------------------------
# Error recovery (sql_aborted)
# ---------------------------------------------------------------------------


def test_bad_sql_exhausts_retries_and_sets_aborted(real_conn):
    """When every LLM attempt returns invalid SQL, sql_aborted is set after retries."""
    with patch("slack_bot.agent.tools.run_sql._startup.CTX") as ctx:
        ctx.conn = real_conn
        ctx.schema_by_table = {}
        # Consistently return SQL that passes validation but fails at execution
        ctx.llm.invoke.return_value = _llm_returning(
            "SELECT nonexistent_column FROM customers LIMIT 1"
        )
        obs, updates = _execute_sql("bad query", ["customers"], {})

    assert updates.get("sql_aborted") is True
    assert "aborted" in obs.lower()
