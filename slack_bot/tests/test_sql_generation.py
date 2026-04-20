"""
Tests for SQL generation quality.

SQL strings are hand-written to represent what the LLM is expected to generate
for common query patterns. Each string is run through _validate_sql and then
executed directly against the real DB. Tests assert that:
  - queries the LLM is supposed to generate pass validation
  - those queries return data (not empty results)
  - common hallucination patterns are caught by _validate_sql

No LLM calls are made.
"""

import re

import pytest

from slack_bot.agent.tools.run_sql import _validate_sql
from slack_bot.config import SQL_MAX_ROWS
from slack_bot.startup import CTX

_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)


def _run(sql: str) -> tuple[str | None, list[dict]]:
    """
    Validate then directly execute sql. Returns (error, rows).

    error is None when the SQL is valid; rows is empty on any failure.
    """
    error = _validate_sql(sql)
    if error:
        return error, []

    if not _LIMIT_RE.search(sql):
        sql = sql.rstrip(";").rstrip() + f" LIMIT {SQL_MAX_ROWS}"

    try:
        cursor = CTX.conn.execute(sql)
        cols = [d[0] for d in cursor.description]
        rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
        return None, rows
    except Exception as exc:
        return str(exc), []


# ---------------------------------------------------------------------------
# Queries that must pass validation AND return data
# ---------------------------------------------------------------------------


def test_customers_basic():
    error, rows = _run("SELECT customer_id, name, account_health FROM customers LIMIT 5")
    assert error is None
    assert len(rows) > 0
    assert "customer_id" in rows[0]


def test_at_risk_customers():
    error, rows = _run(
        "SELECT c.name, c.account_health, i.status "
        "FROM implementations i "
        "JOIN customers c ON i.customer_id = c.customer_id "
        "WHERE LOWER(i.status) LIKE '%at risk%' "
        "LIMIT 20"
    )
    assert error is None
    assert len(rows) > 0


def test_employees_department_filter():
    error, rows = _run(
        "SELECT full_name, title, department, region "
        "FROM employees "
        "WHERE LOWER(department) LIKE '%customer success%' "
        "LIMIT 20"
    )
    assert error is None
    assert len(rows) > 0


def test_company_profile_single_row():
    error, rows = _run(
        "SELECT name, mission, differentiation FROM company_profile LIMIT 1"
    )
    assert error is None
    assert len(rows) == 1


def test_competitors_full():
    error, rows = _run(
        "SELECT name, segment, pricing_position FROM competitors ORDER BY name LIMIT 20"
    )
    assert error is None
    assert len(rows) > 0


def test_json_each_risks():
    """json_each expansion of risks_json array must return rows."""
    error, rows = _run(
        "SELECT c.name, r.value AS risk_item "
        "FROM implementations i "
        "JOIN customers c ON i.customer_id = c.customer_id, "
        "json_each(i.risks_json) r "
        "LIMIT 10"
    )
    assert error is None
    assert len(rows) > 0
    assert "risk_item" in rows[0]


def test_json_extract_blueprint_valid_key():
    """json_extract with a real blueprint_json key should work."""
    error, rows = _run(
        "SELECT s.trigger_event, json_extract(s.blueprint_json, '$.summary') AS bp_summary "
        "FROM scenarios s "
        "LIMIT 5"
    )
    assert error is None
    assert len(rows) > 0


def test_fts_ids_where_in():
    """WHERE customer_id IN (...) pattern must return rows for real IDs."""
    cursor = CTX.conn.execute("SELECT customer_id FROM customers LIMIT 1")
    real_id = cursor.fetchone()[0]

    error, rows = _run(
        f"SELECT name, account_health FROM customers "
        f"WHERE customer_id IN ('{real_id}') LIMIT 5"
    )
    assert error is None
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# Hallucination patterns that must be caught
# ---------------------------------------------------------------------------


def test_hallucinated_table_rejected():
    error, _ = _run("SELECT * FROM renewal_proof_plans LIMIT 5")
    assert error is not None
    assert "renewal_proof_plans" in error


def test_hallucinated_join_table_rejected():
    error, _ = _run(
        "SELECT c.name, p.plan_text "
        "FROM customers c JOIN proof_plans p ON c.customer_id = p.customer_id "
        "LIMIT 5"
    )
    assert error is not None


def test_non_select_rejected():
    error, _ = _run("DELETE FROM customers WHERE 1=1")
    assert error is not None


def test_invalid_json_key_returns_null_not_error():
    """
    A hallucinated blueprint_json key passes validation (SQLite does not
    validate JSON paths at EXPLAIN time) but json_extract returns NULL — the
    query runs without crashing, just returns nulls.
    """
    error, rows = _run(
        "SELECT json_extract(blueprint_json, '$.hallucinated_key') AS val "
        "FROM scenarios LIMIT 3"
    )
    assert error is None
    assert all(row["val"] is None for row in rows)
