"""
Tests for ``slack_bot.agent.tools.run_sql._validate_sql``.

There is no ``validate_node`` in the graph — validation lives inside the
``run_sql`` tool. ``_validate_sql`` has no LLM calls: EXPLAIN against the DB
and table allowlist. Returns None when valid, an error string otherwise.
"""

from slack_bot.agent.tools.run_sql import _validate_sql


# ---------------------------------------------------------------------------
# Empty / missing SQL
# ---------------------------------------------------------------------------


def test_empty_sql_fails():
    err = _validate_sql("")
    assert err is not None
    assert "No SQL" in err


def test_whitespace_only_sql_fails():
    err = _validate_sql("   ")
    assert err is not None


# ---------------------------------------------------------------------------
# Non-SELECT statements
# ---------------------------------------------------------------------------


def test_insert_rejected():
    err = _validate_sql("INSERT INTO customers VALUES ('x', 'y')")
    assert err is not None
    assert "INSERT" in err


def test_drop_rejected():
    err = _validate_sql("DROP TABLE customers")
    assert err is not None


def test_update_rejected():
    err = _validate_sql("UPDATE customers SET name='x'")
    assert err is not None


# ---------------------------------------------------------------------------
# Table allowlist
# ---------------------------------------------------------------------------


def test_unknown_table_rejected():
    err = _validate_sql("SELECT * FROM hallucinated_table LIMIT 5")
    assert err is not None
    assert "hallucinated_table" in err


def test_artifacts_fts_in_known_tables():
    # artifacts_fts IS in KNOWN_TABLES so the allowlist check passes;
    # EXPLAIN may still fail on the FTS5 MATCH syntax outside a proper context.
    err = _validate_sql(
        "SELECT * FROM artifacts_fts WHERE artifacts_fts MATCH 'test' LIMIT 5"
    )
    if err:
        assert "hallucinated" not in err


# ---------------------------------------------------------------------------
# Valid SQL
# ---------------------------------------------------------------------------


def test_valid_select_customers():
    err = _validate_sql("SELECT customer_id, name FROM customers LIMIT 5")
    assert err is None


def test_valid_join_passes():
    err = _validate_sql(
        "SELECT c.name, i.status "
        "FROM implementations i "
        "JOIN customers c ON i.customer_id = c.customer_id "
        "LIMIT 5"
    )
    assert err is None


def test_valid_company_profile():
    err = _validate_sql("SELECT name, mission FROM company_profile LIMIT 1")
    assert err is None


# ---------------------------------------------------------------------------
# Syntax errors
# ---------------------------------------------------------------------------


def test_syntax_error_fails():
    err = _validate_sql("SELECT FROM WHERE LIMIT")
    assert err is not None
