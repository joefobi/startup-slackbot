"""
tools/run_sql.py — validated SQL execution and the ``run_sql`` tool for ``create_agent``.

Uses ``import slack_bot.startup as _startup`` and ``_startup.CTX`` inside
functions so loading this module during ``build_startup_context`` does not
require ``CTX`` to exist yet.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Annotated, Any

import slack_bot.startup as _startup
from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from slack_bot.agent.utils import _extract_table_names, _strip_fences
from slack_bot.config import SQL_MAX_RETRIES, SQL_MAX_ROWS
from slack_bot.few_shots.sql_plan import FEW_SHOT_EXAMPLES as SQL_FEW_SHOTS
from slack_bot.prompts.sql import CORRECT_USER_PROMPT, PLAN_USER_PROMPT
from slack_bot.prompts.sql import build_system_prompt as build_sql_prompt
from slack_bot.startup import KNOWN_TABLES

_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Validation + execution helpers
# ---------------------------------------------------------------------------


def _validate_sql(sql: str) -> str | None:
    """Validate that SQL is a safe read-only query against known tables.

    Runs ``EXPLAIN``, enforces ``SELECT``-only, and checks table names against
    ``KNOWN_TABLES``.

    Args:
        sql (str): Candidate SQL string.

    Returns:
        str | None: Human-readable error, or ``None`` if valid.

    """
    sql = sql.strip()
    if not sql:
        return "No SQL was generated."
    if sql.split()[0].upper() != "SELECT":
        return f"Only SELECT statements are allowed; got {sql.split()[0].upper()}."
    try:
        _startup.CTX.conn.execute(f"EXPLAIN {sql}")  # noqa: S608
    except sqlite3.OperationalError as exc:
        return str(exc)
    referenced = _extract_table_names(sql)
    known_lower = {t.lower() for t in KNOWN_TABLES}
    unknown = referenced - known_lower
    if unknown:
        return f"Query references unknown table(s): {', '.join(sorted(unknown))}."
    return None


def _execute_sql(
    sql_intent: str,
    relevant_tables: list[str],
    fts_ids: dict[str, list[str]],
) -> tuple[str, dict]:
    """Generate SQL from intent, validate, execute, and retry on failure.

    On success returns rows and ``generated_sql``; on exhaustion sets
    ``sql_aborted``.

    Args:
        sql_intent (str): Natural-language description of the needed data.
        relevant_tables (list[str]): Tables to expose in the schema block;
            empty falls back to all known tables.
        fts_ids (dict[str, list[str]]): Entity id lists (e.g. from hybrid_search)
            passed into the planner prompt.

    Returns:
        tuple[str, dict]: ``(observation, updates)`` with optional keys
            ``sql_results``, ``generated_sql``, ``sql_aborted``.

    """
    relevant = relevant_tables or list(_startup.CTX.schema_by_table.keys())
    schema_block = "\n".join(
        _startup.CTX.schema_by_table[t]
        for t in relevant
        if t in _startup.CTX.schema_by_table
    ) or "\n".join(_startup.CTX.schema_by_table.values())
    system_prompt = build_sql_prompt(schema_block, SQL_FEW_SHOTS)

    fts_ids_str = str(fts_ids) if fts_ids else "(none)"
    plan_prompt = PLAN_USER_PROMPT.format(sql_intent=sql_intent, fts_ids=fts_ids_str)

    result = _startup.CTX.llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=plan_prompt),
    ])
    sql = _strip_fences(result.content)

    last_error: str | None = None

    for attempt in range(SQL_MAX_RETRIES + 1):
        error = _validate_sql(sql)

        if not error:
            if not _LIMIT_RE.search(sql):
                sql = sql.rstrip(";").rstrip() + f" LIMIT {SQL_MAX_ROWS}"
            try:
                cursor = _startup.CTX.conn.execute(sql)
                cols = [d[0] for d in cursor.description]
                rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
                if not rows:
                    observation = f"SQL executed successfully but returned no rows.\nSQL used: {sql}"
                else:
                    preview = json.dumps(rows[:5], indent=2, default=str)
                    tail = f"\n... ({len(rows)} total rows)" if len(rows) > 5 else ""
                    observation = f"{len(rows)} row(s) returned:\n{preview}{tail}"
                return observation, {"sql_results": rows, "generated_sql": sql, "sql_aborted": False}
            except sqlite3.OperationalError as exc:
                error = str(exc)

        last_error = error
        if attempt == SQL_MAX_RETRIES:
            break

        correct_prompt = CORRECT_USER_PROMPT.format(sql=sql, error=error)
        result = _startup.CTX.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=correct_prompt),
        ])
        sql = _strip_fences(result.content)

    observation = f"SQL aborted after {SQL_MAX_RETRIES} retries. Last error: {last_error}"
    return observation, {"sql_aborted": True, "generated_sql": sql}


def _fts_ids_from_state(
    state: dict[str, Any] | None,
    fts_customer_ids: list[str],
    fts_competitor_ids: list[str],
    fts_product_ids: list[str],
    fts_scenario_ids: list[str],
) -> dict[str, list[str]]:
    """Merge explicit FTS id args with ids mined from checkpointed ``fts_results``.

    Args:
        state (dict[str, Any] | None): Agent runtime state, if present.
        fts_customer_ids (list[str]): Caller-supplied customer ids.
        fts_competitor_ids (list[str]): Caller-supplied competitor ids.
        fts_product_ids (list[str]): Caller-supplied product ids.
        fts_scenario_ids (list[str]): Caller-supplied scenario ids.

    Returns:
        dict[str, list[str]]: Keys such as ``fts_customer_ids`` for the SQL
            planner prompt.

    """
    fts_ids: dict[str, list[str]] = {}
    if fts_customer_ids:
        fts_ids["fts_customer_ids"] = fts_customer_ids
    if fts_competitor_ids:
        fts_ids["fts_competitor_ids"] = fts_competitor_ids
    if fts_product_ids:
        fts_ids["fts_product_ids"] = fts_product_ids
    if fts_scenario_ids:
        fts_ids["fts_scenario_ids"] = fts_scenario_ids
    if fts_ids or not state:
        return fts_ids

    accumulated = state.get("fts_results") or []
    if not accumulated:
        return fts_ids

    def col(name: str) -> list[str]:
        return list({row.get(name) for row in accumulated if row.get(name)})

    cid = col("customer_id")
    if cid:
        fts_ids["fts_customer_ids"] = cid
    comp = col("competitor_id")
    if comp:
        fts_ids["fts_competitor_ids"] = comp
    pid = col("product_id")
    if pid:
        fts_ids["fts_product_ids"] = pid
    sid = col("scenario_id")
    if sid:
        fts_ids["fts_scenario_ids"] = sid
    return fts_ids


@tool
def run_sql(
    sql_intent: str,
    relevant_tables: list[str],
    fts_customer_ids: list[str],
    fts_competitor_ids: list[str],
    fts_product_ids: list[str],
    fts_scenario_ids: list[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
    runtime: ToolRuntime,
) -> Command:
    """Generate, validate, and run a ``SELECT`` against structured tables.

    Use after ``hybrid_search`` when you have entity ids to narrow the query.

    Args:
        sql_intent (str): One-sentence retrieval goal for the SQL LLM.
        relevant_tables (list[str]): Tables to include in the dynamic schema.
        fts_customer_ids (list[str]): Customer ids (often from hybrid_search).
        fts_competitor_ids (list[str]): Competitor ids from search.
        fts_product_ids (list[str]): Product ids from search.
        fts_scenario_ids (list[str]): Scenario ids from search.
        tool_call_id (str): Injected id for the tool result message.
        runtime (ToolRuntime): Graph state for merging prior ``fts_results``.

    Returns:
        Command: Updates ``messages``, ``sql_results``, ``generated_sql``, and
            ``sql_aborted``.

    """
    state = runtime.state if runtime.state else None
    fts_ids = _fts_ids_from_state(
        state,
        fts_customer_ids,
        fts_competitor_ids,
        fts_product_ids,
        fts_scenario_ids,
    )
    observation, updates = _execute_sql(sql_intent, relevant_tables, fts_ids)
    return Command(
        update={
            "messages": [
                ToolMessage(content=observation, tool_call_id=tool_call_id),
            ],
            "sql_results": updates.get("sql_results", []),
            "generated_sql": updates.get("generated_sql", ""),
            "sql_aborted": updates.get("sql_aborted", False),
        }
    )
