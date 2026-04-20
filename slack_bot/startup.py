"""
startup.py — runs once before the FastAPI app begins serving requests.

Everything in this module is computed once at import time and exposed as
module-level singletons. Every LangGraph node that needs the DB connection,
schema, or system prompt reads from CTX — never re-initialises per request.
"""

import os
import sqlite3
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from slack_bot.agent.types import UnifiedGraphState
from slack_bot.prompts.agent import build_system_prompt as _build_agent_system_prompt
from slack_bot.prompts.answer import SYSTEM_PROMPT as _ANSWER_SYSTEM_PROMPT
from slack_bot.prompts.synthesize import SYSTEM_PROMPT as _SYNTHESIZE_SYSTEM_PROMPT
from slack_bot.prompts.update_summary import SYSTEM_PROMPT as _SUMMARY_SYSTEM_PROMPT
from slack_bot.vector_index import (
    maybe_populate,
    open_vector_db,
    setup_vector_tables,
)

load_dotenv()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWN_TABLES: frozenset[str] = frozenset(
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

# FTS internal table suffixes that PRAGMA table_list / sqlite_master returns
# but which are not user-queryable tables we want in the schema block.
_FTS_INTERNAL_SUFFIXES = (
    "_content",
    "_segdir",
    "_segments",
    "_stat",
    "_docsize",
    "_config",
    "_data",
    "_idx",
)

# Bounded-value columns — enum-like, values queried from the DB at startup.
# Filter with = or IN only, never LIKE or invented values.
_BOUNDED_COLUMNS: dict[str, list[str]] = {
    "customers": [
        "account_health",
        "crm_stage",
        "region",
        "country",
        "size_band",
        "industry",
    ],
    "scenarios": ["trigger_event"],
    "implementations": ["deployment_model"],
    "products": ["category", "pricing_model"],
    "competitors": ["pricing_position", "segment"],
    "employees": ["department", "management_level"],
}

# Free-prose columns — LLM instruction only, no value enumeration possible.
_FREE_PROSE_COLUMNS: dict[str, dict[str, str]] = {
    "customers": {
        "name": "free-prose — do NOT filter with LIKE unless the question names a specific account",
        "tech_stack_summary": "free-prose — do NOT filter with LIKE",
        "notes": "free-prose — do NOT filter with LIKE",
    },
    "scenarios": {
        "pain_point": "free-prose — do NOT filter with LIKE; use fts_ids to scope by topic instead",
        "scenario_summary": "free-prose — do NOT filter with LIKE",
    },
    "implementations": {
        "status": "free-prose (many variants) — use LOWER(status) LIKE '%keyword%' only for status",
    },
    "products": {
        "description": "free-prose — do NOT filter with LIKE",
    },
}

# Inline JSON shape annotations appended after each table's CREATE TABLE line.
# These are derived from the actual data — every key listed is real, no others exist.
# Use json_each() to expand arrays into rows; use json_extract(col, '$.key') for objects.
_JSON_COLUMN_SHAPES: dict[str, dict[str, str]] = {
    "implementations": {
        "success_metrics_json": (
            '["metric string", ...]'
            " -- array of strings (no named keys)."
            " Expand with json_each(success_metrics_json); access via value."
        ),
        "risks_json": (
            '["risk string", ...]'
            " -- array of strings (no named keys)."
            " Expand with json_each(risks_json); access via value."
            " NEVER use json_extract(risks_json, '$.anything') — there are no object keys."
        ),
    },
    "competitors": {
        "strengths_json": (
            '["strength string", ...]'
            " -- array of strings (no named keys). Expand with json_each()."
        ),
        "weaknesses_json": (
            '["weakness string", ...]'
            " -- array of strings (no named keys). Expand with json_each()."
        ),
    },
    "scenarios": {
        "blueprint_json": (
            "object — exact keys (no others exist):"
            " account_health, annual_revenue_band, blueprint_seed, call_type,"
            " company_size_band, country, crm_stage, deployment_model,"
            " employee_count_hint, industry, internal_doc_type, issue_category,"
            " pain_point, primary_competitor_id, primary_competitor_name,"
            " primary_product_id, primary_product_name, region, root_cause_hint,"
            " scenario_id, secondary_product_id, secondary_product_name,"
            " slack_channel_name, slack_visibility, subindustry, summary,"
            " support_severity, target_bundle_tokens, trigger_event, uniqueness_key."
            " Access with json_extract(blueprint_json, '$.key')."
            " NEVER use any key not in this list."
            " Bounded-value keys (use = not LIKE, never invent values):"
            " issue_category: 'analytics pipeline'|'data integration'|'event orchestration'|'identity lifecycle'|'knowledge retrieval'|'observability'|'rules engine'|'workflow automation'."
            " call_type: 'discovery'|'executive escalation'|'quarterly business review'|'renewal negotiation'|'technical validation'."
            " deployment_model: 'hybrid'|'multi-tenant SaaS'|'private cloud'|'single-tenant cloud'."
            " account_health: 'at risk'|'expanding'|'healthy'|'recovering'|'watch list'."
            " pain_point and root_cause_hint are free-prose — do NOT filter with LIKE or =."
        ),
    },
    "products": {
        "features_json": (
            "array of objects — each object has exactly these keys:"
            " capability_type, code, dependencies, keywords, maturity, name, sku_tier, summary."
            " Expand with json_each(features_json) AS f,"
            " then access fields with json_extract(f.value, '$.key')."
            " NEVER use any key not in this list."
        ),
        "core_use_cases_json": (
            '["use case string", ...]'
            " -- array of strings (no named keys). Expand with json_each()."
        ),
    },
}


# ---------------------------------------------------------------------------
# Schema introspection
# ---------------------------------------------------------------------------


def get_schema_by_table(conn: sqlite3.Connection) -> dict[str, str]:
    """
    Build a per-table schema block dict: {table_name: DDL + annotation lines}.
    Only includes user-facing tables (excludes FTS internals).
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    all_tables: list[str] = [
        row[0]
        for row in cursor.fetchall()
        if not any(row[0].endswith(sfx) for sfx in _FTS_INTERNAL_SUFFIXES)
    ]

    result: dict[str, str] = {}
    for table in all_tables:
        try:
            cursor.execute(f"PRAGMA table_info({table})")  # noqa: S608
            cols = cursor.fetchall()
        except sqlite3.OperationalError:
            continue
        if not cols:
            continue
        col_defs = ", ".join(f"{c[1]} {c[2]}" for c in cols)
        lines: list[str] = [f"CREATE TABLE {table} ({col_defs});"]
        for col in _BOUNDED_COLUMNS.get(table, []):
            cursor.execute(  # noqa: S608
                f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL ORDER BY {col}"
            )
            vals = ", ".join(f"'{row[0]}'" for row in cursor.fetchall())
            lines.append(
                f"--   {table}.{col}: exact values: {vals}. Filter with = or IN only, never LIKE."
            )
        for col, note in _FREE_PROSE_COLUMNS.get(table, {}).items():
            lines.append(f"--   {table}.{col}: {note}")
        for col, shape in _JSON_COLUMN_SHAPES.get(table, {}).items():
            lines.append(f"--   {table}.{col}: {shape}")
        result[table] = "\n".join(lines)

    return result


def get_schema(conn: sqlite3.Connection) -> str:
    """Full schema string (all tables joined)."""
    return "\n".join(get_schema_by_table(conn).values())


# ---------------------------------------------------------------------------
# StartupContext dataclass — the singleton payload
# ---------------------------------------------------------------------------


@dataclass
class StartupContext:
    conn: sqlite3.Connection  # read-only main DB, reused across all requests
    schema_block: str  # full schema string
    schema_by_table: (
        dict  # {table_name: schema_block} for per-query SQL prompt filtering
    )
    llm: ChatOpenAI  # shared chat LLM client
    embedder: OpenAIEmbeddings  # shared embedding client
    vec_conn: sqlite3.Connection  # writable vector DB (sqlite-vec)
    react_agent_prompt: str  # tool-use instructions + few-shots
    react_agent_graph: Any  # LangGraph from create_agent (UnifiedGraphState)
    answer_prompt: str  # answer generation rules
    synthesize_prompt: str  # Slack formatting rules
    summary_prompt: str  # conversation compression rules


def build_startup_context(db_path: str) -> StartupContext:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    schema_by_table = get_schema_by_table(conn)
    schema = "\n".join(schema_by_table.values())

    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        api_key=os.environ["OPENAI_API_KEY"],
        max_tokens=2048,
    )
    embedder = OpenAIEmbeddings(
        model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        api_key=os.environ["OPENAI_API_KEY"],
    )

    vec_db_path = os.environ.get("VECTOR_DB_PATH", "vectors.db")
    vec_conn = open_vector_db(vec_db_path)
    setup_vector_tables(vec_conn)
    maybe_populate(vec_conn, conn, embedder)  # embeds title + content_text (truncated)

    react_prompt = _build_agent_system_prompt()
    # Imported here so hybrid_search / run_sql modules load after embedder exists.
    from slack_bot.agent.tools.hybrid_search import hybrid_search
    from slack_bot.agent.tools.run_sql import run_sql

    # Retrieval agent: LangChain create_agent builds the model↔tools loop; we do not
    # replicate that loop with separate LangGraph nodes (tools_condition, etc.).
    react_agent_graph = create_agent(
        llm,
        tools=[hybrid_search, run_sql],
        system_prompt=react_prompt,
        state_schema=UnifiedGraphState,
    )

    return StartupContext(
        conn=conn,
        schema_block=schema,
        schema_by_table=schema_by_table,
        llm=llm,
        embedder=embedder,
        vec_conn=vec_conn,
        react_agent_prompt=react_prompt,
        react_agent_graph=react_agent_graph,
        answer_prompt=_ANSWER_SYSTEM_PROMPT,
        synthesize_prompt=_SYNTHESIZE_SYSTEM_PROMPT,
        summary_prompt=_SUMMARY_SYSTEM_PROMPT,
    )


# ---------------------------------------------------------------------------
# Module-level singleton — imported by every node that needs DB or LLM access
# ---------------------------------------------------------------------------

_db_path = os.environ.get("DB_PATH", "synthetic_startup.sqlite")
CTX: StartupContext = build_startup_context(_db_path)
