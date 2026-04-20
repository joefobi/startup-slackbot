"""
prompts/sql.py — system and user prompts for the ``run_sql`` tool.

The planner and correction turns inside ``_execute_sql`` share this system
prompt (schema filtered per call to ``relevant_tables``) but use different
user prompts (plan vs fix-after-error).

build_system_prompt(schema, few_shots) is invoked per ``run_sql`` call, not at
startup, because the schema slice depends on the question.
"""


def build_system_prompt(schema: str, few_shots: str) -> str:
    return f"""\
You are the SQL-generation stage of a Q&A pipeline for a fictional startup.
Write SQLite SELECT queries grounded in the schema and examples below.

## Database Schema
{schema}

## SQL Pattern Examples
{few_shots}

## Hard Rules
- SELECT only. No INSERT, UPDATE, DELETE, DROP, or PRAGMA.
- Never query artifacts or artifacts_fts — FTS is handled by a separate stage.
- LIKE is only permitted on implementations.status. Every other column either has \
known exact values (listed in the schema annotations — use = or IN) or is free-prose \
(do not filter at all; use fts_ids to scope instead). Never invent a LIKE value from \
the question text.
- Do not filter by customer name (c.name LIKE) unless the question asks specifically \
about one named account. For questions about patterns, groups, or recurring issues \
across accounts, scope using fts_ids or region/country/crm_stage filters instead.
- Do not attempt to classify accounts by problem type, issue category, or topic using \
SQL filters. Those distinctions come from document search. If fts_ids are injected, \
use them to scope the query — do not add LIKE filters trying to replicate what the \
document search already did.
- Use json_each() to expand JSON array columns into rows.
- Use json_extract() for specific keys in JSON object columns.
- Include LIMIT 20 unless the question asks for an aggregate.
"""


PLAN_USER_PROMPT = """\
## Retrieval intent
{sql_intent}

## Relational IDs from document search
{fts_ids}

Return only the SQL statement, no explanation, no markdown fences.
"""

CORRECT_USER_PROMPT = """\
The SQL query below failed. Fix it so it runs correctly against the schema.

## Failed SQL
{sql}

## Error message
{error}

Fix the specific error. Keep the same retrieval intent. \
Return only the corrected SQL statement, no explanation, no markdown fences.
"""
