"""
utils/sql.py — SQL string utilities shared across agent tools.
"""

import re

_TABLE_RE = re.compile(r"\b(?:FROM|JOIN)\s+(\w+)", re.IGNORECASE)


def _strip_fences(text: str) -> str:
    """Strip markdown fences from an LLM SQL reply.

    Args:
        text (str): Raw model output, often wrapped in markdown code fences.

    Returns:
        str: Inner SQL text without fence lines.

    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()
    return text


def _extract_table_names(sql: str) -> set[str]:
    """Collect bare table names appearing after ``FROM`` / ``JOIN``.

    Args:
        sql (str): SQL text to scan.

    Returns:
        set[str]: Lowercased identifiers (best-effort regex parse).

    """
    return {m.group(1).lower() for m in _TABLE_RE.finditer(sql)}
