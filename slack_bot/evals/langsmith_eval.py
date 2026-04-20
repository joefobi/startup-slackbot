"""
langsmith_eval.py — LangSmith eval suite for the ReAct agent.

Evaluators:
  recall          — fraction of expected_facts found in final_answer
  tool_calls      — number of tool invocations across all loop iterations
  efficiency      — binary: tool_calls <= max_tool_calls
  iterations      — number of LLM call rounds in the ReAct loop
  confidence      — high=1.0 / medium=0.5 / low=0.0
  sql_completed   — 1.0 if SQL did not abort

tool_calls and iterations are written to unified graph state after the
create_agent subgraph; evaluators read them from run.outputs.

Usage:
  python -m slack_bot.evals.langsmith_eval --upload
  python -m slack_bot.evals.langsmith_eval --run
  python -m slack_bot.evals.langsmith_eval --upload --run
  python -m slack_bot.evals.langsmith_eval --run --dataset my-dataset-name

Required env vars:
  LANGCHAIN_API_KEY
  LANGCHAIN_TRACING_V2=true
  OPENAI_API_KEY
"""

import argparse
import os

import structlog
from langsmith import Client
from langsmith.evaluation import evaluate

from slack_bot.agent.graph import build_graph
from slack_bot.config import MAX_REACT_ITERATIONS
from slack_bot.evals.cases import EVAL_CASES

_EVAL_RECURSION_LIMIT = 2 * MAX_REACT_ITERATIONS + 12
from slack_bot.log import configure
from datetime import datetime

configure()
log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_NAME = os.environ.get("LANGSMITH_DATASET", "slack-bot-qa-react")
EXPERIMENT_PREFIX = "slack-bot-qa-react"

_graph = build_graph(checkpointer=None)
_client = Client()


# ---------------------------------------------------------------------------
# Dataset upload (one-time)
# ---------------------------------------------------------------------------


def upload_dataset(client: Client, name: str = DATASET_NAME) -> None:
    """
    Create the LangSmith dataset from EVAL_CASES.
    Safe to call multiple times — creates a versioned copy if dataset exists.

    Args:
        client (Client): LangSmith client.
        name (str): Dataset name to create.
    """
    existing = [d for d in client.list_datasets() if d.name == name]
    if existing:
        log.info("dataset_exists", name=name, id=str(existing[0].id))
        return

    dataset = client.create_dataset(
        name,
        description="Slack Q&A bot eval cases (ReAct) — grounded in synthetic_startup.sqlite",
    )

    # required_nodes / forbidden_nodes are omitted — not applicable in ReAct mode.
    examples = [
        {
            "inputs": {"question": case["question"]},
            "outputs": {
                "expected_facts": case["expected_facts"],
                "max_tool_calls": case["max_tool_calls"],
                "notes": case.get("notes", ""),
            },
        }
        for case in EVAL_CASES
    ]

    client.create_examples(examples=examples, dataset_id=dataset.id)
    log.info(
        "dataset_uploaded", name=name, id=str(dataset.id), example_count=len(examples)
    )


# ---------------------------------------------------------------------------
# Target function
# ---------------------------------------------------------------------------


def run_agent(inputs: dict) -> dict:
    """
    Invoke the ReAct graph and return fields for evaluators and the experiment table.

    Args:
        inputs (dict): Must contain "question" key.

    Returns:
        dict: Flat output dict with final_answer, confidence, and debug columns.
    """
    raw = _graph.invoke(
        {
            "question": inputs["question"],
            "summary": "",
            "recent_turns": [],
            "replan_attempted": False,
        },
        config={"recursion_limit": _EVAL_RECURSION_LIMIT},
    )

    fts_rows = raw.get("fts_results") or []
    fts_hits = []
    for r in fts_rows:
        if isinstance(r, dict):
            fts_hits.append(
                {
                    "artifact_id": r.get("artifact_id", ""),
                    "title": r.get("title", ""),
                    "type": r.get("artifact_type", ""),
                }
            )

    return {
        # Evaluator-facing
        "final_answer": raw.get("final_answer") or "",
        "confidence": raw.get("confidence") or "",
        "sql_aborted": raw.get("sql_aborted"),
        "tool_call_count": raw.get("tool_call_count") or 0,
        "react_iterations": raw.get("react_iterations") or 0,
        # Experiment-table columns
        "fts_hits": fts_hits,
        "generated_sql": raw.get("generated_sql") or "(none)",
        "sql_results": raw.get("sql_results") or [],
    }


# ---------------------------------------------------------------------------
# Evaluators
#
# tool_call_count and react_iterations come directly from run.outputs —
# answer_node writes them to unified state; run_agent surfaces them
# as flat columns. No child run traversal needed.
# ---------------------------------------------------------------------------


def recall(run, example) -> dict:
    """Fraction of expected_facts present (case-insensitive) in final_answer."""
    answer = (run.outputs.get("final_answer") or "").lower()
    facts = example.outputs.get("expected_facts", [])

    if not facts:
        return {"key": "recall", "score": None, "comment": "no expected facts defined"}

    found_list = [f for f in facts if f.lower() in answer]
    missing_list = [f for f in facts if f.lower() not in answer]
    score = len(found_list) / len(facts)
    return {
        "key": "recall",
        "score": score,
        "comment": (
            f"output: {found_list} | "
            f"expected: {facts} | "
            f"missing: {missing_list or '(none)'}"
        ),
    }


def tool_calls(run, example) -> dict:
    """Total tool invocations; read directly from run output (no child run traversal)."""
    calls = run.outputs.get("tool_call_count", 0)
    max_calls = example.outputs.get("max_tool_calls", 8)
    return {
        "key": "tool_calls",
        "score": calls,
        "comment": f"{calls} calls (max {max_calls}) — {'efficient' if calls <= max_calls else 'over budget'}",
    }


def efficiency(run, example) -> dict:
    """Binary: 1.0 if tool_call_count <= max_tool_calls, else 0.0."""
    calls = run.outputs.get("tool_call_count", 0)
    max_calls = example.outputs.get("max_tool_calls", 8)
    return {
        "key": "efficient",
        "score": 1.0 if calls <= max_calls else 0.0,
    }


def iterations(run, example) -> dict:
    """Number of LLM call rounds in the ReAct loop (1 round may invoke multiple tools)."""
    n = run.outputs.get("react_iterations", 0)
    return {
        "key": "iterations",
        "score": n,
        "comment": f"{n} LLM round(s)",
    }


def confidence_score(run, example) -> dict:
    """Map confidence string to numeric: high=1.0, medium=0.5, low=0.0."""
    mapping = {"high": 1.0, "medium": 0.5, "low": 0.0}
    conf = run.outputs.get("confidence", "")
    return {
        "key": "confidence",
        "score": mapping.get(conf),
        "comment": conf or "unknown",
    }


def sql_abort_rate(run, example) -> dict:
    """1.0 if SQL completed without aborting, 0.0 if sql_aborted=True."""
    aborted = run.outputs.get("sql_aborted", False)
    return {
        "key": "sql_completed",
        "score": 0.0 if aborted else 1.0,
    }


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------


def run_eval(dataset_name: str = DATASET_NAME) -> None:
    """
    Run the eval experiment against the named dataset.

    Args:
        dataset_name (str): LangSmith dataset to evaluate against.
    """
    _client.flush()
    results = evaluate(
        run_agent,
        data=dataset_name,
        evaluators=[
            recall,
            tool_calls,
            efficiency,
            iterations,
            confidence_score,
            sql_abort_rate,
            # required_nodes_called and no_spurious_nodes omitted —
            # routing is dynamic in ReAct mode.
        ],
        experiment_prefix=EXPERIMENT_PREFIX,
        max_concurrency=1,
        metadata={
            "model": os.environ.get("OPENAI_MODEL", "gpt-5.4"),
            "embedding": os.environ.get(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"
            ),
            "mode": "react",
        },
    )

    log.info("experiment_complete", name=results.experiment_name)

    scores_by_key: dict[str, list[float]] = {}
    for result in results:
        for fb in result.get("feedback", []):
            if fb.score is not None:
                scores_by_key.setdefault(fb.key, []).append(fb.score)

    for key, scores in sorted(scores_by_key.items()):
        avg = sum(scores) / len(scores)
        log.info("evaluator_score", key=key, avg=round(avg, 2), n=len(scores))

    _client.flush()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LangSmith eval suite for the ReAct Slack Q&A bot."
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload EVAL_CASES as a LangSmith dataset.",
    )
    parser.add_argument("--run", action="store_true", help="Run the eval experiment.")
    parser.add_argument(
        "--dataset",
        default=DATASET_NAME,
        help=f"Dataset name (default: {DATASET_NAME}).",
    )
    args = parser.parse_args()

    if not args.upload and not args.run:
        parser.print_help()
        raise SystemExit(1)

    client = Client()

    if args.upload:
        upload_dataset(client, name=args.dataset)

    if args.run:
        run_eval(dataset_name=args.dataset)
