# Slack Q&A Bot


## DEMO

[View the demo here](https://drive.google.com/file/d/1c958d7PlzoK1FwGD-HerwD2CWeBk6H5h/view?usp=sharing)

A Slack chatbot that answers questions grounded in a SQLite database of startup data. An outer **LangGraph** pipeline resets state each turn, runs a LangChain **`create_agent`** subgraph for retrieval (**BM25 + vector + SQL**), then formats answers for Slack.

## How it works

1. **`prepare_react_node`** clears the ReAct **`messages`** channel and stale retrieval fields so each Slack message starts fresh.
2. **`react_agent_graph`** is **`create_agent`** with two tools:
   - **`hybrid_search`** — FTS5 BM25 + sqlite-vec kNN, merged with RRF; returns document summaries and relational IDs; optionally loads full **`content_text`** for top hits.
   - **`run_sql`** — Generates validated **SELECT** SQL, runs it on the read-only DB, retries on errors up to a fixed cap.

The LLM chooses tools and order until it stops calling tools or hits **`MAX_REACT_ITERATIONS`**.

3. **`answer_node`** builds a grounded answer (dedicated **`prompts/answer.py`**), sets **`confidence`**, and records **`react_iterations`** and **`tool_call_count`** from the agent **`messages`** (for LangSmith evals).

4. **`synthesize_node`** formats Slack markdown (including length splitting); **`update_summary_node`** updates rolling **`summary`** / **`recent_turns`**.

Checkpoint schema: **`UnifiedGraphState`** (`slack_bot/agent/types/graph_state.py`). Shared domain types: **`ArtifactResult`**, **`AnswerOutput`** (`slack_bot/agent/types/domain.py`).

## Prerequisites

- Python 3.11+
- `synthetic_startup.sqlite` in the project root (or set **`DB_PATH`**)
- OpenAI API key
- Slack app with `chat:write`, `reactions:write`, `channels:history`, and `im:history` bot token scopes

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env` and fill in your credentials:

```bash
cp .env .env.local   # or edit .env directly
```

Required environment variables:

```
SLACK_SIGNING_SECRET=...
SLACK_BOT_TOKEN=...
OPENAI_API_KEY=...
```

Optional:

```
OPENAI_MODEL=gpt-5.4                            # default: gpt-5.4
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002  # default: text-embedding-ada-002
DB_PATH=synthetic_startup.sqlite               # default: synthetic_startup.sqlite
VECTOR_DB_PATH=vectors.db                      # default: vectors.db
CHECKPOINT_DB_PATH=checkpoints.db              # default: checkpoints.db
LANGCHAIN_TRACING_V2=true                      # enables LangSmith tracing
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=slack-bot-qa
```

## Running the bot

```bash
uvicorn slack_bot.app:app --reload --port 8000
```

On first start, the bot builds the vector index from `synthetic_startup.sqlite`. This takes about 30 seconds and only runs when `vectors.db` is empty.

To expose the server to Slack, use ngrok:

```bash
ngrok http 8000
```

Set the Slack event subscription URL to `https://<your-ngrok-subdomain>.ngrok.io/slack/events`. The bot listens for `app_mention` and `message.im` events.

While the graph runs, placeholder messages can show status from outer nodes (**`prepare_react_node`**, **`react_agent_graph`**, **`answer_node`**, **`synthesize_node`**) and from **`hybrid_search`** / **`run_sql`** tool starts (see `slack_bot/app.py`).

## Running evals

### LangSmith experiment

`langsmith_eval.py` runs the graph against a LangSmith dataset and scores metrics such as:

- **recall** — fraction of expected facts found in the final answer
- **tool_calls** — total tool invocations in the ReAct loop
- **efficient** — 1.0 if tool calls are within the per-case budget, 0.0 otherwise
- **iterations** — ReAct rounds derived from agent messages
- **confidence** — numeric mapping of **`answer_node`** confidence (high / medium / low)

**`react_iterations`** and **`tool_call_count`** are written in **`answer_node`** and appear on **`run.outputs`**, so evaluators do not need to traverse LangSmith child runs for those counts.

Required env vars:

```
LANGCHAIN_API_KEY=...
LANGCHAIN_TRACING_V2=true
OPENAI_API_KEY=...
```

Upload the eval cases to LangSmith as a dataset (only needs to run once):

```bash
python -m slack_bot.evals.langsmith_eval --upload
```

Run the eval experiment against the uploaded dataset:

```bash
python -m slack_bot.evals.langsmith_eval --run
```

Upload and run in one command:

```bash
python -m slack_bot.evals.langsmith_eval --upload --run
```

Run against a specific named dataset:

```bash
python -m slack_bot.evals.langsmith_eval --run --dataset my-dataset-name
```

The dataset name defaults to `slack-bot-qa-react`. If a dataset with that name already exists, `--upload` creates a timestamped copy rather than overwriting it.

Results appear in the LangSmith UI at `smith.langchain.com` under the `slack-bot-qa-react` experiment prefix. Average scores for each evaluator are also printed to stdout after the run completes.

### Unit tests

```bash
python -m pytest slack_bot/tests/ -q
```

## Project structure

```
/                 # project root
  requirements.txt
  DESIGN.md                   # architecture and tradeoffs (longer writeup)
  CLAUDE.md                     # condensed reference for agents / tooling
  synthetic_startup.sqlite      # expected at repo root unless DB_PATH points elsewhere
  slack_bot/
    app.py                    # FastAPI app, Slack events, streaming status updates
    startup.py                # CTX: DB, schema, create_agent subgraph
    vector_index.py           # sqlite-vec setup and kNN search
    config.py                 # Tunable constants (RRF, SQL retries, MAX_REACT_ITERATIONS, etc.)
    log.py                    # Shared structlog configuration
    agent/
      graph.py                # prepare_react → react_agent_graph → answer → synthesize → update_summary
      types/
        __init__.py           # re-exports domain + graph_state types
        domain.py             # ArtifactResult, AnswerOutput (Pydantic)
        graph_state.py        # UnifiedGraphState (checkpoint schema)
      nodes/
        prepare_react.py      # Per-turn ReAct + field reset
        answer.py             # Grounded answer + metrics for evals
        synthesize.py         # Slack markdown formatting
        update_summary.py     # Rolling conversation summary
      tools/
        __init__.py           # package marker (tools are hybrid_search.py + run_sql.py)
        hybrid_search.py      # BM25 + vector RRF, Command tool + helpers
        run_sql.py            # SQL plan / validate / execute / correct, Command tool
      utils/
        __init__.py
        sql.py                # Shared SQL string helpers
    prompts/
      agent.py                # ReAct agent system prompt + user template
      answer.py               # Answer node system prompt
      synthesize.py           # Synthesize node system prompt
      update_summary.py       # Summary node system prompt
      sql.py                  # SQL system prompt builder (used by run_sql)
    few_shots/
      agent.py                # Tool-use trace examples for the ReAct agent
      sql_plan.py             # SQL generation examples
    evals/
      cases.py                # Ground truth eval cases
      langsmith_eval.py       # LangSmith dataset upload and eval experiment
    tests/
      conftest.py               # Pytest fixtures
      test_execute_sql.py       # _execute_sql (run_sql tool)
      test_rrf_merge.py         # hybrid_search._rrf_merge
      test_sql_generation.py    # _validate_sql + DB fixtures
      test_update_summary_node.py
      test_validate_sql.py      # _validate_sql (run_sql tool)
```

