# Design Document

## Overview

This document describes a Slack chatbot that answers questions grounded in a SQLite database containing structured startup data and long-form artifacts in the form of call transcripts, internal communications, support tickets, etc.

The bot uses a LangGraph outer pipeline plus an inner LangChain `create_agent` ReAct loop with a hybrid retrieval strategy of keyword search (BM25 via SQLite FTS5) combined with semantic vector search (sqlite-vec), merged with Reciprocal Rank Fusion. Structured questions are answered with SQL and document questions with FTS + vector search. The LLM decides which tools to call and in what order based on intermediate results.

## Architecture

### Agent Graph

```
START
  │
  ▼
prepare_react_node          (clears ReAct messages, injects summary / recent_turns / question)
  │
  ▼
react_agent_graph           LangChain create_agent: internal LLM ↔ tools loop
  │   ┌──────────────────────────────────────────────┐
  │   │  LLM → if tool_calls are needed → hybrid_search           │
  │   │                   → run_sql                  │
  │   │       no tool_calls → exit subgraph          │
  │   └──────────────────────────────────────────────┘
  │
  ▼
answer_node
  │
  ▼
synthesize_node
  │
  ▼
update_summary_node
  │
  ▼
END
```

`prepare_react_node` runs once per Slack message. It wipes the prior turn's ReAct `messages` so tool traces do not accumulate forever, and it fills a fresh user message from `summary`, `recent_turns`, and the current `question`. 

`react_agent_graph` is the compiled subgraph from `create_agent`. The LLM decides which tools to call and in what order from each observation. The loop ends when the LLM returns with no tool calls, or when `MAX_REACT_ITERATIONS` is reached. Retrieval outputs are provided on a unified checkpoint state (`UnifiedGraphState`) so `answer_node` can read outputs from `react_agent_graph`.

### Node Responsibilities

- **`prepare_react_node`**: reset ReAct `messages`, clear stale retrieval fields, inject USER_PROMPT from memory + question.
- **`react_agent_graph`**: Tool loop calling `hybrid_search` and `run_sql`; accumulates results in state. 
- **`answer_node`**: Grounded answer from retrieval plus structured confidence; tallies `react_iterations` / `tool_call_count` for evals. 
- **`synthesize_node`**: Formats answer as Slack markdown with citations. 
- **`update_summary_node`**: Evicts oldest verbatim turn, compresses to rolling summary.

**Tools inside `react_agent_graph`:**

- **`hybrid_search`**: BM25 (FTS5) + kNN vector search + RRF merge
- **`run_sql`**: SQL plan + validate + execute + correct loop

---

## Retrieval Strategy

### Two Retrieval Branches

I observed that answerable questions could fall into three categories:

1. **Structured**: "Which accounts are at risk?" requires SQL against `implementations`, `customers`, etc.
2. **Unstructured**: "What did the team say about the taxonomy rollout?" requires a document search over `artifacts`.
3. **Hybrid**: "Which customer is most likely to churn to a cheaper competitor?" requires both SQL for structured risk signals and FTS for call transcript context.

The sequential pipeline of fts/hybrid then sql covers all three cases.

### Hybrid Search (BM25 + Vector), Not Pure BM25

BM25 (FTS5) works for exact-match retrieval for proper nouns, product names, and technical terms but fails on paraphrases.

Vector search (sqlite-vec with `text-embedding-ada-002`) handles semantic similarity and can catch cases where the user's phrasing differs from the document's vocabulary.

The two lists are merged with **Reciprocal Rank Fusion (k=60)**:

```
score(doc) = Σ  1 / (60 + rank_i)
```

A document that appears in both lists gets contributions from both terms. This can then produce results that are both keyword-relevant and semantically relevant.

### FTS Before SQL

When questions mention customer or competitor names, the agent prompt instructs the LLM to call `hybrid_search` first. The observation it returns includes `customer_id`, `competitor_id`, and other relational IDs extracted from the matched artifacts. The LLM then passes those IDs explicitly to `run_sql` as entity filters, which scopes the SQL query to the accounts that appeared in document search rather than relying on imprecise name-matching with LIKE.

In the ReAct architecture, the agent system prompt and few-shot examples illustrate the FTS-first pattern, and the LLM applies it when the question warrants it.

The `run_sql` tool receives a schema block with `artifacts` and `artifacts_fts` excluded, and the SQL few-shot examples contain no FTS patterns, so the LLM cannot accidentally write FTS queries through the SQL tool.

---

## SQL Safety

### Read-Only Connection

The main database is opened with:

```python
sqlite3.connect("file:db.sqlite?mode=ro", uri=True)
```

Any non-SELECT statement raises `sqlite3.OperationalError` at the SQLite level, independent of any application-layer checks.

### Validate Before Execute

The `run_sql` tool runs these checks before executing any query:

1. **EXPLAIN**: SQLite parses and plans the query and any syntax error raises immediately.
2. **Table allowlist**: every table referenced in FROM/JOIN must be in `KNOWN_TABLES`. This was done to catch any hallucinated table names are caught here before execution.
3. **SELECT-only guard**: prevents non-SELECT queires from running to prevent editing of the db.

### Retry Loop (max 3)

If validation fails or execution raises an error, the tool rewrites the SQL using the error message as context. This loop runs up to `SQL_MAX_RETRIES` times. At the limit, `sql_aborted=True` is set and the tool returns an error observation. The agent can still generate a partial answer from FTS results.

---

## Prompt Architecture

### Separation of Concerns

Each node gets only the prompt it needs. Static system prompts are stored in `StartupContext` once at boot. The `run_sql` tool is the only tool that calls `build_system_prompt(schema_block, FEW_SHOT_EXAMPLES)` at request time with the schema filtered to the tables the LLM said it needed, so the SQL LLM only sees the tables relevant to the current query.

The retrieval agent runs under `prompts/agent.py`, which describes the two tools, when to use each, the FTS-before-SQL strategy, and when to stop retrieving. `answer_node` runs under a separate answer prompt (`prompts/answer.py`). These two prompts are intentionally distinct because the agent prompt optimises for retrieval decisions and the answer prompt optimises for answer quality. 

I observed more fine-grained, better-grounded final answers when I split that second step out as a dedicated `answer_node` with its own prompt and structured output, instead of treating the ReAct loop's last assistant message as the user-facing reply. That said, a separate answer pass is not 100% required for correctness. You could ship the subgraph's exit text straight to Slack for fewer tokens and lower latency; the dedicated node is a quality and control knob I kept because it consistently helped on the eval cases.

### Few-Shot Examples

There are 12 annotated examples for the `run_sql` tool covering patterns that I observed to be problematic:

- `LOWER(col) LIKE '%value%'` for free-text status fields
- `json_each()` for JSON array columns (`risks_json`, `success_metrics_json`, `strengths_json`)
- `json_extract()` for JSON object columns (`blueprint_json`, `features_json`)
- Direct column preference over `json_extract` when both work
- `LIMIT 1` for single-row tables (`company_profile`)
- Date arithmetic with `date('now')` for overdue detection
- `full_name` (not `name`) for the `employees` table
- Guardrail examples: `json_each` array expansion, and no `LIKE` on bounded-value columns

`few_shots/agent.py` contains 3 tool-use traces for the retrieval agent covering pure FTS, pure SQL, and hybrid (FTS first for entity IDs then SQL). Full document text is included directly in the `hybrid_search` observation, so no separate tool call is needed to fetch it.

---

## Conversation Memory

Each Slack thread maps to a LangGraph checkpointer `thread_id` (keyed by `thread_ts`). State is persisted in `checkpoints.db` via `AsyncSqliteSaver`.

### Verbatim Window + Rolling Summary

Storing the full conversation verbatim would make token cost grow linearly with thread length. The memory strategy keeps cost flat:

- `recent_turns`: last 4 turns (8 `AnyMessage` objects) verbatim (tunable parameter in `config.py`)
- `summary`: compressed prose of all older turns, updated after each response.

`update_summary_node` uses the `add_messages` reducer. After every synthesis, it appends the current turn as a `HumanMessage` + `AIMessage`. When the window exceeds `VERBATIM_WINDOW`, it returns `RemoveMessage` objects for the evicted pair alongside the new messages.

An obvious tradeoff here is that verbatim context is lost for facts established more than 4 turns ago. The summary captures decisions and named entities but not every detail.

---

## Full Content Loading

`hybrid_search` always fetches `content_text` for the top `LAZY_LOAD_MAX_ARTIFACTS` (10) results after the RRF merge, using a single batched `SELECT ... WHERE artifact_id IN (...)` query. The full text is appended to each artifact block in the observation under a `--- full text ---` marker, so the LLM sees complete document content in the same tool response without needing a second tool call.

The tradeoff is higher token usage per turn compared to returning summaries only. At this scale (~250 artifacts), the top-10 content fetch is fast and the additional context reliably improves answer quality, so the cost is acceptable.

---

## Slack Integration

### 3-Second Deadline

Slack requires an HTTP 200 within 3 seconds or it retries the event. The handler returns 200 immediately and dispatches the agent as an `asyncio.create_task`. The LangGraph graph runs entirely in the background.

### UX for Slow Responses

1. `:thinking_face:` reaction is added to the user's message immediately.
2. A `_Thinking..._` placeholder is posted in the thread immediately.
3. The placeholder is updated in-place with a human-readable status string as outer nodes and tool starts complete (throttled to one Slack API call per second to stay within rate limits).
4. When the agent finishes, the placeholder is replaced with the real answer via `chat_update`.
5. Answers over `SLACK_CHAR_LIMIT` (3,800) characters are split at `[CONTINUED]` markers and posted as follow-up messages in the same thread.

### Security

- **HMAC-SHA256**: every request is verified against `SLACK_SIGNING_SECRET` before any payload is parsed.
- **Constant-time comparison**: `hmac.compare_digest` prevents timing attacks on signature validation.
- **Replay protection**: requests with timestamps older than 5 minutes are rejected.
- **Event deduplication**: `event_id` values are tracked in a TTL cache (10 min) to handle Slack's retry behaviour.
- **Bot loop prevention**: messages with `bot_id` or `subtype=bot_message` are ignored before any processing.

---

## Startup Caching

`startup.py` builds a `StartupContext` singleton at import time:

```python
CTX: StartupContext = build_startup_context(_db_path)
```

This is imported by every node. It contains:

- The read-only DB connection
- The writable vector DB connection
- `schema_block`, the full schema string
- `schema_by_table` used by the `run_sql` tool to build a per-request filtered schema prompt
- The shared `ChatOpenAI` and `OpenAIEmbeddings` clients
- `react_agent_graph` (compiled `create_agent` subgraph) and `react_agent_prompt`
- System prompts for `answer_node`, `synthesize_node`, `update_summary_node`

Nothing is re-initialized per request. Schema introspection (including `SELECT DISTINCT` queries for bounded-value column annotations) and static prompt assembly happen once at boot and are reused for every LLM call. The `run_sql` tool builds its prompt dynamically per request with the schema filtered to the tables the agent needs.

---

## Evaluation

`evals/langsmith_eval.py` uploads cases to LangSmith as a dataset and runs the full graph against each with five evaluators:

- **Recall**: fraction of `expected_facts` that appear (case-insensitive) in `final_answer`.
- **Tool calls**: total tool invocations in the ReAct loop, read from `run.outputs["tool_call_count"]` (`answer_node` derives this from `messages`; no child-run traversal needed).
- **Efficiency**: binary score, 1.0 if `tool_call_count` is within `max_tool_calls`.
- **Iterations**: LLM rounds in the ReAct loop from `run.outputs["react_iterations"]`.
- **Confidence**: numeric mapping of the confidence string from `answer_node` (high=1.0, medium=0.5, low=0.0).

---

## Key Tradeoffs Summary


| Decision                                    | Alternative                                                  | Why this way                                                                                                                                                                                                               |
| ------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dedicated `answer_node` after retrieval     | Use the ReAct subgraph's last assistant message as the reply | I saw finer, more grounded outputs with a purpose-built answer prompt and structured confidence. Not strictly required (you could stream the loop exit text and save a call), but worth it for quality on the evals I ran. |
| BM25 + vector hybrid                        | Pure BM25 or pure vector                                     | BM25 for exact names; vector for paraphrases. Hybrid covers both.                                                                                                                                                          |
| Per-call schema filtering                   | Full schema always                                           | LLM only sees tables relevant to the current query, which reduces hallucination surface.                                                                                                                                   |
| Verbatim window + rolling summary           | Full history                                                 | O(1) token cost per turn. Precision loss on old turns is acceptable for Q&A.                                                                                                                                               |
| Always load full content in `hybrid_search` | Separate `load_artifacts` tool call                          | One fewer round-trip per turn. The LLM sees full document text in the same observation as the search results, which produces more grounded answers without an extra tool invocation.                                       |


---

## Future Directions

### Pinecone (or similar) for the vector store

`sqlite-vec` is appropriate at this scale (~250 artifacts) but does not support horizontal scaling, approximate nearest neighbour indexing at large volume, or metadata filtering at the vector layer. Migrating to a hosted store (Pinecone, Weaviate, Qdrant) would provide:

- Filtering by `artifact_type`, `customer_id`, or `created_at` at query time without post-filtering in Python
- Faster kNN at scale without loading the full index into memory
- Multi-instance deployment without a shared writable `vectors.db`

### Adaptive retrieval selection

The ReAct agent decides at runtime whether to call `hybrid_search`, `run_sql`, both, or neither. It does not currently self-assess retrieval quality between calls: it gets the results, forms an observation, and decides whether to call another tool or stop. A possible improvement is to give the agent explicit signals about retrieval confidence, such as the number of BM25 hits per query, the RRF score distribution, or the fraction of matched artifacts that share a customer ID. With those signals the agent could make more informed decisions, such as retrying with broader queries when BM25 returns very few results, or skipping SQL when FTS already surfaces a clear entity match with no ambiguity.

### RAG for schema pruning

The `run_sql` tool currently receives a schema block filtered to the tables the agent asked for. The agent asks for tables by name, which means it needs to know what tables exist. Those table names come from the agent system prompt, which lists all of them. An alternative is to embed the per-table schema blocks at startup and retrieve the top-k most relevant ones using the question as a query, rather than having the agent select tables from a fixed list. This would allow the schema to scale to many more tables without growing the system prompt linearly, and would surface the right tables even when the question vocabulary does not match the hardcoded table names.

### More systematic retrieval ranking evaluation

The current eval harness measures end-to-end recall (facts in final answer) and routing correctness (which nodes ran), but not retrieval ranking quality in isolation. Improvements:

- Add `expected_artifact_ids` to eval cases and score Mean Reciprocal Rank (MRR) and Recall@k directly on `fts_results`
- Measure BM25-only vs vector-only vs RRF merged recall separately to quantify the contribution of each retrieval branch
- Add a LangSmith evaluator that scores whether the top-ranked artifact is the one most relevant to the question
- Use this signal to tune the RRF k parameter, `MAX_FTS_QUERIES`, and `RRF_FINAL_LIMIT`

### Long-term memory with LangGraph Store

Each Slack thread is currently isolated. If a user opens a new thread and asks "what about those at-risk customers we discussed?", the bot has no context from prior threads. LangGraph's cross-thread `Store` API provides a user-scoped key-value store that survives across threads. This requires migrating from `AsyncSqliteSaver` (thread-scoped) to a store-aware checkpointer, and adding a conditional write step after `update_summary_node`.

### Rate limiting on the Slack endpoint

The FastAPI handler currently accepts all inbound Slack events without any per-user or per-workspace rate limit. Under load, a single user could queue many concurrent graph runs, each holding open LLM and DB connections. Additions:

- Per-user token bucket (e.g. max 5 requests per minute) enforced in the handler before dispatching `asyncio.create_task`
- Global concurrency cap on in-flight graph runs using an `asyncio.Semaphore`
- Return a user-visible message rather than silently queuing

### Additional tools for the ReAct agent

The agent currently has two tools: `hybrid_search` and `run_sql`. Additional tools could be added without changing the graph structure, since the agent decides at runtime which ones to call:

- **Calculator / code interpreter**: for questions requiring arithmetic over SQL results, such as weighted averages or year-over-year growth rates, that are error-prone to express in SQL alone
- **Table visualisation**: render SQL results as a formatted table or chart posted as a Slack block rather than prose
- **LangMem**: structured long-term memory with automatic fact extraction, conflict resolution, and retrieval, which is a more principled alternative to the manual store approach above

