

# Design Document

## Overview
This document describes a chatbot built to answer questions from a SQLite database containing both startup data and call transcripts. The system uses a LangGraph outer pipeline and an inner ReAct loop. Structured questions require SQL, while document questions require a hybrid retrieval strategy. The LLM must decide which tools to call based on the intermediate results it sees.

## Architecture

START -> `prepare_react_node` -> `react_agent_graph` (calls hybrid search and sql tools in a loop) -> `answer_node` -> `synthesize_node` -> `update_summary_node` -> END

The **`prepare_react_node`** runs once per message to clear prior tool traces. It fills a fresh message from the summary and the current question. Then it enters the **`react_agent_graph`**. The loop ends only when the LLM returns with no tool calls or reaches the iteration limit. 

### Node Responsibilities
* **`answer_node`**: This provides the grounded answer plus a structured confidence score. I kept this as a dedicated step because it allows for better control over quality than simply using the last message of a loop.
* **`synthesize_node`**: Formats the final answer as Slack markdown with citations.
* **`update_summary_node`**: This manages the reality of token costs. We cannot keep everything verbatim. This node compresses older turns into a rolling summary to keep the cost flat even as the thread grows.

## Retrieval Strategy
I observed that questions fall into three categories: structured, unstructured, and the hybrid ground between them. The sequential pipeline covers all three by allowing the bot to choose its path.

### Hybrid Search over Pure BM25
BM25 works for exact names but fails on paraphrases. I tried a pure FTS solution first but found better results by combining it with vector search. I merge these lists using **Reciprocal Rank Fusion (k=60)**. This ensures that a document appearing in both lists gets contributions from both terms, producing results that are both keyword-relevant and semantically relevant.

### FTS Before SQL
When a question mentions a customer or competitor name, the agent calls `hybrid_search` first. The results include relational IDs extracted from matched artifacts. The LLM then passes those IDs to the SQL tool as filters. This scopes the query to specific accounts rather than relying on imprecise name-matching. By excluding FTS tables from the SQL schema, I ensured the LLM cannot accidentally write invalid FTS queries through the SQL tool.

## SQL Safety and Validation

* The database is opened with `mode=ro`. Any non-SELECT statement raises an error at the SQLite level.
* The tool plans the query with `EXPLAIN` to catch syntax errors. It also checks every table against an allowlist to stop the model from hallucinating table names.
* If validation fails, the tool rewrites the SQL using the error message as context. We allow this up to three times. Even if the limit is reached, the agent can still generate a partial answer from the search results.

## Performance and Conversation Memory
Storing a full conversation verbatim would make token costs grow linearly. My strategy keeps the cost manageable by keeping only the last four turns verbatim and compressing everything else. An obvious tradeoff is that specific details from more than eight turns ago are lost, but the summary captures the essential decisions.

For the search results, I chose to always fetch the full content text for the top 10 results. The tradeoff is higher token usage per turn, but at our scale of 250 artifacts, the fetch is fast and the additional context consistently improves the answer quality.



## Future Directions

### Scaling the Vector Store
The current use of `sqlite-vec` is appropriate for 250 artifacts. It does not support horizontal scaling or the kind of metadata filtering that should happen at the vector layer. Migrating to a hosted store like Pinecone would allow us to filter by customer ID or creation date during the query itself, rather than trying to post-filter the results in Python. 

### Adaptive Retrieval Selection
Right now, the agent blindly accepts the results it gets. It does not yet self-assess the quality of its retrieval between calls. I want to give the agent explicit signals about its own performance, such as the score distribution of the search results. With those signals, the agent could make an informed decision to retry a query if the results are weak, or skip a SQL step if the search results have already provided an answer.

### RAG for Schema Pruning
The `run_sql` tool depends on the agent knowing the names of every table in the database. A better approach is to embed the schema blocks themselves and retrieve only the most relevant tables based on the user's question. This would allow the database to grow to hundreds of tables without overwhelming the model's context window.

### Systematic Ranking Evaluation
We currently measure if the final answer is correct, but we do not measure the retrieval quality in isolation. I want to implement metrics like Mean Reciprocal Rank to see if the most relevant document is actually appearing at the top of the list. We need to quantify the contribution of each retrieval branch so we know exactly which part of the system is failing when a question goes unanswered.

### Long-Term Memory
Each Slack thread is currently isolated. If a user asks a follow-up question in a new thread, the bot has no memory of the previous conversation. Implementing a cross-thread Store would allow the bot to maintain a user-scoped memory that survives across different sessions. 

### Rate Limiting and Safety
The system currently accepts every inbound event from Slack without restriction. Under heavy load, a single user could overwhelm the database connections. We need to implement a token bucket to enforce a per-user limit. 

### Expanding the Toolset
The agent is currently limited to search and SQL. There are questions that require arithmetic or visualization that these tools cannot handle well. Adding a dedicated calculator would stop the LLM from making basic math errors on SQL results, and a table visualization tool would allow the bot to post formatted charts directly to Slack. It is about giving the agent the right tools for the conversation it is trying to have.