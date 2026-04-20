"""
config.py
"""

# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------

# Leave headroom below Slack's hard 4,000-char block limit.
SLACK_CHAR_LIMIT = 3_800

# Seconds before a seen event_id is evicted from the dedup cache.
EVENT_DEDUP_TTL = 600

# Minimum seconds between placeholder status-update API calls.
# Keeps us within Slack's per-message rate limit.
STATUS_UPDATE_MIN_INTERVAL = 1.0

# ---------------------------------------------------------------------------
# Hybrid search — BM25 (FTS5)
# ---------------------------------------------------------------------------

# Maximum number of BM25 query phrases to run per turn.
# understand_node may produce more; extras are silently dropped.
MAX_FTS_QUERIES = 10

# BM25 candidates fetched per phrase. Kept larger than RRF_FINAL_LIMIT so
# RRF has a wide pool; docs that rank well in both BM25 and vector rise higher.
FTS_PER_QUERY = 10

# ---------------------------------------------------------------------------
# Hybrid search — Vector (sqlite-vec)
# ---------------------------------------------------------------------------

# Nearest neighbours returned by kNN search. Diminishing returns past the
# corpus size (~250 artifacts).
VEC_TOP_K = 100

# ada-002 and text-embedding-3-small both produce 1536-dim vectors.
EMBEDDING_DIM = 1536

# Number of artifacts embedded per OpenAI API batch call.
EMBEDDING_BATCH_SIZE = 50

# Content truncation before embedding. ada-002 limit is 8191 tokens
# (~4 chars/token → ~32 000 chars). Truncating to 25 000 chars leaves
# headroom for the title while capturing ~6 250 tokens of the body.
# NOTE: delete vectors.db and restart if you change this.
CONTENT_TRUNCATE_CHARS = 25_000

# ---------------------------------------------------------------------------
# Hybrid search — RRF merge
# ---------------------------------------------------------------------------

# RRF damping constant. Score = 1 / (k + rank). k=60 is the value from
# the original RRF paper; higher k reduces the penalty for lower ranks.
RRF_K = 60

# Merged results forwarded to answer_node / lazy_load_node after RRF.
# Higher = richer context but more tokens; lower = risk of missing relevant docs.
RRF_FINAL_LIMIT = 20

# Recency decay: score halves for every RECENCY_HALF_LIFE_DAYS days of age.
# Only applied when recency_sensitive=True.
RECENCY_HALF_LIFE_DAYS = 90

# Maximum additive recency bonus for a same-day document.
# A pure RRF rank-1 score is 1/(60+1) ≈ 0.016; 0.01 lifts recent docs
# without swamping strongly-ranked older ones.
RECENCY_WEIGHT = 0.01

# ---------------------------------------------------------------------------
# SQL generation
# ---------------------------------------------------------------------------

# Maximum LLM correction attempts before sql_aborted=True.
SQL_MAX_RETRIES = 3

# Hard row cap appended by execute_node if the query has no LIMIT.
SQL_MAX_ROWS = 20

# ---------------------------------------------------------------------------
# Lazy load
# ---------------------------------------------------------------------------

# Only fetch full content_text for artifacts whose token_estimate meets
# this threshold. Small artifacts are well-represented by their summary.
LAZY_LOAD_TOKEN_THRESHOLD = 500

# Maximum number of artifacts to fully load per turn.
LAZY_LOAD_MAX_ARTIFACTS = 10

# ---------------------------------------------------------------------------
# ReAct agent
# ---------------------------------------------------------------------------

# Maximum tool-call rounds per turn before forcing answer generation.
# One round = one LLM call that may produce multiple simultaneous tool_calls.
MAX_REACT_ITERATIONS = 10

# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------

# Maximum items in the verbatim recent_turns window (2 items per turn:
# 1 user + 1 assistant). VERBATIM_WINDOW=8 keeps the last 4 full turns.
# Older turns are compressed into `summary` by update_summary_node.
VERBATIM_WINDOW = 8
