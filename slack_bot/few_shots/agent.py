"""
few_shots/agent.py — tool-use trace examples for the ReAct agent.

Each example shows a question followed by the tool call sequence the agent
should produce. These are baked into the agent system prompt via
build_system_prompt() in prompts/agent.py.

``run_sql`` illustrations list all six callable arguments (pass ``[]`` for any
unused fts_* id lists).

Patterns covered:
  1. Pure FTS  — answer lives in documents only
  2. Pure SQL  — answer is fully derivable from structured data
  3. Hybrid    — FTS first for entity IDs, then SQL scoped to those IDs
"""

FEW_SHOT_EXAMPLES: str = """\
### Example 1 — Pure FTS
Question: "What feedback have customers given about our analytics dashboards in recent calls?"

Reasoning: The answer lives in call transcripts, not structured tables.
           Run hybrid_search with recency_sensitive=True.

hybrid_search({
  "queries": [
    "customer feedback analytics dashboard call",
    "analytics reporting dashboard customer review",
    "dashboard usability concerns customer call",
    "analytics module customer pain points",
    "reporting feature complaints customer success call",
    "dashboard performance issues customer conversation",
    "analytics product customer satisfaction recent",
    "dashboard experience feedback support ticket",
    "analytics feedback quarterly business review",
    "reporting dashboard improvement requests customer"
  ],
  "artifact_types": ["customer_call"],
  "recency_sensitive": true
})
→ Observation: 8 documents found. [art_0031] customer_call — Q1 DataFlow Review
               customer_id: cust_014, tokens: 1450
               Summary: Q1 review discussing analytics dashboard usability concerns...
               --- full text ---
               [full transcript text with exact feedback...]

[Full document content is in the observation. Generate answer.]

---

### Example 2 — Pure SQL
Question: "How many customers do we have in each region broken down by account health?"

Reasoning: Pure aggregate over structured customer data. No documents needed.

run_sql({
  "sql_intent": "Count customers grouped by region and account_health.",
  "relevant_tables": ["customers"],
  "fts_customer_ids": [],
  "fts_competitor_ids": [],
  "fts_product_ids": [],
  "fts_scenario_ids": []
})
→ Observation: 5 rows returned: [{"region": "APAC", "account_health": "healthy", "count": 4}, ...]

[SQL results are sufficient. Generate answer.]

---

### Example 3 — Hybrid: FTS first for entity IDs, SQL scoped to those IDs
Question: "Which Nordics customers are in renewal review and what concerns have they raised?"

Reasoning: Need both document content (concerns raised) and structured data (CRM stage +
           region). Run FTS first — the customer_ids it returns will scope the SQL query
           precisely instead of relying on LIKE matching on free-text name fields.

hybrid_search({
  "queries": [
    "Nordics customer renewal concerns raised",
    "Scandinavia renewal negotiation customer issues",
    "Nordic region renewal objections customer feedback",
    "Nordic account renewal risk customer conversation",
    "renewal review customer concerns Nordic region",
    "Nordics renewal discussion concerns documented",
    "Sweden Norway Denmark renewal escalation",
    "Nordic renewal objections support ticket",
    "Nordics account renewal risk call transcript",
    "renewal concerns Nordic customer call"
  ],
  "artifact_types": ["customer_call", "support_ticket"],
  "recency_sensitive": false
})
→ Observation: 6 documents found with full text. customer_id values seen: cust_014, cust_027, cust_031 ...

run_sql({
  "sql_intent": "Find customers in the Nordics region with crm_stage = 'renewal review'.",
  "relevant_tables": ["customers", "implementations"],
  "fts_customer_ids": ["cust_014", "cust_027", "cust_031"],
  "fts_competitor_ids": [],
  "fts_product_ids": [],
  "fts_scenario_ids": []
})
→ Observation: 2 rows returned: [{"name": "NordTech AS", "region": "Nordics", ...}, ...]

[Both retrieval results available. Generate answer.]
"""
