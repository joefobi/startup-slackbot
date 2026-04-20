"""
cases.py — ground truth eval cases sourced from the take-home spec.

Each case specifies:
  question        — the exact question to ask the agent
  expected_facts  — strings that must appear (case-insensitive) in final_answer
  max_tool_calls  — upper bound on retrieve operations for an "efficient" score
  required_nodes  — retrieval nodes that MUST have run (routing correctness)
  forbidden_nodes — retrieval nodes that must NOT have run (routing efficiency)
  notes           — human-readable description of what the case is testing

Retrieval nodes tracked in required/forbidden:
  hybrid_search_node  — BM25 + vector FTS
  sql_plan_node       — SQL generation (validate + execute always follow)
  execute_node        — SQL execution
  lazy_load_node      — full artifact text fetch

understand_node, answer_node, synthesize_node, update_summary_node always run
and are not tracked here — they are not routing decisions.

Tool call budget reference:
  Pure FTS + lazy load:   1 FTS + up to 5 lazy loads = max 6
  Pure SQL:               1 validate + 1 execute     = max 2
  Hybrid:                 1 FTS + 1 validate + 1 execute + lazy load = 3–8
"""

EVAL_CASES: list[dict] = [
    # ------------------------------------------------------------------
    # Case 1 — hybrid: FTS for proof plan content + SQL for customer date
    #
    # Answer: BlueHarbor Logistics. Northstar proposed a 7-10 business day
    # proof-of-fix: update index weighting, add a taxonomy mapping layer,
    # and run an A/B test on the top 20 saved searches, with success defined
    # as top-5 correct hit rate of at least 80 percent on prioritized queries.
    #
    # Routing: SQL on implementations/customers to find customer whose
    # implementation kicked off after 2026-02-20, then FTS for the
    # proof-of-fix document that contains the specific plan details.
    # Lazy load needed for the full plan text (token_estimate likely high).
    # ------------------------------------------------------------------
    {
        "question": (
            "Which customer's issue started after the 2026-02-20 taxonomy rollout, "
            "and what proof plan did we propose to get them comfortable with renewal?"
        ),
        "expected_facts": [
            "BlueHarbor",
            "7-10 business day",
            "80",  # "80 percent" / "80%" — success threshold
            "A/B test",
            "saved searches",
            "taxonomy mapping",
        ],
        "max_tool_calls": 6,  # FTS + SQL validate + SQL execute + up to 3 lazy loads
        "required_nodes": ["hybrid_search_node", "sql_plan_node", "execute_node"],
        "forbidden_nodes": [],
        "notes": (
            "Hybrid: SQL date filter on implementations/customers, "
            "FTS + lazy load for the specific proof-of-fix plan text. "
            "Verified answer: BlueHarbor Logistics, 7-10 business day A/B test plan."
        ),
    },
    # ------------------------------------------------------------------
    # Case 2 — pure FTS + lazy load: patch playbook lookup
    #
    # Answer: Approved window is 2026-03-24 02:00–04:00 local time.
    # Rollback command: orchestrator rollback --target ruleset=<prior_sha>,
    # which restores the prior ruleset and replays the invalidation hook.
    #
    # Routing: No SQL needed — the patch playbook is an internal_document
    # artifact for City of Verdant Bay. FTS on "Verdant Bay patch window"
    # or "rollback ruleset" finds it; lazy load pulls the full command text.
    # ------------------------------------------------------------------
    {
        "question": (
            "For Verdant Bay, what's the approved live patch window, "
            "and exactly how do we roll back if the validation checks fail?"
        ),
        "expected_facts": [
            "Verdant Bay",
            "2026-03-24",
            "02:00",  # patch window start
            "orchestrator rollback",  # exact command prefix
            "prior_sha",  # rollback target parameter
            "invalidation hook",  # what gets replayed
        ],
        "max_tool_calls": 4,  # FTS + up to 3 lazy loads (playbook may be large)
        "required_nodes": ["hybrid_search_node"],
        "forbidden_nodes": ["sql_plan_node", "execute_node"],
        "notes": (
            "Pure FTS + lazy load: patch playbook is in an internal_document artifact. "
            "No SQL required — Verdant Bay customer record is not needed to answer this. "
            "Verified answer: 2026-03-24 02:00–04:00, orchestrator rollback command."
        ),
    },
    # ------------------------------------------------------------------
    # Case 3 — pure FTS + lazy load: field mapping + workshop deliverables
    #
    # Answer: Router transform maps txn_id→transaction_id, total_amount→amount_cents
    # (coerced to int), preserves store_id and register_id. The 2026-03-23 workshop
    # should agree the canonical schema, define alias mappings and producer migration
    # milestones, and produce a signed schema document for SI-SCHEMA-REG.
    #
    # Routing: FTS on "MapleHarvest transform" or "router transform txn_id" finds
    # the internal_document or internal_communication artifact. Lazy load for full
    # field-mapping detail (schema docs tend to be long).
    # ------------------------------------------------------------------
    {
        "question": (
            "In the MapleHarvest Quebec pilot, what temporary field mappings are we "
            "planning in the router transform, and what is the March 23 workshop "
            "supposed to produce?"
        ),
        "expected_facts": [
            "MapleHarvest",
            "txn_id",  # source field name in transform
            "transaction_id",  # target field name
            "amount_cents",  # coerced target field
            "store_id",  # preserved field
            "SI-SCHEMA-REG",  # destination for signed schema doc
            "2026-03-23",  # workshop date (or "March 23")
        ],
        "max_tool_calls": 4,  # FTS + up to 3 lazy loads
        "required_nodes": ["hybrid_search_node"],
        "forbidden_nodes": ["sql_plan_node", "execute_node"],
        "notes": (
            "Pure FTS + lazy load: all answer facts live in a document artifact. "
            "No SQL needed — the question is about plan content, not structured DB fields. "
            "Verified answer: txn_id→transaction_id, total_amount→amount_cents, "
            "workshop produces signed schema doc for SI-SCHEMA-REG."
        ),
    },
    # ------------------------------------------------------------------
    # Case 4 — pure FTS + lazy load: SCIM conflict + engineering fast fix
    #
    # Answer: Aureum was sending both `department` and `businessUnit` variants.
    # Jin's fast fix: hot-reloadable Signal Ingest preprocessing rule to normalise
    # those attributes into one canonical field, plus SCIM tracing to observe
    # where approval latency is happening.
    #
    # Routing: FTS on "Aureum SCIM" or "department businessUnit" finds the
    # support_ticket or internal_communication. Lazy load for the fix details.
    # No SQL needed — this is entirely in document artifacts.
    # ------------------------------------------------------------------
    {
        "question": (
            "What SCIM fields were conflicting at Aureum, and what fast fix did "
            "Jin propose so we don't have to wait on Okta change control?"
        ),
        "expected_facts": [
            "department",  # one of the conflicting SCIM fields
            "businessUnit",  # the other conflicting SCIM field
            "hot-reloadable",  # key property of the fix
            "SCIM tracing",  # observability addition in the fix
        ],
        "max_tool_calls": 4,  # FTS + up to 3 lazy loads
        "required_nodes": ["hybrid_search_node"],
        "forbidden_nodes": ["sql_plan_node", "execute_node"],
        "notes": (
            "Pure FTS + lazy load: SCIM conflict details are in a support_ticket "
            "or internal_communication artifact. No SQL. "
            "Verified answer: department/businessUnit conflict, Jin's hot-reloadable "
            "preprocessing rule + SCIM tracing."
        ),
    },
    # ------------------------------------------------------------------
    # Case 5 — hard hybrid: competitor defection risk + milestone details
    #
    # Answer: BlueHarbor Logistics. NoiseGuard is explicitly framed as a
    # low-cost tactical dedupe layer that can buy time if Northstar misses.
    # Next milestone: BlueHarbor sends schema export + 14 days of query logs
    # by 2026-03-19; Northstar starts A/B test 2026-03-22; success = top-5
    # correct hit rate >= 80% on top 20 saved searches with no suppression
    # regression.
    #
    # Routing: SQL on competitors (NoiseGuard pricing_position) + SQL on
    # customers/implementations (risk fields) to identify the at-risk account,
    # then FTS for the milestone document with specific dates.
    # This is the hardest case: requires synthesising competitor SQL + customer
    # risk SQL + FTS document content into one coherent answer.
    # ------------------------------------------------------------------
    {
        "question": (
            "Which customer looks most likely to defect to a cheaper tactical "
            "competitor if we miss the next promised milestone, and what exactly "
            "is that milestone?"
        ),
        "expected_facts": [
            "BlueHarbor",
            "NoiseGuard",
            "2026-03-19",  # deadline for BlueHarbor to send schema export + logs
            "2026-03-22",  # Northstar A/B test start date
            "80",  # success threshold (80 percent / 80%)
            "tactical",  # how NoiseGuard is characterised in the risk framing
        ],
        "max_tool_calls": 8,  # FTS + SQL×2 (competitors + implementations) + lazy loads
        "required_nodes": ["hybrid_search_node", "sql_plan_node", "execute_node"],
        "forbidden_nodes": [],
        "notes": (
            "Hard hybrid: SQL for competitor pricing + customer risk, FTS + lazy load "
            "for specific milestone dates. Requires cross-source synthesis. "
            "Verified answer: BlueHarbor at risk of moving to NoiseGuard; "
            "milestone dates 2026-03-19 and 2026-03-22, 80% success bar."
        ),
    },
    # ------------------------------------------------------------------
    # Case 6 — hard hybrid: multi-account categorisation across a product
    #
    # Answer: Taxonomy/search semantics group: Arcadia Cloudworks, BlueHarbor
    # Logistics, CedarWind Renewables, HelioFab Systems, Pacific Health Network,
    # Pioneer Freight Solutions. Duplicate-action group: Helix Assemblies Inc.,
    # LedgerBright Analytics, LedgerPeak Software, MedLogix Distribution,
    # Peregrine Logistics Group, Pioneer Grid Retail LLC.
    #
    # Routing: SQL on customers JOIN implementations to get NAW Event Nexus
    # accounts + their status/risks; FTS to find which artifacts discuss
    # taxonomy degradation vs duplicate incidents for each account.
    # Requires reading multiple artifacts — lazy load budget likely hit.
    # ------------------------------------------------------------------
    {
        "question": (
            "Among the North America West Event Nexus accounts, which ones are "
            "really dealing with taxonomy/search semantics problems versus "
            "duplicate-action problems?"
        ),
        "expected_facts": [
            "Arcadia Cloudworks",  # taxonomy group
            "BlueHarbor",  # taxonomy group
            "CedarWind",
            "HelioFab",
            "Pacific Health Network",  # taxonomy group
            "Pioneer Freight",
            "Helix Assemblies",  # duplicate group
            "LedgerBright",  # duplicate group
            "Pioneer Grid Retail",  # duplicate group
            "LedgerPeak",
            "MedLogix Distribution",
            "Peregrine Logistics",
        ],
        "max_tool_calls": 8,  # FTS + SQL (customers/products/implementations) + lazy loads
        "required_nodes": ["hybrid_search_node", "sql_plan_node", "execute_node"],
        "forbidden_nodes": [],
        "notes": (
            "Hard hybrid: SQL to enumerate NAW Event Nexus customers, FTS + lazy load "
            "to classify each by issue type. Requires multi-artifact synthesis. "
            "Verified answer: 6 taxonomy accounts, 6 duplicate-action accounts listed above."
        ),
    },
    # ------------------------------------------------------------------
    # Case 7 — hard hybrid: cross-account pattern detection
    #
    # Answer: Recurring pattern, not a one-off. Accounts: MapleBridge Insurance,
    # City of Verdant Bay, Maple Regional Transit Authority, MapleBay Marketplace,
    # MapleFork Franchise Systems, MaplePath Career Institute, MapleWest Bank.
    # Shared failure: bad precedence metadata / stale caches / field alias
    # mismatches / delayed schema propagation → global rules win when
    # province/city/Canada-specific approval rules should win.
    #
    # Routing: FTS on "approval bypass" / "Canada approval" / "precedence" to
    # surface artifacts across all seven accounts; SQL on customers (Canada region)
    # to confirm account names. Lazy load needed to extract the shared pattern.
    # ------------------------------------------------------------------
    {
        "question": (
            "Do we have a recurring Canada approval-bypass pattern across accounts, "
            "or is MapleBridge basically a one-off? Give me the customer names and "
            "the shared failure pattern in plain English."
        ),
        "expected_facts": [
            "MapleBridge Insurance",  # anchor account named in question
            "MapleFork Franchise Systems",  # recurring account
            "Maple Regional Transit Authority",  # recurring account
            "MapleBay Marketplace",  # recurring account
            "MapleWest Bank",  # recurring account
            "MaplePath Career Institute",
            "City of Verdant Bay",
        ],
        "max_tool_calls": 8,  # FTS (multiple queries) + SQL (Canada customers) + lazy loads
        "required_nodes": ["hybrid_search_node", "sql_plan_node", "execute_node"],
        "forbidden_nodes": [],
        "notes": (
            "Hard hybrid: FTS across multiple accounts for approval-bypass artifacts, "
            "SQL on customers for Canada region. Requires pattern synthesis across 7 accounts. "
            "Verified answer: recurring pattern, 7 named accounts, shared root cause is "
            "precedence/metadata/alias issues letting global rules override regional ones."
        ),
    },
    # ------------------------------------------------------------------
    # Case 8 — pure SQL: at-risk accounts
    # ------------------------------------------------------------------
    {
        "question": "Which accounts are currently at risk or stalled?",
        "expected_facts": [
            "at risk",
        ],
        "max_tool_calls": 2,
        "required_nodes": ["sql_plan_node", "execute_node"],
        "forbidden_nodes": ["hybrid_search_node"],
        "notes": "Pure SQL: LOWER(status) LIKE pattern on implementations.",
    },
    # ------------------------------------------------------------------
    # Case 8b — conversational follow-up to Case 8
    #
    # Tests multi-turn coherence: the agent must resolve "those customers"
    # from the prior turn's context (injected via recent_turns).
    # chain_from causes the runner to execute Case 8 first and pass its
    # Q&A as recent_turns before invoking this case.
    # ------------------------------------------------------------------
    {
        "question": "What are the names of those customers?",
        "chain_from": "Which accounts are currently at risk or stalled?",
        "expected_facts": [
            "BlueHarbor",
            "Harborline",
            "Maple Ridge University",
            "MapleHarvest",
            "NordFryst",
            "Department of Regional Services",
            "Province of Laurentia",
        ],
        "max_tool_calls": 2,
        "required_nodes": ["sql_plan_node", "execute_node"],
        "forbidden_nodes": ["hybrid_search_node"],
        "notes": (
            "Multi-turn follow-up to Case 8: agent must resolve 'those customers' "
            "from prior turn context. Routing should be pure SQL — no FTS needed. "
            "Verified against DB: BlueHarbor Logistics, Department of Regional "
            "Services (DRS), Harborline Hospitality Group, Maple Ridge University "
            "System, MapleHarvest Grocers, NordFryst AB, Province of Laurentia."
        ),
    },
    # ------------------------------------------------------------------
    # Case 9 — pure SQL: competitor landscape
    # ------------------------------------------------------------------
    {
        "question": "What are the main competitors and their pricing models?",
        "expected_facts": [
            "NoiseGuard",
        ],
        "max_tool_calls": 2,
        "required_nodes": ["sql_plan_node", "execute_node"],
        "forbidden_nodes": ["hybrid_search_node"],
        "notes": "Pure SQL: competitors table scan.",
    },
    # ------------------------------------------------------------------
    # Case 10 — pure SQL: company overview
    # ------------------------------------------------------------------
    {
        "question": "What does the company do and what is its core differentiation?",
        "expected_facts": [],  # facts depend on actual DB content
        "max_tool_calls": 2,
        "required_nodes": ["sql_plan_node", "execute_node"],
        "forbidden_nodes": ["hybrid_search_node"],
        "notes": "Pure SQL: single-row company_profile table.",
    },
    # ------------------------------------------------------------------
    # Case 11 — SQL with JSON extraction: blueprint milestones
    # ------------------------------------------------------------------
    {
        "question": "What milestones were promised to BlueHarbor in their blueprint?",
        "expected_facts": [
            "BlueHarbor",
        ],
        "max_tool_calls": 2,
        "required_nodes": ["sql_plan_node", "execute_node"],
        "forbidden_nodes": [],  # hybrid acceptable but SQL is sufficient
        "notes": "SQL: json_extract on scenarios.blueprint_json milestones field.",
    },
    # ------------------------------------------------------------------
    # Case 12 — hybrid: support ticket FTS + customer SQL
    # ------------------------------------------------------------------
    {
        "question": "Are there any open support tickets related to search relevance issues?",
        "expected_facts": [
            "search",
        ],
        "max_tool_calls": 4,
        "required_nodes": ["hybrid_search_node"],
        "forbidden_nodes": [],  # SQL join may or may not occur
        "notes": "FTS on artifact_type=support_ticket + optional customer join.",
    },
]
