"""
Annotated SQL examples for the ``run_sql`` tool's SQL planner.

Each block mirrors what the planner sees in ``PLAN_USER_PROMPT``:
  sql_intent — natural-language retrieval goal from the ReAct agent
  fts_ids    — relational IDs from ``hybrid_search`` / tool args, or (none)

When fts_ids are present, use WHERE <id_col> IN (...) over LIKE matching.
Fall back to LOWER(col) LIKE only when fts_ids is (none).

Do NOT query artifacts or artifacts_fts — document search is ``hybrid_search``.
"""

FEW_SHOT_EXAMPLES: str = """\
-- =========================================================================
-- Example 1
-- sql_intent: Retrieve all implementations whose status indicates risk or stalling.
-- fts_ids: (none)
-- Reasoning: No FTS IDs available — pure SQL scan. status is free-text so
--            use LOWER() LIKE to catch all variants of risk/stalled wording.
-- =========================================================================
SELECT
    c.name              AS customer_name,
    c.account_health,
    c.crm_stage,
    c.region,
    i.status,
    i.contract_value,
    i.risks_json
FROM implementations i
JOIN customers c ON i.customer_id = c.customer_id
WHERE LOWER(i.status) LIKE '%at risk%'
   OR LOWER(i.status) LIKE '%stalled%'
   OR LOWER(i.status) LIKE '%remediation%'
   OR LOWER(i.status) LIKE '%escalation%'
ORDER BY i.contract_value DESC
LIMIT 20;

-- =========================================================================
-- Example 2
-- sql_intent: Retrieve pricing position and strengths/weaknesses for all competitors.
-- fts_ids: (none)
-- Reasoning: No fts_ids — return full competitor table.
--            strengths_json / weaknesses_json are JSON arrays; return raw.
-- =========================================================================
SELECT
    name,
    segment,
    pricing_position,
    strengths_json,
    weaknesses_json
FROM competitors
ORDER BY pricing_position, name
LIMIT 20;

-- =========================================================================
-- Example 3
-- sql_intent: Retrieve implementation risks, status, and scenario context for
--             customers identified in document search, scoped to the competitor
--             also mentioned in those documents.
-- fts_ids: {"fts_customer_ids": ["cus_10762173c26d", "cus_92bc8f749347"],
--           "fts_competitor_ids": ["cmp_88dc528f7db7"]}
-- Reasoning: Multiple fts_ids injected — use WHERE IN (...) for both
--            customers and competitors rather than LIKE matching.
--            Join scenarios for full customer context; join competitors
--            separately so answer_node can compare risk vs competitor threat.
-- =========================================================================
SELECT
    c.name              AS customer_name,
    c.account_health,
    i.status,
    i.contract_value,
    r.value             AS risk_item,
    s.trigger_event,
    s.pain_point,
    comp.name           AS competitor_name,
    comp.pricing_position
FROM implementations i
JOIN customers c    ON i.customer_id = c.customer_id
JOIN scenarios s    ON c.scenario_id = s.scenario_id,
     json_each(i.risks_json) r
JOIN competitors comp ON comp.competitor_id IN ('cmp_88dc528f7db7')
WHERE i.customer_id IN ('cus_10762173c26d', 'cus_92bc8f749347')
ORDER BY i.contract_value DESC
LIMIT 20;

-- =========================================================================
-- Example 4
-- sql_intent: Retrieve pricing and competitive positioning for the competitor
--             referenced in document search results.
-- fts_ids: {"fts_competitor_ids": ["cmp_88dc528f7db7"]}
-- Reasoning: Single fts_competitor_id — target that row directly rather
--            than scanning all competitors.
-- =========================================================================
SELECT
    name,
    segment,
    pricing_position,
    strengths_json,
    weaknesses_json
FROM competitors
WHERE competitor_id IN ('cmp_88dc528f7db7')
LIMIT 20;

-- =========================================================================
-- Example 5
-- sql_intent: Retrieve trigger events and blueprint summaries for customers
--             from document search who are in renewal review stage.
-- fts_ids: {"fts_customer_ids": ["cus_92bc8f749347", "cus_b430f59e0caf"],
--           "fts_scenario_ids": ["scn_23e3314ca20f", "scn_bf54536c2c07"]}
-- Reasoning: Both fts_customer_ids and fts_scenario_ids injected — use
--            scenario_id IN (...) as primary filter (more direct than
--            joining through customers). crm_stage filter from intent.
--            blueprint_json is an object; valid keys include: summary,
--            pain_point, trigger_event, crm_stage, industry, region,
--            deployment_model, primary_competitor_name, primary_product_name,
--            account_health, annual_revenue_band, company_size_band,
--            root_cause_hint, subindustry. ONLY use these exact key names.
-- =========================================================================
SELECT
    c.name              AS customer_name,
    c.crm_stage,
    c.account_health,
    s.trigger_event,
    s.pain_point,
    json_extract(s.blueprint_json, '$.summary') AS blueprint_summary
FROM scenarios s
JOIN customers c ON c.scenario_id = s.scenario_id
WHERE s.scenario_id IN ('scn_23e3314ca20f', 'scn_bf54536c2c07')
  AND c.crm_stage LIKE '%renewal%'
LIMIT 20;

-- =========================================================================
-- Example 6
-- sql_intent: Retrieve success metrics promised to customers currently in
--             the expansion cycle.
-- fts_ids: (none)
-- Reasoning: No FTS IDs — filter by crm_stage from intent. success_metrics_json
--            is a JSON array; use json_each() to expand into individual rows.
-- =========================================================================
SELECT
    c.name          AS customer_name,
    c.crm_stage,
    i.status,
    m.value         AS success_metric
FROM implementations i
JOIN customers c ON i.customer_id = c.customer_id,
     json_each(i.success_metrics_json) m
WHERE c.crm_stage = 'expansion cycle'
ORDER BY c.name
LIMIT 20;

-- =========================================================================
-- Example 7
-- sql_intent: Retrieve employees working in Customer Success and their regions.
-- fts_ids: (none)
-- Reasoning: Pure SQL — employees table, structured department field.
--            Column is full_name (not name). Use LIKE for partial match.
-- =========================================================================
SELECT
    full_name,
    title,
    department,
    region,
    management_level
FROM employees
WHERE LOWER(department) LIKE '%customer success%'
ORDER BY management_level, full_name
LIMIT 20;

-- =========================================================================
-- Example 8
-- sql_intent: Retrieve features and pricing for the product referenced in
--             document search results.
-- fts_ids: {"fts_product_ids": ["prd_f8d861694bac"]}
-- Reasoning: Single fts_product_id — target that product directly.
--            Without fts_ids, omit WHERE and return all products.
-- =========================================================================
SELECT
    name,
    category,
    description,
    pricing_model,
    core_use_cases_json,
    features_json
FROM products
WHERE product_id IN ('prd_f8d861694bac')
LIMIT 20;

-- =========================================================================
-- Example 9
-- sql_intent: Retrieve the company mission, differentiation, and pricing overview.
-- fts_ids: (none)
-- Reasoning: company_profile is a single-row table — no WHERE clause needed.
--            Always include LIMIT 1.
-- =========================================================================
SELECT
    name,
    mission,
    ideal_customer_profile,
    architecture_summary,
    differentiation,
    pricing_overview,
    compliance_posture
FROM company_profile
LIMIT 1;

-- =========================================================================
-- Example 10
-- sql_intent: Retrieve implementations that have been kicked off but are not
--             yet live, to identify overdue deployments.
-- fts_ids: (none)
-- Reasoning: Date arithmetic on structured fields — kickoff_date IS NOT NULL
--            and go_live_date is either NULL or in the future, combined with
--            status not yet active/deployed.
-- =========================================================================
SELECT
    c.name              AS customer_name,
    c.region,
    i.status,
    i.kickoff_date,
    i.go_live_date,
    i.contract_value,
    i.scope_summary
FROM implementations i
JOIN customers c ON i.customer_id = c.customer_id
WHERE i.kickoff_date IS NOT NULL
  AND (i.go_live_date IS NULL OR i.go_live_date > date('now'))
  AND LOWER(i.status) NOT LIKE '%active%'
  AND LOWER(i.status) NOT LIKE '%deployed%'
ORDER BY i.kickoff_date ASC
LIMIT 20;

-- =========================================================================
-- Example 11
-- sql_intent: Retrieve at-risk customer implementations with their individual
--             risk items and success metrics listed as separate rows.
-- fts_ids: {"fts_customer_ids": ["cus_10762173c26d", "cus_92bc8f749347"]}
-- Reasoning: risks_json and success_metrics_json are ARRAYS OF STRINGS, not
--            objects. NEVER use json_extract(col, '$.key') on these columns —
--            there are no named keys. Use json_each() to expand each array
--            into one row per item. Two separate queries or a UNION is cleaner
--            than a cross-join; here we expand risks only and return metrics raw.
-- =========================================================================
SELECT
    c.name              AS customer_name,
    c.account_health,
    i.status,
    i.contract_value,
    r.value             AS risk_item,
    i.success_metrics_json
FROM implementations i
JOIN customers c ON i.customer_id = c.customer_id,
     json_each(i.risks_json) r
WHERE i.customer_id IN ('cus_10762173c26d', 'cus_92bc8f749347')
  AND (LOWER(i.status) LIKE '%risk%'
       OR LOWER(i.status) LIKE '%remediation%'
       OR LOWER(i.status) LIKE '%stalled%')
ORDER BY i.contract_value DESC
LIMIT 20;

-- =========================================================================
-- Example 12
-- sql_intent: Retrieve customer names, CRM stage, and account health for
--             accounts involved in a taxonomy-related renewal situation,
--             scoped to scenarios from document search.
-- fts_ids: {"fts_customer_ids": ["cus_10762173c26d", "cus_b430f59e0caf"],
--           "fts_scenario_ids": ["scn_bf54536c2c07", "scn_61cd84f1589c"]}
-- Reasoning: The question mentions "taxonomy rollout" and a specific date —
--            these are document-layer concepts, not structured column values.
--            trigger_event values are short generic phrases like
--            "new regional launch" — they never contain dates or topic labels
--            invented from the question. DO NOT filter on trigger_event or
--            blueprint_json.trigger_event using = or LIKE with a value from
--            the question. The fts_ids already scope to the right accounts
--            and scenarios — use scenario_id IN (...) and customer_id IN (...)
--            as the only filters.
-- =========================================================================
SELECT
    c.name              AS customer_name,
    c.crm_stage,
    c.account_health,
    s.trigger_event,
    s.pain_point
FROM scenarios s
JOIN customers c ON c.scenario_id = s.scenario_id
WHERE s.scenario_id IN ('scn_bf54536c2c07', 'scn_61cd84f1589c')
  AND c.customer_id IN ('cus_10762173c26d', 'cus_b430f59e0caf')
LIMIT 20;
"""
