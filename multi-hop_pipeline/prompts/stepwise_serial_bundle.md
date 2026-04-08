STEPWISE SERIAL BUNDLE (single file)

This file bundles all templates needed for ToolACE-style stepwise serial generation.
`generate_via_api.py` will split templates by the exact markers below.

===== TEMPLATE: QUERY_ONLY_JSON =====
Query Generation (Query-only)

You are generating synthetic user queries for multi-hop tool-using agents.

Requirements:
- Use English only.
- Produce ONE realistic business-style user query that is designed around the provided tools.
- The query should naturally require multiple tool calls (typically 2–5).
- Do NOT write any tool calls, trajectories, XML, or markdown sections.
- Include all concrete values needed for the workflow (dates, ranges, names, URLs, etc.) directly in the query text if they are meant to come from the user.

Selected tools (with schemas when available):
{tools_context_detailed}

Call-count preference: {target_calls} (soft guidance; keep it realistic).
Strategy target: {target_strategy} (serial means later steps should depend on earlier tool outputs; parallel means steps can be independent).

Output format (STRICT JSON):
{
  "query": "..."
}

===== TEMPLATE: AGENT_NEXT_CALL_JSON =====
Stepwise Serial Trajectory: Next Tool Call (Agent)

You are an agent that solves the user's query by calling tools step-by-step.

Hard rules:
- Use English only.
- You MUST output exactly ONE JSON object (no markdown, no code fences).
- Choose ONLY from the provided tools.
- Produce AT MOST ONE tool call in this step.
- If you are done (no more tool calls needed), set "done": true and do NOT include tool_name/arguments.

Serial dependency rule (CRITICAL):
- Starting from step 2, your tool call MUST use at least one specific identifier/value that appears in previous tool results.
- That identifier/value MUST NOT already appear in the original user query text.
- Prefer IDs/codes with digits or special characters (e.g., "CUST_10293", "order-7781", "0x1a2b...").

User query:
{query_text}

Selected tools:
{tools_context_detailed}

Conversation so far (previous tool calls and results):
{history_text}

Target total calls: {target_calls} (keep within 2–5 overall).
Current step index (1-based): {step_index}

Output JSON schema:
If continuing:
{
  "done": false,
  "tool_name": "EXACT_TOOL_NAME",
  "arguments": { "param": "value" }
}

If finished:
{
  "done": true
}

===== TEMPLATE: TOOL_EXECUTE_JSON =====
Stepwise Serial Trajectory: Tool Execution (Simulator)

You simulate the tool's output given the tool schema and the provided arguments.

Hard rules:
- Use English only.
- Output exactly ONE JSON object (no markdown, no code fences).
- The output MUST be self-consistent with the arguments.
- If appropriate, include at least one concrete identifier/value (ID/code) in the results that could be used by later steps.

Original user query (for context only):
{query_text}

Tool:
{tool_block}

Arguments (JSON):
{arguments_json}

Output format (STRICT JSON):
{
  "summary": "1-2 sentences describing the outcome and key values (include IDs/codes when appropriate).",
  "list_title": "Key Output",
  "list_items": ["item 1", "item 2", "item 3"]
}

===== TEMPLATE: FINAL_ANSWER_JSON =====
Stepwise Serial Trajectory: Final Answer

You are writing the final user-facing answer based ONLY on the user query and the tool results so far.

Rules:
- Use English only.
- Do NOT include any tool calls.
- Be concise (1–3 short paragraphs).
- Every specific value should be traceable to the tool outputs or the query text.

User query:
{query_text}

Tool calls and results:
{history_text}

Output format (STRICT JSON):
{
  "final_answer": "..."
}


