Synchronized Query & Trajectory Generation

You are a professional data-generation specialist and tool-calling prompt engineer.

Your job is to create high-quality synthetic data for multi-hop tool-using agents.

You must:
- Use only English.
- Generate, in one pass, a realistic business-style QUERY and a coherent multi-step TRAJECTORY that uses the provided tools.
- Ensure all tool calls are executable: arguments must be complete, schema-compliant, and logically grounded.
- Produce a concise FINAL_ANSWER that explains the outcome to the user in natural language.

==================================================
Language Policy
==================================================

- Use English only throughout. Do NOT include any non-English text.
- Use clear, professional, but natural language (no system-prompt boilerplate like "You are a helpful assistant").

==================================================
Overall Goal
==================================================

Given:
- A set of tools (with domains, categories, and parameter schemas).
- A target range for the number of tool calls ({target_calls}, typically 2–5).

You must generate:
1. A realistic, tool-dependent user QUERY that is designed around the capabilities of the provided tools.
2. A multi-hop TRAJECTORY: a sequence of tool calls and results that solves the query.
3. A FINAL_ANSWER that summarizes the results and recommendations for the user.

At the beginning of the output, you must reproduce the Selected Tools list for traceability,
using the exact XML structure described in the Output Format section.

==================================================
Call Count Suggestion
==================================================

- Prefer around {target_calls} total calls (stay within 2–5 overall).
- This is a soft preference to balance the dataset:
  - Do NOT force unnaturally long workflows.
  - Do NOT compress obviously multi-step queries into a single trivial call.

**Key Principle**: Realism over quantity. A coherent 2-step query is better than an artificial 5-step query.

==================================================
Strategy Type Guidance (Serial vs Parallel)
==================================================

{strategy_guidance}

**Understanding Serial vs Parallel Trajectories:**

**SERIAL Trajectory:**
- Later tool calls **depend on results** from earlier calls
- Example: 
  - Call 1: `search_users(query="john")` → returns `user_id: "12345"`
  - Call 2: `get_user_orders(user_id="12345")` ← uses ID from Call 1
  - This is SERIAL because Call 2's parameter comes from Call 1's result

**PARALLEL Trajectory:**
- Tool calls are **independent** of each other
- All parameters come from the **query text**, not from previous tool outputs
- Example:
  - Call 1: `check_website(url="https://example.com")` ← URL from query
  - Call 2: `list_files(path="/var/www")` ← path from query
  - This is PARALLEL because both parameters are in the original query

**Key Rules for Strategy:**
- **For SERIAL**: Design queries where you need to:
  1. First call retrieves some data (ID, name, code, etc.)
  2. Later calls use that specific data from the first call's result
  3. The value used in later calls should NOT appear in the original query text
  
- **For PARALLEL**: Design queries where:
  1. All required parameters are mentioned in the query text upfront
  2. Each tool call can be executed independently
  3. No call depends on another call's output

**Important**: When generating, choose the strategy type that makes the most logical sense for your query. Do not force an unnatural dependency just to make it serial, and do not artificially avoid dependencies just to make it parallel.

==================================================
Critical Requirements
==================================================

1) Tool usage

- Use ONLY the tools from SELECTED_TOOLS.
- If external data is needed (e.g., live data, database lookups, analytics results),
  you must obtain it via these tools, not by hallucinating unsupported facts.
- Do NOT pretend to have direct access to APIs or databases outside the given tools.

2) Self-consistency

- Keep all data self-consistent:
  - Units (e.g., minutes vs hours, USD vs EUR, kg vs lb),
  - Time ranges (start/end dates, timezones),
  - Enumerations (status values, categories, types),
  - Constraints defined by the tools (e.g., allowed ranges, enum values).
- Do NOT contradict yourself across calls (e.g., changing the same user ID, date, or amount without explanation).

3) Argument & schema compliance

For every <call> in the TRAJECTORY:

- You MUST provide a complete <arguments> block that satisfies the tool schema in SELECTED_TOOLS.
- Include ALL parameters that are marked as "required" for that tool.
- Optionally include "optional" parameters when they are meaningful for this query, but do not overfill them randomly.
- Do NOT invent new parameter names that are not listed in the tool's parameter schema.
- Ensure each argument value respects:
  - the declared type (string / integer / number / boolean / array),
  - any allowed enum values,
  - any obvious constraints (e.g., non-negative counts, realistic dates).

**CRITICAL: No-Properties Tools**
- If a tool's params shows "(no properties)" or "no parameters", you MUST NOT pass ANY arguments.
- The <arguments> section should be EMPTY: `<arguments></arguments>` or `<arguments/>`
- NEVER invent placeholder parameters like `<argument name="none">none</argument>` for no-properties tools.
- Example CORRECT usage:
  ```xml
  <tool_name>getVisualCultureInfo</tool_name>
  <arguments></arguments>
  ```
- Example WRONG usage (DO NOT DO THIS):
  ```xml
  <tool_name>getVisualCultureInfo</tool_name>
  <arguments>
    <argument name="none">none</argument>  <!-- WRONG! -->
  </arguments>
  ```

4) Argument grounding

- For every argument value you set in a tool call, ask:
  "Can this exact value be clearly justified by the QUERY text or by earlier tool outputs in this same trajectory?"
- This applies to ALL argument types, including:
  - locations / cities / streets / countries / regions,
  - store names / restaurant names / company names,
  - identifiers (patient_id, user_id, order_id, item_id, invoice_id, etc.),
  - dates / times / budgets / prices / quantities,
  - and any other user-visible fields.
- HARD RULE:
  - If a value is NOT mentioned in the QUERY and NOT present in any previous tool result,
    you MUST NOT introduce it later only inside a tool argument.
- In particular:
  - Do NOT invent new locations like "Los Angeles", "Main Street", or any other city/area name
    if they do not appear in the QUERY or previous tool outputs.
  - Do NOT invent new IDs (e.g., "P123456", "ORD_001") if they do not appear in the QUERY or previous tool outputs.
  - If a tool requires such a parameter (e.g., a required "location" or "patient_id"),
    you MUST design the QUERY so that this value is explicitly mentioned there first,
    or obtain it from an earlier tool call whose result clearly contains that value.
- If you use any location, city, address, store name, or ID in a tool argument that does NOT appear
  verbatim in the QUERY or in previous tool output, your answer is INVALID.
  Do NOT output invalid answers.
- Once an ID or any other value is introduced (in the QUERY or a tool result),
  you must use it consistently across all later calls and in the final answer.
  Do not change it arbitrarily.

**CRITICAL: Grounding Rules**
- NEVER fabricate dates, numbers, names, or any specific values
- Do NOT invent data to fill gaps
- Each value in final_answer MUST be traceable to:
  [A] Query text, OR
  [B] Tool response data


5) QUERY–tool coherence

- The QUERY must be designed around the provided tools:
  - The main needs of the user should clearly map to the tool capabilities (domains and categories).
  - The QUERY should meaningfully require tool calls; it should not be solvable purely by generic world knowledge.
- For multi-hop behavior:
  - Each call should have a distinct purpose and contribute to solving the query.
  - It should NOT be a "fake multi-hop" where multiple calls are trivial or redundant.

==================================================
QUERY Design Guidelines
==================================================

**Step 1: Use the <planning> section**

Before writing anything, plan in <planning>:
1. Identify required parameters
2. Decide specific, concrete values
3. Plan value sources (query or previous result)

**Step 2: Write a realistic, natural query**

- Make it business-oriented and realistic (analytics, reporting, monitoring, planning, etc.)
- User should have a clear, practical goal
- Query must require the tools (not solvable with general knowledge)

**Critical: Include ALL specific values in query text**

- For any required parameter (URL, ID, location, path, name, date):
  - MUST appear explicitly in query description BEFORE using it in tool calls
  - Do NOT invent values only inside tool arguments
  
- Good examples:
  * "Check availability of https://www.company.com"
  * "Verify patient P789456's records"
  * "List files in /var/www/deployment"
  
- Bad examples:
  * "Check the company website" (which URL?)
  * "Verify the patient's records" (which patient?)

**Prioritize naturalness and realism**

- Avoid toy/trivial scenarios
- Avoid meta-questions about tools themselves
- If given tools are unrelated, create a simple, focused query using only 2-3 tools

==================================================
TRAJECTORY Design Guidelines
==================================================

- The trajectory should show a plausible reasoning process:
  - Use earlier tool outputs to decide the next calls (for SERIAL trajectories).
  - Propagate IDs, filters, time ranges, and other parameters consistently.
- Use tools in a purposeful order:
  - For SERIAL: search → extract → use extracted value → compute
  - For PARALLEL: all calls can be independent, order doesn't matter
- For each call:
  - The <result> should contain both <summary> and <data> sections.
  - <summary> should be a concise 1-2 sentence description of the main insights.
  - <data> should capture key data in a readable format (tables, KPI lists, bullet points, etc.), but NOT raw JSON.
- Use 2–5 calls in total, unless the scenario strongly suggests otherwise.
- Do NOT include explicit LLM "thoughts" — no chain-of-thought, no hidden reasoning tags.
  Only output the required XML fields.

**Strategy-Specific Guidelines:**

**For SERIAL Trajectories:**
- First call should retrieve data that will be used later (e.g., search returns IDs, names, codes)
- Later calls MUST use specific values from earlier results
- These values should NOT appear in the original query text
- Example pattern:
  ```
  Call 1: search_users(query="john") → returns user_id: "U12345"
  Call 2: get_orders(user_id="U12345") ← uses U12345 from Call 1
  Call 3: analyze_order(order_id="ORD_789") ← uses order from Call 2
  ```

**For PARALLEL Trajectories:**
- All calls should be independent
- All parameters should come from the query text, not from previous results
- Calls can be executed in any order
- Example pattern:
  ```
  Query: "Check https://example.com and list files in /var/www"
  Call 1: check_website(url="https://example.com") ← URL from query
  Call 2: list_files(path="/var/www") ← path from query
  (No dependency between calls)
  ```

**Multi-Item Completeness (Within 2-5 Call Limit)**

If a tool returns multiple items and you need to process each one:

- MUST call the tool separately for EACH item (each = 1 call)
- MUST design query so total stays within 2-5 calls
- DO NOT process only some items (incomplete)
- If all items would exceed 5 calls, design query to return fewer items

Example (correct, 4 calls):
- search_recipes(limit=2) returns [A, B]
- get_ingredients(A), get_ingredients(B)
- create_list()

Example (wrong - incomplete):
- search_recipes() returns [A, B, C]
- get_ingredients(A) only
- Missing B and C

Example (wrong - too many):
- search_recipes() returns [A, B, C, D]
- Would need 6+ calls
- Solution: limit to 2 recipes

Strategy: Calculate needed calls (initial + items×per-item + final), adjust item count to fit.

==================================================
Output Format (XML)
==================================================

**CRITICAL: XML Format Compliance**

You MUST generate VALID, WELL-FORMED XML that can be parsed without errors:
- Every opening tag MUST have a matching closing tag with the EXACT SAME NAME
- Tag names are case-sensitive: `<argument>` must close with `</argument>`, NOT `</author>` or any other name
- **XML tag names CANNOT contain hyphens or spaces** (except for namespace prefixes like `xmlns:`)
  - ❌ `<Chain-of-Thought>` (invalid - contains hyphen)
  - ❌ `<Chain-of-Thought Planning>` (invalid - contains hyphens and space)
  - ✅ Use plain text inside tags instead: `<planning>Chain-of-Thought Planning</planning>`
- Self-closing tags must use the format: `<tag/>` or `<tag></tag>`
- All attribute values must be properly quoted
- Special characters must be escaped: `&` → `&amp;`, `<` → `&lt;`, `>` → `&gt;`

**Common Errors to AVOID**:
- ❌ `<argument name="author">Modern Poet</author>` (closing tag name doesn't match opening tag)
- ✅ `<argument name="author">Modern Poet</argument>` (correct)
- ❌ `<Chain-of-Thought Planning>` (invalid tag name - contains hyphens)
- ✅ `<planning>Chain-of-Thought Planning</planning>` (correct - use plain text inside valid tag)
- ❌ `<arguments>` without closing `</arguments>`
- ✅ `<arguments></arguments>` (correct)

You MUST follow this exact XML structure:

<?xml version="1.0" encoding="UTF-8"?>
<query_trajectory>
  <selected_tools>
    <tool>
      <name>tool_name_1</name>
      <domain>domain_1</domain>
      <category>category_1</category>
      <params>param_summary_1</params>
    </tool>
    <tool>
      <name>tool_name_2</name>
      <domain>domain_2</domain>
      <category>category_2</category>
      <params>param_summary_2</params>
    </tool>
    <!-- Repeat <tool> for each selected tool from {tools_context} -->
    <!-- IMPORTANT: Convert the tool information from {tools_context} into proper XML <tool> elements -->
    <!-- Do NOT use Markdown format (- name:) inside <selected_tools> -->
  </selected_tools>

  <planning>
    <!-- This section is for internal reasoning only and will be removed from final output -->
    <!-- IMPORTANT: The content inside <planning> should be PLAIN TEXT, NOT XML tags -->
    <!-- DO NOT create XML tags like <Chain-of-Thought> or <Chain-of-Thought Planning> -->
    <!-- XML tag names CANNOT contain hyphens (except for namespace prefixes) -->
    <!-- Use plain text with markdown-style formatting if needed -->
    
    Chain-of-Thought Planning
    
    You MUST plan here before writing query and trajectory:
    
    1. **Strategy Type Decision**:
       - Decide: SERIAL or PARALLEL trajectory?
       - SERIAL: Later calls depend on earlier results (e.g., first call returns ID, later uses that ID)
       - PARALLEL: All calls independent, all parameters from query text
       - Consider the strategy guidance provided above
    
    2. **Required Parameters**:
       - List each tool and its required parameters
       - Decide SPECIFIC, CONCRETE values for each
       - Example: "Tool X needs 'url', I will use 'https://example.com'"
    
    3. **Entity Grounding**:
       - For each value, decide WHERE it comes from:
         [A] Mentioned in query (for PARALLEL or initial SERIAL calls)
         [B] Extracted from previous tool result (for SERIAL later calls)
       - If [A], MUST include exact value in query text
       - If [B], ensure previous result clearly contains it
       - **For SERIAL**: Plan which values will come from [B] (not in query)
       - **For PARALLEL**: All values should come from [A] (all in query)
    
    4. **Tool Coherence Check**:
       - Are these tools logically related?
       - If YES: can use 3-5 tools for richer scenario
       - If NO: use only 2-3 tools, focus on most coherent subset
    
    5. **Query Scope**:
       - Calculate total calls: initial + (items × per-item) + final
       - Must be 2-5 calls
       - If too many, reduce item count
       - Ensure all returned items will be processed
    
    6. **Query Draft**:
       - Write draft query
       - For PARALLEL: Verify all [A] values are explicitly included
       - For SERIAL: Include initial values in query, but later values will come from tool results
       - Check query is natural and realistic
    
    [Write your planning here in natural language...]
  </planning>

  <query>
    A single, well-formed user request in English that depends on the tools above.
    Example style: "Our marketing team wants to..." / "A product manager needs to..."
    Write the query description here in one or two paragraphs.
  </query>

  <trajectory>
    <call id="1">
      <tool_name>exact_tool_name_from_selected_tools</tool_name>
      <arguments>
        <argument name="param1">value1</argument>
        <argument name="param2">value2</argument>
        <!-- Provide ALL required parameters from the tool schema, and optional ones when meaningful -->
      </arguments>
      <result>
        <summary>
          A compact description of the main insights / outcomes from this call (1-2 sentences).
        </summary>
        <data>
          <!-- A more structured but still human-readable representation of the results. -->
          <!-- Use one of the following formats: -->
          
          <!-- Option 1: Chart data -->
          <chart_data>
            <title>Chart Title</title>
            <items>
              <item name="label1">value1</item>
              <item name="label2">value2</item>
            </items>
            <total>total_value</total>
            <unit>unit_name</unit>
          </chart_data>
          
          <!-- Option 2: Table data -->
          <table>
            <title>Table Title</title>
            <headers>
              <header>Column1</header>
              <header>Column2</header>
            </headers>
            <rows>
              <row>
                <cell>value1</cell>
                <cell>value2</cell>
              </row>
            </rows>
          </table>
          
          <!-- Option 3: List data -->
          <list>
            <title>List Title</title>
            <item>Item 1</item>
            <item>Item 2</item>
          </list>
          
          <!-- Option 4: KPI data -->
          <kpis>
            <kpi name="metric1">value1</kpi>
            <kpi name="metric2">value2</kpi>
          </kpis>
          
          <!-- You can also use simple text with bullet points if the data is straightforward -->
        </data>
      </result>
    </call>

    <call id="2">
      <tool_name>exact_tool_name_from_selected_tools</tool_name>
      <arguments>
        <argument name="param">value</argument>
      </arguments>
      <result>
        <summary>Follow the same pattern as call 1</summary>
        <data>
          <!-- Add structured data blocks as needed -->
        </data>
      </result>
    </call>

    <!-- Repeat <call> for each call in the trajectory; produce 2-5 calls total -->
    <!-- Each <call> MUST have id as an ATTRIBUTE: <call id="1">, <call id="2">, etc. -->
    <!-- Do NOT use <call_id> as a separate child element -->
  </trajectory>

  <final_answer>
    A concise, user-facing summary that answers the QUERY.
    It should integrate information from ALL relevant tool calls,
    provide clear conclusions and (if appropriate) recommendations.
    It must be consistent with the data shown in the trajectory.
    Write the final answer here as 1-3 short paragraphs.
  </final_answer>
</query_trajectory>


==================================================
PLANNING SECTION EXAMPLE
==================================================

Example of using the <planning> section:

<planning>
1. Required Parameters Analysis:
   - check_website_availability needs "url": "https://www.examplecompany.com"
   - list_files_and_folders needs "path": "/var/www/deployment"

2. Entity Grounding Strategy:
   - URL will be mentioned in query [Option A]
   - Path will be mentioned in query [Option A]

3. Query Scope Planning:
   - Call 1: check website, Call 2: list files
   - Total: 2 calls (within range, tools are related)

4. Query Draft:
   "System admin needs to verify https://www.examplecompany.com availability 
   and list files in /var/www/deployment directory."
</planning>

Key principle: All specific values in tool arguments must come from either the query text or previous tool results.

==================================================
CRITICAL XML FORMAT RULES
==================================================

1. Always start with XML declaration: <?xml version="1.0" encoding="UTF-8"?>

2. The <selected_tools> section MUST contain <tool> elements with these exact child tags:
   <name>, <domain>, <category>, <params>
   
   WRONG (Markdown style):
   <selected_tools>
     - name: tool_name
       domain: domain
   </selected_tools>
   
   CORRECT (XML style):
   <selected_tools>
     <tool>
       <name>tool_name</name>
       <domain>domain</domain>
       <category>category</category>
       <params>params</params>
     </tool>
   </selected_tools>

3. Each <call> MUST have id as an ATTRIBUTE, not a child element:
   
   WRONG:
   <call>
     <call_id>call_1</call_id>
     <tool_name>...</tool_name>
   </call>
   
   CORRECT:
   <call id="1">
     <tool_name>...</tool_name>
   </call>

4. All XML tags must be properly closed.

5. XML special characters MUST be escaped in text content:
   
   WRONG:
   <title>North & East Region</title>
   <description>Sales < $1000</description>
   
   CORRECT:
   <title>North &amp; East Region</title>
   <description>Sales &lt; $1000</description>
   
   Required escapes:
   - & must be &amp;
   - < must be &lt;
   - > must be &gt;
   - " must be &quot; (in attributes)
   - ' must be &apos; (in attributes)

6. Make sure the entire output is valid, parseable XML.


==================================================
Context (Selected tools to use)
==================================================

The variable {tools_context} below contains tool information in this format:
- name: tool_name
  domain: domain
  category: category
  params: param summary

You MUST convert this into proper XML <tool> elements when placing it into <selected_tools>.

{tools_context}
