Instructions: You will generate a high-quality single-turn tool invocation sample data for English business scenarios based on the given individual tool definition. Please strictly follow the output format and constraints below.

1. Known Tool Definition (JSON)
{tool_json}

2. Generation Requirements
1) Language: Use English throughout; professional terminology, natural expression, business-oriented; avoid phrases like "as an AI". Human text should be conversational and casual.
2) Objective: Generate four parts of single-turn dialogue around the tool's purpose:
   - human.value: User's natural language requirements, clear and able to trigger the tool; must not contain function names or system implementation details. Avoid precise ratios/numbers/thresholds; dates can use concise ranges.
   - function_call.value: Stringified JSON, in the format:
     {"name": "<tool_name>", "arguments": {<parameter_key_value_pairs>}}
     Note: Parameter fields, enums, formats, and required fields must fully satisfy the tool's inputSchema constraints; any time/company/resort fields must comply with the range and enum limitations described in the tool information; time-based tools must ensure time bounds are consistent with "earliest/latest/cannot be past/cannot span" rules; resort fields must match enum lists; if there are default values and range descriptions, set them reasonably.
    - observation.value: Stringified JSON (array), used to simulate tool returns. Fixed output order and format:
      1) First item must be detailed text overview:
         {"type": "text", "text": "{text_prefix}.\n\nAnalysis Period: YYYY-MM-DD to YYYY-MM-DD\nCore Metrics: Specific values and units\n\nKey Findings:\n• Key Point 1: Specific data\n• Key Point 2: Specific data\n• Key Point 3: Specific data\n\nSummary: Brief conclusion"}
         Note: text field must start with "{text_prefix}." and then provide detailed analysis report including time range, core metrics, key findings and summary
      2) Second item onwards are structured data blocks (1-2 as needed): {"type": "chart_data" | "table" | "list" | "kpis", ...}
      Requirements: Clear field naming, reasonable values, consistent with input; self-consistent values (sums/percentages/units). Avoid putting lengthy descriptions in text; try to put metrics in structured blocks. These structured data blocks contain specific analysis results, statistical data, chart data, etc.
     - gpt.value: Finally provide model reply, summarizing and providing insights on observation in English (do not repeat all data, focus on conclusions)
3) Data authenticity: Simulated data but must be self-consistent, correct units, reasonable value ranges; do not generate values that violate tool constraints.


3. Style Constraints
- human: Natural, casual, conversational; has business context; can include approximate time/location/objects; avoid rigid official style, templated and tool-like expressions.
  - Do not actively provide precise percentages or strict numerical targets in human
  - Can mention activities/holidays as background, but do not quantify their specific impact
  - Date expressions more conversational: such as "July 10th to 15th", "during National Day"
- observation: Stable format, unified field naming; concise text; if containing chart data, provide title, items/series, total and units.
  - Fixed order: first text overview, then structured data (chart_data/table/list/kpis)
  - text overview only serves as "introduction/conclusion", put specific numbers in structured blocks
- gpt: Give overall conclusion first, then key points; avoid redundancy.

4. Output Format (Strict JSON wrapping four string values, given section by section)
Please output only one JSON object containing the following keys, with all sub-field contents as strings:
{
  "human_value": "...human.value string...",
  "function_call_value": "...function_call.value stringified JSON...",
  "observation_value": "...observation.value stringified JSON array...",
  "gpt_value": "...assistant summary (string)"
}

5. Critical JSON Format Requirements
- ALL string values containing quotes MUST be properly escaped
- Use \" to escape quotes within string values
- Examples of correct escaping:
  * "The Wandering Earth 2" → "The Wandering Earth 2"
  * "Album "Jay" released" → "Album \"Jay\" released"
  * "The Romance of the Three Kingdoms" → "The Romance of the Three Kingdoms"
- Ensure ALL generated JSON can be parsed by standard JSON parsers
- Double-check that function_call_value and observation_value are valid JSON strings

6. Additional Rule Reminders
- If tool inputSchema contains dates: ensure format YYYY-MM-DD, and satisfy "start≤end", "earliest/latest date limits", "prediction from tomorrow", etc.
- If containing resort/company/channel enums: must take values from enum lists.
- If containing boolean/default values: if not emphasized, can use default values, but must be legal values.
- If containing array parameters (like event impacts): need to provide correctly structured object arrays, containing required fields.

7. Semantic Time Parsing and Alignment (Extremely Important)】
- Relative time expressions in human must be converted to strict date ranges in function_call, completely consistent with semantics:
  - "this month": use 01st of current month to today (current date)
  - "next month": use 01st of next month to last day of next month (prohibited to mix in current month dates)
  - "this weekend": use the date range of the nearest Saturday-Sunday
  - "next week": use next Monday to next Sunday
  - "recent month": use today minus 30 days interval (including today)
  - "July 10th to 15th", "two weeks before National Day", etc.: need to precisely expand to YYYY-MM-DD~YYYY-MM-DD, and ensure correct year
- If prediction tool: start_date must ≥ tomorrow; if human says "next month", then start_date=1st of next month, end_date=last day of next month
- Strictly prohibited: human says "next month" but function_call contains current month dates; once conflict occurs, prioritize human semantics to rewrite dates
- Today's date is October 13, 2025

8. Example (illustration only, do not copy field names)
Target tool: predict_visitor_flow (assuming category is analysis)
Possible human:
"Will Qiandao Lake be particularly crowded two weeks before National Day? Just give a rough daily estimate."
Corresponding function_call.value (string):
"{\"name\": \"predict_visitor_flow\", \"arguments\": {\"resort_name\": \"Qiandao Lake\", \"start_date\": \"2025-09-16\", \"end_date\": \"2025-09-30\", \"prediction_granularity\": \"daily\", \"include_confidence_interval\": true}}"
Corresponding observation.value (stringified array, illustration):
"[{\"type\": \"text\", \"text\": \"{text_prefix}.\\n\\nPrediction Period: 2025-09-16 to 2025-09-30\\nExpected Total Visitors: 285,000\\n\\nKey Findings:\\n• Weekday Average: 18,000-23,000\\n• Weekend Average: 30,000-36,000\\n• Peak: September 28th, expected 38,000\\n\\nSummary: Qiandao Lake visitor volume will significantly increase two weeks before National Day, recommend preparing for reception in advance\"}, {\"type\": \"chart_data\", \"chart_type\": \"line\", \"title\": \"Qiandao Lake Visitor Volume Prediction Two Weeks Before National Day\", \"data\": {...}}]"
Note: {text_prefix} will be automatically replaced with corresponding fixed prefix based on tool's category


