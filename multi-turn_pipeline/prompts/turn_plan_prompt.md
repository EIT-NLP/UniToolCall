You are generating a turn-level plan for a tool-using assistant.

Inputs:
- user_message: {{user_message}}
- conversation_history (JSON): {{conversation_history}}
- global_state (JSON): {{global_state}}
- selected_tools (JSON array): {{selected_tools}}

Output only valid JSON with this schema:
{
  "turn_goal": "...",
  "turn_strategy": "serial|parallel|mixed",
  "planned_calls": [
    {
      "tool_name": "...",
      "purpose": "...",
      "args_source": {
        "field": "user",
        "details": "User message must explicitly contain every argument value"
      },
      "depends_on_call_index": null
    }
  ],
  "required_anchors": ["..."],
  "expected_output": "..."
}

Rules:
- 1-3 planned calls.
- serial: later calls depend on prior outputs.
- parallel: no dependencies among calls.
- mixed: first parallel batch, then one dependent call.
- Required anchors must exist in history or global_state.
- Every tool call argument value must be explicitly stated in the user_message (verbatim or exact value), with no inference or fabrication.
- Do not plan any call if the required argument values are not present in the user_message.
- Do not include markdown or extra commentary.
