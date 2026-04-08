You are generating tool calls, tool outputs, and the assistant response for a single turn.

Inputs:
- turn_plan (JSON): {{turn_plan}}
- user_message: {{user_message}}
- conversation_history (JSON): {{conversation_history}}
- global_state (JSON): {{global_state}}
- selected_tools (JSON array): {{selected_tools}}

Output only valid JSON with this schema:
{
  "tool_calls": [
    {
      "tool_name": "...",
      "arguments": {"...": "..."}
    }
  ],
  "tool_outputs": [
    {
      "tool_name": "...",
      "output": {"...": "..."}
    }
  ],
  "assistant_response": "..."
}

Rules:
- The number and order of tool_calls must match turn_plan.planned_calls.
- All tool call argument values must be copied from the user_message (no invention, no inference).
- Do not introduce any new IDs, codes, names, or numbers that are not explicitly present in the user_message.
- Tool outputs must look realistic and consistent with tool schemas, and must not introduce new argument values.
- The assistant_response must be grounded in tool_outputs and history only.
- Do not include markdown or extra commentary.
