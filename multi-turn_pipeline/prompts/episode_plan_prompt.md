You are generating a multi-turn episode-level plan for a tool-using assistant.

Inputs:
- domain_lock: {{domain_lock}}
- selected_tools (JSON array): {{selected_tools}}
- usage_stats_summary: {{usage_stats_summary}}
- constraints:
  - num_turns: 2-4
  - tools_per_episode: ~10
  - calls_per_turn: 1-3
  - anchor_required: true
  - monotonic_progress: true

Your task:
1) Create a coherent real-world scenario and persona.
2) Define a single global objective, plus at most one natural extension.
3) Define state variables that must persist across turns.
4) Create a 2-4 turn outline where each turn:
   - advances the global objective
   - introduces a missing piece that requires tool calls
   - references anchors from the previous turn (for turn >= 2)

Output only valid JSON with this schema:
{
  "scenario": "...",
  "persona": "...",
  "domain": "...",
  "global_objective": "...",
  "optional_extension": "... or empty",
  "state_variables": [
    {"name": "...", "type": "...", "description": "..."}
  ],
  "turn_outline": [
    {
      "turn_index": 1,
      "turn_goal": "...",
      "expected_output": "...",
      "anchors_from_prev": [],
      "anchor_to_produce": ["..."],
      "notes": "..."
    },
    {
      "turn_index": 2,
      "turn_goal": "...",
      "expected_output": "...",
      "anchors_from_prev": ["..."],
      "anchor_to_produce": ["..."],
      "notes": "..."
    }
  ]
}

Rules:
- Use only tools from selected_tools.
- Provide realistic anchors (IDs, names, dates, amounts).
- Ensure turn 2+ references anchors_from_prev.
- Do not include markdown or extra commentary.
