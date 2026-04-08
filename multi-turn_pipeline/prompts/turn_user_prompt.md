You are generating the next user message for a multi-turn episode.

Inputs:
- episode_level_plan (JSON): {{episode_level_plan}}
- selected_tools (JSON array): {{selected_tools}}
- conversation_history (JSON): {{conversation_history}}
- global_state (JSON): {{global_state}}
- turn_index: {{turn_index}}

Requirements:
- The message must reflect the persona and scenario.
- For turn_index >= 2, the message must explicitly reference an anchor from the previous turn (ID, name, date, or value).
- The request must introduce a missing piece that cannot be answered without 1-3 tool calls.
- The user must explicitly provide every required argument value for the intended tool call(s) in the message text.
- Do not imply or infer tool arguments; spell them out directly (IDs, names, numbers, booleans, dates, etc.).
- Use the anchor multiple times if natural (e.g., restate the keyword or ID) to strengthen cross-turn linkage.
- Use a natural, first-person user tone. Avoid meta-instructions like "Please provide..." or "Use the tool...".
- Do not narrate the assistant's actions; speak as the user asking for help.
- Explicitly include all numeric values and boolean values (e.g., speeds, distances, counts, true/false) needed for the tool calls.
- Avoid meta language about planning, prompts, or tools.

Output only the user message (plain text). No JSON, no markdown.
