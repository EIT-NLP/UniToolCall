Query Augmentation (Two Types)

You are a professional data augmentation specialist for multi-hop tool-calling datasets.

Your job is to rewrite the QUERY text in controlled ways so that:
- the augmented queries remain solvable by the same fixed TRAJECTORY,
- the same tools, tool-call sequence, and argument structure remain valid,
- and the overall difficulty and realism of the queries are preserved or slightly improved.

==================================================
Language Policy
==================================================

- Use English only for all outputs.
- Do NOT include any non-English text.

==================================================
Goal
==================================================

Given:
- ONE original QUERY (user request),
- ONE fixed TRAJECTORY (sequence of tool calls, observations, and final answer),
- A list of SELECTED TOOLS (for context only),

You must produce exactly TWO augmented versions of the QUERY:

1) noisy_overspecified
   - Add extra, possibly irrelevant details or minor conditions.
   - The model is expected to ignore unnecessary information and still use the correct parameters.
   - Do not introduce new question to the query.

2) ambiguous_simplified
   - Make the user query shorter, more informal, and slightly more vague.
   - The query should look more like a real, high-level user request, but still contain all essential information.

==================================================
Very Important Invariants
==================================================

For BOTH augmentation types:

- Do NOT modify the trajectory or the final answer.
- Do NOT modify which tools are used, or the order of tool calls.
- Do NOT change the meaning of any key argument values used in the fixed trajectory.
- **Do NOT add requirements that the existing trajectory cannot satisfy.**

Key entities (for example, but not limited to):
- Cities / locations (e.g., "New York", "Los Angeles"),
- Date ranges / time periods (e.g., "Q1 2024", "2023-01-01 to 2023-03-31"),
- IDs (e.g., patient_id, user_id, order_id),
- Main objects (e.g., which product, which account, which branch),
- Any other value that appears as a critical argument in the fixed trajectory.

You may rephrase how these entities are described,
but you must NOT change them to different values
(e.g., you must not change "New York" to "Chicago", or "Q1 2024" to "last 5 years").

**Critical Check Before Finalizing**:
For each augmented query, ask yourself:
"Can the EXISTING trajectory (with its exact tool calls and arguments) fully satisfy this augmented query?"
- If YES → Good augmentation
- If NO → You added too much, remove the extra requirements

==================================================
Augmentation Types (exactly one variation per type)
==================================================

1) noisy_overspecified

Goal:
- Make the query look more detailed and slightly messy, like a real user who over-shares.
- Add extra information that does NOT change what the trajectory needs to do.

**CRITICAL: Stay Within Trajectory Capability**

Rules:
- You MAY add:
  - **Background context**: personal role, team name, department, company background
    Example: "As a marketing analyst at TechCorp..." (does not change the actual task)
  
  - **Irrelevant side information**: mentions of tangential concerns that do NOT require action
    Example: "...and I'm also thinking about Q2 planning later" (just a thought, not a requirement)
  
  - **Descriptive elaboration**: rephrasing the same requirement in more verbose ways
    Example: "I need to check" → "I need to carefully review and verify"
  
  - **Redundant confirmations**: repeating what is already implied
    Example: "find Italian restaurants" → "find Italian restaurants that serve Italian cuisine"
  
  - **Non-binding preferences**: soft preferences that do not create new filtering requirements
    Example: "preferably with good reviews" (soft, not a hard filter)

- You MUST NOT add:
  - ❌ **New filtering conditions**: date ranges, price thresholds, quantity limits, comparison criteria
    Example: "past 3 years", "under $200", "top 5", "grants over private investments"
  
  - ❌ **New analytical requirements**: comparisons, rankings, prioritizations, trend analysis
    Example: "compare A vs B", "rank by X", "identify the best", "analyze recent changes"
  
  - ❌ **New output requirements**: specific formats, additional fields, extra data points
    Example: "include revenue breakdown", "show year-by-year trends", "list all subcategories"
  
  - ❌ **Temporal constraints**: specific time periods, deadlines, time-based filtering
    Example: "from 2020-2023", "by next week", "recent updates in the last quarter"
  
  - ❌ **Conditional logic**: if-then requirements, multi-step decision rules
    Example: "if X is available, then do Y", "choose A unless B is better"

**Key Principle**: 
If your addition would require the trajectory to do something different, filter differently, 
call additional tools, or return different data → DO NOT add it.

Only add "noise" that a smart model should ignore to reach the same solution.

Intuition:
- The query becomes wordier and contains extra context/thoughts,
  but the SAME trajectory with the SAME tool calls and arguments still perfectly solves it.

2) ambiguous_simplified

Goal:
- Make the query shorter, more informal, and slightly more vague in phrasing,
  while still giving all essential information needed by the fixed trajectory.

Rules:
- You MAY:
  - remove implementation details (no mention of tools, no "call API", no "run this query"),
  - phrase the request as a high-level business goal or personal need,
  - simplify structure (fewer sentences, more conversational tone).
- You MUST:
  - explicitly mention all key entities required by the existing trajectory,
    such as the city, time range, product, main account, or ID.
  - preserve the same overall intent and scope:
    the same tools and argument values should still naturally solve the query.
- You MUST NOT:
  - drop any essential entity so that the trajectory becomes under-specified
    (for example, removing the city name when the trajectory filters by that city),
  - add new major queries or new required outputs,
  - suggest using a different set of tools or a different type of analysis.

Intuition:
- The user sounds more "lazy" or high-level,
  but a good agent can still infer that the same operations as in the fixed trajectory are appropriate.

==================================================
Constraints
==================================================

1) Do NOT modify the TRAJECTORY or the final answer in any way.
   - You only produce new QUERY texts.

2) Each augmented QUERY MUST be solvable by the SAME TRAJECTORY:
   - same tools,
   - same call order,
   - same argument structure and key values.

3) Do NOT contradict the fixed trajectory.
   - If the trajectory clearly uses a specific city/date/ID, your augmented QUERY must remain consistent with that.

4) Each augmented QUERY must be a single, coherent user request in natural English.

==================================================
Examples of Good vs Bad Augmentations
==================================================

**Example 1: Noisy Overspecified**

Original Query:
"Find Italian restaurants in New York."

✅ GOOD augmentation:
"As a food blogger based in Manhattan, I'm working on my next article and need to find 
Italian restaurants in New York. I'm particularly interested in authentic Italian cuisine, 
though I'm also considering French restaurants for a future post."

Why good:
- Adds background (food blogger, Manhattan, article)
- Adds soft preference (authentic Italian)
- Mentions tangential info (French restaurants for future) that doesn't require action NOW
- The trajectory can still just find Italian restaurants in New York

❌ BAD augmentation:
"Find Italian restaurants in New York with prices under $50, ratings above 4.5, 
open on weekends, and compare them with French restaurants to determine which cuisine 
has better value."

Why bad:
- Adds filtering requirements (price, ratings, open days) that trajectory may not support
- Adds comparison requirement (Italian vs French) that needs additional tools
- Adds analytical requirement (determine value) that trajectory doesn't do

---

**Example 2: Noisy Overspecified**

Original Query:
"Check the library stats and translate this Urdu sentence."

✅ GOOD augmentation:
"As an educational administrator planning the annual report, I need to check the library 
system statistics to understand our current capacity. I also need to translate this Urdu 
sentence for our multilingual documentation project. I'm hoping to complete this review 
by end of month, though the timeline is flexible."

Why good:
- Adds context (administrator, annual report, documentation project)
- Adds soft timeline (end of month, flexible) - not a hard requirement
- Same trajectory with same tool calls works perfectly

❌ BAD augmentation:
"Check the library stats for the past 3 years, compare them with the previous period, 
identify trends in user growth, and translate this Urdu sentence. Also prioritize 
libraries with more than 10,000 books."

Why bad:
- "past 3 years" - temporal filtering
- "compare with previous period" - new analytical requirement
- "identify trends" - new analysis not in trajectory
- "more than 10,000 books" - quantitative filtering

==================================================
Output Format (JSON)
==================================================

Return a single JSON object with the following structure:

{
  "augmented_queries": {
    "noisy_overspecified": {
      "query": "<string: the full rewritten user request>",
      "how": "<10-20 words: briefly describe what you changed>"
    },
    "ambiguous_simplified": {
      "query": "<string: the full rewritten user request>",
      "how": "<10-20 words: briefly describe what you changed>"
    }
  }
}

Notes:
- "query": one coherent paragraph per type (can be 2–4 sentences).
- "how": a short explanation of the augmentation strategy for that type,
         mentioning what was added/removed/changed at a high level.

**Final Checklist Before Submitting**:

For noisy_overspecified, verify:
- [ ] Did I add any date ranges, time periods, or "past X years"? → If YES, REMOVE them
- [ ] Did I add any comparison requirements (A vs B, better/worse, rank)? → If YES, REMOVE them
- [ ] Did I add any quantitative filters (price limits, rating thresholds, top N)? → If YES, REMOVE them
- [ ] Did I add any new output requirements beyond the original? → If YES, REMOVE them
- [ ] Can the EXACT SAME trajectory with EXACT SAME tool calls satisfy my augmented query? → Must be YES

For ambiguous_simplified, verify:
- [ ] Did I keep all essential entities (city names, IDs, key parameters)? → Must be YES
- [ ] Would a model still know to use the same tools in the same order? → Must be YES
- [ ] Is the query still specific enough for the trajectory to be the right solution? → Must be YES

==================================================
Inputs
==================================================

Original QUERY:
{query_text}

Fixed TRAJECTORY (do not change):
{trajectory_text}

Selected tools list (for context only, do not modify):
{tools_context}

Now, generate the JSON object with the two augmented queries.
