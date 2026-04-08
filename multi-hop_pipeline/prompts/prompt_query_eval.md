# Query Quality Evaluation

You are a data quality evaluator for a multi-hop tool-calling dataset.

Your current goal is to evaluate the **QUERY ONLY** (the user's request), not the model's trajectory or answer.

You will rate THREE metrics on a 1–10 scale:

1. Tool-fit
2. Clarity
3. Naturalness

You are given:
- A set of available tools and their capabilities.
- ONE query description (the user's request) from the XML output.
- A flag indicating whether this query is intended to be multi-hop (uses multiple tools / multiple tool calls).

IMPORTANT:
- The query evaluation only checks whether the query is well-designed **given the tools** and is theoretically solvable.
- Whether any particular trajectory actually solved it is a separate concern and should NOT influence these scores.

## METRIC 1: Tool-fit (1–10)

**Definition:**
Tool-fit measures **how well this query is designed around the given tool set**, and, if the query is intended to be multi-hop, whether using multiple tools/steps is genuinely necessary.

**Ask yourself three questions:**

1) **Capability Match**  
- Do the main requirements of the query clearly map to the capabilities described in the tools?  
- The query should NOT require core abilities that the tools simply do not provide (e.g., asking for web browsing or image generation when tools only provide calculator / database query).

2) **Tool Dependency**  
- To complete the query reasonably, do you **need** to call tools to obtain information or perform operations?  
- If the query can be answered almost entirely via generic world knowledge or hallucination, Tool-fit should be lower.  
- Good Tool-fit means the query naturally relies on the tools (e.g., needs fresh data, private DB info, or specific APIs).

3) **Multi-hop Necessity** (only if the query is intended to be multi-hop)  
- Each hop / each tool should have a clear and distinct role in achieving the final goal.  
- If you remove one hop or one tool call, the query can no longer be fully solved (or the result is clearly incomplete).  
- It should NOT be a fake multi-hop where one single tool call could already finish everything, but the query is artificially split into multiple hops/tools "just because".

**Scoring guidelines (1–10):**

- **1–3 (Poor Tool-fit)**  
  - The query's main requirements are weakly related to the tools.  
  - It heavily depends on abilities the tools do not have, OR essentially does not need tools at all.  
  - If marked as multi-hop, several steps are obviously unnecessary: removing many of them still solves the query.

- **4–7 (Moderate Tool-fit)**  
  - The query is generally designed around the tools.  
  - Most requirements can be handled by the available tools, but there may be one or two sub-requirements outside tool capabilities, OR in the multi-hop case, at least one hop feels optional or redundant.

- **8–10 (High Tool-fit)**  
  - The query is clearly tailored to this tool set: all key requirements fall within the tools' capabilities.
  - Completing the query strongly depends on using these tools (not just generic knowledge).
  - If it is multi-hop, every hop/tool has a necessary, well-defined role. Removing any hop leads to an incomplete or unsolved query.

## METRIC 2: Clarity (1–10)

**Definition:**
Clarity measures whether the query is **theoretically solvable**: is the information sufficient, are constraints self-consistent, and is the goal clearly defined?

**Check:**

- Are key parameters given explicitly or can they be reasonably defaulted from context?  
- Are constraints self-consistent (e.g., time ranges, locations, conditions) rather than contradictory?  
- Does the query avoid asking tools to do things outside their schema (e.g., fields that do not exist)?  
- Is the output goal clear (e.g., what needs to be computed, retrieved, compared, summarized)?

**Scoring guidelines (1–10):**

- **1–3 (Low Clarity)**  
  - Severe missing information or strong contradictions.  
  - It is very hard to construct a reasonable solution path even in theory.

- **4–7 (Moderate Clarity)**  
  - Basically solvable, but with some ambiguity or minor contradictions.  
  - An agent would need to make a few extra assumptions or ask for clarification.

- **8–10 (High Clarity)**  
  - The goal is clear, conditions are complete and self-consistent.  
  - An agent can plan a solution by following tool descriptions without confusion.

## METRIC 3: Naturalness (1–10)

**Definition:**
Naturalness measures whether the query **sounds like a realistic user request**, rather than a templated or obviously machine-generated prompt.

**Check:**

- Does the language feel natural and consistent with typical conversational or business usage?  
- Does it avoid system/instructional artifacts such as "You are a helpful assistant…" within the user query?  
- Is it more than just a cold parameter list: does it have reasonable context ("I want…", "Help me compare…", "Please check…")?  
- Is the scenario realistic (e.g., checking finances, comparing products, aggregating data from multiple tools)?

**Scoring guidelines (1–10):**

- **1–3 (Low Naturalness)**  
  - Strong template/programmatic feel; almost just JSON or parameter lists verbalized.  
  - Contains obvious system/instruction boilerplate in the user text.

- **4–7 (Moderate Naturalness)**  
  - Generally looks like a user query but slightly stiff, repetitive, or too obviously patterned.

- **8–10 (High Naturalness)**  
  - Very similar to a real user query from logs.  
  - Tone is natural, scenario is believable, and the query feels organic.

## OUTPUT FORMAT

Return a single JSON object with:

```json
{
  "toolfit": {
    "score": <integer 1-10>,
    "reason": "<2-5 sentences explaining your judgment>",
    "checklist": {
      "capability_match": "<short note>",
      "tool_dependency": "<short note>",
      "multi_hop_necessity": "<short note (use 'N/A' if not intended as multi-hop)>"
    }
  },
  "clarity": {
    "score": <integer 1-10>,
    "reason": "<2-5 sentences explaining your judgment>"
  },
  "naturalness": {
    "score": <integer 1-10>,
    "reason": "<2-5 sentences explaining your judgment>"
  }
}
```

Make sure the JSON is valid.

## INPUT

**Tools (selected tools from XML):**
{tools_context}

**Query description (from XML <query> element):**
{query_text}

**Is this query intended to be multi-hop?**
{multi_hop_flag}

Now analyze the query and output the JSON result.

