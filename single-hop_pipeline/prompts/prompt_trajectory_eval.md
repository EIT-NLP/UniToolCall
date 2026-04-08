# Trajectory Quality Evaluation

You are a data quality evaluator for a single-hop tool-calling dataset.

Your current goal is to evaluate the **TRAJECTORY** (the single function call, observation, and final answer) for ONE sample.

You will rate THREE metrics on a 1–10 scale:

1. Success
2. Grounding
3. Efficiency

You are given:
- A tool definition (name, description, inputSchema).
- ONE query description (the user's request).
- The function call (single tool call with arguments).
- The observation (tool response).
- The final answer (assistant's response).

IMPORTANT:
- You are now evaluating the **actual behavior** in this single-hop trajectory: did it solve the query, did it use the tool correctly, and is the reasoning grounded and efficient?
- Do NOT change the query text. Evaluate how well this trajectory handled the given query.

## METRIC 1: Success (1–10)

**Definition:**
Success measures whether this trajectory **actually completes the query**, given the tool and the query requirements.

**Check:**

- Does the function call move toward solving the query, rather than random or irrelevant operations?  
- Are function call parameters consistent with the tool schema (field names, types, value ranges)?  
- Does the final assistant answer cover the key requirements stated in the query (all major questions/constraints)?  
- For numeric or data-based answers, can the final result be supported by the observation (tool response)?

**Scoring guidelines (1–10):**

- **1–3 (Low Success)**  
  - The trajectory largely fails to solve the query or severely deviates from the request.  
  - Function call is incorrect, or the final answer is missing the main requested outcome.

- **4–7 (Moderate Success)**  
  - The main goal is achieved, but some details are missing or partially incorrect.

- **8–10 (High Success)**  
  - The trajectory clearly and correctly solves the query.  
  - Tool usage is appropriate, and the final answer is reliable and complete with respect to the query.

## METRIC 2: Grounding (1–10)

**Definition:**
Grounding measures whether **each important parameter and conclusion** in the trajectory is well-supported by the query description or tool observation, and whether the reasoning is internally consistent.

**Check:**

- Are function call parameters derived from:
  - the original query/user input, OR
  - clearly reasonable default values?  
- Does the trajectory avoid fabricating IDs, cities, prices, dates, or other key facts that are not supported by context?  
- Is the observation (tool response) actually used in the final answer?  
- Does the final answer accurately reflect the information from the observation?

**Scoring guidelines (1–10):**

- **1–3 (Low Grounding)**  
  - Many parameters appear hallucinated or unsupported.  
  - The final answer contradicts the observation or earlier facts.

- **4–7 (Moderate Grounding)**  
  - Most parameters and conclusions are grounded, but there are occasional unsupported details or minor inconsistencies, OR some observation details are only partially reflected in the answer.

- **8–10 (High Grounding)**  
  - Parameters and conclusions are highly traceable to query input or observation.  
  - There is strong consistency throughout the trajectory, with little to no hallucination.

## METRIC 3: Efficiency (1–10)

**Definition:**
Efficiency measures whether the trajectory is **clean and well-structured**: the function call is appropriate, the observation is properly formatted, and the answer is concise and relevant.

**Check:**

- Is the function call appropriate for the query's complexity (not over-complicated or too simple)?  
- Are all required parameters provided, and are optional parameters used reasonably?  
- Is the observation properly formatted (valid JSON if it's JSON, clear structure)?  
- Is the final answer concise and directly addresses the query without unnecessary repetition or verbosity?

**Scoring guidelines (1–10):**

- **1–3 (Low Efficiency)**  
  - The trajectory is cluttered with unnecessary parameters or redundant information.  
  - There are frequent format/protocol issues that make it hard to use directly.

- **4–7 (Moderate Efficiency)**  
  - Generally usable, but with some unnecessary parameters or minor structural issues.  
  - Some redundancy or small formatting problems, yet still fixable with moderate cleaning.

- **8–10 (High Efficiency)**  
  - The trajectory is concise and well-structured.  
  - Function call parameters are necessary and well-justified; the format closely adheres to the protocol and can be used almost as-is.

## OUTPUT FORMAT

Return a single JSON object with:

```json
{
  "success": {
    "score": <integer 1-10>,
    "reason": "<2-5 sentences explaining your judgment>"
  },
  "grounding": {
    "score": <integer 1-10>,
    "reason": "<2-5 sentences explaining your judgment>"
  },
  "efficiency": {
    "score": <integer 1-10>,
    "reason": "<2-5 sentences explaining your judgment>"
  }
}
```

Make sure the JSON is valid.

## INPUT

**Tool definition:**
{tools_context}

**Query description:**
{query_text}

**Function call (JSON string):**
{function_call_value}

**Observation (tool response, JSON string):**
{observation_value}

**Final answer:**
{gpt_value}

Now analyze the trajectory and output the JSON result.
