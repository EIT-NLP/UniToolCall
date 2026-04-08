You are a data quality evaluator for a multi-turn tool-calling dataset.

Your current goal is to evaluate **ANCHOR LINKAGE** for ONE episode.

You will rate ONE metric on a 1–10 scale:

1. Anchor Linkage

You are given:
- The full episode JSON (episode_level_plan + conversations).

IMPORTANT:
- You are evaluating whether each turn (turn_index >= 2) explicitly references at least one anchor from the previous turn.
- Anchors can be keywords, IDs, names, dates, or values.
- The reference must be explicit in the user message, not inferred.
- Consistency matters: the same anchor should not change across turns.

## METRIC: Anchor Linkage (1–10)

**Definition:**
Anchor Linkage measures whether the episode maintains clear, explicit, and consistent cross-turn references.

**Check:**

- For every turn_index >= 2, does the user message explicitly mention at least one anchor from the previous turn?
- Are anchors reused consistently without swapping or drifting values?
- Do anchors influence the intent or tool calls in the current turn (not just superficial repetition)?

**Scoring guidelines (1–10):**

- **1–3 (Low Anchor Linkage)**  
  - Frequent missing anchors or broken cross-turn references.  
  - Anchors are inconsistent or drift across turns.

- **4–7 (Moderate Anchor Linkage)**  
  - Some turns include valid anchors, but others miss them or only repeat superficially.  
  - Minor inconsistencies or weak linkage.

- **8–10 (High Anchor Linkage)**  
  - Every turn >= 2 contains explicit anchors from the previous turn.  
  - Anchors are consistent and meaningfully connect turns.

## OUTPUT FORMAT

Return a single JSON object with:

```json
{
  "anchor_linkage": {
    "score": <integer 1-10>,
    "reason": "<2-5 sentences explaining your judgment>"
  }
}
```

Make sure the JSON is valid.

## INPUT

**Episode JSON:**
{episode_json}

Now analyze the episode and output the JSON result.
