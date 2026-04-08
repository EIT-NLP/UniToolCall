#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

UNI_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TOOLSET_JSON = UNI_ROOT / "tool_set" / "apis" / "toolset.json"
DEFAULT_SYSTEM_PROMPT_MD = UNI_ROOT / "test_set" / "prompt_and_format" / "system_prompt.md"


def load_tools_db(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "tools" in data:
        data = data["tools"]
    elif isinstance(data, dict):
        data = list(data.values())
    if not isinstance(data, list):
        raise ValueError("Unsupported tools json format")
    return data


def build_tools_index(tools: List[dict]) -> Dict[str, dict]:
    idx = {}
    for t in tools:
        name = t.get("name") or t.get("tool_name")
        if name:
            idx[str(name)] = t
            idx[str(name).lower()] = t
    return idx


def load_passed_stems(scored_dir: Path) -> set:
    passed = set()
    if not scored_dir.exists():
        return passed
    for p in sorted(scored_dir.glob("episode_*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("passed") is True:
                passed.add(p.stem)
        except Exception:
            continue
    return passed


def extract_turn_messages(turn: dict) -> Tuple[str, list, list, str]:
    user_msg = ""
    tool_calls = []
    tool_outputs = []
    assistant_final = ""
    for msg in turn.get("messages", []):
        role = msg.get("role")
        if role == "user":
            user_msg = msg.get("content", "")
        elif role == "assistant" and msg.get("tool_calls"):
            tool_calls = msg.get("tool_calls") or []
        elif role == "tool":
            output = {}
            raw = msg.get("content", "")
            try:
                output = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                output = {"raw": raw}
            tool_outputs.append({"tool_name": msg.get("tool_name"), "output": output})
        elif role == "assistant" and msg.get("content"):
            assistant_final = msg.get("content", "")
    return user_msg, tool_calls, tool_outputs, assistant_final


def _build_params_table(args: dict) -> dict | None:
    if not args:
        return None
    rows = []
    for k, v in args.items():
        if isinstance(v, (dict, list)):
            vv = json.dumps(v, ensure_ascii=False)
        else:
            vv = v
        rows.append({"parameter": str(k), "value": vv})
    return {"type": "table", "title": "Input Parameters", "headers": ["parameter", "value"], "rows": rows}


def _obs_value_from_output(output_obj: object, arguments: dict) -> str:
    prefix = "Result analysis completed."
    try:
        body = json.dumps(output_obj, ensure_ascii=False)
    except Exception:
        body = str(output_obj)
    arr = [{"type": "text", "text": f"{prefix}\n\n{body}".strip()}]
    tbl = _build_params_table(arguments or {})
    if tbl:
        arr.append(tbl)
    return json.dumps(arr, ensure_ascii=False)


def detect_turn_hop_type(tool_calls: list) -> str:
    return "multi-hop" if len(tool_calls or []) >= 2 else "single_hop"


def detect_turn_strategy(turn: dict, user_msg: str, tool_calls: list, tool_outputs: list) -> str:
    strategy = ((turn.get("turn_plan") or {}).get("turn_strategy") or "").strip().lower()
    if strategy in ("parallel", "serial"):
        return strategy
    if strategy == "mixed":
        # mixed contains at least one dependency; map to serial for binary labeling.
        return "serial"

    if len(tool_calls or []) <= 1:
        return "parallel"

    user_lower = (user_msg or "").lower()
    seen_outputs = []
    for i, call in enumerate(tool_calls):
        args = call.get("arguments") or {}
        arg_values = []
        for v in args.values():
            if isinstance(v, (dict, list)):
                arg_values.append(json.dumps(v, ensure_ascii=False))
            else:
                arg_values.append(str(v))
        for av in arg_values:
            av_l = av.lower().strip()
            if len(av_l) < 3:
                continue
            if av_l in user_lower:
                continue
            for prev_out in seen_outputs:
                if av_l in prev_out or prev_out in av_l:
                    return "serial"
        if i < len(tool_outputs):
            try:
                seen_outputs.append(json.dumps(tool_outputs[i].get("output", {}), ensure_ascii=False).lower())
            except Exception:
                seen_outputs.append(str(tool_outputs[i].get("output", "")).lower())
    return "parallel"


def make_conversations_and_properties(episode: dict) -> Tuple[List[dict], dict, List[str]]:
    conversations = []
    properties = {}
    used_tool_names = []

    turns = episode.get("conversations", [])
    for idx, turn in enumerate(turns, start=1):
        user_msg, tool_calls, tool_outputs, assistant_final = extract_turn_messages(turn)
        conversations.append({"from": "human", "value": user_msg or ""})

        for i, call in enumerate(tool_calls):
            tool_name = call.get("tool_name")
            args = call.get("arguments") or {}
            conversations.append(
                {
                    "from": "function_call",
                    "value": json.dumps({"name": tool_name, "arguments": args}, ensure_ascii=False),
                }
            )
            if tool_name and tool_name not in used_tool_names:
                used_tool_names.append(tool_name)

            out_obj = {}
            if i < len(tool_outputs):
                out_obj = tool_outputs[i].get("output", {})
            conversations.append(
                {
                    "from": "observation",
                    "value": _obs_value_from_output(out_obj, args),
                }
            )

        conversations.append({"from": "gpt", "value": assistant_final or ""})

        properties[f"turn_{idx}"] = detect_turn_hop_type(tool_calls)
        properties[f"turn_{idx}_strategy"] = detect_turn_strategy(turn, user_msg, tool_calls, tool_outputs)

    properties["num_turns"] = len(turns)
    properties["turn"] = "multi-turn"
    return conversations, properties, used_tool_names


def main():
    ap = argparse.ArgumentParser(description="Standardize passed multi-turn episodes into a single final dataset JSON.")
    ap.add_argument("--source-dir", default="outputs/raw", help="Directory containing episode_XXXX.json")
    ap.add_argument("--scored-dir", default="outputs/scored", help="Directory containing episode score JSONs")
    ap.add_argument(
        "--tools-json",
        default=str(DEFAULT_TOOLSET_JSON),
        help="Path to full tools database JSON (default: tool_set/apis/toolset.json)",
    )
    ap.add_argument(
        "--system-file",
        default=str(DEFAULT_SYSTEM_PROMPT_MD),
        help="Path to system prompt markdown file (default: test_set/prompt_and_format/system_prompt.md)",
    )
    ap.add_argument(
        "--output",
        default="outputs/standardized/final_dataset.json",
        help="Output JSON file path",
    )
    ap.add_argument("--require-pass", action="store_true", default=True, help="Only include episodes with passed=true in scored-dir")
    args = ap.parse_args()

    source_dir = Path(args.source_dir)
    scored_dir = Path(args.scored_dir)
    tools_db = load_tools_db(Path(args.tools_json))
    tools_idx = build_tools_index(tools_db)
    system_text = Path(args.system_file).read_text(encoding="utf-8")

    passed_stems = load_passed_stems(scored_dir) if args.require_pass else set()

    items = []
    for p in sorted(source_dir.glob("episode_*.json")):
        if args.require_pass and p.stem not in passed_stems:
            continue
        try:
            episode = json.loads(p.read_text(encoding="utf-8"))
            conversations, properties, used_tool_names = make_conversations_and_properties(episode)

            used_tools_full = []
            for nm in used_tool_names:
                t = tools_idx.get(nm) or tools_idx.get(str(nm).lower())
                if t:
                    used_tools_full.append(t)

            # fallback: if no used tool resolved, keep selected_tools from episode.
            if not used_tools_full:
                used_tools_full = episode.get("selected_tools", [])

            item = {
                "conversations": conversations,
                "system": system_text,
                "tools": json.dumps(used_tools_full, ensure_ascii=False),
                "properties": properties,
            }
            items.append(item)
            print(f"  ✓ Added {p.name}")
        except Exception as e:
            print(f"  ✗ Error processing {p.name}: {e}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote standardized dataset: {out_path} (items={len(items)})")


if __name__ == "__main__":
    main()
