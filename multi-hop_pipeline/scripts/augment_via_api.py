#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

from core import OUT_DIR, parse_sections, load_manifest, parse_selected_tools, parse_trajectory
from generate_via_api import call_gemini, call_openai_compatible


def extract_json_object(text: str) -> dict:
    cleaned = re.sub(r"```[a-zA-Z]*", "", text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        segment = cleaned[start : end + 1].strip()
        try:
            return json.loads(segment)
        except Exception:
            pass
    return json.loads(text)


def main():
    ap = argparse.ArgumentParser(description="Generate query augmentations via API for high-quality samples")
    ap.add_argument("--provider", choices=["gemini", "openai"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-base", default="https://api.siliconflow.cn")
    ap.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds for API calls (default 120)")
    args = ap.parse_args()

    raw_dir = OUT_DIR / "raw"
    augmented_dir = OUT_DIR / "augmented"
    augmented_dir.mkdir(parents=True, exist_ok=True)

    tpl = (Path(__file__).resolve().parent / "prompts" / "query_augment.md").read_text(encoding="utf-8")
    manifest = load_manifest()
    # Get quality evaluation passed samples
    qc_passed = set(manifest.get("quality_evaluation", {}).get("passed", []))
    # If QC not run, default to all raw files except basic_filter.failed
    if not qc_passed:
        bf_failed = set(manifest.get("basic_filter", {}).get("failed", []))
        qc_passed = set([p.name for p in list(raw_dir.glob("*.md")) + list(raw_dir.glob("*.txt")) + list(raw_dir.glob("*.xml")) if p.name not in bf_failed])

    for name in sorted(qc_passed):
        p = raw_dir / name
        stem = p.stem
        raw = p.read_text(encoding="utf-8")
        secs = parse_sections(raw)
        tools_section = secs.get("SELECTED_TOOLS", "")
        query_text = secs.get("QUERY", "")
        traj_text = secs.get("TRAJECTORY", "")
        prompt = (
            tpl.replace("{query_text}", query_text)
               .replace("{trajectory_text}", traj_text)
               .replace("{tools_context}", tools_section)
        )

        if args.provider == "gemini":
            resp = call_gemini(args.model, prompt, timeout=args.timeout)
        else:
            resp = call_openai_compatible(args.model, prompt, api_base=args.api_base, timeout=args.timeout)

        data = extract_json_object(resp)
        # Normalize augmented structure
        augmented = {}
        if isinstance(data, dict) and isinstance(data.get("augmented_queries"), dict):
            augmented = data["augmented_queries"]
        elif isinstance(data, dict) and isinstance(data.get("augmented_queries"), list) and len(data["augmented_queries"]) >= 2:
            lst = data["augmented_queries"]
            augmented = {
                "noisy_overspecified": {"query": lst[0], "how": "Added extra details and minor conditions"},
                "ambiguous_simplified": {"query": lst[1], "how": "Simplified and made more informal"},
            }
        else:
            # If model returned plain strings, wrap defensively
            augmented = {
                "noisy_overspecified": {"query": str(data), "how": "Auto-wrapped from response"},
                "ambiguous_simplified": {"query": "", "how": ""},
            }

        # Merge with original content
        merged = {
            "selected_tools": parse_selected_tools(raw),
            "query": query_text,
            "trajectory": parse_trajectory(traj_text),
            "final_answer": secs.get("FINAL_ANSWER", ""),
            "augmented_queries": augmented,
        }
        (augmented_dir / f"{stem}.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Augmented (merged): {stem}")


if __name__ == "__main__":
    main()
