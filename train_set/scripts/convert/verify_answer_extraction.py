#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

WORKSPACE_ROOT = Path("/home/yijuan_liang/10.12Tool_Set").resolve()
TOUCAN_ORIGINAL_DIR = WORKSPACE_ROOT / "train_set" / "data" / "data_origin" / "TOUCAN_converted"

SOURCE_FILES = {
    "qwen3": TOUCAN_ORIGINAL_DIR / "qwen3_converted_all.json",
    "oss": TOUCAN_ORIGINAL_DIR / "oss_converted_all.json",
    "kimi": TOUCAN_ORIGINAL_DIR / "kimi_converted_all.json",
}


def normalize_text(text: str) -> str:
    """Collapse whitespace."""
    if not isinstance(text, str):
        return ""
    return "".join(text.split())


def extract_all_gpt_answers(conversations):
    """Collect all gpt message values."""
    gpt_answers = []
    for conv in conversations:
        if conv.get("from") == "gpt":
            gpt_answers.append(conv.get("value", "").strip())
    return gpt_answers


def verify_answer_extraction(source_file: Path, source_name: str, sample_size: int = 1000):
    """Sanity-check whether the last gpt turn matches expected final answer."""
    print(f"\nChecking {source_name}...")
    
    if not source_file.exists():
        print(f"  Warning: file not found: {source_file}")
        return
    
    try:
        with source_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"  Warning: {source_name} root is not a list")
            return
        
        print(f"  Rows: {len(data)}")
        print(f"  Sample size: {min(sample_size, len(data))}")
        
        stats = {
            "total_checked": 0,
            "single_gpt": 0,
            "multiple_gpt": 0,
            "last_is_answer": 0,
            "last_not_answer": 0,
            "no_gpt": 0,
        }
        
        sample_count = min(sample_size, len(data))
        for i, item in enumerate(data[:sample_count]):
            conversations = item.get("conversations", [])
            gpt_answers = extract_all_gpt_answers(conversations)
            
            stats["total_checked"] += 1
            
            if len(gpt_answers) == 0:
                stats["no_gpt"] += 1
            elif len(gpt_answers) == 1:
                stats["single_gpt"] += 1
                stats["last_is_answer"] += 1
            else:
                stats["multiple_gpt"] += 1
                last_answer = normalize_text(gpt_answers[-1])
                second_last_answer = normalize_text(gpt_answers[-2]) if len(gpt_answers) >= 2 else ""
                
                if last_answer != second_last_answer or len(gpt_answers[-1]) > len(gpt_answers[-2]):
                    stats["last_is_answer"] += 1
                else:
                    stats["last_not_answer"] += 1
                    
                    if stats["last_not_answer"] == 1:
                        print(f"\n  Example: last gpt reply may not be final answer")
                        print(f"    GPT turns: {len(gpt_answers)}")
                        print(f"    Last: {gpt_answers[-1][:200]}...")
                        if len(gpt_answers) >= 2:
                            print(f"    Second-to-last: {gpt_answers[-2][:200]}...")
        
        print(f"\n  Stats:")
        print(f"    Checked: {stats['total_checked']:,}")
        print(f"    No gpt: {stats['no_gpt']:,} ({stats['no_gpt']/stats['total_checked']*100:.2f}%)")
        print(f"    Single GPT reply: {stats['single_gpt']:,} ({stats['single_gpt']/stats['total_checked']*100:.2f}%)")
        print(f"    Multiple GPT replies: {stats['multiple_gpt']:,} ({stats['multiple_gpt']/stats['total_checked']*100:.2f}%)")
        print(f"    Last reply likely answer: {stats['last_is_answer']:,} ({stats['last_is_answer']/stats['total_checked']*100:.2f}%)")
        if stats['last_not_answer'] > 0:
            print(f"    Warning: last reply may not be answer: {stats['last_not_answer']:,} ({stats['last_not_answer']/stats['total_checked']*100:.2f}%)")
        
        return stats
        
    except Exception as exc:
        print(f"  Error: {exc}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """CLI entry point"""
    print("=" * 70)
    print("Verify answer extraction (last gpt = final answer)")
    print("=" * 70)
    print("\nChecking whether last gpt reply is the final answer...")
    
    all_stats = {}
    for source_name, source_file in SOURCE_FILES.items():
        stats = verify_answer_extraction(source_file, source_name, sample_size=5000)
        if stats:
            all_stats[source_name] = stats
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    for source_name, stats in all_stats.items():
        print(f"\n{source_name}:")
        if stats['last_not_answer'] > 0:
            print(f"  Warning: {stats['last_not_answer']} rows where last gpt may not be final answer")
            print(f"  Tip: inspect structure; you may need to change answer extraction logic")
        else:
            print(f"  OK: last gpt is final answer for sampled rows")
    
    print("\n" + "=" * 70)
    print("Done")
    print("=" * 70)


if __name__ == "__main__":
    main()
