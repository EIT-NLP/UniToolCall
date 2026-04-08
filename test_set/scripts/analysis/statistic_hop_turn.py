#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Paths
BASE_DIR = Path("/home/yijuan_liang/10.12Tool_Set/test_set")
DATA_DIR = BASE_DIR / "data" / "data_nonull"
OUTPUT_DIR = BASE_DIR / "data" / "data_analysis"


def analyze_conversation(conversations: List[Dict]) -> Tuple[List[int], List[str]]:
    """Analyze turns: function_call counts per turn and hop label per turn.

    Args:
        conversations: List of conversation messages.

    Returns:
        turn_fc_counts: function_call count for each turn.
        turn_hop_types: hop type per turn ("single-hop" or "multi-hop").
    """
    turn_fc_counts = []
    turn_hop_types = []
    
    human_indices = []
    for idx, conv in enumerate(conversations):
        if conv.get("from") == "human":
            human_indices.append(idx)
    
    if not human_indices:
        return turn_fc_counts, turn_hop_types
    
    for turn_idx in range(len(human_indices)):
        turn_start = human_indices[turn_idx]
        turn_end = human_indices[turn_idx + 1] if turn_idx + 1 < len(human_indices) else len(conversations)
        
        fc_count = 0
        for idx in range(turn_start, turn_end):
            if conversations[idx].get("from") == "function_call":
                fc_count += 1
        
        turn_fc_counts.append(fc_count)
        
        if fc_count == 1:
            turn_hop_types.append("single-hop")
        elif fc_count >= 2:
            turn_hop_types.append("multi-hop")
        else:
            turn_hop_types.append("single-hop")
    
    return turn_fc_counts, turn_hop_types


def process_json_file(file_path: Path) -> Dict[str, Any]:
    """Process one JSON file and return stats."""
    try:
        with file_path.open("r", encoding="utf-8") as f:
            records = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Failed to read {file_path}: {exc}")
        return {
            "success": False,
            "error": str(exc),
            "file_name": file_path.name,
            "relative_path": str(file_path.relative_to(DATA_DIR))
        }
    
    if not isinstance(records, list):
        return {
            "success": False,
            "error": "Invalid format: root must be a JSON array",
            "file_name": file_path.name,
            "relative_path": str(file_path.relative_to(DATA_DIR))
        }
    
    stats = {
        "total": len(records),
        "single_turn": 0,
        "multi_turn": 0,
        "single_hop": 0,
        "multi_hop": 0,
        "turn_distribution": Counter(),
        "hop_distribution": Counter(),
    }
    
    for item in records:
        conversations = item.get("conversations", [])
        if not conversations:
            continue
        
        human_count = sum(1 for conv in conversations if conv.get("from") == "human")
        
        if human_count == 1:
            stats["single_turn"] += 1
        elif human_count >= 2:
            stats["multi_turn"] += 1
        
        stats["turn_distribution"][human_count] += 1
        
        turn_fc_counts, turn_hop_types = analyze_conversation(conversations)
        
        for hop_type in turn_hop_types:
            if hop_type == "single-hop":
                stats["single_hop"] += 1
            elif hop_type == "multi-hop":
                stats["multi_hop"] += 1
            
            stats["hop_distribution"][hop_type] += 1
    
    return {
        "success": True,
        "file_name": file_path.name,
        "relative_path": str(file_path.relative_to(DATA_DIR)),
        **stats
    }


def generate_markdown_report(results: List[Dict[str, Any]]) -> str:
    """Build Markdown report text."""
    lines = []
    lines.append("# Hop and turn statistics")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("**Notes**: Turn count follows human queries; hop type follows function_call count per turn.")
    lines.append("")
    lines.append("- **single-turn**: exactly one human query in the conversation")
    lines.append("- **multi-turn**: two or more human queries")
    lines.append("- **single-hop**: one function_call in that turn")
    lines.append("- **multi-hop**: two or more function_calls in that turn")
    lines.append("")
    
    successful_results = [r for r in results if r.get("success", False)]
    failed_results = [r for r in results if not r.get("success", False)]
    
    successful_results.sort(key=lambda x: x["file_name"])
    
    total_stats = {
        "total": 0,
        "single_turn": 0,
        "multi_turn": 0,
        "single_hop": 0,
        "multi_hop": 0,
        "turn_distribution": Counter(),
        "hop_distribution": Counter(),
    }
    
    for result in successful_results:
        total_stats["total"] += result["total"]
        total_stats["single_turn"] += result["single_turn"]
        total_stats["multi_turn"] += result["multi_turn"]
        total_stats["single_hop"] += result["single_hop"]
        total_stats["multi_hop"] += result["multi_hop"]
        total_stats["turn_distribution"] += result["turn_distribution"]
        total_stats["hop_distribution"] += result["hop_distribution"]
    
    lines.append("## Per file")
    lines.append("")
    lines.append("| File | Relative path | Total | single-turn | multi-turn | single-hop | multi-hop |")
    lines.append("|--------|----------|------|-------------|------------|-----------|----------|")
    
    for result in successful_results:
        lines.append(
            f"| {result['file_name']} | {result['relative_path']} | "
            f"{result['total']} | {result['single_turn']} | {result['multi_turn']} | "
            f"{result['single_hop']} | {result['multi_hop']} |"
        )
    
    lines.append("")
    lines.append(f"| **Total** | | **{total_stats['total']}** | **{total_stats['single_turn']}** | "
                f"**{total_stats['multi_turn']}** | **{total_stats['single_hop']}** | "
                f"**{total_stats['multi_hop']}** |")
    lines.append("")
    
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- **Total conversations**: {total_stats['total']}")
    lines.append("")
    
    lines.append("### Turn breakdown")
    lines.append("")
    if total_stats['total'] > 0:
        single_turn_pct = total_stats['single_turn'] / total_stats['total'] * 100
        multi_turn_pct = total_stats['multi_turn'] / total_stats['total'] * 100
        lines.append(f"- **single-turn**: {total_stats['single_turn']} ({single_turn_pct:.2f}%)")
        lines.append(f"- **multi-turn**: {total_stats['multi_turn']} ({multi_turn_pct:.2f}%)")
        lines.append("")
    
    if total_stats['turn_distribution']:
        lines.append("#### Turn count distribution")
        lines.append("")
        lines.append("| Turns | Conversations | Share |")
        lines.append("|----------|------------------|------|")
        
        sorted_turns = sorted(total_stats['turn_distribution'].items())
        for turn_count, conv_count in sorted_turns:
            percentage = (conv_count / total_stats['total']) * 100
            lines.append(f"| {turn_count} | {conv_count} | {percentage:.2f}% |")
        lines.append("")
    
    lines.append("### Hop breakdown")
    lines.append("")
    total_hops = total_stats['single_hop'] + total_stats['multi_hop']
    if total_hops > 0:
        single_hop_pct = total_stats['single_hop'] / total_hops * 100
        multi_hop_pct = total_stats['multi_hop'] / total_hops * 100
        lines.append(f"- **single-hop**: {total_stats['single_hop']} ({single_hop_pct:.2f}%)")
        lines.append(f"- **multi-hop**: {total_stats['multi_hop']} ({multi_hop_pct:.2f}%)")
        lines.append("")
    
    if total_stats['hop_distribution']:
        lines.append("#### Hop type distribution")
        lines.append("")
        lines.append("| Hop type | Turns | Share |")
        lines.append("|----------|---------|------|")
        
        sorted_hops = sorted(total_stats['hop_distribution'].items())
        for hop_type, turn_count in sorted_hops:
            percentage = (turn_count / total_hops) * 100 if total_hops > 0 else 0
            lines.append(f"| {hop_type} | {turn_count} | {percentage:.2f}% |")
        lines.append("")
    
    if failed_results:
        lines.append("## Failed files")
        lines.append("")
        for result in failed_results:
            lines.append(f"- **{result['file_name']}** ({result.get('relative_path', 'N/A')}): {result.get('error', 'unknown error')}")
    
    return "\n".join(lines)


def main() -> None:
    """CLI entry point"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Counting hop/turn under {DATA_DIR} ...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    json_files = list(DATA_DIR.rglob("*.json"))
    json_files = [f for f in json_files if not f.name.endswith(".bak")]
    
    if not json_files:
        print(f"No JSON files under {DATA_DIR}.")
        return
    
    print(f"Found {len(json_files)} JSON file(s)")
    print()
    
    results: List[Dict[str, Any]] = []
    for file_path in sorted(json_files):
        print(f"Processing: {file_path.relative_to(DATA_DIR)}")
        result = process_json_file(file_path)
        results.append(result)
        
        if result.get("success"):
            print(
                f"  - Conversations: {result['total']}, "
                f"single-turn: {result['single_turn']}, multi-turn: {result['multi_turn']}, "
                f"single-hop: {result['single_hop']}, multi-hop: {result['multi_hop']}"
            )
        else:
            print(f"  - Error: {result.get('error', 'unknown error')}")
    
    print()
    
    report_content = generate_markdown_report(results)
    output_file = OUTPUT_DIR / "hop_and_turn_statistics.md"
    
    with output_file.open("w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"Report written: {output_file}")
    
    successful_results = [r for r in results if r.get("success", False)]
    if successful_results:
        total_stats = {
            "total": 0,
            "single_turn": 0,
            "multi_turn": 0,
            "single_hop": 0,
            "multi_hop": 0,
        }
        
        for result in successful_results:
            total_stats["total"] += result["total"]
            total_stats["single_turn"] += result["single_turn"]
            total_stats["multi_turn"] += result["multi_turn"]
            total_stats["single_hop"] += result["single_hop"]
            total_stats["multi_hop"] += result["multi_hop"]
        
        print()
        print("=== Aggregate ===")
        print(f"Total conversations: {total_stats['total']}")
        print()
        print("Turn:")
        if total_stats['total'] > 0:
            print(f"  - single-turn: {total_stats['single_turn']} ({total_stats['single_turn']/total_stats['total']*100:.2f}%)")
            print(f"  - multi-turn: {total_stats['multi_turn']} ({total_stats['multi_turn']/total_stats['total']*100:.2f}%)")
        print()
        print("Hop:")
        total_hops = total_stats['single_hop'] + total_stats['multi_hop']
        if total_hops > 0:
            print(f"  - single-hop: {total_stats['single_hop']} ({total_stats['single_hop']/total_hops*100:.2f}%)")
            print(f"  - multi-hop: {total_stats['multi_hop']} ({total_stats['multi_hop']/total_hops*100:.2f}%)")


if __name__ == "__main__":
    main()
