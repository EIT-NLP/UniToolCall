#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Dict, Any, Set, Tuple


def extract_tool_combination(tool: Dict[str, Any]) -> Tuple[str, str] | None:
    """Extract (name, description) from a tool dict."""
    if not isinstance(tool, dict):
        return None
    
    name = tool.get("name", "")
    description = tool.get("description", "")
    
    if not isinstance(name, str) or not name or not isinstance(description, str):
        return None
    
    return (name, description)


def collect_tools_from_file(file_path: Path, seen_combinations: Set[Tuple[str, str]], 
                           count_only: bool = False) -> int:
    """
    Count tools and optionally add (name, description) pairs to seen_combinations.

    Args:
        file_path: JSON file
        seen_combinations: Global or local set
        count_only: If True, only count; do not mutate seen_combinations

    Returns:
        Tool count
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    count = 0
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict) and "name" in value and "description" in value:
                count += 1
                combination = extract_tool_combination(value)
                if combination and not count_only:
                    seen_combinations.add(combination)
            elif isinstance(value, list):
                for tool in value:
                    if isinstance(tool, dict) and "name" in tool and "description" in tool:
                        count += 1
                        combination = extract_tool_combination(tool)
                        if combination and not count_only:
                            seen_combinations.add(combination)
    elif isinstance(data, list):
        for tool in data:
            if isinstance(tool, dict) and "name" in tool and "description" in tool:
                count += 1
                combination = extract_tool_combination(tool)
                if combination and not count_only:
                    seen_combinations.add(combination)
    
    return count


def get_file_tool_combinations(file_path: Path) -> Set[Tuple[str, str]]:
    """All (name, description) pairs in a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    file_combinations = set()
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict) and "name" in value and "description" in value:
                combination = extract_tool_combination(value)
                if combination:
                    file_combinations.add(combination)
            elif isinstance(value, list):
                for tool in value:
                    if isinstance(tool, dict) and "name" in tool and "description" in tool:
                        combination = extract_tool_combination(tool)
                        if combination:
                            file_combinations.add(combination)
    elif isinstance(data, list):
        for tool in data:
            if isinstance(tool, dict) and "name" in tool and "description" in tool:
                combination = extract_tool_combination(tool)
                if combination:
                    file_combinations.add(combination)
    
    return file_combinations


def dedup_file(file_path: Path, seen_combinations: Set[Tuple[str, str]], 
                file_combinations: Set[Tuple[str, str]],
                file_combinations_map: Dict[str, Set[Tuple[str, str]]],
                current_file_name: str) -> tuple:
    """
    Dedupe one file: drop duplicates within file and tools that appear in other files.

    Returns:
        (deduped data, removed count, total count)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    other_files_combinations = set()
    for file_name, combinations in file_combinations_map.items():
        if file_name != current_file_name:
            other_files_combinations.update(combinations)
    
    result = {}
    removed = 0
    total = 0
    file_seen = set()
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict) and "name" in value and "description" in value:
                total += 1
                combination = extract_tool_combination(value)
                if combination:
                    if combination in file_seen:
                        removed += 1
                        continue
                    file_seen.add(combination)
                    
                    if combination in other_files_combinations:
                        removed += 1
                        continue
                result[key] = value
            elif isinstance(value, list):
                deduped_tools = []
                for tool in value:
                    total += 1
                    if not isinstance(tool, dict):
                        deduped_tools.append(tool)
                        continue
                    
                    combination = extract_tool_combination(tool)
                    if combination:
                        if combination in file_seen:
                            removed += 1
                            continue
                        file_seen.add(combination)
                        
                        if combination in other_files_combinations:
                            removed += 1
                            continue
                    deduped_tools.append(tool)
                
                if deduped_tools:
                    result[key] = deduped_tools
            else:
                result[key] = value
        
        return result, removed, total
    elif isinstance(data, list):
        deduped_list = []
        for tool in data:
            total += 1
            if not isinstance(tool, dict):
                deduped_list.append(tool)
                continue
            
            combination = extract_tool_combination(tool)
            if combination:
                if combination in file_seen:
                    removed += 1
                    continue
                file_seen.add(combination)
                
                if combination in other_files_combinations:
                    removed += 1
                    continue
            deduped_list.append(tool)
        
        return deduped_list, removed, total
    
    return data, 0, 0


def main() -> None:
    apis_dir = Path("/home/yijuan_liang/10.12Tool_Set/tool_set/apis/apis_origin")
    apis_dup_dir = Path("/home/yijuan_liang/10.12Tool_Set/tool_set/apis/apis_dup")
    
    apis_dup_dir.mkdir(parents=True, exist_ok=True)
    
    special_files = ["test_set_tool.json", "train_set_tool.json"]
    
    stats = {
        "file_stats": [],
        "special_files_stats": [],
        "summary": {}
    }
    
    print("Step 1: build global combination set from all files...")
    seen_combinations: Set[Tuple[str, str]] = set()
    file_combinations_map: Dict[str, Set[Tuple[str, str]]] = {}
    
    all_json_files = list(apis_dir.glob("*.json"))
    
    total_tools_in_all_files = 0
    for file_path in sorted(all_json_files):
        if file_path.exists():
            file_combinations = get_file_tool_combinations(file_path)
            file_combinations_map[file_path.name] = file_combinations
            
            seen_combinations.update(file_combinations)
            
            count = len(file_combinations)
            total_tools_in_all_files += collect_tools_from_file(file_path, set(), count_only=True)
            print(f"  Collected {count} unique combos from {file_path.name} (total tools: {collect_tools_from_file(file_path, set(), count_only=True)})")
    
    print(f"  Unique (name, description) pairs: {len(seen_combinations)}")
    print(f"  Total tool rows (all files): {total_tools_in_all_files}")
    print()
    
    print("Step 2: copy special files unchanged...")
    
    special_total = 0
    for special_file in special_files:
        file_path = apis_dir / special_file
        if file_path.exists():
            print(f"  Copy: {special_file} (no cross-file dedup)")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            count = collect_tools_from_file(file_path, set(), count_only=True)
            special_total += count
            
            output_file = apis_dup_dir / f"{file_path.stem}_dedup.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"    Tools: {count}, all kept")
            print(f"    -> {output_file}")
            
            stats["special_files_stats"].append({
                "file": special_file,
                "total": count,
                "kept": count,
                "removed": 0,
                "removed_rate": 0.0
            })
        else:
            print(f"  Warning: {special_file} not found, skip")
    
    print()
    
    print("Step 3: dedupe remaining files...")
    
    files_to_process = [f for f in all_json_files if f.name not in special_files]
    
    total_removed = 0
    total_processed = 0
    total_kept = 0
    
    for file_path in sorted(files_to_process):
        print(f"  Process: {file_path.name}")
        
        file_combinations = get_file_tool_combinations(file_path)
        
        output_file = apis_dup_dir / f"{file_path.stem}_dedup.json"
        
        result, removed, total = dedup_file(file_path, seen_combinations, file_combinations, 
                                             file_combinations_map, file_path.name)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        kept = 0
        if isinstance(result, dict):
            for value in result.values():
                if isinstance(value, dict) and "name" in value:
                    kept += 1
                elif isinstance(value, list):
                    kept += len(value)
        elif isinstance(result, list):
            kept = len(result)
        
        removed_rate = (removed / total * 100) if total > 0 else 0.0
        
        print(f"    Total: {total}, kept: {kept}, removed: {removed} ({removed_rate:.2f}%)")
        print(f"    -> {output_file}")
        
        stats["file_stats"].append({
            "file": file_path.name,
            "total": total,
            "kept": kept,
            "removed": removed,
            "removed_rate": removed_rate
        })
        
        total_removed += removed
        total_processed += total
        total_kept += kept
    
    total_all_files = special_total + total_processed
    total_all_kept = special_total + total_kept
    total_all_removed = total_removed
    overall_removed_rate = (total_all_removed / total_all_files * 100) if total_all_files > 0 else 0.0
    
    stats["summary"] = {
        "total_files": len(all_json_files),
        "special_files_count": len(special_files),
        "processed_files_count": len(files_to_process),
        "unique_combinations": len(seen_combinations),
        "total_tools_in_all_files": total_tools_in_all_files,
        "special_files_total": special_total,
        "processed_files_total": total_processed,
        "total_all_files": total_all_files,
        "special_files_kept": special_total,
        "processed_files_kept": total_kept,
        "total_all_kept": total_all_kept,
        "processed_files_removed": total_removed,
        "total_all_removed": total_all_removed,
        "overall_removed_rate": overall_removed_rate
    }
    
    print()
    print("=" * 80)
    print("Deduplication statistics")
    print("=" * 80)
    print()
    print("[Special files] (copied as-is)")
    print("-" * 80)
    for stat in stats["special_files_stats"]:
        print(f"  File: {stat['file']}")
        print(f"    Total tools: {stat['total']}")
        print(f"    Kept: {stat['kept']}")
        print(f"    Removed: {stat['removed']}")
        print()
    
    print("[Other files] (cross-file dedup)")
    print("-" * 80)
    for stat in stats["file_stats"]:
        print(f"  File: {stat['file']}")
        print(f"    Total tools: {stat['total']}")
        print(f"    Kept: {stat['kept']}")
        print(f"    Removed: {stat['removed']} ({stat['removed_rate']:.2f}%)")
        print()
    
    print("[Overall]")
    print("-" * 80)
    summary = stats["summary"]
    print(f"  JSON files: {summary['total_files']}")
    print(f"    - Special (passthrough): {summary['special_files_count']}")
    print(f"    - Deduped: {summary['processed_files_count']}")
    print()
    print(f"  Unique (name, description) pairs: {summary['unique_combinations']}")
    print()
    print(f"  Tool counts:")
    print(f"    - All files (raw rows): {summary['total_tools_in_all_files']}")
    print(f"    - Special files: {summary['special_files_total']}")
    print(f"    - Deduped files: {summary['processed_files_total']}")
    print(f"    - Combined input total: {summary['total_all_files']}")
    print()
    print(f"  After dedup:")
    print(f"    - Special kept: {summary['special_files_kept']}")
    print(f"    - Deduped files kept: {summary['processed_files_kept']}")
    print(f"    - Total kept: {summary['total_all_kept']}")
    print(f"    - Total removed: {summary['total_all_removed']}")
    print(f"    - Overall removal rate: {summary['overall_removed_rate']:.2f}%")
    print()
    
    stats_file = apis_dup_dir / "dedup_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"Stats written to: {stats_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()


