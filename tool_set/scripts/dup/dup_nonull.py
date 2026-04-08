#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Dict, Any, List, Union, Set
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Metadata-only keys (not real user parameters)
METADATA_FIELDS: Set[str] = {
    'additionalProperties',
    '$schema',
    'dummy',
    'title',
    'random_string',
    'weekly_id',
    'id',
    'type',
    'description'
}


def is_properties_empty(tool: Dict[str, Any]) -> bool:
    """
    True if inputSchema.properties is missing, empty, or only metadata keys.

    Args:
        tool: Tool dict

    Returns:
        True if the tool should be filtered out
    """
    if not isinstance(tool, dict):
        return False
    
    input_schema = tool.get("inputSchema")
    if not isinstance(input_schema, dict):
        return False
    
    properties = input_schema.get("properties")
    if properties is None:
        return True
    if not isinstance(properties, dict):
        return False
    if len(properties) == 0:
        return True
    
    real_params = [key for key in properties.keys() if key not in METADATA_FIELDS]
    if len(real_params) == 0:
        return True
    
    return False


def filter_tools_with_properties(data: Union[Dict, List]) -> tuple:
    """
    Remove tools whose properties are empty/metadata-only.

    Args:
        data: JSON root (dict or list)

    Returns:
        (filtered data, removed count, total tool count)
    """
    total = 0
    removed = 0
    
    if isinstance(data, dict):
        filtered_data = {}
        
        for key, value in data.items():
            if isinstance(value, list):
                filtered_list = []
                for tool in value:
                    total += 1
                    if not is_properties_empty(tool):
                        filtered_list.append(tool)
                    else:
                        removed += 1
                filtered_data[key] = filtered_list
            elif isinstance(value, dict) and "name" in value and "inputSchema" in value:
                total += 1
                if not is_properties_empty(value):
                    filtered_data[key] = value
                else:
                    removed += 1
            else:
                filtered_data[key] = value
        
        return filtered_data, removed, total
    
    elif isinstance(data, list):
        filtered_list = []
        for tool in data:
            total += 1
            if not is_properties_empty(tool):
                filtered_list.append(tool)
            else:
                removed += 1
        return filtered_list, removed, total
    
    return data, 0, 0


def process_file(input_file: Path, output_file: Path) -> Dict[str, Any]:
    """
    Process one JSON file.

    Args:
        input_file: Source path
        output_file: Destination path

    Returns:
        Stats dict
    """
    print(f"\nProcessing file: {input_file.name}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    filtered_data, removed, total = filter_tools_with_properties(data)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    stats = {
        "file": input_file.name,
        "total_tools": total,
        "removed_tools": removed,
        "remaining_tools": total - removed,
        "removal_rate": f"{(removed / total * 100):.2f}%" if total > 0 else "0.00%"
    }
    
    print(f"  Total tools: {total}")
    print(f"  Removed: {removed}")
    print(f"  Kept: {total - removed}")
    print(f"  Removal rate: {stats['removal_rate']}")
    print(f"  Saved to: {output_file}")
    
    return stats


def main() -> None:
    """CLI entry point"""
    apis_notime_dir = Path("/home/yijuan_liang/10.12Tool_Set/tool_set/apis/apis_notime")
    apis_nonull_dir = Path("/home/yijuan_liang/10.12Tool_Set/tool_set/apis/apis_nonull_v2")
    
    apis_nonull_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = list(apis_notime_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files in {apis_notime_dir}")
        return
    
    print(f"Found {len(json_files)} JSON file(s)")
    print("=" * 60)
    
    all_stats = []
    total_tools_all = 0
    total_removed_all = 0
    
    for input_file in sorted(json_files):
        output_file = apis_nonull_dir / input_file.name
        stats = process_file(input_file, output_file)
        all_stats.append(stats)
        total_tools_all += stats["total_tools"]
        total_removed_all += stats["removed_tools"]
    
    print("\n" + "=" * 60)
    print("Overall statistics")
    print("=" * 60)
    print(f"Files processed: {len(json_files)}")
    print(f"Total tools: {total_tools_all}")
    print(f"Removed: {total_removed_all}")
    print(f"Kept: {total_tools_all - total_removed_all}")
    if total_tools_all > 0:
        print(f"Overall removal rate: {(total_removed_all / total_tools_all * 100):.2f}%")
    
    print("\nPer-file breakdown:")
    print("-" * 60)
    for stats in all_stats:
        print(f"{stats['file']:50s} | "
              f"total: {stats['total_tools']:6d} | "
              f"removed: {stats['removed_tools']:6d} | "
              f"kept: {stats['remaining_tools']:6d} | "
              f"rate: {stats['removal_rate']:>7s}")
    
    print("\nDone. Output directory:", apis_nonull_dir)


if __name__ == "__main__":
    main()
