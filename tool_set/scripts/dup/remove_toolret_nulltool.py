#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Dict, Any, Set
import sys

sys.stdout.reconfigure(encoding='utf-8')

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

    Returns:
        True if this entry should be removed
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


def filter_tools(input_file: Path, output_file: Path) -> Dict[str, Any]:
    """
    Drop tools with empty/metadata-only properties.

    Returns:
        Stats dict
    """
    print("=" * 80)
    print("Remove placeholder tools (empty real parameters)")
    print("=" * 80)
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print()
    
    print("Reading file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, dict):
        print("Error: root must be an object")
        return {}
    
    print(f"Original entries: {len(data)}")
    print()
    
    print("Filtering...")
    filtered_data = {}
    removed_tools = []
    total = len(data)
    removed = 0
    
    for key, tool in data.items():
        if is_properties_empty(tool):
            removed += 1
            removed_tools.append({
                'key': key,
                'name': tool.get('name', 'N/A')
            })
        else:
            filtered_data[key] = tool
        
        if (removed + len(filtered_data)) % 1000 == 0:
            processed = removed + len(filtered_data)
            print(f"  Progress: {processed}/{total} ({processed/total*100:.1f}%)")
    
    print()
    print("=" * 80)
    print("Result")
    print("=" * 80)
    print(f"Original entries: {total}")
    print(f"Removed: {removed}")
    print(f"Kept: {len(filtered_data)}")
    print(f"Removal rate: {(removed / total * 100):.2f}%")
    
    if removed_tools:
        print()
        print("Sample removed (first 20):")
        for i, tool_info in enumerate(removed_tools[:20], 1):
            print(f"  {i}. [{tool_info['key']}] {tool_info['name']}")
        if len(removed_tools) > 20:
            print(f"  ... and {len(removed_tools) - 20} more")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print()
    print(f"Saving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    print("Save complete.")
    print("=" * 80)
    
    stats = {
        "total_tools": total,
        "removed_tools": removed,
        "remaining_tools": len(filtered_data),
        "removal_rate": f"{(removed / total * 100):.2f}%"
    }
    
    return stats


def main() -> None:
    """CLI entry point"""
    input_file = Path("/home/yijuan_liang/10.12Tool_Set/tool_set/apis/apis_cosdup/toolret_tool_dedup_cosdup.json")
    
    if not input_file.exists():
        print(f"Error: input not found: {input_file}")
        return
    
    output_file = input_file.parent / f"{input_file.stem}_filtered{input_file.suffix}"
    
    stats = filter_tools(input_file, output_file)
    
    if stats:
        print()
        print("Done.")
        print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
