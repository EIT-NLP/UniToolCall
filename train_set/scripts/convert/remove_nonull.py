#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import List, Dict, Any, Set
import sys

sys.stdout.reconfigure(encoding='utf-8')


def load_allowed_tool_names(tool_file: Path) -> Set[str]:
    """
    Load allowed tool names from a tools JSON file.

    Args:
        tool_file: Path to tools JSON

    Returns:
        Set of trimmed tool names
    """
    try:
        with open(tool_file, 'r', encoding='utf-8') as f:
            tools_data = json.load(f)
        
        tool_names: Set[str] = set()
        
        if isinstance(tools_data, dict):
            for tool_obj in tools_data.values():
                if isinstance(tool_obj, dict):
                    tool_name = tool_obj.get('name', '').strip()
                    if tool_name:
                        tool_names.add(tool_name)
        elif isinstance(tools_data, list):
            for tool_obj in tools_data:
                if isinstance(tool_obj, dict):
                    tool_name = tool_obj.get('name', '').strip()
                    if tool_name:
                        tool_names.add(tool_name)
        
        print(f"Loaded {len(tool_names)} tool name(s) from {tool_file.name}")
        return tool_names
        
    except Exception as e:
        print(f"Error: failed to load tool file - {e}")
        import traceback
        traceback.print_exc()
        return set()


def has_invalid_tools(conversation: Dict[str, Any], allowed_tool_names: Set[str]) -> bool:
    """
    True if tools JSON contains a tool name not in allowed_tool_names.

    Args:
        conversation: Record with optional tools string field
        allowed_tool_names: Allowed names

    Returns:
        True if any tool name is not allowed
    """
    if 'tools' not in conversation:
        return False
    
    tools_str = conversation.get('tools', '')
    if not isinstance(tools_str, str):
        return False
    
    try:
        tools_list = json.loads(tools_str)
        if not isinstance(tools_list, list):
            return False
        
        for tool in tools_list:
            if not isinstance(tool, dict):
                continue
            
            tool_name = tool.get('name', '').strip()
            
            if not tool_name:
                continue
            
            if tool_name not in allowed_tool_names:
                return True
        
        return False
        
    except json.JSONDecodeError:
        return False
    except Exception as e:
        print(f"  Warning: error parsing tools field: {e}")
        return False


def process_data_file(
    input_file: Path, 
    output_file: Path, 
    allowed_tool_names: Set[str]
) -> tuple[int, int]:
    """
    Filter one data file: drop conversations with disallowed tool names.

    Returns:
        (original_count, kept_count)
    """
    print(f"Processing file: {input_file.name}")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print("  Warning: root is not an array, skip")
            return 0, 0

        original_count = len(data)
        filtered_data: List[Dict[str, Any]] = []
        removed_count = 0

        for item in data:
            if has_invalid_tools(item, allowed_tool_names):
                removed_count += 1
            else:
                filtered_data.append(item)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)

        print(f"  Original: {original_count}, removed: {removed_count}, kept: {len(filtered_data)}")
        return original_count, len(filtered_data)

    except json.JSONDecodeError as e:
        print(f"  Error: JSON parse error - {e}")
        print(f"  Skip file, continue others")
        return 0, 0
    except Exception as e:
        print(f"  Error: failed to process file - {e}")
        print(f"  Skip file, continue others")
        import traceback
        traceback.print_exc()
        return 0, 0


def process_data_directory(
    input_dir: Path, 
    output_dir: Path, 
    allowed_tool_names: Set[str],
    base_input_dir: Path = None
) -> tuple[int, int]:
    """
    Recursively process all JSON under input_dir.

    Returns:
        (total_original, total_kept)
    """
    if not input_dir.exists():
        print(f"Error: input directory not found: {input_dir}")
        return 0, 0

    if base_input_dir is None:
        base_input_dir = input_dir

    json_files = sorted([f for f in input_dir.glob('*.json') 
                        if not f.name.endswith('.bak') and not f.name.endswith('.bak2')])

    total_original = 0
    total_filtered = 0

    if json_files:
        print(f"\nDirectory: {input_dir}")
        print(f"Found {len(json_files)} JSON file(s)")
        
        for json_file in json_files:
            relative_path = json_file.relative_to(base_input_dir)
            output_file = output_dir / relative_path
            original, filtered = process_data_file(json_file, output_file, allowed_tool_names)
            total_original += original
            total_filtered += filtered

    for subdir in sorted(input_dir.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith('.'):
            sub_output_dir = output_dir / subdir.relative_to(base_input_dir)
            original, filtered = process_data_directory(
                subdir, sub_output_dir, allowed_tool_names, base_input_dir
            )
            total_original += original
            total_filtered += filtered

    return total_original, total_filtered


def main() -> None:
    """CLI entry point"""
    tool_file = Path("/home/yijuan_liang/10.12Tool_Set/tool_set/apis/apis_cosdup/train_set_tool_dedup_cosdup.json")
    data_dir = Path("/home/yijuan_liang/10.12Tool_Set/train_set/data/data_nonull")
    data_output_dir = Path("/home/yijuan_liang/10.12Tool_Set/train_set/data/data_nonull_v2")
    
    print("=" * 80)
    print("Remove conversations whose tools list references disallowed tool names")
    print("=" * 80)
    
    print("\nLoading tool list...")
    allowed_tool_names = load_allowed_tool_names(tool_file)
    
    if not allowed_tool_names:
        print("Error: empty tool list, abort")
        return
    
    print(f"Allowed tools: {len(allowed_tool_names)}")
    
    print("\n" + "=" * 80)
    print("Process data_nonull tree")
    print("=" * 80)
    data_original, data_filtered = process_data_directory(data_dir, data_output_dir, allowed_tool_names)
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"data_nonull:")
    print(f"  Original conversations: {data_original:,}")
    print(f"  Kept: {data_filtered:,}")
    print(f"  Removed: {data_original - data_filtered:,}")
    if data_original > 0:
        print(f"  Retention: {data_filtered / data_original * 100:.2f}%")
        print(f"  Removal rate: {(data_original - data_filtered) / data_original * 100:.2f}%")
    
    print(f"\nOutput written to: {data_output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
