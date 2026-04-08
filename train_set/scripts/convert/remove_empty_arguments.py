#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import List, Dict, Any, Set
import sys

# Force UTF-8 stdout on platforms that need it
sys.stdout.reconfigure(encoding='utf-8')


def find_empty_argument_tools(data_dir: Path) -> Set[str]:
    """
    Scan the data tree and collect tool names that appear in empty-argument function_call rows.

    Args:
        data_dir: Root data directory.

    Returns:
        Set of tool names.
    """
    empty_arg_tools: Set[str] = set()

    def process_file(file_path: Path):
        """Process one JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                return

            for item in data:
                conversations = item.get('conversations', [])
                for conv in conversations:
                    if conv.get('from') == 'function_call':
                        value = conv.get('value', '')
                        try:
                            tool_call = json.loads(value)
                            if 'name' in tool_call:
                                arguments = tool_call.get('arguments')
                                if arguments is None or (isinstance(arguments, dict) and len(arguments) == 0):
                                    tool_name = tool_call['name'].strip()
                                    if tool_name:
                                        empty_arg_tools.add(tool_name)
                        except Exception:
                            pass
        except Exception as e:
            print(f"  Warning: error processing {file_path.name}: {e}")

    json_files = list(data_dir.rglob('*.json'))
    json_files = [f for f in json_files if not f.name.endswith('.bak') and not f.name.endswith('.bak2')]

    print(f"Scanning {len(json_files)} file(s) for tools used with empty arguments...")
    for i, json_file in enumerate(json_files, 1):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(json_files)}")
        process_file(json_file)

    return empty_arg_tools


def has_empty_argument_call(conversation: Dict[str, Any]) -> bool:
    """
    Return True if any function_call has missing or empty `arguments`.

    Args:
        conversation: One training item.

    Returns:
        True if an empty-argument call exists.
    """
    conversations = conversation.get('conversations', [])
    for conv in conversations:
        if conv.get('from') == 'function_call':
            value = conv.get('value', '')
            try:
                tool_call = json.loads(value)
                if 'name' in tool_call:
                    arguments = tool_call.get('arguments')
                    if arguments is None or (isinstance(arguments, dict) and len(arguments) == 0):
                        return True
            except Exception:
                pass
    return False


def remove_tools_from_file(tool_file: Path, tools_to_remove: Set[str]) -> tuple[int, int]:
    """
    Remove named tools from a tools JSON file.

    Args:
        tool_file: Path to tools file.
        tools_to_remove: Tool names to drop.

    Returns:
        (original_count, remaining_count)
    """
    if not tool_file.exists():
        print(f"  Warning: tools file not found: {tool_file}")
        return 0, 0

    try:
        with open(tool_file, 'r', encoding='utf-8') as f:
            tools_data = json.load(f)

        original_count = 0
        removed_count = 0

        if isinstance(tools_data, dict):
            original_count = len(tools_data)
            filtered_data = {}
            for key, tool_obj in tools_data.items():
                if isinstance(tool_obj, dict):
                    tool_name = tool_obj.get('name', '').strip()
                    if tool_name in tools_to_remove:
                        removed_count += 1
                    else:
                        filtered_data[key] = tool_obj
                else:
                    filtered_data[key] = tool_obj
            tools_data = filtered_data
        elif isinstance(tools_data, list):
            original_count = len(tools_data)
            filtered_data = []
            for tool_obj in tools_data:
                if isinstance(tool_obj, dict):
                    tool_name = tool_obj.get('name', '').strip()
                    if tool_name in tools_to_remove:
                        removed_count += 1
                    else:
                        filtered_data.append(tool_obj)
                else:
                    filtered_data.append(tool_obj)
            tools_data = filtered_data

        with open(tool_file, 'w', encoding='utf-8') as f:
            json.dump(tools_data, f, ensure_ascii=False, indent=2)

        return original_count, original_count - removed_count
    except Exception as e:
        print(f"  Error: failed to update tools file - {e}")
        import traceback
        traceback.print_exc()
        return 0, 0


def process_data_file(
    input_file: Path,
    output_file: Path
) -> tuple[int, int]:
    """
    Drop conversations that contain an empty-argument function_call.

    Args:
        input_file: Input JSON path.
        output_file: Output JSON path.

    Returns:
        (original_count, kept_count)
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"  Warning: file is not a JSON array; skipping: {input_file.name}")
            return 0, 0

        original_count = len(data)
        filtered_data: List[Dict[str, Any]] = []
        removed_count = 0

        for item in data:
            if has_empty_argument_call(item):
                removed_count += 1
            else:
                filtered_data.append(item)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)

        return original_count, len(filtered_data)

    except json.JSONDecodeError as e:
        print(f"  Error: invalid JSON - {e}")
        return 0, 0
    except Exception as e:
        print(f"  Error: failed to process file - {e}")
        return 0, 0


def process_data_directory(
    input_dir: Path,
    output_dir: Path,
    base_input_dir: Path = None
) -> tuple[int, int]:
    """
    Recursively process JSON files under input_dir.

    Args:
        input_dir: Input root.
        output_dir: Output root.
        base_input_dir: Base for relative paths.

    Returns:
        (total_original, total_kept)
    """
    if not input_dir.exists():
        print(f"Error: input directory does not exist: {input_dir}")
        return 0, 0

    if base_input_dir is None:
        base_input_dir = input_dir

    json_files = sorted([f for f in input_dir.glob('*.json')
                        if not f.name.endswith('.bak') and not f.name.endswith('.bak2')])

    total_original = 0
    total_filtered = 0

    if json_files:
        for json_file in json_files:
            relative_path = json_file.relative_to(base_input_dir)
            output_file = output_dir / relative_path
            original, filtered = process_data_file(json_file, output_file)
            total_original += original
            total_filtered += filtered

    for subdir in sorted(input_dir.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith('.'):
            sub_output_dir = output_dir / subdir.relative_to(base_input_dir)
            original, filtered = process_data_directory(
                subdir, sub_output_dir, base_input_dir
            )
            total_original += original
            total_filtered += filtered

    return total_original, total_filtered


def main() -> None:
    """CLI entry point"""
    print("=" * 80)
    print("Remove conversations that contain empty-argument function_call entries")
    print("=" * 80)

    data_dir = Path("/home/yijuan_liang/10.12Tool_Set/train_set/data/data_nonull_v2")
    output_dir = Path("/home/yijuan_liang/10.12Tool_Set/train_set/data/data_nonull_v2")
    tool_file_1 = Path("/home/yijuan_liang/10.12Tool_Set/tool_set/apis/apis_nonull/train_set_tool_dedup.json")
    tool_file_2 = Path("/home/yijuan_liang/10.12Tool_Set/tool_set/apis/apis_cosdup/train_set_tool_dedup_cosdup.json")

    print("\nStep 1: Scan data and collect tools seen with empty arguments...")
    print("-" * 80)
    empty_arg_tools = find_empty_argument_tools(data_dir)

    print(f"\nFound {len(empty_arg_tools)} tool(s) with empty-argument calls:")
    for i, tool_name in enumerate(sorted(empty_arg_tools)[:20], 1):
        print(f"  {i}. {tool_name}")
    if len(empty_arg_tools) > 20:
        print(f"  ... and {len(empty_arg_tools) - 20} more")

    if not empty_arg_tools:
        print("\nNo empty-argument tools found; nothing to do.")
        return

    print("\nStep 2: Remove those tools from the tool list files...")
    print("-" * 80)

    print(f"\nTools file 1: {tool_file_1.name}")
    original_1, remaining_1 = remove_tools_from_file(tool_file_1, empty_arg_tools)
    print(f"  Original: {original_1}, after removal: {remaining_1}, removed: {original_1 - remaining_1}")

    print(f"\nTools file 2: {tool_file_2.name}")
    original_2, remaining_2 = remove_tools_from_file(tool_file_2, empty_arg_tools)
    print(f"  Original: {original_2}, after removal: {remaining_2}, removed: {original_2 - remaining_2}")

    print("\nStep 3: Drop conversations that still contain empty-argument function_call...")
    print("-" * 80)
    data_original, data_filtered = process_data_directory(data_dir, output_dir)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"\nTool list stats:")
    print(f"  {tool_file_1.name}:")
    print(f"    Original count: {original_1:,}")
    print(f"    Removed count: {original_1 - remaining_1:,}")
    print(f"    Kept count: {remaining_1:,}")
    if original_1 > 0:
        print(f"    Removal rate: {(original_1 - remaining_1) / original_1 * 100:.2f}%")

    print(f"\n  {tool_file_2.name}:")
    print(f"    Original count: {original_2:,}")
    print(f"    Removed count: {original_2 - remaining_2:,}")
    print(f"    Kept count: {remaining_2:,}")
    if original_2 > 0:
        print(f"    Removal rate: {(original_2 - remaining_2) / original_2 * 100:.2f}%")

    print(f"\nData file stats:")
    print(f"  Original conversations: {data_original:,}")
    print(f"  Kept conversations: {data_filtered:,}")
    print(f"  Removed conversations: {data_original - data_filtered:,}")
    if data_original > 0:
        print(f"  Removal rate: {(data_original - data_filtered) / data_original * 100:.2f}%")

    print(f"\nEmpty-argument tool names: {len(empty_arg_tools)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
