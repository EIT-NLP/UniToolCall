#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple


def load_allowed_tools(tool_file: Path) -> Set[Tuple[str, str]]:
    """
    Load all allowed tools as (name, description) pairs from a tools file.

    Args:
        tool_file: Path to the tools JSON file.

    Returns:
        Set of (name, description) tuples; names and descriptions are stripped.
    """
    try:
        with open(tool_file, 'r', encoding='utf-8') as f:
            tools_data = json.load(f)

        tool_set: Set[Tuple[str, str]] = set()

        if isinstance(tools_data, dict):
            # Shape: {"1": {"name": "...", "description": "...", ...}, ...}
            for tool_obj in tools_data.values():
                if isinstance(tool_obj, dict):
                    tool_name = tool_obj.get('name', '').strip()
                    tool_desc = tool_obj.get('description', '').strip()
                    if tool_name:
                        tool_set.add((tool_name, tool_desc))
        elif isinstance(tools_data, list):
            # Shape: [{"name": "...", "description": "...", ...}, ...]
            for tool_obj in tools_data:
                if isinstance(tool_obj, dict):
                    tool_name = tool_obj.get('name', '').strip()
                    tool_desc = tool_obj.get('description', '').strip()
                    if tool_name:
                        tool_set.add((tool_name, tool_desc))

        print(f"Loaded {len(tool_set)} tool(s) from {tool_file.name}")
        return tool_set

    except Exception as e:
        print(f"Error: failed to load tools file - {e}")
        import traceback
        traceback.print_exc()
        return set()


def has_invalid_tools(conversation: Dict[str, Any], allowed_tools: Set[Tuple[str, str]]) -> bool:
    """
    Return True if conversation's `tools` (ground truth) lists a tool not in the allowlist.

    Args:
        conversation: Object with a `tools` field.
        allowed_tools: Allowed (name, description) pairs.

    Returns:
        True if any tool in `tools` is not in the allowlist.
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
            tool_desc = tool.get('description', '').strip()

            if not tool_name:
                continue

            tool_key = (tool_name, tool_desc)
            if tool_key not in allowed_tools:
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
    allowed_tools: Set[Tuple[str, str]]
) -> tuple[int, int]:
    """
    Process one data file: drop conversations that reference disallowed tools.

    Args:
        input_file: Input JSON path.
        output_file: Output JSON path.
        allowed_tools: Allowed (name, description) pairs.

    Returns:
        (original_count, kept_count)
    """
    print(f"Processing file: {input_file.name}")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print("  Warning: file is not a JSON array; skipping")
            return 0, 0

        original_count = len(data)
        filtered_data: List[Dict[str, Any]] = []
        removed_count = 0

        for item in data:
            if has_invalid_tools(item, allowed_tools):
                removed_count += 1
            else:
                filtered_data.append(item)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)

        print(f"  Original: {original_count}, removed: {removed_count}, kept: {len(filtered_data)}")
        return original_count, len(filtered_data)

    except json.JSONDecodeError as e:
        print(f"  Error: invalid JSON - {e}")
        print(f"  At line {e.lineno}, column {e.colno}")
        print(f"  Skipping this file; continuing with others")
        return 0, 0
    except Exception as e:
        print(f"  Error: failed to process file - {e}")
        print(f"  Skipping this file; continuing with others")
        import traceback
        traceback.print_exc()
        return 0, 0


def process_data_directory(
    input_dir: Path,
    output_dir: Path,
    allowed_tools: Set[Tuple[str, str]],
    base_input_dir: Path = None
) -> tuple[int, int]:
    """
    Recursively process all JSON files under a data directory.

    Args:
        input_dir: Input root.
        output_dir: Output root.
        allowed_tools: Allowed (name, description) pairs.
        base_input_dir: Base for relative paths; defaults to input_dir.

    Returns:
        (total_original_count, total_kept_count)
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
        print(f"\nProcessing directory: {input_dir}")
        print(f"Found {len(json_files)} JSON file(s)")

        for json_file in json_files:
            relative_path = json_file.relative_to(base_input_dir)
            output_file = output_dir / relative_path
            original, filtered = process_data_file(json_file, output_file, allowed_tools)
            total_original += original
            total_filtered += filtered

    for subdir in sorted(input_dir.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith('.'):
            sub_output_dir = output_dir / subdir.relative_to(base_input_dir)
            original, filtered = process_data_directory(
                subdir, sub_output_dir, allowed_tools, base_input_dir
            )
            total_original += original
            total_filtered += filtered

    return total_original, total_filtered


def main():
    """CLI entry point"""
    tool_file = Path("/home/yijuan_liang/10.12Tool_Set/tool_set/apis/apis_nonull/train_set_tool_dedup.json")
    data_dir = Path("/home/yijuan_liang/10.12Tool_Set/train_set/data")
    data_output_dir = Path("/home/yijuan_liang/10.12Tool_Set/train_set/data_nonull")

    print("=" * 80)
    print("Remove conversations whose tools are not in the allowlist")
    print("=" * 80)

    print("\nLoading tool list...")
    allowed_tools = load_allowed_tools(tool_file)

    if not allowed_tools:
        print("Error: could not load tools; aborting")
        return

    print(f"Allowed tools: {len(allowed_tools)}")

    print("\n" + "=" * 80)
    print("Processing data directory")
    print("=" * 80)
    data_original, data_filtered = process_data_directory(data_dir, data_output_dir, allowed_tools)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"data directory:")
    print(f"  Original rows: {data_original}")
    print(f"  Kept rows: {data_filtered}")
    print(f"  Removed rows: {data_original - data_filtered}")
    if data_original > 0:
        print(f"  Keep rate: {data_filtered / data_original * 100:.2f}%")

    print(f"\nOutput written under: {data_output_dir}")


if __name__ == "__main__":
    main()
