#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set


def normalize_string(s: str) -> str:
    """Normalize a string and strip formatting differences."""
    if not isinstance(s, str):
        return ""
    # Strip leading/trailing whitespace
    s = s.strip()
    # Normalize newlines: collapse consecutive whitespace/newlines to a single space
    import re
    s = re.sub(r'\s+', ' ', s)  # Collapse all whitespace (including newlines, tabs) to one space
    return s.strip()


def normalize_description(desc: str) -> str:
    """Normalize description and strip formatting differences."""
    return normalize_string(desc)


def normalize_name(name: str) -> str:
    """Normalize name and strip formatting differences."""
    return normalize_string(name)


def load_reference_tools(ref_file: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    """
    Load the reference file and build an exact map (name, normalized_description) -> {category, domain}.
    Normalization removes formatting differences.

    Args:
        ref_file: Path to the reference tools file

    Returns:
        Exact mapping dict: {(name, description): {category: ..., domain: ...}}
    """
    print(f"Loading reference file: {ref_file.name}")

    if not ref_file.exists():
        print(f"  Error: reference file not found: {ref_file}")
        return {}

    try:
        with open(ref_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Error: could not load reference file: {e}")
        return {}

    tool_map = {}  # Exact map: (name, description) -> {category, domain}
    count = 0

    if isinstance(data, dict):
        # Standard format: numeric keys
        for tool_id, tool in data.items():
            if not isinstance(tool, dict):
                continue

            name = tool.get("name", "")
            description = tool.get("description", "")
            category = tool.get("category")
            domain = tool.get("domain")

            if name and (category or domain):
                normalized_name = normalize_name(name)
                normalized_desc = normalize_description(description)
                key = (normalized_name, normalized_desc)

                # Exact mapping (prefer entries with both category and domain)
                if key not in tool_map or (category and domain):
                    tool_map[key] = {
                        "category": category,
                        "domain": domain
                    }
                count += 1
    elif isinstance(data, list):
        # List format
        for tool in data:
            if not isinstance(tool, dict):
                continue

            name = tool.get("name", "")
            description = tool.get("description", "")
            category = tool.get("category")
            domain = tool.get("domain")

            if name and (category or domain):
                normalized_name = normalize_name(name)
                normalized_desc = normalize_description(description)
                key = (normalized_name, normalized_desc)

                if key not in tool_map or (category and domain):
                    tool_map[key] = {
                        "category": category,
                        "domain": domain
                    }
                count += 1

    print(f"  Loaded {count} tools, built {len(tool_map)} exact mappings")
    return tool_map


def add_category_domain_to_tool(
    tool: Dict[str, Any],
    ref_map: Dict[Tuple[str, str], Dict[str, str]],
    unmatched_tools: Set[Tuple[str, str]]
) -> bool:
    """
    Add category and domain to a single tool if missing (skip if both already set).
    Uses exact match on (name + description) after normalization.

    Args:
        tool: Tool dict
        ref_map: Exact map (name, description) -> {category, domain}
        unmatched_tools: Set of unmatched tools (for logging, deduped automatically)
    """
    modified = False

    name = tool.get("name", "")
    description = tool.get("description", "")

    if not name:
        return False

    # Check if category and domain already exist
    has_category = "category" in tool and tool.get("category")
    has_domain = "domain" in tool and tool.get("domain")

    # Skip if both category and domain are already present
    if has_category and has_domain:
        return False

    # Normalize name and description
    normalized_name = normalize_name(name)
    normalized_desc = normalize_description(description)
    key = (normalized_name, normalized_desc)

    # Exact match (name + description)
    if key in ref_map:
        ref_info = ref_map[key]

        # Only fill missing fields
        if not has_category and ref_info.get("category"):
            tool["category"] = ref_info["category"]
            modified = True

        if not has_domain and ref_info.get("domain"):
            tool["domain"] = ref_info["domain"]
            modified = True
    else:
        # Record unmatched tools (Set dedupes)
        unmatched_tools.add((normalized_name, normalized_desc))

    return modified


def process_json_file(
    file_path: Path,
    ref_map: Dict[Tuple[str, str], Dict[str, str]],
    unmatched_tools: Set[Tuple[str, str]]
) -> Tuple[int, int, int]:
    """
    Process a single JSON file.

    Args:
        file_path: File path
        ref_map: Reference mapping
        unmatched_tools: Unmatched tools collection

    Returns:
        (modified_tool_count, total_tool_count, modified_conversation_count)
    """
    print(f"Processing file: {file_path.name}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  Error: JSON parse failed: {e}")
        print(f"  Skipping this file")
        return 0, 0, 0
    except Exception as e:
        print(f"  Error: failed to read file: {e}")
        return 0, 0, 0

    if not isinstance(data, list):
        print(f"  Warning: {file_path.name} is not a list, skipping")
        return 0, 0, 0

    modified_count = 0
    total_tools = 0
    modified_items = 0

    for item in data:
        if "tools" not in item:
            continue

        tools_str = item["tools"]
        if not isinstance(tools_str, str):
            continue

        # Parse tools field (with retries)
        tools = None
        max_retries = 3
        for retry in range(max_retries):
            try:
                tools = json.loads(tools_str)
                break
            except json.JSONDecodeError as e:
                if retry < max_retries - 1:
                    # Try to fix common JSON issues
                    tools_str = tools_str.rstrip().rstrip(',')
                    continue
                else:
                    print(f"  Warning: could not parse tools JSON (failed after {max_retries} retries): {e}")
                    print(f"    First 100 chars of tools field: {tools_str[:100]}")
                    tools = None
                    break

        if tools is None or not isinstance(tools, list):
            continue

        # Process each tool
        tool_modified = False
        for tool in tools:
            if not isinstance(tool, dict):
                continue

            total_tools += 1
            if add_category_domain_to_tool(tool, ref_map, unmatched_tools):
                modified_count += 1
                tool_modified = True

        # If any tool was modified, update the tools field
        if tool_modified:
            item["tools"] = json.dumps(tools, ensure_ascii=False)
            modified_items += 1

    # Save if modified
    if modified_items > 0:
        # Create backup
        backup_path = file_path.with_suffix('.json.bak')
        if not backup_path.exists():
            try:
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"  Warning: failed to create backup: {e}")

        # Write updated file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  Error: failed to save file: {e}")

    print(f"  Modified {modified_count}/{total_tools} tools across {modified_items} conversations")
    return modified_count, total_tools, modified_items


def process_directory(
    data_dir: Path,
    ref_map: Dict[Tuple[str, str], Dict[str, str]],
    unmatched_tools: Set[Tuple[str, str]],
    base_dir: Path = None
) -> Tuple[int, int, int]:
    """
    Recursively process all JSON files under a directory (including subdirs).

    Args:
        data_dir: Data directory path
        ref_map: Reference mapping
        unmatched_tools: Unmatched tools collection
        base_dir: Base directory (for relative paths in logs)

    Returns:
        (total_modified_tools, total_tools, total_modified_conversations)
    """
    if not data_dir.exists():
        print(f"Error: directory does not exist: {data_dir}")
        return 0, 0, 0

    if base_dir is None:
        base_dir = data_dir

    total_modified = 0
    total_tools = 0
    total_conversations = 0

    # JSON files in current directory (exclude backups)
    json_files = sorted([
        f for f in data_dir.glob('*.json')
        if not f.name.endswith('.bak') and not f.name.endswith('.bak2')
    ])

    for json_file in json_files:
        modified, tools, conversations = process_json_file(json_file, ref_map, unmatched_tools)
        total_modified += modified
        total_tools += tools
        total_conversations += conversations

    # Recurse into subdirectories
    for subdir in sorted(data_dir.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith('.'):
            modified, tools, conversations = process_directory(
                subdir, ref_map, unmatched_tools, base_dir
            )
            total_modified += modified
            total_tools += tools
            total_conversations += conversations

    return total_modified, total_tools, total_conversations


def main():
    """CLI entry point"""
    # Paths (same toolset as filtering scripts)
    # Prefer apis_cosdup when it has category/domain; else apis_nonull
    test_ref_file_cosdup = Path("/home/yijuan_liang/10.12Tool_Set/tool_set/apis/apis_cosdup/test_set_tool_dedup_cosdup.json")
    test_ref_file_nonull = Path("/home/yijuan_liang/10.12Tool_Set/tool_set/apis/apis_nonull/test_set_tool_dedup.json")

    test_ref_file = test_ref_file_cosdup if test_ref_file_cosdup.exists() else test_ref_file_nonull

    test_data_dir = Path("/home/yijuan_liang/10.12Tool_Set/test_set/data/data_nonull")

    print("=" * 80)
    print("Add category and domain to dataset from reference tools")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("Load reference tools")
    print("=" * 80)
    test_ref_map = load_reference_tools(test_ref_file)

    print("\n" + "=" * 80)
    print("Process test dataset")
    print("=" * 80)
    test_unmatched_tools = set()
    test_total_modified, test_total_tools, test_total_conversations = process_directory(
        test_data_dir, test_ref_map, test_unmatched_tools
    )

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Test dataset:")
    print(f"  Tools modified: {test_total_modified}")
    print(f"  Total tools: {test_total_tools}")
    print(f"  Conversations modified: {test_total_conversations}")

    if test_unmatched_tools:
        print(f"\nUnmatched tools: {len(test_unmatched_tools)} (deduped)")
        print("All unmatched tools:")
        sorted_unmatched = sorted(test_unmatched_tools, key=lambda x: x[0])
        for name, desc in sorted_unmatched:
            print(f"  - {name}: {desc}")
    else:
        print("\nAll tools matched.")

    print("\nDone.")


if __name__ == "__main__":
    main()
