#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple


def load_allowed_tools(tool_file: Path) -> Set[Tuple[str, str]]:
    """
    Load allowed (name, description) tool pairs from a JSON tool file.

    Args:
        tool_file: Path to the tool JSON file.

    Returns:
        Set of (name, description) tuples with stripped strings.
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
        
        print(f"Loaded {len(tool_set)} tools from {tool_file.name}")
        return tool_set
        
    except Exception as e:
        print(f"Error: failed to load tool file - {e}")
        import traceback
        traceback.print_exc()
        return set()


def has_invalid_tools(conversation: Dict[str, Any], allowed_tools: Set[Tuple[str, str]]) -> bool:
    """
    Return True if any function_call in conversations uses a tool not in allowed_tools.

    Matching uses (name, description) against the conversation's tools list.

    Args:
        conversation: Object with conversations and tools fields.
        allowed_tools: Allowed (name, description) pairs.

    Returns:
        True if any function_call references a disallowed tool.
    """
    if 'conversations' not in conversation:
        return False
    
    conversations = conversation.get('conversations', [])
    if not isinstance(conversations, list):
        return False
    
    tools_map: Dict[str, str] = {}
    if 'tools' in conversation:
        tools_str = conversation.get('tools', '')
        if isinstance(tools_str, str):
            try:
                tools_list = json.loads(tools_str)
                if isinstance(tools_list, list):
                    for tool in tools_list:
                        if isinstance(tool, dict):
                            tool_name = tool.get('name', '').strip()
                            tool_desc = tool.get('description', '').strip()
                            if tool_name:
                                tools_map[tool_name] = tool_desc
            except json.JSONDecodeError:
                pass
            except Exception as e:
                print(f"  Warning: failed to parse tools field: {e}")
    
    try:
        for msg in conversations:
            if not isinstance(msg, dict):
                continue
            
            if msg.get('from') != 'function_call':
                continue
            
            value = msg.get('value', '')
            if not isinstance(value, str):
                continue
            
            try:
                function_call = json.loads(value)
                if not isinstance(function_call, dict):
                    continue
                
                tool_name = function_call.get('name', '').strip()
                if not tool_name:
                    continue
                
                tool_desc = tools_map.get(tool_name, '').strip()
                
                tool_key = (tool_name, tool_desc)
                if tool_key not in allowed_tools:
                    return True
                    
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"  Warning: failed to parse function_call: {e}")
                continue
        
        return False
    
    except Exception as e:
        print(f"  Warning: failed to process conversations: {e}")
        return False


def process_data_file(
    input_file: Path, 
    output_file: Path, 
    allowed_tools: Set[Tuple[str, str]]
) -> Tuple[int, int]:
    """
    Filter one data file: drop conversations that reference disallowed tools.

    Args:
        input_file: Input JSON path.
        output_file: Output JSON path.
        allowed_tools: Allowed tool pairs.

    Returns:
        (original_count, kept_count)
    """
    print(f"Processing: {input_file.name}")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print("  Warning: root is not a list; skip")
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

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0


def process_data_directory(
    input_dir: Path, 
    output_dir: Path, 
    allowed_tools: Set[Tuple[str, str]]
) -> Tuple[int, int]:
    """
    Process all JSON files under input_dir.

    Args:
        input_dir: Input directory.
        output_dir: Output directory.
        allowed_tools: Allowed tool pairs.

    Returns:
        (total_original, total_kept)
    """
    if not input_dir.exists():
        print(f"Error: input directory not found: {input_dir}")
        return 0, 0

    json_files = sorted(input_dir.glob('*.json'))
    json_files = [f for f in json_files if not f.name.endswith('.bak') and not f.name.endswith('.bak2')]

    if not json_files:
        print(f"Warning: no JSON files under {input_dir}")
        return 0, 0

    print(f"\nDirectory: {input_dir}")
    print(f"Found {len(json_files)} JSON file(s)")

    total_original = 0
    total_filtered = 0

    for json_file in json_files:
        output_file = output_dir / json_file.name
        original, filtered = process_data_file(json_file, output_file, allowed_tools)
        total_original += original
        total_filtered += filtered

    return total_original, total_filtered


def main():
    """CLI entry point"""
    tool_file = Path("/home/yijuan_liang/10.12Tool_Set/tool_set/apis/apis_nonull/test_set_tool_dedup.json")
    data_dir = Path("/home/yijuan_liang/10.12Tool_Set/test_set/data/data_origin")
    data_output_dir = Path("/home/yijuan_liang/10.12Tool_Set/test_set/data/data_nonull")
    
    print("=" * 80)
    print("Remove conversations whose tools are not in the allowlist")
    print("=" * 80)
    
    print("\nLoading tool list...")
    allowed_tools = load_allowed_tools(tool_file)
    
    if not allowed_tools:
        print("Error: empty allowlist; abort")
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
    print(f"  Original: {data_original}")
    print(f"  Kept: {data_filtered}")
    print(f"  Removed: {data_original - data_filtered}")
    if data_original > 0:
        print(f"  Keep ratio: {data_filtered / data_original * 100:.2f}%")
    
    print(f"\nOutput written under: {data_output_dir}")


if __name__ == "__main__":
    main()
