#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import shutil
from pathlib import Path
from typing import Dict, Set, Tuple


def normalize_string(s: str) -> str:
    """Normalize string: strip all whitespace."""
    if s is None:
        return ""
    return str(s).replace(" ", "").replace("\t", "").replace("\n", "").replace("\r", "")


def extract_tool_signature(tool: Dict) -> Tuple[str, str]:
    """Return normalized (name, description) from a tool dict."""
    name = normalize_string(tool.get("name", ""))
    description = normalize_string(tool.get("description", ""))
    return name, description


def load_tool_file(file_path: Path) -> Dict:
    """Load a tools JSON file."""
    print(f"Loading tool file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} tools")
    return data


def extract_tools_from_data_nonull(data_dir: Path) -> Set[Tuple[str, str]]:
    """
    Collect signatures from data_nonull JSON (tools field and function_call names).
    Returns: Set[(normalized_name, normalized_description)]
    """
    print("\nExtracting tool signatures from data_nonull...")
    
    tools_set = set()
    json_files = []
    
    exclude_files = ["filter_statistics.json", "direct_answer_statistics.json", 
                     "chinese_content_statistics.json", "chinese_removal_statistics.json",
                     "count_tokens.py"]
    
    for file in data_dir.glob("*.json"):
        if file.name not in exclude_files and not file.name.endswith('.bak'):
            json_files.append(file)
    
    print(f"Found {len(json_files)} JSON files to process")
    
    total_conversations = 0
    processed_files = 0
    tools_from_tools_field = 0
    tools_from_function_call = 0
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                continue
            
            file_tools_count = 0
            for conversation in data:
                total_conversations += 1
                
                tools_value = conversation.get("tools")
                if tools_value:
                    tools_list = []
                    if isinstance(tools_value, str):
                        try:
                            tools_list = json.loads(tools_value)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    elif isinstance(tools_value, list):
                        tools_list = tools_value
                    
                    for tool in tools_list:
                        if isinstance(tool, dict):
                            name, description = extract_tool_signature(tool)
                            if name:
                                tools_set.add((name, description))
                                tools_from_tools_field += 1
                                file_tools_count += 1
                
                conversations = conversation.get("conversations", [])
                if isinstance(conversations, list):
                    for msg in conversations:
                        if not isinstance(msg, dict):
                            continue
                        
                        if msg.get("from") == "function_call":
                            value = msg.get("value", "")
                            if value and isinstance(value, str):
                                try:
                                    fc_data = json.loads(value)
                                    if isinstance(fc_data, dict):
                                        tool_name = fc_data.get("name", "")
                                        if tool_name:
                                            normalized_name = normalize_string(tool_name)
                                            tools_set.add((normalized_name, ""))
                                            tools_from_function_call += 1
                                except (json.JSONDecodeError, TypeError):
                                    pass
            
            if file_tools_count > 0:
                processed_files += 1
                print(f"  {file_path.name}: {file_tools_count} unique tool(s)")
        
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nProcessed {total_conversations} conversation(s)")
    print(f"From {processed_files} file(s): {len(tools_set)} unique (name, description) pairs")
    print(f"  From tools field: {tools_from_tools_field}")
    print(f"  From function_call: {tools_from_function_call}")
    
    return tools_set


def filter_tools(tool_file_data: Dict, used_tools: Set[Tuple[str, str]]) -> Tuple[Dict, Dict]:
    """
    Keep only tools that appear in used_tools.

    Matching:
    1. Exact (name, description) from tools field
    2. If used_tools contains (name, ""), name-only match (from function_call)
    """
    filtered_data = {}
    stats = {
        "total": len(tool_file_data),
        "used": 0,
        "unused": 0,
        "unused_tools": []
    }
    
    for key, tool in tool_file_data.items():
        name, description = extract_tool_signature(tool)
        signature = (name, description)
        
        is_used = False
        
        if signature in used_tools:
            is_used = True
        else:
            for used_name, used_desc in used_tools:
                if used_name == name:
                    is_used = True
                    break
        
        if is_used:
            filtered_data[key] = tool
            stats["used"] += 1
        else:
            stats["unused"] += 1
            stats["unused_tools"].append({
                "key": key,
                "name": tool.get("name", ""),
                "description": tool.get("description", "")[:100] if tool.get("description") else ""
            })
    
    return filtered_data, stats


def main():
    """CLI entry point"""
    tool_file1 = Path("/home/yijuan_liang/10.12Tool_Set/tool_set/apis/apis_nonull/test_set_tool_dedup.json")
    tool_file2 = Path("/home/yijuan_liang/10.12Tool_Set/tool_set/apis/apis_cosdup/test_set_tool_dedup_cosdup.json")
    data_nonull_dir = Path("/home/yijuan_liang/10.12Tool_Set/test_set/data/data_nonull")
    
    print("=" * 80)
    print("Remove unused tools")
    print("=" * 80)
    
    if not tool_file1.exists():
        print(f"Error: tool file not found: {tool_file1}")
        return
    
    if not tool_file2.exists():
        print(f"Error: tool file not found: {tool_file2}")
        return
    
    if not data_nonull_dir.exists():
        print(f"Error: data_nonull not found: {data_nonull_dir}")
        return
    
    used_tools = extract_tools_from_data_nonull(data_nonull_dir)
    
    if not used_tools:
        print("\nWarning: no tools found in data_nonull!")
        return
    
    print("\n" + "=" * 80)
    print("Load tool files")
    print("=" * 80)
    
    tool_data1 = load_tool_file(tool_file1)
    tool_data2 = load_tool_file(tool_file2)
    
    print("\n" + "=" * 80)
    print("Filter tool files")
    print("=" * 80)
    
    filtered_data1, stats1 = filter_tools(tool_data1, used_tools)
    filtered_data2, stats2 = filter_tools(tool_data2, used_tools)
    
    print("\n" + "=" * 80)
    print("Statistics")
    print("=" * 80)
    print(f"\nFile 1: {tool_file1.name}")
    print(f"  Total tools: {stats1['total']}")
    print(f"  Used: {stats1['used']}")
    print(f"  Unused: {stats1['unused']}")
    if stats1['total'] > 0:
        print(f"  Retention: {stats1['used']/stats1['total']*100:.2f}%")
    
    print(f"\nFile 2: {tool_file2.name}")
    print(f"  Total tools: {stats2['total']}")
    print(f"  Used: {stats2['used']}")
    print(f"  Unused: {stats2['unused']}")
    if stats2['total'] > 0:
        print(f"  Retention: {stats2['used']/stats2['total']*100:.2f}%")
    
    print("\n" + "=" * 80)
    print("Save files")
    print("=" * 80)
    
    backup_file1 = tool_file1.with_suffix('.json.bak')
    shutil.copy2(tool_file1, backup_file1)
    print(f"✓ Backed up: {backup_file1.name}")
    
    with open(tool_file1, 'w', encoding='utf-8') as f:
        json.dump(filtered_data1, f, ensure_ascii=False, indent=2)
    print(f"✓ Updated: {tool_file1.name}")
    
    backup_file2 = tool_file2.with_suffix('.json.bak')
    shutil.copy2(tool_file2, backup_file2)
    print(f"✓ Backed up: {backup_file2.name}")
    
    with open(tool_file2, 'w', encoding='utf-8') as f:
        json.dump(filtered_data2, f, ensure_ascii=False, indent=2)
    print(f"✓ Updated: {tool_file2.name}")
    
    stats_file = tool_file1.parent / "test_set_tool_filter_statistics.json"
    statistics = {
        "used_tools_count": len(used_tools),
        "data_source": str(data_nonull_dir),
        "file1": {
            "path": str(tool_file1),
            "stats": stats1
        },
        "file2": {
            "path": str(tool_file2),
            "stats": stats2
        }
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Stats saved to: {stats_file.name}")
    
    if stats1['unused_tools'] or stats2['unused_tools']:
        print("\n" + "=" * 80)
        print("Sample unused tools (first 10)")
        print("=" * 80)
        
        if stats1['unused_tools']:
            print(f"\nFile 1 unused examples:")
            for i, tool in enumerate(stats1['unused_tools'][:10], 1):
                print(f"  {i}. Key: {tool['key']}, Name: {tool['name']}")
            if len(stats1['unused_tools']) > 10:
                print(f"  ... and {len(stats1['unused_tools']) - 10} more")
        
        if stats2['unused_tools']:
            print(f"\nFile 2 unused examples:")
            for i, tool in enumerate(stats2['unused_tools'][:10], 1):
                print(f"  {i}. Key: {tool['key']}, Name: {tool['name']}")
            if len(stats2['unused_tools']) > 10:
                print(f"  ... and {len(stats2['unused_tools']) - 10} more")
    
    print("\n" + "=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
