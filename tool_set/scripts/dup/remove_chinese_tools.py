#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import shutil
from pathlib import Path

# Regex for CJK characters (used to detect Chinese in tool text)
CHINESE_PATTERN = re.compile(r'[\u4e00-\u9fa5\u3400-\u4dbf\uf900-\ufaff]')

def contains_chinese(text):
    """Return True if text contains CJK characters."""
    if text is None:
        return False
    if not isinstance(text, str):
        text = str(text)
    return bool(CHINESE_PATTERN.search(text))

def tool_has_chinese(tool):
    """Return True if any relevant tool field contains CJK text."""
    if "name" in tool:
        if contains_chinese(tool["name"]):
            return True
    
    if "description" in tool:
        if contains_chinese(tool["description"]):
            return True
    
    if "inputSchema" in tool:
        schema_str = json.dumps(tool["inputSchema"], ensure_ascii=False)
        if contains_chinese(schema_str):
            return True
    
    for key, value in tool.items():
        if isinstance(value, str):
            if contains_chinese(value):
                return True
    
    return False

def remove_chinese_tools(file_path):
    """Remove tools that contain Chinese from a JSON tool file."""
    print(f"Processing: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_count = len(data)
    filtered_data = {}
    removed_tools = []
    
    for key, tool in data.items():
        if tool_has_chinese(tool):
            removed_tools.append({
                "key": key,
                "name": tool.get("name", ""),
                "description": tool.get("description", "")[:100] if tool.get("description") else ""
            })
        else:
            filtered_data[key] = tool
    
    removed_count = len(removed_tools)
    kept_count = len(filtered_data)
    
    print(f"  Original tools: {original_count}")
    print(f"  Removed: {removed_count}")
    print(f"  Kept: {kept_count}")
    
    backup_path = file_path.with_suffix('.json.bak')
    shutil.copy2(file_path, backup_path)
    print(f"  Backup: {backup_path}")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    print(f"  Updated: {file_path}")
    
    return {
        "file": str(file_path),
        "original": original_count,
        "removed": removed_count,
        "kept": kept_count,
        "removed_tools": removed_tools
    }

def main():
    file_path = Path("/home/yijuan_liang/10.12Tool_Set/tool_set/apis/apis_cosdup/toolret_tool_dedup_cosdup.json")
    
    print("="*80)
    print("Remove tools that contain Chinese text")
    print("="*80)
    print()
    
    result = remove_chinese_tools(file_path)
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"File: {result['file']}")
    print(f"Original: {result['original']}")
    print(f"Removed: {result['removed']}")
    print(f"Kept: {result['kept']}")
    print(f"Removal rate: {result['removed']/result['original']*100:.2f}%")
    
    if result['removed_tools']:
        print("\nRemoved tools:")
        for tool in result['removed_tools']:
            print(f"  Key: {tool['key']}, Name: {tool['name']}")
    
    stats_file = file_path.parent / "chinese_tools_removal_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nStats written: {stats_file}")
    print("\n" + "="*80)
    print("Done.")
    print("="*80)

if __name__ == "__main__":
    main()
