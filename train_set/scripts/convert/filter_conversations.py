#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import shutil
from pathlib import Path

def has_direct_answer(conversations):
    """
    True if some human message is immediately followed by gpt/assistant (no function_call).
    """
    i = 0
    while i < len(conversations):
        if conversations[i].get("from") == "human":
            if i + 1 < len(conversations):
                next_msg = conversations[i + 1]
                if next_msg.get("from") in ["gpt", "assistant"]:
                    return True
                elif next_msg.get("from") == "function_call":
                    j = i + 1
                    while j < len(conversations):
                        if conversations[j].get("from") in ["human", "gpt", "assistant"]:
                            i = j
                            break
                        j += 1
                    else:
                        i = len(conversations)
                    continue
        i += 1
    return False

def is_empty_tools(tools_value):
    """
    True if tools field is an empty JSON array (string or list).
    """
    if tools_value is None:
        return False
    
    if isinstance(tools_value, str):
        stripped = tools_value.strip()
        if stripped == "[]" or stripped == "":
            return True
        try:
            tools = json.loads(tools_value)
            if isinstance(tools, list):
                return len(tools) == 0
        except (json.JSONDecodeError, TypeError):
            return False
    
    if isinstance(tools_value, list):
        return len(tools_value) == 0
    
    return False

def should_filter_item(item):
    """
    True if item should be dropped (direct text answer after query, or empty tools).
    """
    if "tools" in item:
        if is_empty_tools(item["tools"]):
            return True
    
    if "conversations" in item:
        if has_direct_answer(item["conversations"]):
            return True
    
    return False

def filter_file(input_path, output_path):
    """Filter one JSON array file."""
    print(f"Processing: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        original_count = len(data)
        filtered_data = []
        filtered_count = 0
        
        for item in data:
            if should_filter_item(item):
                filtered_count += 1
            else:
                filtered_data.append(item)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        
        print(f"  Original: {original_count}, filtered out: {filtered_count}, kept: {len(filtered_data)}")
        
        return {
            "file": str(input_path),
            "original": original_count,
            "filtered": filtered_count,
            "kept": len(filtered_data)
        }
    except Exception as e:
        print(f"  Error processing {input_path}: {e}")
        return {
            "file": str(input_path),
            "original": 0,
            "filtered": 0,
            "kept": 0,
            "error": str(e)
        }

def main():
    input_dir = Path("/home/yijuan_liang/10.12Tool_Set/train_set/data/data_nonull")
    output_dir = Path("/home/yijuan_liang/10.12Tool_Set/train_set/data/data_nonull")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = []
    
    for file in input_dir.glob("*.json"):
        json_files.append((file, output_dir / file.name))
    
    toucan_input_dir = input_dir / "TOUCAN_converted"
    toucan_output_dir = output_dir / "TOUCAN_converted"
    
    if toucan_input_dir.exists():
        for file in toucan_input_dir.rglob("*.json"):
            rel_path = file.relative_to(toucan_input_dir)
            output_path = toucan_output_dir / rel_path
            json_files.append((file, output_path))
    
    print(f"Found {len(json_files)} JSON file(s)\n")
    print("="*80)
    
    results = []
    total_original = 0
    total_filtered = 0
    total_kept = 0
    
    for input_path, output_path in sorted(json_files):
        result = filter_file(input_path, output_path)
        results.append(result)
        total_original += result["original"]
        total_filtered += result["filtered"]
        total_kept += result["kept"]
    
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    print(f"{'File':<60} {'Original':<12} {'Filtered':<12} {'Kept':<12}")
    print("-"*80)
    
    for result in results:
        file_name = os.path.basename(result["file"])
        if "TOUCAN_converted" in result["file"]:
            rel_path = os.path.relpath(result["file"], input_dir)
            file_name = rel_path
        
        original = result["original"]
        filtered = result["filtered"]
        kept = result["kept"]
        
        print(f"{file_name:<60} {original:<12} {filtered:<12} {kept:<12}")
    
    print("-"*80)
    print(f"{'Total':<60} {total_original:<12} {total_filtered:<12} {total_kept:<12}")
    print("="*80)
    
    stats_file = output_dir / "filter_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nStats saved to: {stats_file}")
    print(f"\nFiltered files written under: {output_dir}")

if __name__ == "__main__":
    main()
