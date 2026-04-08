#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from pathlib import Path
from collections import defaultdict

def has_direct_answer(conversations):
    """
    Returns (True, example_info) if a human turn is immediately followed by gpt/assistant
    without an intervening function_call.
    example_info has human_query and direct_answer text.
    """
    i = 0
    while i < len(conversations):
        if conversations[i].get("from") == "human":
            if i + 1 < len(conversations):
                next_msg = conversations[i + 1]
                if next_msg.get("from") in ["gpt", "assistant"]:
                    example_info = {
                        "human_query": conversations[i].get("value", ""),
                        "direct_answer": next_msg.get("value", "")
                    }
                    return True, example_info
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
    return False, None

def is_empty_tools(tools_value):
    """True if tools is empty JSON array (string or list)."""
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

def analyze_file(file_path):
    """Analyze one file; return stats and optional examples."""
    print(f"Analyzing: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_count = len(data)
        direct_answer_count = 0
        empty_tools_count = 0
        example_item = None
        empty_tools_example = None
        
        for item in data:
            if "tools" in item:
                if is_empty_tools(item["tools"]):
                    empty_tools_count += 1
                    if empty_tools_example is None:
                        empty_tools_example = item.copy()
            
            if "conversations" in item:
                has_direct, example_info = has_direct_answer(item["conversations"])
                if has_direct:
                    direct_answer_count += 1
                    if example_item is None:
                        example_item = item.copy()
                        example_item["_example_info"] = example_info
        
        result = {
            "file": str(file_path),
            "total": total_count,
            "direct_answer": direct_answer_count,
            "empty_tools": empty_tools_count
        }
        
        if example_item is not None:
            result["example"] = example_item
        if empty_tools_example is not None:
            result["empty_tools_example"] = empty_tools_example
        
        return result
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {
            "file": str(file_path),
            "total": 0,
            "direct_answer": 0,
            "empty_tools": 0,
            "error": str(e)
        }

def main():
    base_dir = Path("/home/yijuan_liang/10.12Tool_Set/train_set/data/data_nonull")
    output_dir = Path("/home/yijuan_liang/10.12Tool_Set/train_set/data/data_analysis")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = []
    
    for file in base_dir.glob("*.json"):
        json_files.append(file)
    
    toucan_dir = base_dir / "TOUCAN_converted"
    if toucan_dir.exists():
        for file in toucan_dir.rglob("*.json"):
            json_files.append(file)
    
    print(f"Found {len(json_files)} JSON file(s)\n")
    
    results = []
    examples = []
    empty_tools_examples = []
    
    for json_file in sorted(json_files):
        result = analyze_file(json_file)
        results.append(result)
        
        if "example" in result:
            examples.append({
                "source_file": os.path.basename(json_file),
                "source_path": str(json_file),
                "example": result["example"]
            })
        
        if "empty_tools_example" in result:
            empty_tools_examples.append({
                "source_file": os.path.basename(json_file),
                "source_path": str(json_file),
                "example": result["empty_tools_example"]
            })
    
    print("\n" + "="*100)
    print("Statistics:")
    print("="*100)
    print(f"{'File':<60} {'Total':<10} {'Direct ans':<12} {'Empty tools':<12} {'Ex direct':<10} {'Ex empty':<12}")
    print("-"*100)
    
    total_all = 0
    direct_answer_all = 0
    empty_tools_all = 0
    
    for result in results:
        file_name = os.path.basename(result["file"])
        if "TOUCAN_converted" in result["file"]:
            rel_path = os.path.relpath(result["file"], base_dir)
            file_name = rel_path
        
        total = result["total"]
        direct_answer = result["direct_answer"]
        empty_tools = result.get("empty_tools", 0)
        has_direct_example = "yes" if "example" in result else "no"
        has_empty_tools_example = "yes" if "empty_tools_example" in result else "no"
        total_all += total
        direct_answer_all += direct_answer
        empty_tools_all += empty_tools
        
        print(f"{file_name:<60} {total:<10} {direct_answer:<12} {empty_tools:<12} {has_direct_example:<10} {has_empty_tools_example:<12}")
    
    print("-"*100)
    print(f"{'Total':<60} {total_all:<10} {direct_answer_all:<12} {empty_tools_all:<12}")
    print("="*100)
    
    stats_results = []
    for result in results:
        stats_result = {
            "file": result["file"],
            "total": result["total"],
            "direct_answer": result["direct_answer"],
            "empty_tools": result.get("empty_tools", 0)
        }
        if "error" in result:
            stats_result["error"] = result["error"]
        stats_results.append(stats_result)
    
    output_file = output_dir / "direct_answer_statistics.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats_results, f, ensure_ascii=False, indent=2)
    print(f"\nStats saved to: {output_file}")
    
    if examples:
        examples_file = output_dir / "direct_answer_examples.json"
        with open(examples_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        print(f"Direct-answer examples saved to: {examples_file}")
        print(f"Files with example: {len(examples)}")
    else:
        print("\nNo direct-answer examples collected")
    
    if empty_tools_examples:
        empty_tools_file = output_dir / "empty_tools_examples.json"
        with open(empty_tools_file, 'w', encoding='utf-8') as f:
            json.dump(empty_tools_examples, f, ensure_ascii=False, indent=2)
        print(f"Empty-tools examples saved to: {empty_tools_file}")
        print(f"Files with example: {len(empty_tools_examples)}")
    else:
        print("\nNo empty-tools examples collected")

if __name__ == "__main__":
    main()


