#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
from pathlib import Path
from collections import defaultdict

# CJK codepoint ranges (for detection only)
CHINESE_PATTERN = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]')

def contains_chinese(text):
    """Return True if text contains CJK characters."""
    if text is None:
        return False
    
    if not isinstance(text, str):
        text = str(text)
    
    return bool(CHINESE_PATTERN.search(text))

def check_conversation_has_chinese(conversation):
    """
    Return True if any text field in the conversation contains CJK:
    system, tools, conversations[].value, etc.
    """
    if "system" in conversation:
        if contains_chinese(conversation["system"]):
            return True
    
    if "tools" in conversation:
        tools_value = conversation["tools"]
        if contains_chinese(tools_value):
            return True
    
    if "conversations" in conversation:
        conversations = conversation["conversations"]
        if isinstance(conversations, list):
            for conv in conversations:
                if "value" in conv:
                    if contains_chinese(conv["value"]):
                        return True
    
    return False

def analyze_file(file_path):
    """Analyze one JSON file; count conversations containing CJK."""
    print(f"Analyzing: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_count = len(data)
        chinese_count = 0
        chinese_example = None
        
        for item in data:
            if check_conversation_has_chinese(item):
                chinese_count += 1
                if chinese_example is None:
                    chinese_example = item.copy()
        
        result = {
            "file": str(file_path),
            "total": total_count,
            "chinese_count": chinese_count,
            "chinese_ratio": round(chinese_count / total_count * 100, 2) if total_count > 0 else 0
        }
        
        if chinese_example is not None:
            result["example"] = chinese_example
        
        return result
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {
            "file": str(file_path),
            "total": 0,
            "chinese_count": 0,
            "chinese_ratio": 0,
            "error": str(e)
        }

def main():
    base_dir = Path("/home/yijuan_liang/10.12Tool_Set/train_set/data/data_nonull")
    output_dir = Path("/home/yijuan_liang/10.12Tool_Set/train_set/data/data_analysis")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = []
    exclude_files = ["filter_statistics.json", "direct_answer_statistics.json", 
                     "chinese_content_statistics.json"]
    
    for file in base_dir.glob("*.json"):
        if file.name not in exclude_files:
            json_files.append(file)
    
    toucan_dir = base_dir / "TOUCAN_converted"
    if toucan_dir.exists():
        for file in toucan_dir.rglob("*.json"):
            json_files.append(file)
    
    print(f"Found {len(json_files)} JSON file(s)\n")
    print("="*80)
    
    results = []
    examples = []
    
    for json_file in sorted(json_files):
        result = analyze_file(json_file)
        results.append(result)
        
        if "example" in result:
            examples.append({
                "source_file": os.path.basename(json_file),
                "source_path": str(json_file),
                "example": result["example"]
            })
    
    print("\n" + "="*100)
    print("Statistics:")
    print("="*100)
    print(f"{'File':<60} {'Total':<10} {'CJK rows':<12} {'CJK %':<12} {'Example':<8}")
    print("-"*100)
    
    total_all = 0
    chinese_all = 0
    
    for result in results:
        file_name = os.path.basename(result["file"])
        if "TOUCAN_converted" in result["file"]:
            rel_path = os.path.relpath(result["file"], base_dir)
            file_name = rel_path
        
        total = result["total"]
        chinese_count = result["chinese_count"]
        chinese_ratio = result["chinese_ratio"]
        has_example = "yes" if "example" in result else "no"
        total_all += total
        chinese_all += chinese_count
        
        print(f"{file_name:<60} {total:<10} {chinese_count:<12} {chinese_ratio:<12} {has_example:<8}")
    
    print("-"*100)
    total_ratio = round(chinese_all / total_all * 100, 2) if total_all > 0 else 0
    print(f"{'TOTAL':<60} {total_all:<10} {chinese_all:<12} {total_ratio:<12}")
    print("="*100)
    
    stats_results = []
    for result in results:
        stats_result = {
            "file": result["file"],
            "total": result["total"],
            "chinese_count": result["chinese_count"],
            "chinese_ratio": result["chinese_ratio"]
        }
        if "error" in result:
            stats_result["error"] = result["error"]
        stats_results.append(stats_result)
    
    output_file = output_dir / "chinese_content_statistics.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats_results, f, ensure_ascii=False, indent=2)
    print(f"\nStats written: {output_file}")
    
    if examples:
        examples_file = output_dir / "chinese_content_examples.json"
        with open(examples_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        print(f"Examples written: {examples_file}")
        print(f"Datasets with at least one CJK example: {len(examples)}")
    else:
        print("\nNo conversations with CJK text found.")

if __name__ == "__main__":
    main()
