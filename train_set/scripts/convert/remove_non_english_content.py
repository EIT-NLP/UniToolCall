#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import shutil
from pathlib import Path

# Regex for CJK / non-Latin scripts (CJK Unified, extensions, Hiragana, Katakana, Hangul, etc.)
# Ranges:
# - CJK Unified Ideographs: \u4e00-\u9fff
# - Extension A: \u3400-\u4dbf
# - Extension B and beyond: \uf900-\ufaff
# - Hiragana: \u3040-\u309f
# - Katakana: \u30a0-\u30ff
# - CJK symbols/punctuation: \u3000-\u303f
# - Hangul syllables: \uac00-\ud7af
# - Hangul compatibility jamo: \u3130-\u318f
# ASCII letters, digits, common punctuation, spaces, Latin extensions (e.g. é, ñ) are kept.
# This pattern covers the common CJK ranges plus Japanese and Korean.
CJK_PATTERN = re.compile(r'[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u3130-\u318f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]')

def contains_non_english(text):
    """
    True if text contains CJK or related non-Latin-script characters.
    Keeps ASCII and common Latin extensions.
    """
    if text is None:
        return False
    
    if not isinstance(text, str):
        text = str(text)
    
    return bool(CJK_PATTERN.search(text))

def check_conversation_has_non_english(conversation):
    """
    True if any checked text field contains CJK/non-Latin content.
    Scans: system, tools (string), conversations[].value
    """
    if "system" in conversation:
        if contains_non_english(conversation["system"]):
            return True
    
    if "tools" in conversation:
        tools_value = conversation["tools"]
        if contains_non_english(tools_value):
            return True
    
    if "conversations" in conversation:
        conversations = conversation["conversations"]
        if isinstance(conversations, list):
            for conv in conversations:
                if "value" in conv:
                    if contains_non_english(conv["value"]):
                        return True
    
    return False

def find_non_english_text(text):
    """Return sample CJK characters found in text (up to 10 unique)."""
    if text is None:
        return []
    
    if not isinstance(text, str):
        text = str(text)
    
    matches = CJK_PATTERN.findall(text)
    return list(set(matches))[:10]

def filter_file(file_path, max_examples=5):
    """
    Drop conversations that contain CJK/non-Latin text.
    Returns stats dict including removed examples.
    """
    print(f"Processing: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        original_count = len(data)
        filtered_data = []
        removed_count = 0
        removed_examples = []
        
        for item in data:
            if check_conversation_has_non_english(item):
                removed_count += 1
                if len(removed_examples) < max_examples:
                    example = item.copy()
                    non_english_info = {}
                    
                    if "system" in item:
                        non_english_chars = find_non_english_text(item["system"])
                        if non_english_chars:
                            non_english_info["system"] = non_english_chars
                    
                    if "tools" in item:
                        non_english_chars = find_non_english_text(item["tools"])
                        if non_english_chars:
                            non_english_info["tools"] = non_english_chars
                    
                    if "conversations" in item:
                        conv_non_english = []
                        for i, conv in enumerate(item["conversations"]):
                            if "value" in conv:
                                non_english_chars = find_non_english_text(conv["value"])
                                if non_english_chars:
                                    conv_non_english.append({
                                        "index": i,
                                        "from": conv.get("from", ""),
                                        "non_english_chars": non_english_chars
                                    })
                        if conv_non_english:
                            non_english_info["conversations"] = conv_non_english
                    
                    example["_non_english_info"] = non_english_info
                    removed_examples.append(example)
            else:
                filtered_data.append(item)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        
        print(f"  Original: {original_count}, removed: {removed_count}, kept: {len(filtered_data)}")
        
        return {
            "file": str(file_path),
            "original": original_count,
            "removed": removed_count,
            "kept": len(filtered_data),
            "examples": removed_examples
        }
    except Exception as e:
        print(f"  Error processing {file_path}: {e}")
        return {
            "file": str(file_path),
            "original": 0,
            "removed": 0,
            "kept": 0,
            "examples": [],
            "error": str(e)
        }

def main():
    base_dir = Path("/home/yijuan_liang/10.12Tool_Set/train_set/data/data_nonull")
    
    json_files = []
    exclude_files = ["filter_statistics.json", "direct_answer_statistics.json", 
                     "chinese_content_statistics.json", "chinese_removal_statistics.json",
                     "non_english_removal_statistics.json"]
    
    for file in base_dir.glob("*.json"):
        if file.name not in exclude_files:
            json_files.append(file)
    
    toucan_dir = base_dir / "TOUCAN_converted"
    if toucan_dir.exists():
        for file in toucan_dir.rglob("*.json"):
            json_files.append(file)
    
    print(f"Found {len(json_files)} JSON file(s)\n")
    print("="*80)
    print("WARNING: This overwrites files in place, removing conversations with CJK/non-Latin text!")
    print("="*80)
    
    results = []
    total_original = 0
    total_removed = 0
    total_kept = 0
    all_examples = []
    
    for file_path in sorted(json_files):
        result = filter_file(file_path, max_examples=3)
        results.append(result)
        total_original += result["original"]
        total_removed += result["removed"]
        total_kept += result["kept"]
        
        if result.get("examples"):
            for example in result["examples"]:
                all_examples.append({
                    "source_file": os.path.basename(file_path),
                    "source_path": str(file_path),
                    "example": example
                })
    
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    print(f"{'File':<60} {'Original':<12} {'Removed':<12} {'Kept':<12}")
    print("-"*80)
    
    for result in results:
        file_name = os.path.basename(result["file"])
        if "TOUCAN_converted" in result["file"]:
            rel_path = os.path.relpath(result["file"], base_dir)
            file_name = rel_path
        
        original = result["original"]
        removed = result["removed"]
        kept = result["kept"]
        
        print(f"{file_name:<60} {original:<12} {removed:<12} {kept:<12}")
    
    print("-"*80)
    print(f"{'Total':<60} {total_original:<12} {total_removed:<12} {total_kept:<12}")
    print("="*80)
    
    stats_file = base_dir / "non_english_removal_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nStats saved to: {stats_file}")
    
    if all_examples:
        examples_file = base_dir.parent / "data_analysis" / "non_english_removed_examples.json"
        examples_file.parent.mkdir(parents=True, exist_ok=True)
        with open(examples_file, 'w', encoding='utf-8') as f:
            json.dump(all_examples, f, ensure_ascii=False, indent=2)
        print(f"Removed examples saved to: {examples_file}")
        print(f"Collected {len(all_examples)} removed conversation sample(s)")
    
    print(f"\nDone. Conversations with CJK/non-Latin text have been removed.")

if __name__ == "__main__":
    main()
