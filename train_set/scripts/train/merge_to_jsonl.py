#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import glob
from pathlib import Path
from typing import Dict, List


def find_json_files(base_dir: str) -> List[str]:
    """Find all JSON files to process (excluding .bak files)."""
    # Only top-level JSON files in this directory (no recursion)
    pattern = os.path.join(base_dir, "*.json")
    files = glob.glob(pattern)
    json_files = [f for f in files if not f.endswith(".bak")]
    
    return sorted(json_files)


def add_answer_tags(conversation: Dict) -> Dict:
    """
    Wrap each gpt message value with <answer>...</answer>.
    Handles empty or None values.
    """
    new_conv = {}
    
    for key, value in conversation.items():
        if key == "conversations":
            conversations = value if isinstance(value, list) else []
            new_conversations = []
            
            for conv in conversations:
                new_conv_item = conv.copy()
                
                if new_conv_item.get("from") == "gpt":
                    value_content = new_conv_item.get("value")
                    
                    if value_content is None:
                        value_content = ""
                    
                    if not isinstance(value_content, str):
                        value_content = str(value_content) if value_content is not None else ""
                    
                    if value_content.strip():
                        new_conv_item["value"] = f"<answer>{value_content}</answer>"
                    else:
                        new_conv_item["value"] = "<answer></answer>"
                
                new_conversations.append(new_conv_item)
            
            new_conv["conversations"] = new_conversations
        else:
            new_conv[key] = value
    
    return new_conv


def remove_properties(conversation: Dict) -> Dict:
    """Drop the properties field."""
    new_conv = {}
    for key, value in conversation.items():
        if key != "properties":
            new_conv[key] = value
    return new_conv


def process_file(file_path: str) -> List[Dict]:
    """
    Process a single file; return the list of processed conversations.
    """
    print(f"Processing file: {os.path.basename(file_path)}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Error: could not read file: {e}")
        return []
    
    if not isinstance(data, list):
        print(f"  Warning: invalid file format, skipping")
        return []
    
    processed_conversations = []
    
    for idx, conversation in enumerate(data):
        conversations = conversation.get("conversations", [])
        if not conversations:
            continue
        
        processed_conv = remove_properties(conversation)
        
        processed_conv = add_answer_tags(processed_conv)
        
        processed_conversations.append(processed_conv)
        
        if (idx + 1) % 1000 == 0:
            print(f"    Processed {idx + 1}/{len(data)} conversation(s)...")
    
    print(f"  Done: {len(processed_conversations)} conversation(s)")
    return processed_conversations


def main():
    base_dir = "/home/yijuan_liang/10.12Tool_Set/train_set/data/data_toollist"
    output_dir = "/home/yijuan_liang/LLaMA-Factory/data/dataset/12_27"
    output_file = os.path.join(output_dir, "toollist_dataset.jsonl")
    
    print("=" * 80)
    print("Merge JSON files into JSONL")
    print("=" * 80)
    print(f"Input directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Output file: {output_file}")
    
    json_files = find_json_files(base_dir)
    
    if not json_files:
        print("\nNo JSON files found to process")
        return
    
    print(f"\nFound {len(json_files)} file(s) to process:")
    for f in json_files:
        print(f"  - {os.path.basename(f)}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_conversations = []
    total_files = len(json_files)
    
    for idx, file_path in enumerate(json_files, 1):
        print(f"\n[{idx}/{total_files}] ", end="")
        conversations = process_file(file_path)
        all_conversations.extend(conversations)
    
    print("\n" + "=" * 80)
    print(f"Writing {len(all_conversations)} conversation(s) to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for conv in all_conversations:
            json_line = json.dumps(conv, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"Saved toollist_dataset.jsonl")
    
    print("\n" + "=" * 80)
    print("Done.")
    print("=" * 80)
    print(f"Files processed: {len(json_files)}")
    print(f"Total conversations: {len(all_conversations)}")
    print(f"Output file: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
