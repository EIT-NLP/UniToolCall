#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import glob
import random
from pathlib import Path
from typing import Dict, List, Tuple


def find_json_files(base_dir: str) -> List[str]:
    """Find all JSON files to process (excluding .bak files)."""
    pattern = os.path.join(base_dir, "*.json")
    files = glob.glob(pattern)
    json_files = [f for f in files if not f.endswith(".bak")]
    return sorted(json_files)


def count_total_samples(json_files: List[str]) -> int:
    """Count total samples across all JSON files."""
    total = 0
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    total += len(data)
        except Exception as e:
            print(f"  Warning: could not read file {os.path.basename(file_path)}: {e}")
    return total


def calculate_sample_counts(json_files: List[str], total_samples: int, target_count: int) -> List[Tuple[str, int]]:
    """
    Compute per-file sample counts (proportional stratified sampling).
    Returns: [(file_path, sample_count), ...]
    """
    file_counts = []
    total_available = 0
    
    # First count rows per file
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                count = len(data) if isinstance(data, list) else 0
                file_counts.append((file_path, count))
                total_available += count
        except Exception as e:
            print(f"  Warning: could not read file {os.path.basename(file_path)}: {e}")
            file_counts.append((file_path, 0))
    
    # Allocate counts proportionally
    sample_counts = []
    remaining = target_count
    
    for i, (file_path, count) in enumerate(file_counts):
        if total_available == 0:
            sample_counts.append((file_path, 0))
            continue
        
        if i == len(file_counts) - 1:
            # Last file gets all remaining quota
            sample_counts.append((file_path, remaining))
        else:
            proportion = count / total_available
            sample_count = max(0, min(count, int(target_count * proportion)))
            sample_counts.append((file_path, sample_count))
            remaining -= sample_count
    
    return sample_counts


def remove_properties(conversation: Dict) -> Dict:
    """Drop the properties field."""
    new_conv = {}
    for key, value in conversation.items():
        if key != "properties":
            new_conv[key] = value
    return new_conv


def process_conversation(conversation: Dict) -> Dict:
    """
    Process a single conversation:
    1. Remove properties
    2. Set observation value to empty string ""
    3. Set gpt value to "<answer></answer>"
    """
    processed_conv = remove_properties(conversation)
    
    if "conversations" in processed_conv:
        conversations = processed_conv.get("conversations", [])
        new_conversations = []
        
        for conv in conversations:
            new_conv_item = conv.copy()
            
            if new_conv_item.get("from") == "observation":
                new_conv_item["value"] = ""
            
            elif new_conv_item.get("from") == "gpt":
                new_conv_item["value"] = "<answer></answer>"
            
            new_conversations.append(new_conv_item)
        
        processed_conv["conversations"] = new_conversations
    
    return processed_conv


def sample_from_file(file_path: str, sample_count: int, seed: int = 42) -> List[Dict]:
    """
    Randomly sample a given number of items from a single file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Error: could not read file: {e}")
        return []
    
    if not isinstance(data, list):
        print(f"  Warning: invalid file format, skipping")
        return []
    
    if sample_count >= len(data):
        samples = data
    else:
        random.seed(seed)
        samples = random.sample(data, sample_count)
    
    processed_samples = []
    for conv in samples:
        processed_conv = process_conversation(conv)
        processed_samples.append(processed_conv)
    
    return processed_samples


def process_dataset(base_dir: str, output_file: str, target_count: int, dataset_name: str):
    """
    Run proportional sampling and processing on the whole dataset.
    """
    print("=" * 80)
    print(f"Processing dataset: {dataset_name}")
    print("=" * 80)
    print(f"Input directory: {base_dir}")
    print(f"Output file: {output_file}")
    print(f"Target sample count: {target_count}")
    
    json_files = find_json_files(base_dir)
    
    if not json_files:
        print("\nNo JSON files found to process")
        return
    
    print(f"\nFound {len(json_files)} file(s):")
    for f in json_files:
        print(f"  - {os.path.basename(f)}")
    
    print("\nCounting total samples...")
    total_samples = count_total_samples(json_files)
    print(f"Total samples: {total_samples}")
    
    print("\nComputing per-file sample counts (proportional)...")
    sample_counts = calculate_sample_counts(json_files, total_samples, target_count)
    
    print("\nSampling plan per file:")
    for file_path, count in sample_counts:
        print(f"  {os.path.basename(file_path)}: {count} rows")
    
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nSampling and processing...")
    all_samples = []
    total_files = len(json_files)
    
    for idx, (file_path, sample_count) in enumerate(sample_counts, 1):
        if sample_count == 0:
            continue
        
        print(f"\n[{idx}/{total_files}] {os.path.basename(file_path)} (sample {sample_count} rows)...")
        samples = sample_from_file(file_path, sample_count, seed=42 + idx)
        all_samples.extend(samples)
        print(f"  Done: sampled {len(samples)} rows")
    
    print("\n" + "=" * 80)
    print(f"Writing {len(all_samples)} samples to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for conv in all_samples:
            json_line = json.dumps(conv, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"Saved to {output_file}")
    
    print("\n" + "=" * 80)
    print("Done.")
    print("=" * 80)
    print(f"Files processed: {len(json_files)}")
    print(f"Samples written: {len(all_samples)}")
    print(f"Target count: {target_count}")
    print(f"Output file: {output_file}")
    print("=" * 80)


def main():
    base_dir = "/home/yijuan_liang/10.12Tool_Set/train_set/data/data_toollist"
    output_dir = "/home/yijuan_liang/LLaMA-Factory/data/dataset/01_26"
    
    # Match reference line count (e.g. yifan_toollist1_processed.jsonl)
    target_count = 979
    
    output_file = os.path.join(output_dir, "toollist1_979.jsonl")
    
    process_dataset(base_dir, output_file, target_count, "toollist1")


if __name__ == "__main__":
    main()
