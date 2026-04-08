#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import shutil
from pathlib import Path
from typing import List, Dict, Any


def has_empty_function_call(conversation: Dict[str, Any]) -> bool:
    """
    Return True if any function_call in conversations has an empty or invalid value.

    Args:
        conversation: Object with a conversations field.

    Returns:
        True if any function_call is empty/invalid.
    """
    if 'conversations' not in conversation:
        return False
    
    conversations = conversation.get('conversations', [])
    if not isinstance(conversations, list):
        return False
    
    for msg in conversations:
        if not isinstance(msg, dict):
            continue
        
        if msg.get('from') == 'function_call':
            value = msg.get('value', '')
            if not value or not value.strip():
                return True
            
            try:
                fc_data = json.loads(value)
                if not isinstance(fc_data, dict) or not fc_data.get('name'):
                    return True
            except (json.JSONDecodeError, TypeError):
                return True
    
    return False


def process_file(input_file: Path, output_file: Path = None) -> Dict[str, int]:
    """
    Drop conversations that have an empty function_call.

    Args:
        input_file: Input path.
        output_file: Output path; None means overwrite input.

    Returns:
        Stats dict with counts.
    """
    print(f"Processing: {input_file}")
    
    if output_file is None:
        output_file = input_file
    
    try:
        bak_file = input_file.with_suffix(input_file.suffix + '.bak')
        print(f"  Backup: {bak_file}")
        shutil.copy2(input_file, bak_file)
        print(f"  Backup OK")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("  Error: root is not a list; skip")
            return {
                "original_count": 0,
                "removed_count": 0,
                "kept_count": 0
            }
        
        original_count = len(data)
        filtered_data: List[Dict[str, Any]] = []
        removed_count = 0
        
        for item in data:
            if has_empty_function_call(item):
                removed_count += 1
            else:
                filtered_data.append(item)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        
        stats = {
            "original_count": original_count,
            "removed_count": removed_count,
            "kept_count": len(filtered_data)
        }
        
        return stats
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "original_count": 0,
            "removed_count": 0,
            "kept_count": 0
        }


def main():
    """CLI entry point"""
    input_file = Path("/home/yijuan_liang/10.12Tool_Set/test_set/data/data_nonull/test_converted_bfcl.json")
    
    print("=" * 80)
    print("Remove conversations with empty function_call")
    print("=" * 80)
    
    stats = process_file(input_file)
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Original conversations: {stats['original_count']}")
    print(f"Removed: {stats['removed_count']}")
    print(f"Kept: {stats['kept_count']}")
    
    if stats['original_count'] > 0:
        removal_rate = stats['removed_count'] / stats['original_count'] * 100
        keep_rate = stats['kept_count'] / stats['original_count'] * 100
        print(f"Removal rate: {removal_rate:.2f}%")
        print(f"Keep rate: {keep_rate:.2f}%")
    
    print(f"\nOutput: {input_file}")
    print(f"Backup: {input_file.with_suffix(input_file.suffix + '.bak')}")


if __name__ == "__main__":
    main()
