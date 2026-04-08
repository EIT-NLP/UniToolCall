#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import sys

# UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

def remove_null_fields(obj):
    """
    Recursively drop None/null fields from dicts and lists.
    """
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if value is not None:
                cleaned_value = remove_null_fields(value)
                result[key] = cleaned_value
        return result
    elif isinstance(obj, list):
        return [remove_null_fields(item) for item in obj]
    else:
        return obj

def main():
    """CLI entry point."""
    input_file = Path(r"D:\Desktop\10.12Tool_Set\tool_set\apis\mcpuniverse_servers_tools.json")
    
    print(f"Reading: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded OK")
    print(f"Server count: {len(data)}")
    
    null_count = 0
    def count_nulls(obj):
        nonlocal null_count
        if isinstance(obj, dict):
            for key, value in obj.items():
                if value is None:
                    null_count += 1
                else:
                    count_nulls(value)
        elif isinstance(obj, list):
            for item in obj:
                count_nulls(item)
    
    count_nulls(data)
    print(f"Found {null_count} null fields")
    
    print("\nRemoving null fields...")
    cleaned_data = remove_null_fields(data)
    
    backup_file = input_file.with_suffix('.json.backup')
    if backup_file.exists():
        backup_file.unlink()
    input_file.rename(backup_file)
    print(f"Backup saved: {backup_file}")
    
    print("Writing cleaned file...")
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nDone.")
    print(f"Removed {null_count} null fields")
    print(f"Output: {input_file}")

if __name__ == "__main__":
    main()
