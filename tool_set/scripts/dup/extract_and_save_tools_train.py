#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import shutil
from pathlib import Path
from typing import Dict, Tuple


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


def extract_tools_from_data_nonull(data_dir: Path) -> Dict[Tuple[str, str], Dict]:
    """
    Load full tool objects from all JSON under data_nonull (tools field).
    Returns: Dict[(normalized_name, normalized_description), tool_object]
    """
    print("\nExtracting tools from data_nonull...")
    
    tools_dict = {}
    json_files = []
    
    exclude_files = ["filter_statistics.json", "direct_answer_statistics.json", 
                     "chinese_content_statistics.json", "chinese_removal_statistics.json",
                     "count_tokens.py"]
    
    for file in data_dir.rglob("*.json"):
        if file.name not in exclude_files and not file.name.endswith('.bak'):
            json_files.append(file)
    
    print(f"Found {len(json_files)} JSON files to process")
    
    total_conversations = 0
    processed_files = 0
    total_tools_extracted = 0
    
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
                                signature = (name, description)
                                if signature not in tools_dict:
                                    tools_dict[signature] = tool
                                    file_tools_count += 1
                                total_tools_extracted += 1
            
            if file_tools_count > 0:
                processed_files += 1
                relative_path = file_path.relative_to(data_dir)
                print(f"  {relative_path}: {file_tools_count} new unique tool(s)")
        
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nProcessed {total_conversations} conversation(s)")
    print(f"From {processed_files} file(s): {len(tools_dict)} unique (name, description) pairs")
    print(f"  Total extraction events: {total_tools_extracted}")
    
    return tools_dict


def save_tools_to_file(tools_dict: Dict[Tuple[str, str], Dict], output_file: Path, backup: bool = True):
    """
    Save tools dict as indexed JSON { "0": tool, ... }.
    """
    if backup and output_file.exists():
        backup_file = output_file.with_suffix('.json.bak')
        shutil.copy2(output_file, backup_file)
        print(f"✓ Backed up: {backup_file}")
    
    indexed_tools = {}
    for idx, (signature, tool) in enumerate(sorted(tools_dict.items()), start=0):
        indexed_tools[str(idx)] = tool
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(indexed_tools, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved: {output_file}")
    print(f"  {len(indexed_tools)} tools")


def main():
    """CLI entry point"""
    output_file1 = Path("/home/yijuan_liang/10.12Tool_Set/tool_set/apis/apis_nonull/train_set_tool_dedup.json")
    output_file2 = Path("/home/yijuan_liang/10.12Tool_Set/tool_set/apis/apis_cosdup/train_set_tool_dedup_cosdup.json")
    data_nonull_dir = Path("/home/yijuan_liang/10.12Tool_Set/train_set/data/data_nonull")
    
    print("=" * 80)
    print("Extract tools from data_nonull and save (train set)")
    print("=" * 80)
    
    if not data_nonull_dir.exists():
        print(f"Error: data_nonull not found: {data_nonull_dir}")
        return
    
    tools_dict = extract_tools_from_data_nonull(data_nonull_dir)
    
    if not tools_dict:
        print("\nWarning: no tools found in data_nonull!")
        return
    
    print("\n" + "=" * 80)
    print("Save tool files")
    print("=" * 80)
    
    output_file1.parent.mkdir(parents=True, exist_ok=True)
    output_file2.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to: {output_file1}")
    save_tools_to_file(tools_dict, output_file1, backup=True)
    
    print(f"\nSaving to: {output_file2}")
    save_tools_to_file(tools_dict, output_file2, backup=True)
    
    stats_file = output_file1.parent / "train_set_tool_extraction_statistics.json"
    statistics = {
        "unique_tools_count": len(tools_dict),
        "data_source": str(data_nonull_dir),
        "output_files": [
            str(output_file1),
            str(output_file2)
        ],
        "sample_tools": [
            {
                "name": tool.get("name", ""),
                "description": tool.get("description", "")[:100] if tool.get("description") else ""
            }
            for _, tool in list(tools_dict.items())[:10]
        ]
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Stats saved to: {stats_file}")
    
    print("\n" + "=" * 80)
    print("Done.")
    print("=" * 80)
    print(f"\nSaved {len(tools_dict)} unique tools")
    print(f"File 1: {output_file1}")
    print(f"File 2: {output_file2}")


if __name__ == "__main__":
    main()


