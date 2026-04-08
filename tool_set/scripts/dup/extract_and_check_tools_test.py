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


def check_field(value) -> Tuple[bool, str]:
    """
    Check that a field is present and non-empty.
    Returns: (is_valid, status_tag)
    """
    if value is None:
        return False, "null"
    if isinstance(value, str):
        if value.strip() == "":
            return False, "empty_string"
        return True, "valid"
    if isinstance(value, list):
        if len(value) == 0:
            return False, "empty_list"
        return True, "valid"
    return True, "valid"


def extract_tools_from_data_nonull(data_dir: Path) -> Tuple[Dict[Tuple[str, str], Dict], Dict]:
    """
    Load full tool objects from JSON under data_nonull (tools field).
    Returns: (tools dict keyed by signature, check stats)
    """
    print("\nExtracting tools from data_nonull...")
    
    tools_dict = {}
    json_files = []
    
    exclude_files = ["filter_statistics.json", "direct_answer_statistics.json", 
                     "chinese_content_statistics.json", "chinese_removal_statistics.json",
                     "count_tokens.py"]
    
    for file in data_dir.glob("*.json"):
        if file.name not in exclude_files and not file.name.endswith('.bak'):
            json_files.append(file)
    
    print(f"Found {len(json_files)} JSON files to process")
    
    total_conversations = 0
    processed_files = 0
    total_tools_extracted = 0
    
    check_stats = {
        "missing_category": [],
        "missing_domain": [],
        "missing_both": [],
        "category_empty": [],
        "domain_empty": [],
        "valid_tools": 0,
        "category_status": {},
        "domain_status": {}
    }
    
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
                print(f"  {file_path.name}: {file_tools_count} new unique tool(s)")
        
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nProcessed {total_conversations} conversation(s)")
    print(f"From {processed_files} file(s): {len(tools_dict)} unique (name, description) pairs")
    print(f"  Total extraction events: {total_tools_extracted}")
    
    print("\nChecking category and domain on each tool...")
    for idx, (signature, tool) in enumerate(tools_dict.items()):
        tool_name = tool.get("name", f"tool_{idx}")
        
        has_category = "category" in tool
        category_valid, category_status = check_field(tool.get("category"))
        
        has_domain = "domain" in tool
        domain_valid, domain_status = check_field(tool.get("domain"))
        
        if not has_category:
            check_stats["category_status"]["missing"] = check_stats["category_status"].get("missing", 0) + 1
        else:
            check_stats["category_status"][category_status] = check_stats["category_status"].get(category_status, 0) + 1
        
        if not has_domain:
            check_stats["domain_status"]["missing"] = check_stats["domain_status"].get("missing", 0) + 1
        else:
            check_stats["domain_status"][domain_status] = check_stats["domain_status"].get(domain_status, 0) + 1
        
        tool_info = {
            "signature": signature,
            "name": tool_name,
            "category": tool.get("category"),
            "domain": tool.get("domain")
        }
        
        if not has_category:
            check_stats["missing_category"].append(tool_info)
        elif not category_valid:
            check_stats["category_empty"].append(tool_info)
        
        if not has_domain:
            check_stats["missing_domain"].append(tool_info)
        elif not domain_valid:
            check_stats["domain_empty"].append(tool_info)
        
        if not has_category and not has_domain:
            check_stats["missing_both"].append(tool_info)
        
        if has_category and category_valid and has_domain and domain_valid:
            check_stats["valid_tools"] += 1
    
    return tools_dict, check_stats


def print_check_statistics(check_stats: Dict, total_tools: int):
    """Print category/domain check summary."""
    print("\n" + "=" * 80)
    print("Category and domain check")
    print("=" * 80)
    
    print(f"\nTotal tools: {total_tools}")
    print(f"Fully valid tools (non-empty category and domain): {check_stats['valid_tools']} ({check_stats['valid_tools']/total_tools*100:.2f}%)")
    
    print("\n--- Category field ---")
    for status, count in sorted(check_stats["category_status"].items()):
        percentage = count / total_tools * 100
        print(f"  {status}: {count} ({percentage:.2f}%)")
    
    print("\n--- Domain field ---")
    for status, count in sorted(check_stats["domain_status"].items()):
        percentage = count / total_tools * 100
        print(f"  {status}: {count} ({percentage:.2f}%)")
    
    print("\n--- Problem counts ---")
    print(f"Missing category: {len(check_stats['missing_category'])}")
    print(f"Missing domain: {len(check_stats['missing_domain'])}")
    print(f"Missing both: {len(check_stats['missing_both'])}")
    print(f"Empty category: {len(check_stats['category_empty'])}")
    print(f"Empty domain: {len(check_stats['domain_empty'])}")
    
    if check_stats['missing_category']:
        print("\n--- Sample: missing category (first 10) ---")
        for i, tool in enumerate(check_stats['missing_category'][:10], 1):
            print(f"  {i}. Name: {tool['name']}")
        if len(check_stats['missing_category']) > 10:
            print(f"  ... and {len(check_stats['missing_category']) - 10} more")
    
    if check_stats['missing_domain']:
        print("\n--- Sample: missing domain (first 10) ---")
        for i, tool in enumerate(check_stats['missing_domain'][:10], 1):
            print(f"  {i}. Name: {tool['name']}")
        if len(check_stats['missing_domain']) > 10:
            print(f"  ... and {len(check_stats['missing_domain']) - 10} more")
    
    if check_stats['category_empty']:
        print("\n--- Sample: empty category (first 10) ---")
        for i, tool in enumerate(check_stats['category_empty'][:10], 1):
            print(f"  {i}. Name: {tool['name']}, Category: {tool['category']}")
        if len(check_stats['category_empty']) > 10:
            print(f"  ... and {len(check_stats['category_empty']) - 10} more")
    
    if check_stats['domain_empty']:
        print("\n--- Sample: empty domain (first 10) ---")
        for i, tool in enumerate(check_stats['domain_empty'][:10], 1):
            print(f"  {i}. Name: {tool['name']}, Domain: {tool['domain']}")
        if len(check_stats['domain_empty']) > 10:
            print(f"  ... and {len(check_stats['domain_empty']) - 10} more")


def save_tools_to_file(tools_dict: Dict[Tuple[str, str], Dict], output_file: Path, backup: bool = True):
    """Save as indexed JSON { \"0\": tool, ... }."""
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
    output_file1 = Path("/home/yijuan_liang/10.12Tool_Set/tool_set/apis/apis_nonull/test_set_tool_dedup.json")
    output_file2 = Path("/home/yijuan_liang/10.12Tool_Set/tool_set/apis/apis_cosdup/test_set_tool_dedup_cosdup.json")
    data_nonull_dir = Path("/home/yijuan_liang/10.12Tool_Set/test_set/data/data_nonull")
    
    print("=" * 80)
    print("Extract tools from data_nonull and save (test set)")
    print("=" * 80)
    
    if not data_nonull_dir.exists():
        print(f"Error: data_nonull not found: {data_nonull_dir}")
        return
    
    tools_dict, check_stats = extract_tools_from_data_nonull(data_nonull_dir)
    
    if not tools_dict:
        print("\nWarning: no tools found in data_nonull!")
        return
    
    print_check_statistics(check_stats, len(tools_dict))
    
    print("\n" + "=" * 80)
    print("Save tool files")
    print("=" * 80)
    
    output_file1.parent.mkdir(parents=True, exist_ok=True)
    output_file2.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to: {output_file1}")
    save_tools_to_file(tools_dict, output_file1, backup=True)
    
    print(f"\nSaving to: {output_file2}")
    save_tools_to_file(tools_dict, output_file2, backup=True)
    
    stats_file = output_file1.parent / "test_set_tool_extraction_statistics.json"
    statistics = {
        "unique_tools_count": len(tools_dict),
        "data_source": str(data_nonull_dir),
        "output_files": [
            str(output_file1),
            str(output_file2)
        ],
        "category_domain_check": {
            "valid_tools": check_stats["valid_tools"],
            "valid_percentage": check_stats["valid_tools"] / len(tools_dict) * 100,
            "category_status": check_stats["category_status"],
            "domain_status": check_stats["domain_status"],
            "problem_counts": {
                "missing_category": len(check_stats["missing_category"]),
                "missing_domain": len(check_stats["missing_domain"]),
                "missing_both": len(check_stats["missing_both"]),
                "category_empty": len(check_stats["category_empty"]),
                "domain_empty": len(check_stats["domain_empty"])
            }
        },
        "problem_tools": {
            "missing_category": [{"name": t["name"], "category": t.get("category"), "domain": t.get("domain")} 
                                for t in check_stats["missing_category"][:50]],
            "missing_domain": [{"name": t["name"], "category": t.get("category"), "domain": t.get("domain")} 
                              for t in check_stats["missing_domain"][:50]],
            "missing_both": [{"name": t["name"], "category": t.get("category"), "domain": t.get("domain")} 
                            for t in check_stats["missing_both"][:50]]
        }
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
    
    if check_stats["valid_tools"] == len(tools_dict):
        print(f"\n✓ All tools have non-empty category and domain.")
    else:
        print(f"\n⚠ {len(tools_dict) - check_stats['valid_tools']} tools missing or empty category/domain")


if __name__ == "__main__":
    main()


