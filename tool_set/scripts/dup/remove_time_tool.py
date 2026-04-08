#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Dict, Any, List, Set
import re


# Time-related keywords (case-insensitive)
# Focus on date and time, including xxx_date / xxx_time style names
TIME_KEYWORDS: Set[str] = {
    'date', 'dates',
    'time', 'times',
    'datetime', 'timestamp',
    'start_time', 'end_time',
    'start_date', 'end_date',
    'pick_up_time', 'drop_off_time',
    'pickup_time', 'dropoff_time',
    'pick_up_date', 'drop_off_date',
    'pickup_date', 'dropoff_date',
    'duration',
    'when', 'schedule', 'scheduled',
    'day', 'days',
    'hour', 'hours',
    'minute', 'minutes',
    'second', 'seconds',
    'period', 'periods',
    'year', 'years',
    'month', 'months',
    'week', 'weeks'
}


def contains_time_keywords(text: str) -> bool:
    """
    Return True if text contains time-related keywords.

    Args:
        text: Text to scan

    Returns:
        True if a keyword matches, else False
    """
    if not isinstance(text, str):
        return False
    
    text_lower = text.lower()
    
    for keyword in TIME_KEYWORDS:
        keyword_lower = keyword.lower()
        keyword_escaped = re.escape(keyword_lower)
        
        # Patterns (priority order):
        # 1. Word boundary: \\bdate\\b
        # 2. Underscore: _date, date_, _date_
        # 3. Hyphen: -date, date-, -date-
        # 4. Start/end of string: ^date, date$
        # 5. CamelCase: dateX or Xdate (X uppercase)
        
        pattern_lower = (
            r'\b' + keyword_escaped + r'\b' +
            r'|' + r'[_\-\s]' + keyword_escaped + r'[_\-\s]' +
            r'|' + r'^' + keyword_escaped + r'[_\-\s]' +
            r'|' + r'[_\-\s]' + keyword_escaped + r'$' +
            r'|' + r'^' + keyword_escaped + r'$'
        )
        
        if re.search(pattern_lower, text_lower):
            return True
        
        keyword_capitalized = keyword_lower.capitalize()
        pattern_camel = (
            keyword_lower + r'[A-Z]' +
            r'|' + r'[a-z]' + keyword_capitalized +
            r'|' + keyword_capitalized + r'[A-Z]' +
            r'|' + r'[A-Z]' + keyword_capitalized
        )
        if re.search(pattern_camel, text):
            return True
    
    return False


def has_time_in_tool(tool: Dict[str, Any]) -> bool:
    """
    True if the tool mentions time-related fields.

    Checks:
    1. Tool description
    2. Parameter names in inputSchema.properties
    3. Each property's description
    Emphasis on xxx_date, xxx_time style names.

    Args:
        tool: Tool object

    Returns:
        True if any check matches
    """
    tool_description = tool.get('description', '')
    if isinstance(tool_description, str) and contains_time_keywords(tool_description):
        return True
    
    input_schema = tool.get('inputSchema', {})
    if isinstance(input_schema, dict):
        properties = input_schema.get('properties', {})
        if isinstance(properties, dict):
            for param_name, param_value in properties.items():
                if isinstance(param_name, str) and contains_time_keywords(param_name):
                    return True
                
                if isinstance(param_value, dict):
                    param_desc = param_value.get('description', '')
                    if isinstance(param_desc, str) and contains_time_keywords(param_desc):
                        return True
        
        required = input_schema.get('required', [])
        if isinstance(required, list):
            for req_item in required:
                if isinstance(req_item, str) and contains_time_keywords(req_item):
                    return True
    
    return False


def process_tool_file(input_file: Path, output_file: Path) -> Dict[str, Any]:
    """
    Filter one tools JSON: drop tools that match time keywords.

    Args:
        input_file: Input path
        output_file: Output path

    Returns:
        Stats dict (original/removed/kept counts)
    """
    print(f"Processing file: {input_file.name}")
    
    stats = {
        'file_name': input_file.name,
        'tool_sets': {},
        'total_original_tools': 0,
        'total_removed_tools': 0,
        'total_filtered_tools': 0,
        'removed_tool_sets': []
    }
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            print("  Warning: root is not an object; skipping")
            return stats
        
        filtered_data = {}
        
        first_value = next(iter(data.values())) if data else None
        is_array_format = isinstance(first_value, list)
        
        if is_array_format:
            for tool_set_name, tools in data.items():
                if not isinstance(tools, list):
                    filtered_data[tool_set_name] = tools
                    continue
                
                original_count = len(tools)
                filtered_tools = []
                removed_count = 0
                
                for tool in tools:
                    if not isinstance(tool, dict):
                        filtered_tools.append(tool)
                        continue
                    
                    if has_time_in_tool(tool):
                        removed_count += 1
                    else:
                        filtered_tools.append(tool)
                
                if len(filtered_tools) > 0:
                    filtered_data[tool_set_name] = filtered_tools
                
                stats['tool_sets'][tool_set_name] = {
                    'original': original_count,
                    'removed': removed_count,
                    'filtered': len(filtered_tools)
                }
                
                stats['total_original_tools'] += original_count
                stats['total_removed_tools'] += removed_count
                stats['total_filtered_tools'] += len(filtered_tools)
                
                if removed_count > 0:
                    print(f"  Tool set '{tool_set_name}': original {original_count}, removed {removed_count}, kept {len(filtered_tools)}")
        else:
            removed_count = 0
            for tool_key, tool in data.items():
                if not isinstance(tool, dict):
                    filtered_data[tool_key] = tool
                    stats['total_original_tools'] += 1
                    stats['total_filtered_tools'] += 1
                    continue
                
                stats['total_original_tools'] += 1
                
                if has_time_in_tool(tool):
                    removed_count += 1
                else:
                    filtered_data[tool_key] = tool
                    stats['total_filtered_tools'] += 1
            
            stats['total_removed_tools'] = removed_count
            stats['tool_sets']['all_tools'] = {
                'original': stats['total_original_tools'],
                'removed': removed_count,
                'filtered': stats['total_filtered_tools']
            }
            
            if removed_count > 0:
                print(f"  Removed {removed_count} tool(s), kept {stats['total_filtered_tools']}")
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        
        print(f"  Total: original {stats['total_original_tools']} tools, removed {stats['total_removed_tools']}, kept {stats['total_filtered_tools']}")
        if stats['total_original_tools'] > 0:
            print(f"  Retention rate: {stats['total_filtered_tools'] / stats['total_original_tools'] * 100:.2f}%")
        
        return stats
        
    except Exception as e:
        print(f"  Error: failed to process file - {e}")
        import traceback
        traceback.print_exc()
        return stats


def process_directory(input_dir: Path, output_dir: Path) -> List[Dict[str, Any]]:
    """
    Process all JSON files in a directory.

    Args:
        input_dir: Input directory
        output_dir: Output directory

    Returns:
        List of per-file stats
    """
    if not input_dir.exists():
        print(f"Error: input directory not found: {input_dir}")
        return []
    
    json_files = sorted(input_dir.glob('*.json'))
    
    if not json_files:
        print(f"Warning: no JSON files in {input_dir}")
        return []
    
    print(f"\nProcessing directory: {input_dir}")
    print(f"Found {len(json_files)} JSON file(s)")
    print("=" * 80)
    
    all_stats = []
    
    for json_file in json_files:
        output_file = output_dir / json_file.name
        stats = process_tool_file(json_file, output_file)
        all_stats.append(stats)
        print()
    
    return all_stats


def print_summary(all_stats: List[Dict[str, Any]]):
    """Print aggregate stats."""
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    
    total_original = 0
    total_removed = 0
    total_filtered = 0
    
    for stats in all_stats:
        file_name = stats['file_name']
        print(f"\nFile: {file_name}")
        print(f"  Original count: {stats['total_original_tools']}")
        print(f"  Removed: {stats['total_removed_tools']}")
        print(f"  Kept: {stats['total_filtered_tools']}")
        if stats['total_original_tools'] > 0:
            print(f"  Retention rate: {stats['total_filtered_tools'] / stats['total_original_tools'] * 100:.2f}%")
        
        if stats['removed_tool_sets']:
            print(f"  Removed tool sets: {', '.join(stats['removed_tool_sets'])}")
        
        removed_tool_sets_detail = []
        for tool_set_name, tool_set_stats in stats['tool_sets'].items():
            if tool_set_stats['removed'] > 0:
                removed_tool_sets_detail.append(
                    f"    - {tool_set_name}: removed {tool_set_stats['removed']}/{tool_set_stats['original']} tools"
                )
        
        if removed_tool_sets_detail:
            print("  Tool sets with removals:")
            for detail in removed_tool_sets_detail:
                print(detail)
        
        total_original += stats['total_original_tools']
        total_removed += stats['total_removed_tools']
        total_filtered += stats['total_filtered_tools']
    
    print("\n" + "=" * 80)
    print("Totals")
    print("=" * 80)
    print(f"Files processed: {len(all_stats)}")
    print(f"Total original: {total_original}")
    print(f"Total removed: {total_removed}")
    print(f"Total kept: {total_filtered}")
    if total_original > 0:
        print(f"Overall retention: {total_filtered / total_original * 100:.2f}%")


def main():
    """CLI entry point"""
    input_dir = Path("/home/yijuan_liang/10.12Tool_Set/tool_set/apis/apis_dup")
    output_dir = Path("/home/yijuan_liang/10.12Tool_Set/tool_set/apis/apis_notime")
    
    print("=" * 80)
    print("Remove tools that mention time-related keywords")
    print("=" * 80)
    
    all_stats = process_directory(input_dir, output_dir)
    
    print_summary(all_stats)
    
    print(f"\nOutput written to: {output_dir}")


if __name__ == "__main__":
    main()
