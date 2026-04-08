#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Dict, List, Any, Optional


def convert_parameters_to_inputschema(parameters: Dict[str, Any], required: Optional[List[str]] = None) -> Dict[str, Any]:
    """Convert `parameters` dict to inputSchema shape."""
    # Full schema already?
    if "type" in parameters and "properties" in parameters:
        return parameters.copy()
    
    properties = {}
    for param_name, param_def in parameters.items():
        if isinstance(param_def, dict):
            properties[param_name] = param_def.copy()
        else:
            properties[param_name] = {"type": "str", "description": str(param_def)}
    
    return {
        "type": "object",
        "properties": properties,
        "required": required if required else []
    }


def fix_inputschema_format(input_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Fix malformed inputSchema (missing type: object / properties wrapper)."""
    if isinstance(input_schema, dict) and input_schema.get("type") == "object" and "properties" in input_schema:
        return input_schema
    
    if isinstance(input_schema, dict):
        properties = {}
        required = []
        
        for key, value in input_schema.items():
            if key == "required" and isinstance(value, list):
                required = value
            elif key != "type" or (key == "type" and value != "object"):
                if isinstance(value, dict):
                    properties[key] = value.copy()
                else:
                    properties[key] = {"type": "str", "description": str(value)}
        
        if properties:
            return {
                "type": "object",
                "properties": properties,
                "required": required
            }
    
    return input_schema


def convert_tool_to_standard_format(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one tool: parameters -> inputSchema and fix bad inputSchema."""
    converted_tool = tool.copy()
    
    if "inputSchema" not in tool and "parameters" in tool:
        parameters = tool["parameters"]
        
        if parameters is None:
            converted_tool["inputSchema"] = {
                "type": "object",
                "properties": {},
                "required": []
            }
        elif isinstance(parameters, dict) and "type" in parameters and "properties" in parameters:
            converted_tool["inputSchema"] = parameters.copy()
        else:
            required = tool.get("required")
            if required is None:
                required = []
            converted_tool["inputSchema"] = convert_parameters_to_inputschema(parameters, required)
        
        del converted_tool["parameters"]
    
    if "inputSchema" in converted_tool:
        converted_tool["inputSchema"] = fix_inputschema_format(converted_tool["inputSchema"])
    
    return converted_tool


def process_json_file(file_path: Path) -> bool:
    """Process one JSON file."""
    print(f"Processing: {file_path.name}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"  Warning: {file_path.name} is not a list; skip")
            return False
        
        modified = False
        converted_count = 0
        
        for item in data:
            if "tools" not in item:
                continue
            
            tools_str = item["tools"]
            if not isinstance(tools_str, str):
                continue
            
            try:
                tools = json.loads(tools_str)
                if not isinstance(tools, list):
                    continue
                
                converted_tools = []
                for tool in tools:
                    converted_tool = convert_tool_to_standard_format(tool)
                    converted_tools.append(converted_tool)
                    converted_count += 1
                
                item["tools"] = json.dumps(converted_tools, ensure_ascii=False)
                modified = True
                
            except json.JSONDecodeError as e:
                print(f"  Warning: invalid tools JSON: {e}")
                continue
        
        if modified:
            backup_path = file_path.with_suffix('.json.bak')
            if not backup_path.exists():
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"  Backup: {backup_path.name}")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"  Done: converted {converted_count} tools")
            return True
        else:
            print(f"  Skip: nothing to convert")
            return False
            
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """CLI entry point"""
    data_origin_dir = Path("/home/yijuan_liang/10.12Tool_Set/test_set/data/data_origin")
    
    print("=" * 60)
    print("Test set tool format converter")
    print("  1. parameters -> inputSchema")
    print("  2. Fix malformed inputSchema (missing type: object / properties wrapper)")
    print("=" * 60)
    print(f"Directory: {data_origin_dir}")
    print("=" * 60)
    print()
    
    if not data_origin_dir.exists():
        print(f"Error: directory not found: {data_origin_dir}")
        return
    
    json_files = sorted(data_origin_dir.glob("*.json"))
    
    if not json_files:
        print(f"Warning: no JSON files under {data_origin_dir}")
        return
    
    print(f"Found {len(json_files)} JSON file(s)")
    print()
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for file_path in json_files:
        if file_path.suffix == '.bak' or file_path.name.endswith('.bak'):
            continue
        
        if process_json_file(file_path):
            success_count += 1
        else:
            skip_count += 1
    
    print()
    print("=" * 60)
    print("Done")
    print("=" * 60)
    print(f"Succeeded: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Errors: {error_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
