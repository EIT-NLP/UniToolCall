#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import re
from typing import Dict, List, Any

def normalize_server_name(name: str) -> str:
    """
    Normalize server name to a key: lowercase, spaces to hyphens.
    Example: "Time" -> "time", "EdgeOne Pages MCP" -> "edgeone-pages-mcp"
    """
    normalized = name.lower()
    normalized = re.sub(r'[^\w\s-]', '', normalized)
    normalized = re.sub(r'[-\s]+', '-', normalized)
    normalized = normalized.strip('-')
    return normalized

def convert_param_type(param_type: str) -> str:
    """Map parameter type to JSON Schema type string."""
    type_mapping = {
        'string': 'string',
        'number': 'number',
        'integer': 'integer',
        'boolean': 'boolean',
        'array': 'array',
        'object': 'object'
    }
    return type_mapping.get(param_type.lower(), 'string')

def convert_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one tool from source format to corpus-style."""
    converted_tool = {
        'name': tool.get('name', ''),
        'description': tool.get('description', '')
    }
    
    input_schema = {
        'type': 'object',
        'properties': {},
        'required': []
    }
    
    parameters = tool.get('parameters', [])
    for param in parameters:
        param_name = param.get('name', '')
        if not param_name:
            continue
            
        param_type = convert_param_type(param.get('param_type', 'string'))
        param_description = param.get('description', '')
        
        prop = {
            'type': param_type,
            'description': param_description
        }
        
        if 'default' in param:
            prop['default'] = param['default']
        
        if 'enum' in param:
            prop['enum'] = param['enum']
        
        min_value = param.get('min_value')
        max_value = param.get('max_value')
        if min_value is not None:
            prop['minimum'] = min_value
        if max_value is not None:
            prop['maximum'] = max_value
        
        input_schema['properties'][param_name] = prop
        
        if param.get('required', False):
            input_schema['required'].append(param_name)
    
    converted_tool['inputSchema'] = input_schema
    
    return converted_tool

def convert_servers(source_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Convert server list to {server_key: [tools...]}."""
    result = {}
    
    for server in source_data:
        server_name = server.get('name', '')
        if not server_name:
            continue
        
        server_key = normalize_server_name(server_name)
        
        tools = server.get('tools', [])
        converted_tools = []
        
        for tool in tools:
            converted_tool = convert_tool(tool)
            converted_tools.append(converted_tool)
        
        if converted_tools:
            result[server_key] = converted_tools
    
    return result

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    source_file = r'D:\Desktop\9.24Tool_Set\mcp_servers_detailed.json'
    output_dir = os.path.join(project_root, 'tool_set', 'apis')
    output_file = os.path.join(output_dir, 'qiyang_server_tool.json')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading source: {source_file}")
    with open(source_file, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    print(f"Source has {len(source_data)} server(s)")
    
    print("Converting...")
    converted_data = convert_servers(source_data)
    
    print(f"Converted {len(converted_data)} server key(s)")
    
    total_tools = sum(len(tools) for tools in converted_data.values())
    print(f"Total tools: {total_tools}")
    
    print(f"Writing: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print("Conversion done.")
    
    print("\nServers:")
    for server_key, tools in converted_data.items():
        print(f"  {server_key}: {len(tools)} tool(s)")

if __name__ == '__main__':
    main()
