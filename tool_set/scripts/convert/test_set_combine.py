#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
import os

sys.stdout.reconfigure(encoding='utf-8')

class TestSetCombiner:
    def __init__(self, 
                 data_dirs: List[str] = None,
                 output_file: str = "test_set_tool.json",
                 output_dir: str = None):
        """
        Initialize the combiner.

        Args:
            data_dirs: List of data folder paths; if None, use default
            output_file: Output filename
            output_dir: Output directory; if None, use default path
        """
        if output_dir is None:
            output_dir = Path("/home/yijuan_liang/10.12Tool_Set/tool_set/apis/apis_origin")
        else:
            output_dir = Path(output_dir)
        
        self.output_file = output_dir / output_file
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if data_dirs is None:
            default_data_dir = Path("/home/yijuan_liang/10.12Tool_Set/test_set/data/data_origin")
            self.data_dirs = [default_data_dir]
        else:
            self.data_dirs = [Path(d) for d in data_dirs]
        
        self.tool_counter = 1
        self.seen_tools: Set[Tuple[str, str]] = set()  # Dedup by (name, description)
        
        print(f"Found {len(self.data_dirs)} data folder(s)")
        for d in self.data_dirs:
            print(f"  - {d}")
    
    def _find_data_dirs(self, root: Path) -> List[Path]:
        """Recursively find all data folders."""
        data_dirs = []
        for path in root.rglob("data"):
            if path.is_dir():
                json_files = list(path.glob("*.json"))
                if json_files:
                    data_dirs.append(path)
        return data_dirs
    
    def parse_tools_from_string(self, tools_str: str) -> List[Dict[str, Any]]:
        """Parse tool list from a JSON string."""
        try:
            if not tools_str or not isinstance(tools_str, str) or not tools_str.strip():
                return []
            
            tools_list = json.loads(tools_str)
            if not isinstance(tools_list, list):
                return []
            
            return tools_list
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Problematic string: {tools_str[:200]}...")
            return []
        except Exception as e:
            print(f"Error parsing tools string: {e}")
            return []
    
    def convert_parameters_schema(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Convert parameter schema; drop null/empty fields."""
        try:
            if not parameters or not isinstance(parameters, dict):
                return {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            
            properties = parameters.get('properties', {})
            required = parameters.get('required', [])
            
            converted_properties = {}
            for prop_name, prop_def in properties.items():
                if isinstance(prop_def, dict):
                    converted_prop = {}
                    
                    for key, value in prop_def.items():
                        if value is None:
                            continue
                        if isinstance(value, str) and value.strip().lower() == 'null':
                            continue
                        if isinstance(value, str) and not value.strip():
                            continue
                        if isinstance(value, list) and len(value) == 0:
                            continue
                        if isinstance(value, dict) and len(value) == 0:
                            continue
                        
                        converted_prop[key] = value
                    
                    if 'type' not in converted_prop:
                        converted_prop['type'] = 'string'
                    
                    if converted_prop:
                        converted_properties[prop_name] = converted_prop
            
            return {
                "type": "object",
                "properties": converted_properties,
                "required": required if required else []
            }
            
        except Exception as e:
            print(f"Error converting parameter schema: {e}")
            return {
                "type": "object",
                "properties": {},
                "required": []
            }
    
    def convert_tool_to_corpus_format(self, tool: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert one tool to corpus format (inputSchema)."""
        try:
            name = (tool.get('name', '') or '').strip()
            description = (tool.get('description', '') or '').strip()
            
            if not name or not description:
                return None
            
            tool_key = (name, description)
            if tool_key in self.seen_tools:
                return None
            
            self.seen_tools.add(tool_key)
            
            input_schema = None
            
            if 'inputSchema' in tool and tool['inputSchema']:
                input_schema = tool['inputSchema']
                cleaned_schema = self._clean_schema(input_schema)
                if cleaned_schema is None or (isinstance(cleaned_schema, dict) and len(cleaned_schema) == 0):
                    input_schema = {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                else:
                    input_schema = cleaned_schema
            elif 'parameters' in tool and tool['parameters']:
                input_schema = self.convert_parameters_schema(tool['parameters'])
            else:
                input_schema = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            
            corpus_tool = {
                "name": name,
                "description": description,
                "inputSchema": input_schema
            }
            
            return corpus_tool
            
        except Exception as e:
            print(f"Error converting tool: {e}")
            print(f"Tool data: {tool}")
            return None
    
    def _clean_schema(self, schema: Any) -> Any:
        """Recursively remove null/empty values from schema."""
        if schema is None:
            return None
        
        if isinstance(schema, dict):
            cleaned = {}
            for key, value in schema.items():
                if value is None:
                    continue
                if isinstance(value, str) and value.strip().lower() == 'null':
                    continue
                if isinstance(value, str) and not value.strip():
                    continue
                
                cleaned_value = self._clean_schema(value)
                
                if cleaned_value is None:
                    continue
                
                if isinstance(cleaned_value, list) and len(cleaned_value) == 0:
                    continue
                if isinstance(cleaned_value, dict) and len(cleaned_value) == 0:
                    continue
                
                cleaned[key] = cleaned_value
            return cleaned
        elif isinstance(schema, list):
            cleaned_list = []
            for item in schema:
                cleaned_item = self._clean_schema(item)
                if cleaned_item is None:
                    continue
                if isinstance(cleaned_item, list) and len(cleaned_item) == 0:
                    continue
                if isinstance(cleaned_item, dict) and len(cleaned_item) == 0:
                    continue
                cleaned_list.append(cleaned_item)
            return cleaned_list
        else:
            return schema
    
    def process_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process one JSON file and extract all tools."""
        tools = []
        try:
            print(f"Processing file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'tools' in item:
                        tools_str = item['tools']
                        tool_list = self.parse_tools_from_string(tools_str)
                        tools.extend(tool_list)
            elif isinstance(data, dict):
                if 'tools' in data:
                    tools_str = data['tools']
                    tool_list = self.parse_tools_from_string(tools_str)
                    tools.extend(tool_list)
            
            print(f"  Extracted {len(tools)} tool(s) from file")
            return tools
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return []
    
    def combine_all_tools(self):
        """Merge tools from all data folders."""
        all_tools = []
        
        for data_dir in self.data_dirs:
            print(f"\nProcessing folder: {data_dir}")
            json_files = list(data_dir.glob("*.json"))
            json_files = [f for f in json_files if not f.name.endswith('.bak')]
            print(f"Found {len(json_files)} JSON file(s) (backups excluded)")
            
            for json_file in json_files:
                tools = self.process_file(json_file)
                all_tools.extend(tools)
        
        print(f"\nExtracted {len(all_tools)} tool occurrence(s) (with duplicates)")
        
        tools_dict = {}
        converted_count = 0
        skipped_duplicate_count = 0
        skipped_empty_count = 0
        
        for tool in all_tools:
            corpus_tool = self.convert_tool_to_corpus_format(tool)
            if corpus_tool is None:
                name = (tool.get('name', '') or '').strip()
                description = (tool.get('description', '') or '').strip()
                if not name or not description:
                    skipped_empty_count += 1
                else:
                    skipped_duplicate_count += 1
                continue
            
            tools_dict[str(self.tool_counter)] = corpus_tool
            self.tool_counter += 1
            converted_count += 1
            
            if converted_count % 1000 == 0:
                print(f"Converted {converted_count} tools, skipped {skipped_duplicate_count} duplicates, skipped {skipped_empty_count} empty-field tools")
        
        print(f"\nConversion done.")
        print(f"Total occurrences (before dedup): {len(all_tools)}")
        print(f"Tools after conversion: {len(tools_dict)}")
        print(f"Skipped duplicates: {skipped_duplicate_count}")
        print(f"Skipped empty name/description: {skipped_empty_count}")
        
        self.save_output(tools_dict)
        return tools_dict
    
    def save_output(self, tools_dict: Dict[str, Any]):
        """Write output file."""
        try:
            if self.output_file.exists():
                backup_file = self.output_file.with_suffix('.json.backup')
                if backup_file.exists():
                    backup_file.unlink()
                self.output_file.rename(backup_file)
                print(f"Backup created: {backup_file}")
            
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(tools_dict, f, ensure_ascii=False, indent=2)
            
            print(f"Output saved to: {self.output_file}")
            
        except Exception as e:
            print(f"Failed to save output: {e}")
            backup_file = self.output_file.with_suffix('.json.backup')
            if backup_file.exists():
                if self.output_file.exists():
                    self.output_file.unlink()
                backup_file.rename(self.output_file)
                print("Restored backup file")

def main():
    """CLI entry point"""
    print("=== Test set tool combiner ===")
    print("Extract tools from conversations in all files under each data folder")
    print("Convert to inputSchema; dedupe by name+description; drop empty/null parameter fields")
    print()
    
    combiner = TestSetCombiner(
        output_file="test_set_tool_v2.json",
        output_dir=None
    )
    
    combiner.combine_all_tools()
    
    print("\nDone.")

if __name__ == "__main__":
    main()
