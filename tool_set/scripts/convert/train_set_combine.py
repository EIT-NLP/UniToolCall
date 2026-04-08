#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
import os

sys.stdout.reconfigure(encoding='utf-8')

class TrainSetCombiner:
    def __init__(self, 
                 train_set_data_dir: str = None,
                 data_yifan_dir: str = None,
                 output_file: str = "train_set_tool.json",
                 output_dir: str = None):
        """
        Initialize the combiner.

        Args:
            train_set_data_dir: Path to train_set/data; if None, auto-detect
            data_yifan_dir: Optional Data_Yifan folder path
            output_file: Output filename
            output_dir: Output directory; if None, use default under repo
        """
        if output_dir is None:
            root = Path(__file__).resolve().parents[1]
            output_dir = root / "apis" / "apis_origin"
        else:
            output_dir = Path(output_dir)
        
        self.output_file = output_dir / output_file
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Resolve train_set/data directory
        if train_set_data_dir is None:
            # Auto-detect train_set/data
            script_dir = Path(__file__).resolve().parent
            train_set_data_dir = script_dir.parent.parent.parent / "train_set" / "data"
        else:
            train_set_data_dir = Path(train_set_data_dir)
        
        self.train_set_data_dir = train_set_data_dir
        
        # Find all JSON files under train_set/data (including subfolders e.g. TOUCAN_converted)
        self.train_set_files = []
        if self.train_set_data_dir.exists() and self.train_set_data_dir.is_dir():
            # All JSON files recursively; exclude backup files
            all_json_files = list(self.train_set_data_dir.rglob("*.json"))
            self.train_set_files = [f for f in all_json_files if not f.name.endswith('.bak')]
            print(f"Found {len(self.train_set_files)} JSON files under train_set/data (including subfolders)")
            
            # List files grouped by folder
            files_by_dir = {}
            for f in self.train_set_files:
                rel_path = f.relative_to(self.train_set_data_dir)
                dir_name = rel_path.parent if rel_path.parent != Path('.') else Path('.')
                if dir_name not in files_by_dir:
                    files_by_dir[dir_name] = []
                files_by_dir[dir_name].append(rel_path.name)
            
            for dir_name, files in sorted(files_by_dir.items()):
                dir_str = str(dir_name) if dir_name != Path('.') else 'root'
                print(f"  {dir_str}: {len(files)} file(s)")
        else:
            print(f"Warning: train_set/data folder not found: {self.train_set_data_dir}")
        
        # Optional Data_Yifan directory
        self.data_yifan_dir = None
        self.data_yifan_files = []
        if data_yifan_dir:
            data_yifan_dir = Path(data_yifan_dir)
            self.data_yifan_dir = data_yifan_dir
            if self.data_yifan_dir.exists() and self.data_yifan_dir.is_dir():
                self.data_yifan_files = list(self.data_yifan_dir.rglob("*.json"))
                print(f"Found {len(self.data_yifan_files)} JSON files under Data_Yifan")
            else:
                print(f"Warning: Data_Yifan folder not found: {self.data_yifan_dir}")
        
        self.tool_counter = 1
        self.seen_tools: Set[Tuple[str, str]] = set()  # Dedup by (name, description)
        
        print(f"Will process {len(self.train_set_files)} train_set file(s)")
        if self.data_yifan_files:
            print(f"Will process {len(self.data_yifan_files)} Data_Yifan file(s)")
    
    def parse_tools_from_string(self, tools_str: str) -> List[Dict[str, Any]]:
        """Parse tool list from a JSON string."""
        try:
            if not tools_str or not isinstance(tools_str, str) or not tools_str.strip():
                return []
            
            # Parse JSON string
            tools_list = json.loads(tools_str)
            if not isinstance(tools_list, list):
                return []
            
            return tools_list
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Problematic string: {tools_str[:200]}...")
            return []
        except Exception as e:
            print(f"Error while parsing tools string: {e}")
            return []
    
    def convert_parameters_schema(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Convert parameter schema; drop null and empty fields."""
        try:
            if not parameters or not isinstance(parameters, dict):
                return {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            
            # properties and required
            properties = parameters.get('properties', {})
            required = parameters.get('required', [])
            
            # Convert each property; filter null/empty
            converted_properties = {}
            for prop_name, prop_def in properties.items():
                if isinstance(prop_def, dict):
                    converted_prop = {}
                    
                    # Keep original fields except null/empty
                    for key, value in prop_def.items():
                        # Skip null
                        if value is None:
                            continue
                        # Skip string "null" (case-insensitive)
                        if isinstance(value, str) and value.strip().lower() == 'null':
                            continue
                        # Skip empty string (keep non-empty strings)
                        if isinstance(value, str) and not value.strip():
                            continue
                        # Skip empty list
                        if isinstance(value, list) and len(value) == 0:
                            continue
                        # Skip empty dict
                        if isinstance(value, dict) and len(value) == 0:
                            continue
                        
                        converted_prop[key] = value
                    
                    # Ensure type exists
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
            # Basic fields, stripped
            name = (tool.get('name', '') or '').strip()
            description = (tool.get('description', '') or '').strip()
            
            # Skip if name empty
            if not name:
                return None
            
            # If no description, try elsewhere or use empty string
            # Some tools have no description (only name/schema)
            if not description:
                # Try schema description if present
                schema = tool.get('schema') or tool.get('inputSchema') or tool.get('parameters')
                if isinstance(schema, dict) and 'description' in schema:
                    description = (schema.get('description', '') or '').strip()
                if not description:
                    description = ''
            
            # Dedup by (name, description)
            tool_key = (name, description)
            if tool_key in self.seen_tools:
                return None  # Duplicate
            
            self.seen_tools.add(tool_key)
            
            # inputSchema / parameters / schema
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
            elif 'schema' in tool and tool['schema']:
                schema = tool['schema']
                if isinstance(schema, dict) and schema.get('type') == 'object':
                    cleaned_schema = self._clean_schema(schema)
                    if cleaned_schema is None or (isinstance(cleaned_schema, dict) and len(cleaned_schema) == 0):
                        input_schema = {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    else:
                        input_schema = cleaned_schema
                else:
                    input_schema = {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
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
        """Merge tools from all configured files."""
        all_tools = []
        
        print(f"\n=== Processing train_set/data files ===")
        for file_path in self.train_set_files:
            tools = self.process_file(file_path)
            all_tools.extend(tools)
        
        if self.data_yifan_files:
            print(f"\n=== Processing Data_Yifan files ===")
            for file_path in self.data_yifan_files:
                tools = self.process_file(file_path)
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
                # Only name is required; description may be empty
                if not name:
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
        print(f"Skipped empty-name tools: {skipped_empty_count}")
        
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
    print("=== Train set tool combiner ===")
    print("Extract tools from conversations in all files under train_set/data")
    print("Convert to inputSchema; dedupe by name+description; drop empty/null parameter fields")
    print()
    
    train_set_data_dir = None  # Auto-detect
    
    data_yifan_dir = None
    
    combiner = TrainSetCombiner(
        train_set_data_dir=train_set_data_dir,
        data_yifan_dir=data_yifan_dir,
        output_file="train_set_tool_v2.json",
        output_dir="/home/yijuan_liang/10.12Tool_Set/tool_set/apis/apis_origin"
    )
    
    combiner.combine_all_tools()
    
    print("\nDone.")

if __name__ == "__main__":
    main()

