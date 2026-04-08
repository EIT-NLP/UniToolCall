#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from pathlib import Path
from datasets import Dataset
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict

class ToolRetConverter:
    def __init__(self, 
                 output_file: str = "toolret_tool.json",
                 output_dir: str = None):
        """
        Initialize the converter.

        Args:
            output_file: Output filename
            output_dir: Output directory; if None, use apis under the parent of this script
        """
        script_dir = Path(__file__).resolve().parent
        if output_dir is None:
            output_dir = script_dir.parent / "apis"
        else:
            output_dir = Path(output_dir)
        
        self.output_file = output_dir / output_file
        self.output_dir = output_dir
        self.tool_counter = 1
        self.seen_tools: Set[Tuple[str, str]] = set()  # Dedup by (name, description)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_param_type(self, param_type: Any) -> str:
        """Map parameter type to JSON Schema type."""
        if param_type is None:
            return 'string'
        
        if not isinstance(param_type, str):
            return 'string'
        
        type_mapping = {
            'STRING': 'string',
            'INTEGER': 'integer',
            'NUMBER': 'number',
            'BOOLEAN': 'boolean',
            'ARRAY': 'array',
            'OBJECT': 'object',
            'string': 'string',
            'integer': 'integer',
            'number': 'number',
            'boolean': 'boolean',
            'array': 'array',
            'object': 'object'
        }
        return type_mapping.get(param_type.upper(), 'string')
    
    def parse_tool_from_string(self, tool_str: str) -> Dict[str, Any]:
        """Parse tool definition from a JSON string."""
        try:
            if not tool_str or not isinstance(tool_str, str):
                return {}
            
            tool = json.loads(tool_str)
            return tool
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return {}
        except Exception as e:
            print(f"Error parsing tool string: {e}")
            return {}
    
    def convert_parameters_dict_to_schema(self, parameters_dict: Dict[str, Any], required_list: List[str] = None) -> Dict[str, Any]:
        """
        Convert parameters dict to inputSchema.
        parameters_dict: {param_name: {type, description, default, ...}}
        required_list: required parameter names if known
        """
        properties = {}
        required = []
        
        if not isinstance(parameters_dict, dict):
            return {
                "type": "object",
                "properties": {},
                "required": []
            }
        
        if required_list is None:
            required_list = []
            for param_name, param_def in parameters_dict.items():
                if isinstance(param_def, dict):
                    type_str = str(param_def.get('type', '')).lower()
                    has_optional_in_type = 'optional' in type_str
                    has_default_field = 'default' in param_def
                    if not has_optional_in_type and not has_default_field:
                        required_list.append(param_name)
        
        for param_name, param_def in parameters_dict.items():
            if not isinstance(param_def, dict):
                continue
            
            param_type_str = param_def.get('type', '')
            if isinstance(param_type_str, str):
                base_type = param_type_str.split(',')[0].strip()
                param_type = self.convert_param_type(base_type)
            else:
                param_type = self.convert_param_type(param_type_str)
            
            param_description = param_def.get('description', '') or ''
            
            prop = {
                "type": param_type
            }
            
            if param_description and param_description.strip():
                prop['description'] = param_description
            
            if 'default' in param_def:
                default_val = param_def['default']
                if default_val is not None and (not isinstance(default_val, str) or default_val.strip()):
                    prop['default'] = default_val
            
            json_schema_fields = [
                'enum', 'minimum', 'maximum', 'format', 'examples',
                'items', 'properties', 'pattern', 'minLength', 'maxLength',
                'minItems', 'maxItems', 'uniqueItems', 'additionalProperties',
                'const', 'multipleOf', 'exclusiveMinimum', 'exclusiveMaximum',
                'anyOf', 'oneOf', 'allOf', 'not', '$ref', 'title',
                'readOnly', 'writeOnly', 'deprecated', 'x-*'  # x-* vendor extensions
            ]
            
            for key, value in param_def.items():
                if key in ['type', 'description', 'default']:
                    continue
                
                if value is None:
                    continue
                
                if isinstance(value, str) and not value.strip():
                    continue
                
                if isinstance(value, list) and len(value) == 0:
                    continue
                
                if isinstance(value, dict) and len(value) == 0:
                    continue
                
                prop[key] = value
            
            properties[param_name] = prop
            
            if param_name in required_list:
                required.append(param_name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    def convert_parameters_to_schema(self, required_params: List[Dict], optional_params: List[Dict]) -> Dict[str, Any]:
        """Convert required_parameters and optional_parameters to inputSchema."""
        properties = {}
        required = []
        
        for param in required_params:
            if not isinstance(param, dict):
                continue
            
            param_name = param.get('name', '')
            if not param_name:
                continue
            
            param_type = self.convert_param_type(param.get('type'))
            param_description = param.get('description', '') or ''
            
            prop = {
                "type": param_type
            }
            
            if param_description and param_description.strip():
                prop['description'] = param_description
            
            if 'default' in param:
                default_val = param['default']
                if default_val is not None and (not isinstance(default_val, str) or default_val.strip()):
                    prop['default'] = default_val
            
            for key, value in param.items():
                if key in ['type', 'description', 'default', 'name']:
                    continue
                
                if value is None:
                    continue
                
                if isinstance(value, str) and not value.strip():
                    continue
                
                if isinstance(value, list) and len(value) == 0:
                    continue
                
                if isinstance(value, dict) and len(value) == 0:
                    continue
                
                prop[key] = value
            
            properties[param_name] = prop
            required.append(param_name)
        
        for param in optional_params:
            if not isinstance(param, dict):
                continue
            
            param_name = param.get('name', '')
            if not param_name:
                continue
            
            param_type = self.convert_param_type(param.get('type'))
            param_description = param.get('description', '') or ''
            
            prop = {
                "type": param_type
            }
            
            if param_description and param_description.strip():
                prop['description'] = param_description
            
            if 'default' in param:
                default_val = param['default']
                if default_val is not None and (not isinstance(default_val, str) or default_val.strip()):
                    prop['default'] = default_val
            
            for key, value in param.items():
                if key in ['type', 'description', 'default', 'name']:
                    continue
                
                if value is None:
                    continue
                
                if isinstance(value, str) and not value.strip():
                    continue
                
                if isinstance(value, list) and len(value) == 0:
                    continue
                
                if isinstance(value, dict) and len(value) == 0:
                    continue
                
                prop[key] = value
            
            properties[param_name] = prop
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    def convert_tool_to_corpus_format(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Convert one tool to corpus format."""
        try:
            name = tool.get('api_name') or tool.get('name') or f'unknown_tool_{self.tool_counter}'
            description = tool.get('api_description', '') or tool.get('description', '')
            
            if not description or not description.strip():
                return None
            
            tool_key = (name, description)
            if tool_key in self.seen_tools:
                return None
            
            self.seen_tools.add(tool_key)
            
            if 'parameters' in tool and tool['parameters']:
                parameters_value = tool['parameters']
                if isinstance(parameters_value, dict):
                    is_param_dict_format = True
                    if parameters_value:
                        if 'properties' in parameters_value and isinstance(parameters_value.get('properties'), dict):
                            parameters_dict = parameters_value.get('properties', {})
                            required_list = parameters_value.get('required', [])
                        else:
                            first_value = list(parameters_value.values())[0]
                            if isinstance(first_value, dict) and ('type' in first_value or 'description' in first_value):
                                parameters_dict = parameters_value
                                required_list = tool.get('required', [])
                            else:
                                parameters_dict = {}
                                required_list = []
                                is_param_dict_format = False
                    else:
                        parameters_dict = {}
                        required_list = []
                    
                    if is_param_dict_format:
                        input_schema = self.convert_parameters_dict_to_schema(parameters_dict, required_list)
                    else:
                        input_schema = {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                else:
                    input_schema = {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
            else:
                required_params = tool.get('required_parameters', [])
                optional_params = tool.get('optional_parameters', [])
                input_schema = self.convert_parameters_to_schema(required_params, optional_params)
            
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
    
    def load_arrow_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load an .arrow file as a list of dicts.
        Tries several readers for different on-disk layouts.
        """
        try:
            print(f"Loading file: {file_path}")
            from datasets import load_dataset, Features, Value, Sequence
            features = Features({
                'positive': Sequence(Value('string')),
                'negative': Sequence(Value('string')),
            })
            dataset = load_dataset("arrow", data_files=file_path, split="train", features=features)
            data_list = [dict(row) for row in dataset]
            print(f"Loaded via datasets.load_dataset: {len(data_list)} rows")
            return data_list
        except Exception as e1:
            print(f"Method 1 (datasets.load_dataset with features) failed: {e1}")
            try:
                print(f"Trying low-level Arrow stream read...")
                try:
                    with pa.ipc.open_stream(file_path) as reader:
                        batches = []
                        try:
                            while True:
                                batch = reader.read_next_batch()
                                batches.append(batch)
                        except StopIteration:
                            pass
                        if batches:
                            table = pa.Table.from_batches(batches)
                        else:
                            raise ValueError("No batches found")
                except:
                    with pa.ipc.open_file(file_path) as reader:
                        table = reader.read_all()
                
                data_list = []
                column_names = table.column_names
                
                for i in range(table.num_rows):
                    row_dict = {}
                    for col_name in column_names:
                        col = table[col_name]
                        try:
                            value = col[i].as_py()
                        except (AttributeError, TypeError):
                            try:
                                value = col[i]
                                if hasattr(value, 'value'):
                                    value = value.value
                            except:
                                value = None
                        row_dict[col_name] = value
                    data_list.append(row_dict)
                
                print(f"Loaded via low-level Arrow: {len(data_list)} rows")
                return data_list
            except Exception as e1b:
                print(f"Method 1b (low-level Arrow) failed: {e1b}")
        
        try:
            print(f"Trying Parquet reader...")
            table = pq.read_table(file_path)
            data_list = []
            column_names = table.column_names
            
            for i in range(table.num_rows):
                row_dict = {}
                for col_name in column_names:
                    col = table[col_name]
                    try:
                        value = col[i].as_py()
                    except (AttributeError, TypeError):
                        try:
                            value = col[i]
                            if hasattr(value, 'value'):
                                value = value.value
                        except:
                            value = None
                    row_dict[col_name] = value
                data_list.append(row_dict)
            
            print(f"Loaded via Parquet: {len(data_list)} rows")
            return data_list
        except Exception as e2:
            print(f"Method 2 (Parquet) failed: {e2}")
        
        try:
            print(f"Trying Arrow IPC stream...")
            with pa.ipc.open_stream(file_path) as reader:
                batches = []
                try:
                    while True:
                        batch = reader.read_next_batch()
                        batches.append(batch)
                except StopIteration:
                    pass
                if batches:
                    table = pa.Table.from_batches(batches)
                else:
                    raise ValueError("No batches found")
            
            data_list = []
            column_names = table.column_names
            
            for i in range(table.num_rows):
                row_dict = {}
                for col_name in column_names:
                    col = table[col_name]
                    try:
                        value = col[i].as_py()
                    except (AttributeError, TypeError):
                        try:
                            value = col[i]
                            if hasattr(value, 'value'):
                                value = value.value
                        except:
                            value = None
                    row_dict[col_name] = value
                data_list.append(row_dict)
            
            print(f"Loaded via Arrow IPC stream: {len(data_list)} rows")
            return data_list
        except Exception as e3:
            print(f"Method 3 (Arrow IPC stream) failed: {e3}")
        
        try:
            print(f"Trying datasets.Dataset.from_file...")
            dataset = Dataset.from_file(file_path)
            data_list = [dict(row) for row in dataset]
            print(f"Loaded via Dataset.from_file: {len(data_list)} rows")
            return data_list
        except Exception as e4:
            print(f"Method 4 (Dataset.from_file) failed: {e4}")
        
        print(f"All load methods failed")
        return None
    
    def extract_tools_from_dataset(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract tools from positive and negative columns."""
        tools = []
        
        for i, sample in enumerate(data_list):
            positive_tools = sample.get('positive', [])
            if isinstance(positive_tools, list):
                for tool_str in positive_tools:
                    tool = self.parse_tool_from_string(tool_str)
                    if tool:
                        tools.append(tool)
            
            negative_tools = sample.get('negative', [])
            if isinstance(negative_tools, list):
                for tool_str in negative_tools:
                    tool = self.parse_tool_from_string(tool_str)
                    if tool:
                        tools.append(tool)
            
            if (i + 1) % 10000 == 0:
                print(f"Processed {i + 1}/{len(data_list)} rows; extracted {len(tools)} tools so far")
        
        return tools
    
    def convert_all_tools(self, arrow_files: List[str]):
        """Convert all tools from given Arrow shards."""
        all_tools = []
        
        for arrow_file in arrow_files:
            data_list = self.load_arrow_file(arrow_file)
            if data_list is None:
                continue
            
            tools = self.extract_tools_from_dataset(data_list)
            all_tools.extend(tools)
            print(f"Extracted {len(tools)} tools from {arrow_file}")
        
        print(f"\nTotal extracted tool occurrences: {len(all_tools)} (with duplicates)")
        
        tools_dict = {}
        converted_count = 0
        skipped_duplicate_count = 0
        skipped_empty_desc_count = 0
        
        for tool in all_tools:
            description = tool.get('api_description', '') or tool.get('description', '')
            if not description or not description.strip():
                skipped_empty_desc_count += 1
                continue
            
            corpus_tool = self.convert_tool_to_corpus_format(tool)
            if corpus_tool is None:
                skipped_duplicate_count += 1
                continue
            
            tools_dict[str(self.tool_counter)] = corpus_tool
            self.tool_counter += 1
            converted_count += 1
            
            if converted_count % 1000 == 0:
                print(f"Converted {converted_count} tools, skipped {skipped_duplicate_count} duplicates, skipped {skipped_empty_desc_count} with empty description")
        
        print(f"\nConversion done.")
        print(f"Total occurrences (before dedup): {len(all_tools)}")
        print(f"Tools after conversion: {len(tools_dict)}")
        print(f"Skipped duplicates: {skipped_duplicate_count}")
        print(f"Skipped empty description: {skipped_empty_desc_count}")
        
        self.save_output(tools_dict)
        
        return True
    
    def save_output(self, tools_dict: Dict[str, Any]):
        """Write conversion output."""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(tools_dict, f, ensure_ascii=False, indent=2)
            
            print(f"Output saved to: {self.output_file}")
            
        except Exception as e:
            print(f"Failed to save output: {e}")
            raise

def main():
    """CLI entry point"""
    print("=== ToolRet converter ===")
    print("Convert ToolRet dataset tools to corpus format")
    print()
    
    script_dir = Path(__file__).resolve().parent
    base_path = script_dir.parent.parent.parent / "open_dataset" / "tool-ret" / "mangopy___tool_ret-training-20w" / "ToolRet-Training-20w" / "0.0.0" / "fdf5a317455b1e60785de7ba587496aa6cc878e4"
    
    arrow_files = [
        str(base_path / "tool_ret-training-20w-train-00000-of-00006.arrow"),
        str(base_path / "tool_ret-training-20w-train-00001-of-00006.arrow"),
        str(base_path / "tool_ret-training-20w-train-00002-of-00006.arrow"),
        str(base_path / "tool_ret-training-20w-train-00003-of-00006.arrow"),
        str(base_path / "tool_ret-training-20w-train-00004-of-00006.arrow"),
        str(base_path / "tool_ret-training-20w-train-00005-of-00006.arrow")
    ]
    
    output_dir = script_dir.parent / "apis" / "apis_origin"
    converter = ToolRetConverter(output_file="toolret_tool_v2.json", output_dir=str(output_dir))
    
    success = converter.convert_all_tools(arrow_files)
    
    if success:
        print("\nConversion done.")
    else:
        print("\nConversion failed")
    
    print(f"\nOutput file: {converter.output_file}")

if __name__ == "__main__":
    main()
