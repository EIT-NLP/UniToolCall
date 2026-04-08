#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
from datasets import load_dataset
from typing import Dict, List, Any, Optional, Set, Tuple
import time

class FCToolsConverter:
    def __init__(self, 
                 output_file: str = "fc_tool_set.json",
                 cache_dir: str = r"D:\Desktop\9.24Tool_Set\raw_data",
                 progress_file: str = "conversion_progress.json"):
        """
        Initialize the converter.

        Args:
            output_file: Output filename
            cache_dir: Hugging Face dataset cache directory
            progress_file: Progress checkpoint file
        """
        self.output_file = output_file
        self.cache_dir = cache_dir
        self.progress_file = progress_file
        self.dataset = None
        self.tool_counter = 1  # Tool IDs start at 1
        self.seen_tools = set()  # Dedup by (name, description)
        
        self.load_progress()
    
    def load_progress(self):
        """Load conversion progress."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                    self.tool_counter = progress.get('tool_counter', 1)
                    seen_tools_list = progress.get('seen_tools', [])
                    self.seen_tools = set(tuple(t) for t in seen_tools_list)
                    print(f"Found progress file; resuming from tool ID {self.tool_counter}")
                    print(f"Recorded {len(self.seen_tools)} unique tools")
            except Exception as e:
                print(f"Failed to read progress file: {e}")
                self.tool_counter = 1
                self.seen_tools = set()
        else:
            print("No progress file; starting from scratch")
    
    def save_progress(self):
        """Save conversion progress."""
        progress = {
            'tool_counter': self.tool_counter,
            'seen_tools': [list(t) for t in self.seen_tools],
            'timestamp': time.time()
        }
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save progress file: {e}")
    
    def load_dataset(self):
        """Load fc-reward-bench dataset."""
        try:
            print("Loading fc-reward-bench dataset...")
            self.dataset = load_dataset("ibm-research/fc-reward-bench", cache_dir=self.cache_dir)
            print(f"Dataset loaded: {len(self.dataset['data'])} rows")
            return True
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            return False
    
    def parse_tools_from_string(self, tools_str: str) -> List[Dict[str, Any]]:
        """Parse tool definitions from a JSON string."""
        try:
            if not tools_str or not tools_str.strip():
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
    
    def convert_tool_to_corpus_format(self, tool: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert one tool to corpus format."""
        try:
            name = tool.get('name', f'unknown_tool_{self.tool_counter}')
            description = tool.get('description', '') or ''
            parameters = tool.get('parameters', {})
            
            tool_key = (name, description)
            if tool_key in self.seen_tools:
                return None  # Duplicate
            
            self.seen_tools.add(tool_key)
            
            corpus_tool = {
                "name": name,
                "description": description,
                "inputSchema": self.convert_parameters_schema(parameters)
            }
            
            return corpus_tool
            
        except Exception as e:
            print(f"Error converting tool: {e}")
            print(f"Tool data: {tool}")
            return None
    
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
    
    def load_existing_output(self) -> Dict[str, Any]:
        """Load existing output for resume."""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    print(f"Found existing output with {len(existing_data)} tools")
                    
                    for tool_id, tool_data in existing_data.items():
                        if isinstance(tool_data, dict):
                            name = tool_data.get('name', '')
                            description = tool_data.get('description', '') or ''
                            tool_key = (name, description)
                            self.seen_tools.add(tool_key)
                    
                    print(f"Restored {len(self.seen_tools)} unique tools from file")
                    return existing_data
            except Exception as e:
                print(f"Failed to read existing output: {e}")
        
        return {}
    
    def save_output(self, tools_dict: Dict[str, Any]):
        """Save conversion result."""
        try:
            if os.path.exists(self.output_file):
                backup_file = f"{self.output_file}.backup"
                if os.path.exists(backup_file):
                    os.remove(backup_file)
                os.rename(self.output_file, backup_file)
                print(f"Backup created: {backup_file}")
            
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(tools_dict, f, ensure_ascii=False, indent=2)
            
            print(f"Output saved to: {self.output_file}")
            
        except Exception as e:
            print(f"Failed to save output: {e}")
            backup_file = f"{self.output_file}.backup"
            if os.path.exists(backup_file):
                if os.path.exists(self.output_file):
                    os.remove(self.output_file)
                os.rename(backup_file, self.output_file)
                print("Restored backup file")
    
    def convert_all_tools(self, batch_size: int = 100):
        """Convert all tools."""
        if not self.dataset:
            if not self.load_dataset():
                return False
        
        tools_dict = self.load_existing_output()
        
        if tools_dict:
            max_id = max(int(k) for k in tools_dict.keys() if k.isdigit())
            self.tool_counter = max_id + 1
            print(f"Resuming from tool ID {self.tool_counter}")
        
        data_split = self.dataset['data']
        total_samples = len(data_split)
        
        print(f"Converting {total_samples} rows...")
        
        processed_count = 0
        added_tools = 0
        
        try:
            for i, sample in enumerate(data_split):
                if i < processed_count:
                    continue
                
                tools_str = sample.get('tools', '')
                if not tools_str:
                    continue
                
                tools_list = self.parse_tools_from_string(tools_str)
                
                for tool in tools_list:
                    try:
                        corpus_tool = self.convert_tool_to_corpus_format(tool)
                        if corpus_tool is not None:
                            tools_dict[str(self.tool_counter)] = corpus_tool
                            self.tool_counter += 1
                            added_tools += 1
                    except Exception as e:
                        print(f"Error converting tool (sample {i}): {e}")
                        continue
                
                processed_count += 1
                
                if processed_count % batch_size == 0:
                    duplicate_count = len(self.seen_tools) - len(tools_dict)
                    print(f"Processed {processed_count}/{total_samples} samples, added {added_tools} tools, total tools {len(tools_dict)}, unique {len(self.seen_tools)}")
                    self.save_progress()
                    self.save_output(tools_dict)
                    added_tools = 0
            
            duplicate_count = len(self.seen_tools) - len(tools_dict)
            print(f"Conversion done. Processed {processed_count} samples")
            print(f"Final tool count: {len(tools_dict)} (deduped)")
            print(f"Skipped duplicates: {duplicate_count}")
            self.save_output(tools_dict)
            
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
                print("Removed progress file")
            
            return True
            
        except KeyboardInterrupt:
            print("\nConversion interrupted by user")
            print(f"Processed {processed_count} samples; saved {len(tools_dict)} tools")
            self.save_progress()
            self.save_output(tools_dict)
            return False
            
        except Exception as e:
            print(f"Error during conversion: {e}")
            print(f"Processed {processed_count} samples; saved {len(tools_dict)} tools")
            self.save_progress()
            self.save_output(tools_dict)
            return False

def main():
    """CLI entry point"""
    print("=== FC-Reward-Bench tool converter ===")
    print("Convert fc-reward-bench dataset tools field to corpus format")
    print("Supports resume from checkpoint")
    print()
    
    converter = FCToolsConverter()
    
    success = converter.convert_all_tools(batch_size=50)
    
    if success:
        print("\nConversion done.")
    else:
        print("\nConversion incomplete; re-run to resume")
    
    print(f"\nOutput file: {converter.output_file}")
    print(f"Progress file: {converter.progress_file}")

if __name__ == "__main__":
    main()
