#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import sys
from pathlib import Path

_UT_ROOT = Path(__file__).resolve().parents[3]
if str(_UT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_UT_ROOT / "src"))
from uni_toolcall.prompts import read_system_prompt


def parse_function_description(desc_str):
    """
    Parse a Function_Description string into description text and a parameters schema.

    Example shape:
    "Get country info for the given country\\nParameters: {...}\\nOutput: Success.\\n..."
    """
    lines = desc_str.split('\n')
    description = lines[0].strip() if lines else ""

    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }

    params_start = desc_str.find("Parameters:")
    if params_start != -1:
        json_start = desc_str.find("{", params_start)
        if json_start != -1:
            brace_count = 0
            json_end = json_start
            in_string = False
            escape_next = False

            for i in range(json_start, len(desc_str)):
                char = desc_str[i]

                if escape_next:
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break

            try:
                params_json_str = desc_str[json_start:json_end]
                params_dict = json.loads(params_json_str)

                properties = {}
                required = []

                for param_name, param_desc in params_dict.items():
                    param_desc_str = str(param_desc)
                    is_required = "Required" in param_desc_str or param_desc_str.strip().startswith("Required")

                    if is_required:
                        required.append(param_name)

                    param_type = "string"
                    param_desc_lower = param_desc_str.lower()

                    if "integer" in param_desc_lower or "int32" in param_desc_lower or "int64" in param_desc_lower:
                        param_type = "integer"
                    elif "boolean" in param_desc_lower:
                        param_type = "boolean"
                    elif "number" in param_desc_lower or "float" in param_desc_lower or "double" in param_desc_lower:
                        param_type = "number"
                    elif "array" in param_desc_lower:
                        param_type = "array"
                    elif "object" in param_desc_lower:
                        param_type = "object"

                    properties[param_name] = {
                        "type": param_type,
                        "description": param_desc_str
                    }

                parameters = {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            except json.JSONDecodeError:
                try:
                    param_pattern = r'"(\w+)":\s*"((?:[^"\\]|\\.)*)"'
                    matches = re.findall(param_pattern, params_json_str)

                    if matches:
                        properties = {}
                        required = []

                        for param_name, param_desc in matches:
                            param_desc = param_desc.replace('\\"', '"').replace('\\n', '\n')
                            is_required = "Required" in param_desc or param_desc.strip().startswith("Required")

                            if is_required:
                                required.append(param_name)

                            param_type = "string"
                            param_desc_lower = param_desc.lower()

                            if "integer" in param_desc_lower or "int32" in param_desc_lower or "int64" in param_desc_lower:
                                param_type = "integer"
                            elif "boolean" in param_desc_lower:
                                param_type = "boolean"
                            elif "number" in param_desc_lower or "float" in param_desc_lower or "double" in param_desc_lower:
                                param_type = "number"
                            elif "array" in param_desc_lower:
                                param_type = "array"
                            elif "object" in param_desc_lower:
                                param_type = "object"

                            properties[param_name] = {
                                "type": param_type,
                                "description": param_desc
                            }

                        parameters = {
                            "type": "object",
                            "properties": properties,
                            "required": required
                        }
                    else:
                        parameters = {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                except Exception:
                    parameters = {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
            except Exception:
                parameters = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }

    return description, parameters


def convert_function_to_tool_schema(func_name, func_desc_str):
    """Turn a Function_Description block into the project's tool JSON shape."""
    description, parameters = parse_function_description(func_desc_str)

    tool = {
        "name": func_name,
        "description": description,
        "category": "analysis",
        "domain": "business",
        "inputSchema": parameters if parameters else {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
    return tool


def convert_intermediate_steps_to_conversations(intermediate_steps):
    """Convert ToolAlpaca intermediate_steps into our conversation messages."""
    conversations = []

    for step in intermediate_steps:
        if len(step) != 2:
            continue

        function_call_part = step[0]
        observation_part = step[1]

        if len(function_call_part) < 2:
            continue

        action = function_call_part[0]
        action_input_str = function_call_part[1]

        try:
            arguments = json.loads(action_input_str) if isinstance(action_input_str, str) else action_input_str
        except json.JSONDecodeError:
            arguments = {}

        if action:
            function_call_msg = {
                "from": "function_call",
                "value": json.dumps({
                    "name": action,
                    "arguments": arguments
                }, ensure_ascii=False, separators=(',', ':'))
            }
            conversations.append(function_call_msg)

        observation_msg = {
            "from": "observation",
            "value": str(observation_part) if observation_part else ""
        }
        conversations.append(observation_msg)

    return conversations


def determine_properties(conversations):
    """Set metadata properties from the conversation list."""
    properties = {
        "hop": "single-hop",
        "turn": "single-turn",
        "strategy": "",
        "domain": "",
        "other": "train"
    }

    function_call_count = sum(1 for conv in conversations if conv.get("from") == "function_call")

    if function_call_count > 1:
        properties["hop"] = "multi-hop"

    return properties


def convert_alpaca_train_to_format(input_file):
    """Convert ToolAlpaca train JSON to format_mutiturn.json-style rows."""

    with open(input_file, 'r', encoding='utf-8') as f:
        source_data = json.load(f)

    system_prompt = read_system_prompt()

    all_conversations = []

    for item in source_data:
        function_descriptions = item.get("Function_Description", {})
        instances = item.get("Instances", [])

        for instance in instances:
            user_input = instance.get("input", "")
            human_msg = {
                "from": "human",
                "value": user_input
            }

            intermediate_steps = instance.get("intermediate_steps", [])
            step_conversations = convert_intermediate_steps_to_conversations(intermediate_steps)

            used_tool_names = []
            seen_tools = set()
            for step in intermediate_steps:
                if len(step) > 0 and len(step[0]) > 0:
                    action = step[0][0]
                    if action and action not in seen_tools:
                        used_tool_names.append(action)
                        seen_tools.add(action)

            tools = []
            for tool_name in used_tool_names:
                if tool_name in function_descriptions and tool_name != "components":
                    func_desc = function_descriptions[tool_name]
                    tool = convert_function_to_tool_schema(tool_name, func_desc)
                    tools.append(tool)

            conversations = [human_msg]
            conversations.extend(step_conversations)

            output = instance.get("output", "")
            gpt_msg = {
                "from": "gpt",
                "value": output
            }
            conversations.append(gpt_msg)

            properties = determine_properties(conversations)

            conversation_obj = {
                "conversations": conversations,
                "system": system_prompt,
                "tools": json.dumps(tools, ensure_ascii=False),
                "properties": properties
            }

            all_conversations.append(conversation_obj)

    return all_conversations


def main():
    """CLI entry point"""
    base_dir = r"D:\Desktop\10.12Tool_Set"
    input_file = os.path.join(base_dir, "ToolAlpaca", "raw_data", "train_data.json")
    output_dir = os.path.join(base_dir, "train_set", "data")
    output_file = os.path.join(output_dir, "train_converted_alpaca.json")

    if not os.path.exists(input_file):
        print(f"Error: input file not found: {input_file}")
        return

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        print(f"Processing: {os.path.basename(input_file)}")
        conversations = convert_alpaca_train_to_format(input_file)
        print(f"  Converted {len(conversations)} row(s)")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)

        print(f"\nDone: {len(conversations)} row(s) -> {os.path.basename(output_file)}")

    except Exception as e:
        print(f"Conversion failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
