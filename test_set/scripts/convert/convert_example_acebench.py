#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
from pathlib import Path

def extract_properties_from_filename(filename):
    """Extract properties from the filename."""
    properties = {
        "hop": "single-hop",  # defaults
        "turn": "single-turn",  # defaults
        "strategy": "",  # empty by default
        "domain": "",  # empty by default
        "other": filename
    }
    
    filename_lower = filename.lower()
    
    # hop
    if "single-hop" in filename_lower:
        properties["hop"] = "single-hop"
    elif "multi-hop" in filename_lower:
        properties["hop"] = "multi-hop"
    
    # turn
    if "single-turn" in filename_lower:
        properties["turn"] = "single-turn"
    elif "multi-turn" in filename_lower:
        properties["turn"] = "multi-turn"
    
    # strategy (only when explicitly present)
    if "parallel" in filename_lower:
        properties["strategy"] = "Parallel"
    elif "serial" in filename_lower:
        properties["strategy"] = "Serial"
    
    # domain (only when explicitly present)
    if "out_domain" in filename_lower:
        properties["domain"] = "out_domain"
    elif "in_domain" in filename_lower:
        properties["domain"] = "in_domain"
    
    return properties

def get_last_step_for_same_query(data, filename):
    """For special files, keep only the last step per query."""
    if "multi_turn_user_adjust" in filename or "multi_turn_user_switch" in filename:
        # Group by query; keep the last step in each group
        query_groups = {}
        
        for item in data:
            # Parse query index and step index from id
            # id format: "normal_multi_turn_user_adjust_1_0" -> query=1, step=0
            id_parts = item["id"].split("_")
            if len(id_parts) >= 4:
                query_num = int(id_parts[-2])  # second-to-last segment is query index
                step_num = int(id_parts[-1])   # last segment is step index
                
                if query_num not in query_groups:
                    query_groups[query_num] = []
                query_groups[query_num].append((step_num, item))
        
        # Keep the item with the largest step index per group
        filtered_data = []
        for query_num in query_groups:
            max_step_item = max(query_groups[query_num], key=lambda x: x[0])
            filtered_data.append(max_step_item[1])
        
        return filtered_data
    
    return data

def convert_function_to_tool_schema(function_info):
    """Convert function metadata to tools schema."""
    tool = {
        "name": function_info["name"],
        "description": function_info["description"],
        "category": "analysis",  # default category
        "domain": "business",   # default domain
        "inputSchema": function_info.get("parameters", function_info.get("arguments", {}))
    }
    return tool

def convert_ground_truth_to_function_call(ground_truth):
    """Convert ground_truth to function_call shape (new: {"name": "...", "arguments": {...}})."""
    if isinstance(ground_truth, str):
        try:
            # Try parsing as JSON string
            ground_truth = json.loads(ground_truth)
        except:
            # Not JSON; return as-is
            return ground_truth
    
    if isinstance(ground_truth, dict):
        # Already new format?
        if "name" in ground_truth and "arguments" in ground_truth:
            # New format; serialize
            return json.dumps(ground_truth, ensure_ascii=False, separators=(',', ':'))
        
        # Old format: {"tool_name": {...}}
        # New format: {"name": "tool_name", "arguments": {...}}
        if len(ground_truth) == 1:
            tool_name = list(ground_truth.keys())[0]
            arguments = ground_truth[tool_name]
            new_format = {"name": tool_name, "arguments": arguments}
            return json.dumps(new_format, ensure_ascii=False, separators=(',', ':'))
        
        # Other dict shapes
        return json.dumps(ground_truth, ensure_ascii=False, separators=(',', ':'))
    
    return str(ground_truth)

def load_json_lines(file_path):
    """Load a file with one JSON object per line."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        # One JSON object per line
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data

def parse_multi_turn_conversation(question_text):
    """Parse multi-turn text into user/system turns."""
    conversations = []
    
    lines = question_text.strip().split('\n')
    current_speaker = None
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('user:'):
            # Flush previous block
            if current_speaker and current_content:
                conversations.append({
                    "from": current_speaker,
                    "value": '\n'.join(current_content)
                })
            
            current_speaker = "human"
            current_content = [line[5:].strip()]  # strip "user:" prefix
            
        elif line.startswith('system:'):
            if current_speaker and current_content:
                conversations.append({
                    "from": current_speaker,
                    "value": '\n'.join(current_content)
                })
            
            current_speaker = "gpt"
            current_content = [line[7:].strip()]  # strip "system:" prefix
            
        else:
            if current_content is not None:
                current_content.append(line)
    
    if current_speaker and current_content:
        conversations.append({
            "from": current_speaker,
            "value": '\n'.join(current_content)
        })
    
    return conversations

def convert_single_conversation(source_item, answer_item, tools, properties, tool_name_mapping=None):
    """Convert a single conversation row."""
    conversations = []
    
    ground_truth = answer_item.get("ground_truth", {})
    if tool_name_mapping is None:
        tool_name_mapping = {}
    
    question_text = source_item.get("question", "")
    
    if "user:" in question_text and "system:" in question_text:
        parsed_conversations = parse_multi_turn_conversation(question_text)
        
        for conv in parsed_conversations:
            conversations.append(conv)
            
            if conv["from"] == "human":
                if isinstance(ground_truth, dict):
                    for tool_name, tool_args in ground_truth.items():
                        actual_tool_name = tool_name_mapping.get(tool_name, tool_name)
                        # function_call (new format)
                        function_call_msg = {
                            "from": "function_call",
                            "value": json.dumps({"name": actual_tool_name, "arguments": tool_args}, ensure_ascii=False, separators=(',', ':'))
                        }
                        conversations.append(function_call_msg)
                        
                        observation_msg = {
                            "from": "observation",
                            "value": ""
                        }
                        conversations.append(observation_msg)
                else:
                    function_call_msg = {
                        "from": "function_call",
                        "value": convert_ground_truth_to_function_call(ground_truth)
                    }
                    conversations.append(function_call_msg)
                    
                    observation_msg = {
                        "from": "observation",
                        "value": ""
                    }
                    conversations.append(observation_msg)
    else:
        human_msg = {
            "from": "human",
            "value": question_text
        }
        conversations.append(human_msg)
        
        if isinstance(ground_truth, dict):
            for tool_name, tool_args in ground_truth.items():
                actual_tool_name = tool_name_mapping.get(tool_name, tool_name)
                function_call_msg = {
                    "from": "function_call",
                    "value": json.dumps({"name": actual_tool_name, "arguments": tool_args}, ensure_ascii=False, separators=(',', ':'))
                }
                conversations.append(function_call_msg)
                
                observation_msg = {
                    "from": "observation",
                    "value": ""
                }
                conversations.append(observation_msg)
        else:
            function_call_msg = {
                "from": "function_call",
                "value": convert_ground_truth_to_function_call(ground_truth)
            }
            conversations.append(function_call_msg)
            
            observation_msg = {
                "from": "observation",
                "value": ""
            }
            conversations.append(observation_msg)
    
    gpt_msg = {
        "from": "gpt",
        "value": ""
    }
    conversations.append(gpt_msg)
    
    return conversations

def convert_acebench_to_format(source_file, answer_file, output_file):
    """Convert ACEBench inputs to format.json style."""
    
    source_data = load_json_lines(source_file)
    answer_data = load_json_lines(answer_file)
    
    source_data = get_last_step_for_same_query(source_data, os.path.basename(source_file))
    answer_data = get_last_step_for_same_query(answer_data, os.path.basename(answer_file))
    
    min_length = min(len(source_data), len(answer_data))
    source_data = source_data[:min_length]
    answer_data = answer_data[:min_length]
    
    filename = os.path.basename(source_file)
    properties = extract_properties_from_filename(filename)
    
    used_tools = set()
    for answer_item in answer_data:
        ground_truth = answer_item.get("ground_truth", {})
        if isinstance(ground_truth, dict):
            for tool_name in ground_truth.keys():
                used_tools.add(tool_name)
    
    all_tools = []
    for item in source_data:
        if "function" in item:
            for func in item["function"]:
                if func["name"] in used_tools:
                    tool = convert_function_to_tool_schema(func)
                    if not any(t["name"] == tool["name"] for t in all_tools):
                        all_tools.append(tool)
    
    all_conversations = []
    
    for source_item, answer_item in zip(source_data, answer_data):
        ground_truth = answer_item.get("ground_truth", {})
        
        tool_dict = {}
        if "function" in source_item:
            for func in source_item["function"]:
                tool_dict[func["name"]] = convert_function_to_tool_schema(func)
        
        current_tools = []
        seen_base_names = set()
        tool_name_mapping = {}
        if isinstance(ground_truth, dict):
            for tool_name in ground_truth.keys():
                if tool_name in tool_dict:
                    base_name = tool_name
                    tool_name_mapping[tool_name] = base_name
                    if base_name not in seen_base_names:
                        current_tools.append(tool_dict[base_name])
                        seen_base_names.add(base_name)
                else:
                    # parallel_function: strip numeric suffix (e.g. tool_1 -> tool)
                    base_name = re.sub(r'_\d+$', '', tool_name)
                    if base_name in tool_dict and base_name not in seen_base_names:
                        tool_name_mapping[tool_name] = base_name
                        current_tools.append(tool_dict[base_name])
                        seen_base_names.add(base_name)
                    elif base_name in tool_dict:
                        tool_name_mapping[tool_name] = base_name
        
        current_properties = properties.copy()
        
        temp_conversations = convert_single_conversation(source_item, answer_item, current_tools, current_properties, tool_name_mapping)
        
        human_count = sum(1 for msg in temp_conversations if msg["from"] == "human")
        
        if human_count > 1:
            current_properties["turn"] = "multi-turn"
        else:
            current_properties["turn"] = "single-turn"
        
        # hop: multi-hop only if some turn has more than one function_call
        human_indices = [i for i, conv in enumerate(temp_conversations) if conv.get("from") == "human"]
        
        if len(human_indices) > 0:
            max_function_calls_per_turn = 0
            
            for i, human_idx in enumerate(human_indices):
                next_human_idx = human_indices[i + 1] if i + 1 < len(human_indices) else len(temp_conversations)
                
                function_calls_in_turn = sum(
                    1 for conv in temp_conversations[human_idx:next_human_idx] 
                    if conv.get("from") == "function_call"
                )
                
                max_function_calls_per_turn = max(max_function_calls_per_turn, function_calls_in_turn)
            
            if max_function_calls_per_turn > 1:
                current_properties["hop"] = "multi-hop"
            else:
                current_properties["hop"] = "single-hop"
        else:
            current_properties["hop"] = "single-hop"
        
        conversations = convert_single_conversation(source_item, answer_item, current_tools, current_properties, tool_name_mapping)
        
        conversation_obj = {
            "conversations": conversations,
            "system": "\n\n# Role\n\nYou are an AI assistant that can call various tools to help users with their queries. You have access to a comprehensive set of tools for analysis, operations, management, visualization, search, generation and so on.\n\n## Tool Selection\n\n**Important**: The available function signatures are provided in the <tools></tools> XML tags. You must carefully select one or more appropriate tools from the <tools></tools> section that can solve the user's query. Note that the section may contain irrelevant tools that are not suitable for the current request.\n\nIf you determine that there are no suitable tools in the <tools></tools> section to answer the user's request, you should respond with: \"Sorry, there are no suitable tools to answer your request.\"\n\n## Output Rules\n\nYou must strictly follow these rules when responding:\n\n### 1. Function Call Format\nIf you need to call a function, you can output only one function call at a time in the following format:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n\n**Parameter Parsing**: The parameters in <tool_call></tool_call> should be parsed based on the content of the user's query.\n\n**Example**:\n<tool_call>\n{\"name\": \"analyze_customer_feedback\", \"arguments\": {\"start_date\": \"2025-08-25\", \"end_date\": \"2025-09-24\", \"sentiment_type\": \"negative\", \"keyword_filter\": \"Service Attitude\", \"analysis_focus\": \"issue_categories\"}}\n</tool_call>\n\n### 2. Final Answer Format\nIf you have already obtained sufficient information from tool responses or through reasoning, you must immediately stop calling tools and provide your final answer in this format:\n<answer>\nYour final answer here\n</answer>\n\n### 3. Intelligent Process Stage Judgment\n- Carefully analyze the conversation flow history below to understand which stage you are currently in\n- If you see that the Assistant has already called a tool and the User has provided <tool_response>...</tool_response>, it means the tool call is complete\n- If the tool returns empty data (such as total revenue of 0, empty lists, etc.), you should generate an explanatory answer rather than repeating the call\n- If you have obtained sufficient information to answer the user's question, immediately generate the final answer\n\n### 4. Strictly Prohibited Behaviors\n- Never provide both function calls and final answers in the same round of output\n- Never use identical parameters to repeatedly call the same tool\n- Never continue calling the same tool after it has already returned results (including empty results)\n- Never ignore tool calls and response information that already exists in the conversation flow history\n\n**Special Note:** By examining the conversation flow history below, you can clearly see:\n- Previous User and Assistant interactions\n- Tool calls that have already been executed\n- Specific results returned by tools\n- Which stage the current conversation has reached\n\n### 5. Error Handling and Data Quality Assessment\n- If the tool returns `success: False` or clear error codes (such as status_code: 3001), it indicates operation failure\n- If the tool returns total revenue of 0, empty lists, or empty chart data, it means there is indeed no data under the query conditions\n- If the tool returns error messages (such as 'resource not found', 'invalid parameters', etc.), you should not repeat the call\n- In such cases, you should generate an explanatory answer, explaining the specific error cause or data condition\n- Absolutely do not repeat calling the same tool because it returned an error or empty data\n\n### 6. Tool Call History Check\n- Before each tool call, you must check whether the same tool has been called in the conversation history\n- If the same tool call exists in the history and has already returned results, you must generate an answer based on those results\n- If the previous call failed, you should analyze the failure reason and explain it to the user, rather than retrying\n\n**Remember:** Based on the conversation flow history, judge the current stage, and immediately output once you can generate an answer, avoiding meaningless tool repetition.\n\n## Current Time\nCurrent time: {time}\n",
            "tools": json.dumps(current_tools),
            "properties": current_properties
        }
        
        all_conversations.append(conversation_obj)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=2)
    
    print(f"Converted: {filename} -> {os.path.basename(output_file)}")

def main():
    """CLI entry point"""
    data_en_dir = r"D:\Desktop\10.12Tool_Set\ACEBench\data_all\data_en"
    possible_answer_dir = r"D:\Desktop\10.12Tool_Set\ACEBench\data_all\data_en\possible_answer"
    convert_dir = r"D:\Desktop\10.12Tool_Set\ACEBench\data_all\data_en\convert"
    
    os.makedirs(convert_dir, exist_ok=True)
    
    json_files = [f for f in os.listdir(data_en_dir) 
                  if f.endswith('.json') and os.path.isfile(os.path.join(data_en_dir, f))]
    
    for json_file in json_files:
        source_file = os.path.join(data_en_dir, json_file)
        answer_file = os.path.join(possible_answer_dir, json_file)
        output_file = os.path.join(convert_dir, json_file)
        
        if os.path.exists(answer_file):
            try:
                convert_acebench_to_format(source_file, answer_file, output_file)
            except Exception as e:
                print(f"Failed {json_file}: {str(e)}")
        else:
            print(f"Skip {json_file}: answer file not found")

if __name__ == "__main__":
    main()
