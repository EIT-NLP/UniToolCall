import json
import requests
import os
import sys
import time
import re
import glob
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
from uni_toolcall.secrets import get_openai_compatible_key

# API config (by mode)
DEFAULT_BATCH_SIZE = 64  # Default batch size

# Files that already have category/domain
SKIP_FILES = {
    'corpus_9.15_translated_cosdup.json',
    'generate_tool_set_translated_cosdup.json'
}

# Globals for current API config
CURRENT_API_URL = None
CURRENT_API_KEY = None
CURRENT_MODEL = None
CURRENT_BATCH_SIZE = DEFAULT_BATCH_SIZE
CURRENT_ENABLE_THINK = True  # think enabled by default

def setup_api_config(mode='api', batch_size=None, enable_think=True):
    """Configure API per mode (see generate_with_qwen_server.py)."""
    global CURRENT_API_URL, CURRENT_API_KEY, CURRENT_MODEL, CURRENT_BATCH_SIZE, CURRENT_ENABLE_THINK
    
    if batch_size is None:
        CURRENT_BATCH_SIZE = DEFAULT_BATCH_SIZE
    else:
        CURRENT_BATCH_SIZE = batch_size
    
    CURRENT_ENABLE_THINK = enable_think
    
    if mode == 'server':
        # Local vLLM server (see generate_with_qwen_server.py)
        CURRENT_API_URL = 'http://localhost:8000/v1/chat/completions'
        CURRENT_API_KEY = 'EMPTY'
        CURRENT_MODEL = 'Qwen3-8B'
        print(f"Using local server mode: {CURRENT_API_URL}")
    else:  # mode == 'api'
        CURRENT_API_URL = 'https://api.siliconflow.cn/v1/chat/completions'
        CURRENT_API_KEY = get_openai_compatible_key() or ''
        if not CURRENT_API_KEY:
            raise RuntimeError(
                "Please set SILICONFLOW_API_KEY or OPENAI_API_KEY"
            )
        CURRENT_MODEL = 'Qwen/Qwen3-8B'
        print(f"Using API mode: {CURRENT_API_URL}")
    
    print(f"Batch size: {CURRENT_BATCH_SIZE}")
    print(f"Think feature: {'on' if CURRENT_ENABLE_THINK else 'off'}")

def classify_tools_batch_concurrent(tools_batch):
    """Classify tools in parallel with ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=CURRENT_BATCH_SIZE) as executor:
        futures = {executor.submit(classify_tool, tool['name'], tool['description']): idx 
                   for idx, tool in enumerate(tools_batch)}
        
        results = {}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                category, domain = future.result()
                results[idx] = (category, domain)
            except Exception as e:
                results[idx] = (None, None)
                print(f"\n  Batch classify failed for tool {tools_batch[idx]['name']}: {str(e)[:100]}")
        
        return [results.get(i, (None, None)) for i in range(len(tools_batch))]

def classify_tool(name, description, max_retries=10):
    """Classify category/domain with Qwen3-8B; retry until valid."""
    
    # Normalize None to empty string
    name = name or ''
    description = description or ''
    
    valid_categories = {'analysis', 'operations', 'system', 'visualization', 'search', 'generate'}
    valid_domains = {'finance', 'technology', 'education', 'healthcare', 'entertainment', 
                     'travel', 'business', 'lifestyle', 'science', 'social', 'sports', 'environment', 'culture'}
    
    prompt_template = """Analyze this tool and assign one category and one domain.

Tool name: {name}
Tool description: {description}

Categories (pick exactly one):
- analysis: Data analysis and insights (statistical analysis, trend analysis, data mining, predictive analysis, business intelligence, etc.)
- operations: Business process operations (create, update, delete, workflow management, business logic execution, etc.)
- system: System administration and maintenance (system configuration, user management, system monitoring, technical maintenance, etc.)
- visualization: Data visualization and presentation (chart generation, report creation, data display, dashboard creation, etc.)
- search: Information retrieval and search (full-text search, fuzzy search, index query, structured query, data lookup, etc.)
- generate: Content and data generation (content generation, code generation, intelligent recommendation, AI generation, automated creation, etc.)

Domains (pick exactly one):
- finance: Finance related (payment, investment, wealth management, insurance, trading, etc.)
- technology: Technology and software development (programming, system management, software tools, IT infrastructure, etc.)
- education: Education and learning (academic courses, training programs, educational content, learning management, etc.)
- healthcare: Medical and health services (medical treatment, health monitoring, medical devices, healthcare management, etc.)
- entertainment: Entertainment and media (music, games, film/TV, social entertainment, news, content creation, etc.)
- travel: Travel and transportation (tourism, transportation, accommodation, attractions, travel planning, etc.)
- business: Business management (enterprise operations, marketing, customer relations, business processes, etc.)
- lifestyle: Daily life services (shopping, food, housekeeping, personal tools, consumer services, etc.)
- science: Scientific research and analysis (research projects, scientific experiments, academic studies, data analysis, etc.)
- social: Social communication and community (social networking, communication tools, community management, collaboration, etc.)
- sports: Sports and fitness (sports activities, fitness training, sports events, athletic performance, etc.)
- environment: Environment and sustainability (environmental protection, climate monitoring, ecology, sustainable development, etc.)
- culture: Culture and arts (art, literature, history, cultural events, language learning, creative content, etc.)

Pick the best-fitting labels from the name and description.
Return only a JSON object with no other text or markdown:
{{"category": "xxx", "domain": "xxx"}}"""

    prompt = prompt_template.format(name=name, description=description)
    
    # Retry until we get valid category/domain
    for attempt in range(max_retries):
        try:
            # Build request
            headers = {
                'Content-Type': 'application/json'
            }
            
            # Authorization: EMPTY for local server, real key for API
            if CURRENT_API_KEY:
                if CURRENT_API_KEY == 'EMPTY':
                    # Local: Bearer EMPTY
                    headers['Authorization'] = 'Bearer EMPTY'
                else:
                    # API: real key
                    headers['Authorization'] = f'Bearer {CURRENT_API_KEY}'
            
            data = {
                'model': CURRENT_MODEL,
                'messages': [
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': 0.1,
                'max_tokens': 200
            }
            
            # Thinking flags (see generate_with_qwen_server.py)
            if CURRENT_API_KEY == 'EMPTY':
                # Local vLLM: chat_template_kwargs
                data['chat_template_kwargs'] = {'enable_thinking': CURRENT_ENABLE_THINK}
            else:
                # API: enable_thinking on payload
                data['enable_thinking'] = CURRENT_ENABLE_THINK
            
            # Debug: log first tool only
            if attempt == 0 and name == 'webpage_scrape':
                print(f"\n  Debug: tool {name} request params:")
                print(f"    enable_thinking (global): {CURRENT_ENABLE_THINK}")
                print(f"    chat_template_kwargs: {data.get('chat_template_kwargs', 'N/A')}")
                print(f"    enable_thinking (direct): {data.get('enable_thinking', 'N/A')}")
                print(f"    data keys: {list(data.keys())}")
                # Full request for debugging
                import json as json_module
                print(f"    full request payload: {json_module.dumps(data, ensure_ascii=False, indent=2)[:500]}")
            
            response = requests.post(CURRENT_API_URL, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            result_data = response.json()
            
            # Validate response
            if 'choices' not in result_data or not result_data['choices']:
                error_msg = result_data.get('error', {}).get('message', 'unknown error')
                raise ValueError(f"Invalid API response: missing choices. Error: {error_msg}")
            
            choice = result_data['choices'][0]
            if 'message' not in choice:
                raise ValueError(f"API response format error: missing message. choice: {json.dumps(choice, ensure_ascii=False, indent=2)[:500]}")
            
            message = choice['message']
            # Prefer reasoning_content then content (vLLM)
            content = None
            if "reasoning_content" in message and message["reasoning_content"] is not None:
                content = message["reasoning_content"]
                if not CURRENT_ENABLE_THINK and attempt == 0 and name == 'webpage_scrape':
                    print(f"\n  Warning: Think disabled but API returned reasoning_content.")
                    print(f"    Server may force thinking.")
                    print(f"    Using reasoning_content as content.")
                    print(f"    message keys: {list(message.keys())}")
            elif "content" in message and message["content"] is not None:
                content = message["content"]
            
            if content is None:
                message_str = json.dumps(message, ensure_ascii=False, indent=2)
                if 'text' in message:
                    content = message.get('text')
                elif 'delta' in message and 'content' in message['delta']:
                    content = message['delta'].get('content')
                elif 'function_call' in message:
                    raise ValueError(f"API returned function_call instead of content. message: {message_str[:500]}")
                
                if content is None:
                    error_info = result_data.get('error', {})
                    if error_info:
                        error_msg = error_info.get('message', 'unknown error')
                        raise ValueError(f"API content is None. Error: {error_msg}")
                    else:
                        raise ValueError(f"API content is None. message: {message_str[:500]}")
            
            result_text = content.strip()
            
            if '```' in result_text:
                json_match = re.search(r'\{[^}]+\}', result_text)
                if json_match:
                    result_text = json_match.group(0)
            
            result = json.loads(result_text)
            category = (result.get('category') or '').lower().strip()
            domain = (result.get('domain') or '').lower().strip()
            
            if category in valid_categories and domain in valid_domains:
                return category, domain
            else:
                error_msg = []
                if category not in valid_categories:
                    error_msg.append(f"category '{category}' is invalid")
                if domain not in valid_domains:
                    error_msg.append(f"domain '{domain}' is invalid")
                
                print(f"\nWarning: tool {name} returned invalid labels (attempt {attempt + 1}/{max_retries}): {', '.join(error_msg)}")
                print(f"  Valid categories: {', '.join(sorted(valid_categories))}")
                print(f"  Valid domains: {', '.join(sorted(valid_domains))}")
                
                prompt = prompt_template.format(name=name, description=description)
                prompt += f"\n\nCategory must be one of: {', '.join(sorted(valid_categories))}"
                prompt += f"\nDomain must be one of: {', '.join(sorted(valid_domains))}"
                
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    raise ValueError(f"Still invalid after {max_retries} attempts")
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            if attempt < max_retries - 1:
                print(f"\nRetry {attempt + 1}/{max_retries} tool {name}: {str(e)[:100]}")
                time.sleep(1)
                continue
            else:
                raise Exception(f"classify_tool {name} failed after {max_retries} retries: {str(e)[:100]}")
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"\nRetry {attempt + 1}/{max_retries} tool {name}: {str(e)[:100]}")
                time.sleep(wait_time)
            else:
                raise Exception(f"classify_tool {name} failed after {max_retries} retries: {str(e)[:100]}")
    
    raise Exception(f"classify_tool {name} failed: could not obtain valid labels")

def has_category_domain(tool):
    """Return True if tool has both category and domain."""
    return 'category' in tool and 'domain' in tool

def is_standard_format(data):
    """True if top-level keys look like numeric string IDs."""
    if not data:
        return False
    sample_keys = list(data.keys())[:5]
    return all(k.isdigit() for k in sample_keys if k)

def process_standard_format_file(file_path):
    """Process numeric-key tool JSON."""
    print(f"\nStandard-format file: {os.path.basename(file_path)}")
    
    temp_file = file_path.replace('.json', '_temp.json')
    processed_keys = set()
    
    if os.path.exists(temp_file):
        print("  Resume from temp file...")
        with open(temp_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for key, tool in data.items():
            if has_category_domain(tool):
                processed_keys.add(key)
        print(f"    Temp file progress: {len(processed_keys)} tools")
    else:
        print("  Starting from original file...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for key, tool in data.items():
            if has_category_domain(tool):
                processed_keys.add(key)
        print(f"    Source already labeled: {len(processed_keys)} tools")
    
    total_tools = len(data)
    need_process = total_tools - len(processed_keys)
    already_have_count = sum(1 for key, tool in data.items() if has_category_domain(tool) and key not in processed_keys)
    print(f"  Total: {total_tools}, pending: {need_process}, already labeled: {already_have_count}")
    
    if need_process == 0:
        print("  All tools done; skip.")
        if os.path.exists(temp_file):
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            try:
                os.remove(temp_file)
                print("  Saved to source; temp removed.")
            except OSError:
                print("  Warning: could not delete temp file; remove manually.")
        return
    
    keys = sorted([int(k) for k in data.keys() if k.isdigit()])
    processed_count = 0
    error_count = 0
    skipped_count = 0
    
    tools_to_process = []
    tool_keys = []
    for idx in keys:
        key = str(idx)
        tool = data[key]
        
        if key in processed_keys:
            continue
        
        if has_category_domain(tool):
            skipped_count += 1
            continue
        
        name = tool.get('name') or ''
        description = tool.get('description') or ''
        
        if not description and 'annotations' in tool and isinstance(tool['annotations'], dict):
            description = tool['annotations'].get('title') or ''
        
        if not name:
            print(f"\n  Warning: tool {key} has no name; skip")
            continue
        
        if not description:
            description = name
        
        tools_to_process.append({'name': name, 'description': description, 'key': key, 'tool': tool})
        tool_keys.append(key)
    
    total_batches = (len(tools_to_process) + CURRENT_BATCH_SIZE - 1) // CURRENT_BATCH_SIZE
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * CURRENT_BATCH_SIZE
        end_idx = min(start_idx + CURRENT_BATCH_SIZE, len(tools_to_process))
        batch_tools = tools_to_process[start_idx:end_idx]
        
        print(f"\n  Batch {batch_idx + 1}/{total_batches} ({len(batch_tools)} tools)...")
        
        results = classify_tools_batch_concurrent(batch_tools)
        
        for tool_info, (category, domain) in zip(batch_tools, results):
            key = tool_info['key']
            tool = tool_info['tool']
            
            if category and domain:
                tool['category'] = category
                tool['domain'] = domain
                processed_count += 1
                
                try:
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"\n  Warning: temp save failed: {e}")
            else:
                error_count += 1
                print(f"\n  Error: tool {key} ({tool_info['name']}) failed")
        
        if batch_idx < total_batches - 1:
            time.sleep(0.5)
    
    print(f"\n  Writing final output...")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
            print("  Temp file removed.")
        except OSError:
            print("  Warning: could not delete temp file; remove manually.")
    
    print(f"  Done: processed {processed_count} tools, skipped {skipped_count} already filled, errors {error_count}")

def process_server_format_file(file_path):
    """Process server-name -> tool list JSON."""
    print(f"\nServer-format file: {os.path.basename(file_path)}")
    
    temp_file = file_path.replace('.json', '_temp.json')
    processed_tools = set()
    
    if os.path.exists(temp_file):
        print("  Resume from temp file...")
        with open(temp_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for server_name, tools in data.items():
            if isinstance(tools, list):
                for idx, tool in enumerate(tools):
                    if has_category_domain(tool):
                        processed_tools.add((server_name, idx))
        print(f"    Temp file progress: {len(processed_tools)} tools")
    else:
        print("  Starting from original file...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for server_name, tools in data.items():
            if isinstance(tools, list):
                for idx, tool in enumerate(tools):
                    if has_category_domain(tool):
                        processed_tools.add((server_name, idx))
        print(f"    Source already labeled: {len(processed_tools)} tools")
    
    total_tools = sum(len(tools) if isinstance(tools, list) else 0 for tools in data.values())
    need_process = total_tools - len(processed_tools)
    already_have_count = sum(1 for server_name, tools in data.items() 
                            if isinstance(tools, list) 
                            for idx, tool in enumerate(tools)
                            if has_category_domain(tool) and (server_name, idx) not in processed_tools)
    print(f"  Total: {total_tools}, pending: {need_process}, already labeled: {already_have_count}")
    
    if need_process == 0:
        print("  All tools done; skip.")
        if os.path.exists(temp_file):
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            try:
                os.remove(temp_file)
                print("  Saved to source; temp removed.")
            except OSError:
                print("  Warning: could not delete temp file; remove manually.")
        return
    
    processed_count = 0
    error_count = 0
    skipped_count = 0
    
    tools_to_process = []
    tool_refs = []
    
    for server_name, tools in data.items():
        if not isinstance(tools, list):
            continue
        
        for tool_idx, tool in enumerate(tools):
            if (server_name, tool_idx) in processed_tools:
                continue
            
            if has_category_domain(tool):
                skipped_count += 1
                continue
            
            name = tool.get('name') or ''
            if not name and 'annotations' in tool and isinstance(tool['annotations'], dict):
                name = tool['annotations'].get('title') or ''
            
            description = tool.get('description') or ''
            if not description and 'annotations' in tool and isinstance(tool['annotations'], dict):
                description = tool['annotations'].get('title') or ''
            
            if not name:
                print(f"\n  Warning: {server_name}[{tool_idx}] has no name; skip")
                continue
            
            if not description:
                description = name
            
            tools_to_process.append({'name': name, 'description': description})
            tool_refs.append((server_name, tool_idx, tool))
    
    total_batches = (len(tools_to_process) + CURRENT_BATCH_SIZE - 1) // CURRENT_BATCH_SIZE
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * CURRENT_BATCH_SIZE
        end_idx = min(start_idx + CURRENT_BATCH_SIZE, len(tools_to_process))
        batch_tools = tools_to_process[start_idx:end_idx]
        batch_refs = tool_refs[start_idx:end_idx]
        
        print(f"\n  Batch {batch_idx + 1}/{total_batches} ({len(batch_tools)} tools)...")
        
        results = classify_tools_batch_concurrent(batch_tools)
        
        for ref_idx, ((server_name, tool_idx, tool), (category, domain)) in enumerate(zip(batch_refs, results)):
            if category and domain:
                tool['category'] = category
                tool['domain'] = domain
                processed_count += 1
                
                try:
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"\n  Warning: temp save failed: {e}")
            else:
                error_count += 1
                tool_name = batch_tools[ref_idx]['name']
                print(f"\n  Error: {server_name}[{tool_idx}] ({tool_name}) failed")
        
        if batch_idx < total_batches - 1:
            time.sleep(0.5)
    
    print(f"\n  Writing final output...")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
            print("  Temp file removed.")
        except OSError:
            print("  Warning: could not delete temp file; remove manually.")
    
    print(f"  Done: processed {processed_count} tools, skipped {skipped_count} already filled, errors {error_count}")

def main():
    parser = argparse.ArgumentParser(description='Add category and domain fields to tools')
    parser.add_argument('--mode', type=str, default='api', choices=['api', 'server'],
                        help='Mode: api (SiliconFlow) or server (local vLLM)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help=f'Batch size / concurrent requests (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--think', type=str, default='true',
                        help='Enable thinking (default: true; pass false to disable)')
    args = parser.parse_args()
    
    enable_think = args.think.lower() in ['true', '1', 'yes', 'on']
    
    setup_api_config(mode=args.mode, batch_size=args.batch_size, enable_think=enable_think)
    
    apis_dir = "/home/yijuan_liang/10.12Tool_Set/tool_set/apis/apis_cosdup"
    
    print("=" * 60)
    print("Add category/domain to tools under apis_cosdup")
    print("=" * 60)
    print(f"\nWorking directory: {apis_dir}")
    print(f"Mode: {args.mode}")
    
    json_files = glob.glob(os.path.join(apis_dir, '*.json'))
    json_files = [f for f in json_files if os.path.basename(f) not in SKIP_FILES]
    
    if not json_files:
        print("No JSON files to process")
        return
    
    print(f"\nFiles to process: {len(json_files)}")
    for f in json_files:
        print(f"  - {os.path.basename(f)}")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if is_standard_format(data):
                process_standard_format_file(file_path)
            else:
                process_server_format_file(file_path)
        except Exception as e:
            print(f"\nError on {os.path.basename(file_path)}: {str(e)[:200]}")
            continue
    
    print("\n" + "=" * 60)
    print("All files done.")
    print("=" * 60)

if __name__ == "__main__":
    main()
