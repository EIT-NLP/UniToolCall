#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import sys
import time
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import numpy as np

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mcp_generation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
from uni_toolcall.secrets import get_openai_compatible_key

# SiliconFlow OpenAI-compatible API (keys: SILICONFLOW_API_KEY / OPENAI_API_KEY)
CHAT_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
CHAT_MODEL = "Qwen/Qwen3-8B"

# File paths (relative to cwd when run)
CORPUS_PATH = Path("apis/corpus_9.15.json")
PROMPT_PATH = Path("prompt/tool_set_prompt.md")
OUTPUT_PATH = Path("apis/corpus_tool_set.json")  # v1
# Versioned output (internal): legacy naming corpus_tool_set_vN.json
OUTPUT_BASENAME = "apis/corpus_tool_set"

# API pacing
API_DELAY = 2  # seconds between calls
MAX_RETRIES = 3
RETRY_DELAY = 5

# Full domain label space (for inverse-frequency sampling)
ALL_DOMAINS = [
    'finance', 'technology', 'education', 'healthcare', 'entertainment',
    'travel', 'business', 'lifestyle', 'science', 'social',
    'sports', 'environment', 'culture'
]

# Full category label space
CATEGORIES = [
    'analysis',
    'operations',
    'system',
    'visualization',
    'search',
    'generate'
]


def load_corpus_data(corpus_path: Path) -> Dict[str, Any]:
    """Load corpus JSON."""
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded corpus: {len(data)} tools")
        return data
    except Exception as e:
        logger.error(f"Failed to load corpus: {e}")
        raise


def load_json_map(path: Path) -> Dict[str, Any]:
    """Load a tool map JSON; return {} if missing."""
    if not path.exists():
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return {}


def get_versioned_paths() -> List[Path]:
    """Ordered paths: v1, v2, ... (v1 = OUTPUT_PATH, vN = corpus_tool_set_vN.json)."""
    paths: List[Path] = [OUTPUT_PATH]
    for v in range(2, 10):
        paths.append(Path(f"{OUTPUT_BASENAME}_v{v}.json"))
    return paths


def count_tools(tools_map: Dict[str, Any]) -> int:
    return len(tools_map)


def choose_save_target() -> Path:
    """Always append to corpus_tool_set.json."""
    return OUTPUT_PATH


def build_reference_pool() -> List[Dict[str, Any]]:
    """Reference pool: corpus_9.15 + corpus_tool_set."""
    pool: List[Dict[str, Any]] = []
    
    base = load_json_map(CORPUS_PATH)
    pool.extend(base.values())
    
    main_tools = load_json_map(OUTPUT_PATH)
    pool.extend(main_tools.values())
    
    return pool


def aggregate_all_tools_for_stats() -> List[Dict[str, Any]]:
    """All tools for label frequency stats."""
    all_tools: List[Dict[str, Any]] = []
    
    base_tools = load_json_map(CORPUS_PATH)
    all_tools.extend(base_tools.values())
    
    main_tools = load_json_map(OUTPUT_PATH)
    all_tools.extend(main_tools.values())
    
    return all_tools


def compute_inverse_frequency_weights(labels: List[str], universe: List[str], alpha: float = 1.0) -> Dict[str, float]:
    """Inverse-frequency weights over universe, smoothed by alpha."""
    from collections import Counter
    c = Counter([l for l in labels if l in universe])
    weights: Dict[str, float] = {}
    for lab in universe:
        count = c.get(lab, 0) + alpha
        weights[lab] = 1.0 / float(count)
    total = sum(weights.values())
    if total <= 0:
        equal = 1.0 / len(universe)
        return {lab: equal for lab in universe}
    return {lab: w / total for lab, w in weights.items()}


def load_prompt_template(prompt_path: Path) -> str:
    """Load prompt markdown."""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            template = f.read()
        logger.info("Loaded prompt template")
        return template
    except Exception as e:
        logger.error(f"Failed to load prompt template: {e}")
        raise


def sample_domain() -> str:
    """Sample domain with inverse frequency (rarer -> higher prob)."""
    all_tools = aggregate_all_tools_for_stats()
    domain_labels: List[str] = []
    for t in all_tools:
        d = t.get('domain')
        if isinstance(d, str) and d in ALL_DOMAINS:
            domain_labels.append(d)
    weights = compute_inverse_frequency_weights(domain_labels, ALL_DOMAINS, alpha=1.0)
    domains = list(weights.keys())
    probabilities = np.array([weights[d] for d in domains], dtype=float)
    sampled_domain = np.random.choice(domains, p=probabilities)
    logger.info(f"Sampled domain: {sampled_domain} | weights: {weights}")
    return sampled_domain


def sample_category() -> str:
    """Sample category with inverse frequency."""
    all_tools = aggregate_all_tools_for_stats()
    cat_labels: List[str] = []
    for t in all_tools:
        c = t.get('category')
        if isinstance(c, str) and c in CATEGORIES:
            cat_labels.append(c)
    weights = compute_inverse_frequency_weights(cat_labels, CATEGORIES, alpha=1.0)
    cats = list(weights.keys())
    probabilities = np.array([weights[c] for c in cats], dtype=float)
    sampled_category = np.random.choice(cats, p=probabilities)
    logger.info(f"Sampled category: {sampled_category} | weights: {weights}")
    return sampled_category


def select_random_tools(count: int = 5) -> List[Dict[str, Any]]:
    """Random sample from reference pool."""
    pool = build_reference_pool()
    if not pool:
        logger.warning("Empty pool; falling back to 9.15 corpus only")
        base = load_json_map(CORPUS_PATH)
        pool = list(base.values())
    selected_tools = random.sample(pool, min(count, len(pool)))
    logger.info(f"Reference tools={len(selected_tools)} (pool size={len(pool)})")
    return selected_tools


def format_tool_info_for_prompt(tools: List[Dict[str, Any]]) -> str:
    """Format selected tools as tool_info block for the prompt."""
    tool_info_lines = []
    
    for i, tool in enumerate(tools, 1):
        tool_info = f"""Tool {i}:
- name: {tool['name']}
- description: {tool['description']}
- category: {tool.get('category', 'unknown')}
- inputSchema: {json.dumps(tool['inputSchema'], ensure_ascii=False, indent=2)}"""
        tool_info_lines.append(tool_info)
    
    return "\n\n".join(tool_info_lines)


def call_chat_api(prompt: str, max_retries: int = MAX_RETRIES) -> Optional[str]:
    """Call SiliconFlow chat/completions to generate MCP tool JSON."""
    key = get_openai_compatible_key()
    if not key:
        logger.error("No API key: set SILICONFLOW_API_KEY or OPENAI_API_KEY")
        return None
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "top_p": 0.95,
    }

    for attempt in range(max_retries):
        try:
            logger.info(f"Calling SiliconFlow API, attempt {attempt + 1}...")
            response = requests.post(CHAT_API_URL, headers=headers, json=payload, timeout=120)
            if response.status_code != 200:
                logger.error(f"API error HTTP {response.status_code}")
                logger.error(f"Body: {response.text[:500]}")
                if attempt < max_retries - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return None

            response_data = response.json()
            if "error" in response_data:
                logger.error(f"API error field: {response_data.get('error')}")
                if attempt < max_retries - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return None

            try:
                text = response_data["choices"][0]["message"]["content"]
                logger.info(f"API OK, response length: {len(text)} chars")
                return text
            except (KeyError, IndexError) as e:
                logger.error(f"Failed to parse API response: {e}")
                logger.error(f"Response body: {response_data}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retry in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                return None

    return None


def parse_generated_tool(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse model output into a corpus tool dict."""
    try:
        cleaned_text = response_text.strip()
        
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith('```'):
            cleaned_text = cleaned_text[3:]
        
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3]
        
        try:
            tool_data = json.loads(cleaned_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if json_match:
                cleaned_text = json_match.group()
                tool_data = json.loads(cleaned_text)
            else:
                raise
        
        if isinstance(tool_data, list):
            if len(tool_data) > 0:
                tool_data = tool_data[0]
            else:
                logger.error("Generated tool list is empty")
                return None
        
        required_fields = ['name', 'display_name', 'description', 'category', 'domain', 'input_schema']
        missing_fields = [field for field in required_fields if field not in tool_data]
        
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            logger.error(f"Got keys: {list(tool_data.keys())}")
            logger.error(f"Raw response: {response_text[:500]}")
            return None
        
        corpus_format = {
            "name": tool_data["name"],
            "description": tool_data["description"],
            "inputSchema": tool_data["input_schema"],
            "category": tool_data["category"],
            "domain": tool_data.get("domain", "general"),
            "version": None,
            "deprecated": False,
            "experimental": False,
            "tags": None
        }
        
        logger.info(f"Parsed tool: {tool_data['name']}")
        return corpus_format
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed: {e}")
        logger.error(f"Cleaned text: {cleaned_text[:500]}")
        logger.error(f"Raw response: {response_text[:500]}")
        return None
    except Exception as e:
        logger.error(f"Parse error: {e}")
        logger.error(f"Raw response: {response_text[:500]}")
        return None


def load_existing_tools(output_path: Path) -> Dict[str, Any]:
    """Load existing output map."""
    if not output_path.exists():
        return {}
    
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded existing tools: {len(data)}")
        return data
    except Exception as e:
        logger.error(f"Failed to load existing tools: {e}")
        return {}


def check_tool_duplicate(new_tool: Dict[str, Any], existing_maps: List[Dict[str, Any]]) -> bool:
    """True if duplicate vs corpus_9.15 + corpus_tool_set."""
    new_name = new_tool['name']
    new_description = new_tool['description'].lower()

    def iter_all():
        yield from load_json_map(CORPUS_PATH).items()
        yield from load_json_map(OUTPUT_PATH).items()

    for tool_id, existing_tool in iter_all():
        existing_name = existing_tool['name']
        existing_description = existing_tool['description'].lower()
        
        if new_name == existing_name:
            return True
        
        new_keywords = set(new_description.split())
        existing_keywords = set(existing_description.split())
        
        overlap = len(new_keywords & existing_keywords)
        total_unique = len(new_keywords | existing_keywords)
        
        if total_unique > 0 and overlap / total_unique > 0.6:
            return True
    
    return False


def save_tool_incrementally(tool_data: Dict[str, Any]) -> bool:
    """Append one tool to corpus_tool_set.json if not duplicate."""
    try:
        target_path = OUTPUT_PATH

        if check_tool_duplicate(tool_data, []):
            logger.warning(f"Duplicate tool {tool_data['name']}, skip save")
            return False
        
        existing_data = load_json_map(target_path)
        
        if existing_data:
            max_id = max(int(k) for k in existing_data.keys() if k.isdigit())
            new_id = str(max_id + 1)
        else:
            new_id = "1"
        
        existing_data[new_id] = tool_data
        
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(target_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved tool {tool_data['name']}, id {new_id} -> {target_path}")
        return True
        
    except Exception as e:
        logger.error(f"Save failed: {e}")
        return False


def generate_single_tool(prompt_template: str) -> Optional[Dict[str, Any]]:
    """Generate one MCP tool via chat API."""
    target_domain = sample_domain()
    target_category = sample_category()
    
    selected_tools = select_random_tools(5)
    
    tool_info = format_tool_info_for_prompt(selected_tools)
    
    full_prompt = prompt_template.replace("{tool_info}", tool_info)
    full_prompt = full_prompt.replace("{target_domain}", target_domain)
    full_prompt = full_prompt.replace("{target_category}", target_category)
    
    response_text = call_chat_api(full_prompt)
    if not response_text:
        return None
    
    tool_data = parse_generated_tool(response_text)
    
    if tool_data:
        tool_data['domain'] = target_domain
        tool_data['category'] = target_category
        logger.info(f"Tool {tool_data['name']} domain={target_domain}, category={target_category}")
    
    return tool_data


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Generate MCP-style tools via SiliconFlow API")
    parser.add_argument("num", type=int, help="Number of tools to generate")
    
    args = parser.parse_args()
    
    corpus_data = load_corpus_data(CORPUS_PATH)
    prompt_template = load_prompt_template(PROMPT_PATH)
    
    success_count = 0
    failure_count = 0
    
    logger.info(f"Generating {args.num} MCP tool(s)...")
    
    for i in range(args.num):
        logger.info(f"\n=== Tool {i+1}/{args.num} ===")
        
        try:
            tool_data = generate_single_tool(prompt_template)
            
            if tool_data:
                if save_tool_incrementally(tool_data):
                    success_count += 1
                    logger.info(f"OK: tool {i+1} saved")
                else:
                    failure_count += 1
                    logger.error(f"Fail: tool {i+1} save skipped or duplicate")
            else:
                failure_count += 1
                logger.error(f"Fail: tool {i+1} generation")
            
            if i < args.num - 1:
                logger.info(f"Sleep {API_DELAY}s...")
                time.sleep(API_DELAY)
                
        except Exception as e:
            failure_count += 1
            logger.error(f"Exception on tool {i+1}: {e}")
    
    total = success_count + failure_count
    logger.info(f"\n=== Done ===")
    logger.info(f"Saved: {success_count}")
    logger.info(f"Failed: {failure_count}")
    if total > 0:
        logger.info(f"Success rate: {success_count/total*100:.1f}%")
    v_paths = get_versioned_paths()
    sizes = {str(p): count_tools(load_json_map(p)) for p in v_paths if p.exists()}
    logger.info(f"Per-version counts: {sizes}")
    logger.info(f"Reference pool size: {len(build_reference_pool())}")


if __name__ == "__main__":
    main()
