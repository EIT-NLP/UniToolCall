#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import time
import logging
import os
import re
import uuid
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generate_single_hop.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
from uni_toolcall.paths import DEFAULT_SYSTEM_PROMPT_MD
from uni_toolcall.prompts import read_system_prompt
from uni_toolcall.secrets import get_openai_compatible_key

# API: keys from SILICONFLOW_API_KEY / OPENAI_API_KEY
API1_URL = "https://api.siliconflow.cn/v1/chat/completions"
API1_MODEL = "Qwen/Qwen3-32B"  # Default model

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]  # UniToolCall repo root
TOOLS_PATH = BASE_DIR / "tool_set" / "apis" / "toolset.json"
PROMPT_PATH = BASE_DIR / "single-hop_pipeline" / "prompts" / "data_set_prompt.md"
OUTPUT_PATH = BASE_DIR / "single-hop_pipeline" / "data" / "2.8_data_set_2.json"
QUERY_EVAL_PROMPT_PATH = BASE_DIR / "single-hop_pipeline" / "prompts" / "prompt_query_eval.md"
TRAJECTORY_EVAL_PROMPT_PATH = BASE_DIR / "single-hop_pipeline" / "prompts" / "prompt_trajectory_eval.md"

# Fixed text prefix per category
CATEGORY_TEXT_PREFIXES = {
    'analysis': 'Analysis completed',
    'operations': 'Operations completed', 
    'system': 'System processing completed',
    'visualization': 'Data visualization completed',
    'search': 'Search results completed',
    'generate': 'Content generation completed'
}

API_DELAY = 1.5
MAX_RETRIES = 3
RETRY_DELAY = 5
SAMPLE_MAX_RETRIES = 3  # Per-sample max retries
SAMPLE_RETRY_DELAY = 3  # Delay between retries (seconds)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_prompt(path: Path) -> str:
    """Load prompt file"""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def call_api(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.5,
    top_p: float = 0.95,
    max_retries: int = MAX_RETRIES
) -> Optional[str]:
    """Call SiliconFlow OpenAI-compatible chat/completions API."""
    api_key = get_openai_compatible_key()
    if not api_key:
        logger.error("No API key: set SILICONFLOW_API_KEY or OPENAI_API_KEY")
        return None
    api_url = API1_URL
    model_name = model if model else API1_MODEL
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }

    # HTTP request
    for attempt in range(max_retries):
        try:
            logger.info(f"Calling SiliconFlow API, attempt {attempt + 1}...")
            response = requests.post(api_url, json=data, headers=headers, timeout=60)
            
            response.raise_for_status()
            result = response.json()
            
            # Error payload may appear even when HTTP 200
            if "error" in result:
                error_msg = result.get("error", {})
                error_code = error_msg.get("code", "")
                error_message = error_msg.get("message", "")
                error_type = error_msg.get("type", "")
                
                # Insufficient balance?
                error_message_lower = error_message.lower() if error_message else ""
                is_insufficient_balance = (
                    "\u4f59\u989d" in error_message or
                    "balance" in error_message_lower or
                    "insufficient" in error_message_lower or
                    "quota" in error_message_lower or
                    "credits" in error_message_lower or
                    "\u4f59\u989d\u4e0d\u8db3" in error_message or
                    "\u8d26\u6237\u4f59\u989d" in error_message or
                    "\u8d26\u6237\u4f59\u989d\u4e0d\u8db3" in error_message or
                    error_code in ["insufficient_balance", "quota_exceeded", "payment_required"]
                )
                
                if is_insufficient_balance:
                    logger.error(f"Insufficient API balance, stopping: {error_message} (code: {error_code}, type: {error_type})")
                    logger.error("Add credits and rerun")
                    sys.exit(1)  # stop immediately
                
                # Auth error: do not retry
                if "\u4ee4\u724c" in error_message or "token" in error_message_lower or "unauthorized" in error_message_lower or "invalid" in error_message_lower or "api key" in error_message_lower:
                    logger.error(f"API auth failed (invalid or expired token): {error_message} (code: {error_code}, type: {error_type})")
                    return None  # No retry on auth error
                else:
                    # Other errors: log and retry
                    if attempt < max_retries - 1:
                        logger.warning(f"API error; retry {attempt + 1}/{max_retries}: {error_message} (code: {error_code})")
                        time.sleep(RETRY_DELAY)
                        continue
                    logger.error(f"API error: {error_message} (code: {error_code}, type: {error_type})")
                    return None
            
            # OpenAI-compatible response
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    logger.debug(f"API OK; content length: {len(content)}  chars")
                    return content
                logger.error("Bad response: missing message.content")
                if attempt < max_retries - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return None
            logger.error(f"Bad response: {json.dumps(result, ensure_ascii=False)[:500]}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
                continue
            return None
                    
        except requests.exceptions.Timeout as e:
            logger.error(f"Request timeout: {str(e)}")
            if attempt < max_retries - 1:
                logger.warning(f"Timeout; retry {attempt + 1}/{max_retries}")
                time.sleep(RETRY_DELAY)
                continue
            return None
        except requests.exceptions.HTTPError as e:
            # Balance error (402 or body hints)
            error_detail = ""
            error_text = ""
            if hasattr(e.response, 'text'):
                error_text = e.response.text
                error_detail = f", Response body: {error_text[:200]}"
            
            # HTTP status and body
            is_insufficient_balance = False
            if e.response.status_code == 402:  # Payment Required
                is_insufficient_balance = True
            elif error_text:
                error_text_lower = error_text.lower()
                is_insufficient_balance = (
                    "\u4f59\u989d" in error_text or
                    "balance" in error_text_lower or
                    "insufficient" in error_text_lower or
                    "quota" in error_text_lower or
                    "credits" in error_text_lower or
                    "\u4f59\u989d\u4e0d\u8db3" in error_text or
                    "\u8d26\u6237\u4f59\u989d" in error_text or
                    "\u8d26\u6237\u4f59\u989d\u4e0d\u8db3" in error_text
                )
            
            if is_insufficient_balance:
                logger.error(f"Insufficient balance (HTTP {e.response.status_code}), stopping: {str(e)}{error_detail}")
                logger.error("Add credits and rerun")
                sys.exit(1)  # stop immediately
            
            logger.error(f"API request failed: {str(e)}{error_detail}")
            if attempt < max_retries - 1:
                logger.warning(f"HTTP error; retry {attempt + 1}/{max_retries}")
                time.sleep(RETRY_DELAY)
                continue
            return None
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            if attempt < max_retries - 1:
                logger.warning(f"Exception; retry {attempt + 1}/{max_retries}")
                time.sleep(RETRY_DELAY)
                continue
            return None
    
    return None


def fix_json_string(s: str) -> str:
    """Fix common JSON string issues."""
    # Known fixes
    s = s.replace('"New anti-cancer drug "', '"New anti-cancer drug"')
    s = s.replace('"PBS (pH) 7.4)"', '"PBS (pH 7.4)"')
    
    # Quote issues inside values
    def fix_quotes_in_value(match):
        key = match.group(1)
        value = match.group(2)
        # Escape inner quotes
        escaped_value = value.replace('"', '\\"')
        return f'"{key}": "{escaped_value}"'
    
    # Regex quote fix
    s = re.sub(r'"([^"]+)":\s*"([^"]*"[^"]*)"', fix_quotes_in_value, s)
    
    return s


def validate_and_fix_json(json_str: str) -> Optional[str]:
    """Validate and fix JSON string."""
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        # Try common fixes
        fixed = fix_json_string(json_str)
        try:
            json.loads(fixed)
            return fixed
        except:
            return None


def parse_model_output_to_segments(raw_text: str) -> Optional[Dict[str, str]]:
    """Parse model output into four fields."""
    cleaned = raw_text.strip()
    if cleaned.startswith('```json'):
        cleaned = cleaned[7:]
    elif cleaned.startswith('```'):
        cleaned = cleaned[3:]
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3]

    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        # Extract outer JSON object
        m = re.search(r'\{[\s\S]*\}$', cleaned)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None

    required_keys = ["human_value", "function_call_value", "observation_value", "gpt_value"]
    for k in required_keys:
        if k not in obj or not isinstance(obj[k], str):
            logger.error(f"Model output missing or wrong type: {k}")
            return None
    
    # Validate nested JSON strings
    function_call_fixed = validate_and_fix_json(obj["function_call_value"])
    observation_fixed = validate_and_fix_json(obj["observation_value"])
    
    if not function_call_fixed:
        logger.error("Could not fix function_call_value JSON")
        logger.error(f"Raw function_call_value: {obj['function_call_value'][:200]}...")
        return None
    
    if not observation_fixed:
        logger.error("Could not fix observation_value JSON")
        logger.error(f"Raw observation_value: {obj['observation_value'][:200]}...")
        return None
    
    if function_call_fixed != obj["function_call_value"]:
        logger.info("Fixed function_call_value JSON")
    
    if observation_fixed != obj["observation_value"]:
        logger.info("Fixed observation_value JSON")
    
    return {
        "human_value": obj["human_value"],
        "function_call_value": function_call_fixed,
        "observation_value": observation_fixed,
        "gpt_value": obj["gpt_value"],
    }


def format_tool_to_json_string(tool_obj: Dict[str, Any]) -> str:
    """Serialize tool to JSON string for tools field."""
    tool_info = {
        "name": tool_obj.get("name", ""),
        "description": tool_obj.get("description", ""),
        "category": tool_obj.get("category", ""),
        "domain": tool_obj.get("domain", ""),
        "inputSchema": tool_obj.get("inputSchema", {})
    }
    # JSON array of tools
    tools_array = [tool_info]
    return json.dumps(tools_array, ensure_ascii=False)


def build_prompt(template: str, tool_json_obj: Dict[str, Any]) -> str:
    """Build user prompt from template."""
    tool_json_str = json.dumps(tool_json_obj, ensure_ascii=False, indent=2)
    
    # Category -> text prefix
    tool_category = tool_json_obj.get('category', 'analysis')  # default analysis
    text_prefix = CATEGORY_TEXT_PREFIXES.get(tool_category, 'Processing completed')  # fallback
    
    # Template placeholders
    prompt = template.replace("{tool_json}", tool_json_str)
    prompt = prompt.replace("{text_prefix}", text_prefix)
    
    return prompt


def generate_for_tool(
    tool: Dict[str, Any], 
    prompt_template: str,
    system_prompt: str,
    model: str = None,
    temperature: float = 0.5,
    top_p: float = 0.95
) -> Optional[Dict[str, Any]]:
    """Generate one sample for a tool."""
    prompt = build_prompt(prompt_template, tool)
    
    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # Call API
    text = call_api(messages, model=model, temperature=temperature, top_p=top_p)
    if not text:
        return None
    
    # Parse output
    segs = parse_model_output_to_segments(text)
    if not segs:
        return None
    
    return segs


def extract_json_object(text: str) -> dict:
    """Extract JSON object from model text."""
    cleaned = re.sub(r"```[a-zA-Z]*", "", text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        segment = cleaned[start : end + 1].strip()
        try:
            return json.loads(segment)
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return {}


def evaluate_query(
    query_text: str,
    tools_context: str,
    model: str = None,
    timeout: int = 120
) -> dict:
    """Evaluate query (toolfit, clarity, naturalness)."""
    query_eval_prompt_template = load_prompt(QUERY_EVAL_PROMPT_PATH)
    prompt = (
        query_eval_prompt_template.replace("{tools_context}", tools_context)
           .replace("{query_text}", query_text)
    )
    
    messages = [{"role": "user", "content": prompt}]
    response = call_api(messages, model=model, temperature=0.3, top_p=0.95, max_retries=2)
    if not response:
        logger.warning("Query eval API call failed")
        return {}
    
    return extract_json_object(response)


def evaluate_trajectory(
    query_text: str,
    function_call_value: str,
    observation_value: str,
    gpt_value: str,
    tools_context: str,
    model: str = None,
    timeout: int = 120
) -> dict:
    """Evaluate trajectory (success, grounding, efficiency)."""
    trajectory_eval_prompt_template = load_prompt(TRAJECTORY_EVAL_PROMPT_PATH)
    prompt = (
        trajectory_eval_prompt_template.replace("{tools_context}", tools_context)
           .replace("{query_text}", query_text)
           .replace("{function_call_value}", function_call_value)
           .replace("{observation_value}", observation_value)
           .replace("{gpt_value}", gpt_value)
    )
    
    messages = [{"role": "user", "content": prompt}]
    response = call_api(messages, model=model, temperature=0.3, top_p=0.95, max_retries=2)
    if not response:
        logger.warning("Trajectory eval API call failed")
        return {}
    
    return extract_json_object(response)


def extract_scores(query_eval: dict, traj_eval: dict) -> dict:
    """Extract six scores and compute average/min."""
    def _num(x):
        try:
            return float(x)
        except Exception:
            return None
    
    # Query scores
    toolfit = _num((query_eval.get("toolfit") or {}).get("score"))
    clarity = _num((query_eval.get("clarity") or {}).get("score"))
    naturalness = _num((query_eval.get("naturalness") or {}).get("score"))
    
    # Trajectory scores
    success = _num((traj_eval.get("success") or {}).get("score"))
    grounding = _num((traj_eval.get("grounding") or {}).get("score"))
    efficiency = _num((traj_eval.get("efficiency") or {}).get("score"))
    
    scores = {
        "toolfit": toolfit,
        "clarity": clarity,
        "naturalness": naturalness,
        "success": success,
        "grounding": grounding,
        "efficiency": efficiency,
    }
    
    # Average over non-None scores
    valid_scores = [s for s in scores.values() if s is not None]
    average = sum(valid_scores) / len(valid_scores) if valid_scores else None
    scores["average"] = average
    
    # Min score
    min_score = min(valid_scores) if valid_scores else None
    scores["min_score"] = min_score
    
    return scores


def check_quality_threshold(scores: dict, min_threshold: float = 4.0, avg_threshold: float = 8.0) -> tuple:
    """
    Check sample against quality thresholds.
    Returns: (passed: bool, reason: str)
    """
    min_score = scores.get("min_score")
    average = scores.get("average")
    
    if min_score is None or average is None:
        return False, "Missing scores"
    
    if min_score < min_threshold:
        return False, f"Min score ({min_score}) < {min_threshold}"
    
    if average < avg_threshold:
        return False, f"Average score ({average:.2f}) < {avg_threshold}"
    
    return True, "Passed all thresholds"


def main():
    parser = argparse.ArgumentParser(description="Generate single-hop dataset")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default: SiliconFlow Qwen/Qwen3-32B)")
    parser.add_argument("--max", type=int, default=20,
                        help="With --tool_id: how many samples for that tool; else ignored (one per tool)")
    parser.add_argument("--total", type=int, default=None,
                        help="Random mode: total samples (use with --random)")
    parser.add_argument("--random", action="store_true",
                        help="Random tool selection (use with --total)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip first N tools (non-random mode)")
    parser.add_argument("--tool_id", type=str, default=None,
                        help="Only generate for this tool id; use with --max for count")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="temperature (default: 0.5)")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="top_p (default: 0.95)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: OUTPUT_PATH)")
    parser.add_argument("--quality-check", action="store_true",
                        help="Run quality evaluation; only passing samples are saved")
    parser.add_argument("--quality-model", type=str, default=None,
                        help="Model for quality eval (default: same as --model)")
    parser.add_argument("--min-threshold", type=float, default=4.0,
                        help="Min score threshold (default: 4.0)")
    parser.add_argument("--avg-threshold", type=float, default=8.0,
                        help="Average score threshold (default: 8.0)")
    parser.add_argument("--quality-timeout", type=int, default=120,
                        help="Quality eval timeout seconds (default: 120)")
    args = parser.parse_args()

    # Validate args
    if args.random and args.total is None:
        parser.error("--random requires --total")
    if args.total is not None and args.total <= 0:
        parser.error("--total must be > 0")
    if args.random and args.tool_id is not None:
        parser.error("--random cannot be used with --tool_id")

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_PATH
    
    # Ensure output dir
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load tools and prompts
    logger.info(f"Loading tools: {TOOLS_PATH}")
    tools_map = load_json(TOOLS_PATH)
    
    logger.info(f"Loading prompt template: {PROMPT_PATH}")
    prompt_template = load_prompt(PROMPT_PATH)
    
    logger.info(f"Loading system prompt: {DEFAULT_SYSTEM_PROMPT_MD}")
    system_prompt = read_system_prompt()
    
    # If quality check: load eval prompts
    if args.quality_check:
        logger.info("Quality evaluation enabled")
        logger.info(f"  Loading query eval prompt: {QUERY_EVAL_PROMPT_PATH}")
        if not QUERY_EVAL_PROMPT_PATH.exists():
            raise FileNotFoundError(f"Query eval prompt not found: {QUERY_EVAL_PROMPT_PATH}")
        logger.info(f"  Loading trajectory eval prompt: {TRAJECTORY_EVAL_PROMPT_PATH}")
        if not TRAJECTORY_EVAL_PROMPT_PATH.exists():
            raise FileNotFoundError(f"Trajectory eval prompt not found: {TRAJECTORY_EVAL_PROMPT_PATH}")
        logger.info(f"  Quality thresholds: min_score >= {args.min_threshold}, average >= {args.avg_threshold}")
    # Random seed
    if args.seed is not None:
        random.seed(args.seed)
        logger.info(f"Random seed: {args.seed}")

    # Append to existing JSON
    existing_items: List[Dict[str, Any]] = []
    if output_path.exists():
        try:
            existing_items = load_json(output_path)  # type: ignore
            if not isinstance(existing_items, list):
                logger.warning("Existing output is not a list; resetting to []")
                existing_items = []
        except Exception as e:
            logger.warning(f"Could not read existing output; starting fresh: {e}")
            existing_items = []

    new_items: List[Dict[str, Any]] = []

    # Numeric conversation_id start
    def compute_start_conversation_id(items: List[Dict[str, Any]]) -> int:
        max_id = 0
        for it in items:
            cid = it.get("conversation_id")
            try:
                # Numeric ids only
                if isinstance(cid, int):
                    max_id = max(max_id, cid)
                elif isinstance(cid, str) and cid.isdigit():
                    max_id = max(max_id, int(cid))
            except Exception:
                continue
        return max_id + 1

    def append_one_sample(tool_id: str, tool_obj: Dict[str, Any], conversation_id_value: Optional[Any] = None):
        """Generate one sample and append (optional quality eval)."""
        result: Optional[Dict[str, Any]] = None
        for attempt in range(1, SAMPLE_MAX_RETRIES + 1):
            result = generate_for_tool(
                tool_obj, 
                prompt_template,
                system_prompt,
                model=args.model,
                temperature=args.temperature,
                top_p=args.top_p
            )
            if result:
                break
            logger.warning(f"Sample generation failed (attempt {attempt}/{SAMPLE_MAX_RETRIES}), {('retrying soon' if attempt < SAMPLE_MAX_RETRIES else 'max retries reached, skipping')}...")
            if attempt < SAMPLE_MAX_RETRIES:
                time.sleep(SAMPLE_RETRY_DELAY)
        
        if not result:
            return False
        
        # Quality eval
        if args.quality_check:
            logger.info("  Running quality evaluation...")
            tools_json_string = format_tool_to_json_string(tool_obj)
            query_text = result["human_value"]
            
            # Eval query
            logger.info("    Evaluating query...")
            query_eval = evaluate_query(
                query_text,
                tools_json_string,
                model=args.quality_model,
                timeout=args.quality_timeout
            )
            
            # Eval trajectory
            logger.info("    Evaluating trajectory...")
            traj_eval = evaluate_trajectory(
                query_text,
                result["function_call_value"],
                result["observation_value"],
                result["gpt_value"],
                tools_json_string,
                model=args.quality_model,
                timeout=args.quality_timeout
            )
            
            # Scores
            scores = extract_scores(query_eval, traj_eval)
            
            # Thresholds
            passed, reason = check_quality_threshold(scores, args.min_threshold, args.avg_threshold)
            
            if not passed:
                logger.warning(f"  Quality check failed: {reason}")
                average_val = scores.get('average')
                average_str = f"{average_val:.2f}" if average_val is not None else None
                logger.info(f"    Score detail: toolfit={scores.get('toolfit')}, clarity={scores.get('clarity')}, naturalness={scores.get('naturalness')}, success={scores.get('success')}, grounding={scores.get('grounding')}, efficiency={scores.get('efficiency')}, average={average_str}, min_score={scores.get('min_score')}")
                return False
            
            logger.info(f"  Quality check passed: {reason}")
            logger.info(f"    Scores: average={scores.get('average'):.2f}, min_score={scores.get('min_score')}")
        
        conversation = [
            {"from": "human", "value": result["human_value"]},
            {"from": "function_call", "value": result["function_call_value"]},
            {"from": "observation", "value": result["observation_value"]},
            {"from": "gpt", "value": result["gpt_value"]},
        ]
        
        # conversation_id format
        if conversation_id_value is None:
            conv_id = f"{tool_id}-{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"
        else:
            conv_id = conversation_id_value

        # tools JSON string
        tools_json_string = format_tool_to_json_string(tool_obj)

        item = {
            "conversation_id": conv_id,
            "tool_id": tool_id,
            "tool_name": tool_obj.get("name"),
            "conversations": conversation,
            "system": system_prompt,
            "tools": tools_json_string,
        }
        
        # Attach quality block
        if args.quality_check:
            item["quality_evaluation"] = {
                "query_evaluation": query_eval,
                "trajectory_evaluation": traj_eval,
                "scores": scores,
                "quality_check": {"passed": passed, "reason": reason}
            }
        
        # Flush per sample
        existing_items.append(item)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(existing_items, f, ensure_ascii=False, indent=2)
        logger.info(f"Appended 1 sample to {output_path} (total {len(existing_items)})")
        return True

    if args.tool_id is not None:
        # Multiple samples for one tool
        tid = str(args.tool_id)
        if tid not in tools_map:
            raise ValueError(f"Tool id not found: {tid}")
        tool = tools_map[tid]
        logger.info(f"Generating {args.max} samples for {tool.get('name')} (ID={tid})")
        for i in range(args.max):
            ok = append_one_sample(tid, tool)
            if i < args.max - 1:
                time.sleep(API_DELAY)
    elif args.random and args.total is not None:
        # Random mode
        tool_ids = [k for k in tools_map.keys() if k.isdigit()]
        
        if not tool_ids:
            logger.error("No numeric tool ids in toolset")
            return
        
        # Resume count
        existing_count = len(existing_items)
        logger.info(f"Random mode: target {args.total}, existing {existing_count}")
        
        if existing_count >= args.total:
            logger.info(f"Already have {existing_count} >= target {args.total}; nothing to do")
            return
        
        # Track used tool ids
        existing_tool_ids = set()
        for item in existing_items:
            tid = item.get("tool_id")
            if tid and isinstance(tid, str) and tid.isdigit():
                existing_tool_ids.add(tid)
        
        # Available ids
        available_tool_ids = [tid for tid in tool_ids if tid not in existing_tool_ids]
        
        logger.info(f"Existing data uses {len(existing_tool_ids)} tools; {len(available_tool_ids)} remaining")
        
        # Remaining
        remaining_count = args.total - existing_count
        logger.info(f"Need {remaining_count} more samples")
        
        # Sample tool ids
        selected_tool_ids = []
        if remaining_count <= len(available_tool_ids):
            # Enough unique tools
            selected_tool_ids = random.sample(available_tool_ids, remaining_count)
            logger.info(f"Randomly selected {len(selected_tool_ids)} unique tools from {len(available_tool_ids)} available")
        else:
            # Need duplicates
            # Shuffle
            shuffled_tools = available_tool_ids.copy()
            random.shuffle(shuffled_tools)
            # Take all unique first
            selected_tool_ids.extend(shuffled_tools)
            # Fill with repeats
            remaining = remaining_count - len(available_tool_ids)
            for _ in range(remaining):
                selected_tool_ids.append(random.choice(tool_ids))
            logger.info(f"Randomly selected {len(selected_tool_ids)} tools (first {len(available_tool_ids)} unique, then {remaining} may repeat)")
        
        next_conv_id = compute_start_conversation_id(existing_items)
        for idx, tid in enumerate(selected_tool_ids, 1):
            tool = tools_map[tid]
            current_total = existing_count + idx
            logger.info(f"Random mode: item {idx}/{remaining_count} (overall {current_total}/{args.total}), tool: {tool.get('name')} (ID={tid})")
            ok = append_one_sample(tid, tool, conversation_id_value=next_conv_id)
            next_conv_id += 1
            if idx < remaining_count:
                time.sleep(API_DELAY)
    else:
        # One sample per remaining tool
        tool_ids = sorted([k for k in tools_map.keys() if k.isdigit()], key=lambda x: int(x))
        
        def extract_existing_tool_names(items: List[Dict[str, Any]]) -> set:
            """Collect tool names already in output."""
            names = set()
            for it in items:
                n = it.get("tool_name")
                if isinstance(n, str) and n:
                    names.add(n)
                    continue
                try:
                    convs = it.get("conversations", [])
                    if isinstance(convs, list):
                        for seg in convs:
                            if seg.get("from") == "function_call" and isinstance(seg.get("value"), str):
                                try:
                                    fc = json.loads(seg["value"])
                                    if isinstance(fc, dict):
                                        fn = fc.get("name")
                                        if isinstance(fn, str) and fn:
                                            names.add(fn)
                                except Exception:
                                    pass
                except Exception:
                    continue
            return names

        existing_names = extract_existing_tool_names(existing_items)
        remaining_tool_ids = []
        for tid in tool_ids:
            t = tools_map[tid]
            tname = t.get("name")
            if isinstance(tname, str) and tname in existing_names:
                continue
            remaining_tool_ids.append(tid)

        if not remaining_tool_ids:
            logger.info("All tools already have samples; nothing to add.")
            return
        
        # --skip
        if args.skip > 0:
            remaining_tool_ids = remaining_tool_ids[args.skip:]
            logger.info(f"Skipped first {args.skip} tools; {len(remaining_tool_ids)} left")
        
        next_conv_id = compute_start_conversation_id(existing_items)
        for idx, tid in enumerate(remaining_tool_ids, 1):
            tool = tools_map[tid]
            logger.info(f"Remaining-tools mode: {idx}/{len(remaining_tool_ids)} tool: {tool.get('name')} (ID={tid})")
            ok = append_one_sample(tid, tool, conversation_id_value=next_conv_id)
            next_conv_id += 1
            if idx < len(remaining_tool_ids):
                time.sleep(API_DELAY)

    # Summary
    logger.info(f"Done. {len(existing_items)} samples in file.")


if __name__ == "__main__":
    main()
