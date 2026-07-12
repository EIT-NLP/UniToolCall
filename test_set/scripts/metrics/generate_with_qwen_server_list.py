#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
import re
import time
import requests
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import glob
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

_UT_REPO = Path(__file__).resolve().parents[3]
if str(_UT_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_UT_REPO / "src"))
from uni_toolcall.prompts import read_system_prompt
from uni_toolcall.secrets import (
    get_anthropic_key,
    get_google_api_key,
    get_openai_compatible_key,
    get_openai_official_key,
)

# API endpoints (keys via env only; see repo README)
API1_URL = "https://api.siliconflow.cn/v1/chat/completions"
API1_MODEL = "Qwen/Qwen3-32B"  # api1 default (SiliconFlow)
API2_URL = "https://api.openai.com/v1/chat/completions"
API2_MODEL = "gpt-4o-mini"  # api2 OpenAI Chat Completions
API3_URL = "https://api.anthropic.com/v1/messages"
API3_MODEL = "claude-sonnet-4-20250514"  # api3 Anthropic Messages API
API4_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
API4_MODEL = None  # api4 Gemini: pass model via --model
SERVER_URL = "http://localhost:8007/v1/chat/completions"
SERVER_MODEL = "/data/models/Qwen3-32B"
SERVER_MODEL_NAME = "Qwen3-32B"
SFT_SERVER_URL = "http://localhost:8002/v1/chat/completions"
SFT_MODEL_NAME = "my_lora"

# Logging
def setup_logging(log_file: str):
    """Configure logging (append mode)."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8', mode='a'),
            logging.StreamHandler()
        ],
        force=True
    )
    return logging.getLogger(__name__)

def call_qwen_api(messages: List[Dict[str, str]], temperature: float = 0.6, max_retries: int = 3, thinking: bool = True, mode: str = "api1", port: int = None, model: str = None, top_p: float = 0.95) -> Optional[str]:
    """Call the configured chat API with retries.

    Args:
        mode: ``api1`` SiliconFlow; ``api2`` OpenAI official Chat Completions;
            ``api3`` Anthropic Messages; ``api4`` Gemini; ``server`` / ``sft`` local.
        port: Custom port (only for ``sft`` / ``server`` when not None).
        model: Model id (defaults per mode). If the id contains ``gemini``, mode switches to ``api4``.
        top_p: Top-p sampling (default 0.95).
    """
    # Legacy alias: ``anthropic`` is the same as ``api3``
    if mode == "anthropic":
        mode = "api3"
    # Auto-route Gemini model names to api4
    if model and "gemini" in model.lower():
        mode = "api4"

    if mode == "server":
        if port is not None:
            api_url = f"http://localhost:{port}/v1/chat/completions"
        else:
            api_url = SERVER_URL
        model_name = model if model else SERVER_MODEL_NAME
        headers = {
            "Authorization": "Bearer EMPTY",
            "Content-Type": "application/json"
        }
        proxies = None
    elif mode == "sft":
        if port is not None:
            api_url = f"http://localhost:{port}/v1/chat/completions"
        else:
            api_url = SFT_SERVER_URL
        model_name = model if model else SFT_MODEL_NAME
        headers = {
            "Authorization": "Bearer EMPTY",
            "Content-Type": "application/json"
        }
        proxies = None
    elif mode == "api1":
        api_url = API1_URL
        model_name = model if model else API1_MODEL
        k1 = get_openai_compatible_key()
        if not k1:
            raise ValueError(
                "api1 requires SILICONFLOW_API_KEY or OPENAI_API_KEY"
            )
        headers = {
            "Authorization": f"Bearer {k1}",
            "Content-Type": "application/json"
        }
        proxies = None
    elif mode == "api2":
        api_url = API2_URL
        model_name = model if model else API2_MODEL
        k2 = get_openai_official_key()
        if not k2:
            raise ValueError(
                "api2 (OpenAI Chat Completions) requires OPENAI_API_KEY"
            )
        headers = {
            "Authorization": f"Bearer {k2}",
            "Content-Type": "application/json",
        }
        proxies = None
    elif mode == "api3":
        api_url = API3_URL
        model_name = model if model else API3_MODEL
        ka = get_anthropic_key()
        if not ka:
            raise ValueError(
                "api3 (Anthropic Messages) requires ANTHROPIC_API_KEY"
            )
        headers = {
            "x-api-key": ka,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        proxies = None
    elif mode == "api4":
        model_name = model if model else API4_MODEL
        if model_name is None:
            raise ValueError("api4 (Gemini) requires a model name via --model")
        api4_key = get_google_api_key()
        if not api4_key:
            raise ValueError("api4 requires GOOGLE_API_KEY")
        api_url = API4_URL.replace("{model}", model_name)
        is_gemini_official = "generativelanguage.googleapis.com" in api_url
        if is_gemini_official:
            headers = {
                "x-goog-api-key": api4_key,
                "Content-Type": "application/json"
            }
        else:
            headers = {
                "Authorization": f"Bearer {api4_key}",
                "Content-Type": "application/json"
            }
        proxies = {
            "http": "http://localhost:50000",
            "https": "http://localhost:50000"
        }
    else:
        api_url = API1_URL
        model_name = model if model else API1_MODEL
        k0 = get_openai_compatible_key()
        if not k0:
            raise ValueError(
                "Requires SILICONFLOW_API_KEY or OPENAI_API_KEY"
            )
        headers = {
            "Authorization": f"Bearer {k0}",
            "Content-Type": "application/json"
        }
        is_gemini_official = False
        proxies = None

    # Gemini (official) uses contents + systemInstruction instead of messages
    if mode == "api4" and is_gemini_official:
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = content
                continue

            if role == "assistant":
                role = "model"

            contents.append({
                "role": role,
                "parts": [{"text": content}]
            })

        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p
            }
        }

        if thinking:
            data["generationConfig"]["thinkingConfig"] = {
                "thinkingLevel": "HIGH"
            }
        else:
            data["generationConfig"]["thinkingConfig"] = {
                "thinkingLevel": "MINIMAL"
            }

        if system_instruction:
            data["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
    elif mode == "api3":
        system_text = ""
        anth_msgs: List[Dict[str, str]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_text = content
            elif role in ("user", "assistant"):
                anth_msgs.append({"role": role, "content": content})
        data = {
            "model": model_name,
            "max_tokens": 4096,
            "messages": anth_msgs,
            "temperature": temperature,
            "top_p": top_p,
        }
        if system_text:
            data["system"] = system_text
    else:
        # OpenAI-compatible Chat Completions (api1 / api2 / server / sft)
        data = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        data["top_p"] = top_p

    if mode == "server" or mode == "sft":
        data["chat_template_kwargs"] = {"enable_thinking": thinking}
    elif mode == "api1":
        if model_name:
            model_name_lower = model_name.lower()
            if "qwen" in model_name_lower or "deepseek" in model_name_lower:
                data["enable_thinking"] = thinking
    elif mode == "api2":
        pass

    for attempt in range(max_retries):
        try:
            logging.debug(f"API attempt {attempt + 1}/{max_retries}: URL={api_url}, Model={model_name}, Mode={mode}")
            if attempt == 0:
                logging.info(f"Calling API: {api_url}, model: {model_name}")
                import json as json_module
                request_data_str = json_module.dumps(data, ensure_ascii=False)
                logging.debug(f"Request payload size: {len(request_data_str)} chars")
                if mode == "api4" and is_gemini_official:
                    logging.debug(
                        f"Gemini request: contents={len(data.get('contents', []))}, "
                        f"systemInstruction={'systemInstruction' in data}"
                    )

            if mode == "api4":
                response = requests.post(api_url, json=data, headers=headers, timeout=60, proxies=proxies)
            else:
                response = requests.post(api_url, json=data, headers=headers, timeout=60)
            logging.debug(f"HTTP status: {response.status_code}")
            response.raise_for_status()
            result = response.json()

            if "error" in result:
                error_msg = result.get("error", {})
                error_code = error_msg.get("code", "")
                error_message = error_msg.get("message", "")
                error_type = error_msg.get("type", "")

                if (
                    "\u4ee4\u724c" in error_message
                    or "token" in error_message.lower()
                    or "unauthorized" in error_message.lower()
                    or "invalid" in error_message.lower()
                    or "api key" in error_message.lower()
                ):
                    logging.error(
                        f"API auth failed (invalid or expired token): {error_message} "
                        f"(code: {error_code}, type: {error_type})"
                    )
                    return None
                else:
                    if attempt < max_retries - 1:
                        logging.warning(
                            f"API error, retry {attempt + 1}/{max_retries}: {error_message} (code: {error_code})"
                        )
                        time.sleep(2)
                        continue
                    logging.error(
                        f"API error: {error_message} (code: {error_code}, type: {error_type})"
                    )
                    return None

            if mode == "api4" and is_gemini_official:
                if "candidates" in result and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        parts = candidate["content"]["parts"]
                        if parts and len(parts) > 0 and "text" in parts[0]:
                            content = parts[0]["text"]
                            logging.debug(f"Gemini OK, text length: {len(content)}")
                            return content
                    else:
                        logging.warning(
                            f"Gemini response missing content.parts: {json.dumps(result, ensure_ascii=False)[:500]}"
                        )
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            continue
                        return None
                else:
                    if attempt < max_retries - 1:
                        logging.warning(
                            f"Gemini unexpected response, retry {attempt + 1}/{max_retries}: "
                            f"{json.dumps(result, ensure_ascii=False)[:500]}"
                        )
                        time.sleep(1)
                        continue
                    logging.error(f"Gemini unexpected response: {json.dumps(result, ensure_ascii=False)[:1000]}")
                    return None
            elif mode == "api3":
                if "content" in result and isinstance(result["content"], list) and len(result["content"]) > 0:
                    block = result["content"][0]
                    if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                        content = block["text"]
                        logging.debug(f"Anthropic OK, text length: {len(content)}")
                        return content
                logging.warning(f"Anthropic unexpected response: {json.dumps(result, ensure_ascii=False)[:500]}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return None
            else:
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    if "message" in choice:
                        message = choice["message"]
                        content = None
                        if "reasoning_content" in message and message["reasoning_content"] is not None:
                            content = message["reasoning_content"]
                            logging.debug(f"OK from reasoning_content, length: {len(content)}")
                        elif "content" in message and message["content"] is not None:
                            content = message["content"]
                            logging.debug(f"OK from content, length: {len(content)}")

                        if content is not None:
                            return content
                        else:
                            logging.warning(
                                f"OK HTTP but content and reasoning_content empty: "
                                f"{json.dumps(result, ensure_ascii=False)[:500]}"
                            )
                            if attempt < max_retries - 1:
                                time.sleep(1)
                                continue
                            return None
                    else:
                        logging.warning(
                            f"Missing message in choice: {json.dumps(result, ensure_ascii=False)[:500]}"
                        )
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            continue
                        return None
                else:
                    if attempt < max_retries - 1:
                        logging.warning(
                            f"Unexpected response, retry {attempt + 1}/{max_retries}: "
                            f"{json.dumps(result, ensure_ascii=False)[:500]}"
                        )
                        time.sleep(1)
                        continue
                    logging.error(f"Unexpected response: {json.dumps(result, ensure_ascii=False)[:1000]}")
                    return None
                
        except requests.exceptions.Timeout as e:
            logging.error(f"API timeout (60s): {str(e)}")
            logging.error(f"URL: {api_url}")
            logging.error(f"Request payload size: {len(str(data))} chars")
            if attempt < max_retries - 1:
                logging.warning(f"Timeout, retry {attempt + 1}/{max_retries}")
                time.sleep(2)
                continue
            return None
        except requests.exceptions.HTTPError as e:
            error_detail = ""
            error_text = ""
            if hasattr(e.response, 'text'):
                error_text = e.response.text
                error_detail = f", body: {error_text[:200]}"

            if e.response.status_code == 429:
                is_tpm_limit = "TPM limit" in error_text or "tpm" in error_text.lower() or "rate limiting" in error_text.lower()
                if attempt < max_retries - 1:
                    retry_delay = 10 * (2 ** attempt)
                    if is_tpm_limit:
                        logging.warning(
                            f"HTTP 429 (TPM/rate limit), retry {attempt + 1}/{max_retries} "
                            f"(sleep {retry_delay}s): {str(e)}{error_detail}"
                        )
                    else:
                        logging.warning(
                            f"HTTP 429 (rate limit), retry {attempt + 1}/{max_retries} "
                            f"(sleep {retry_delay}s): {str(e)}{error_detail}"
                        )
                    time.sleep(retry_delay)
                    continue
                else:
                    if is_tpm_limit:
                        logging.error(f"HTTP 429 TPM/rate limit, max retries: {str(e)}{error_detail}")
                    else:
                        logging.error(f"HTTP 429 rate limit, max retries: {str(e)}{error_detail}")
                    return None

            if e.response.status_code == 503:
                if "No available channels for model" in error_text or "\u5f53\u524d\u5206\u7ec4" in error_text:
                    logging.error(f"Model unavailable, skip retry: {str(e)}{error_detail}")
                    return None
                else:
                    if attempt < max_retries - 1:
                        retry_delay = 5
                        logging.warning(
                            f"HTTP 503, retry {attempt + 1}/{max_retries} (sleep {retry_delay}s): "
                            f"{str(e)}{error_detail}"
                        )
                        time.sleep(retry_delay)
                        continue
                    else:
                        logging.error(f"HTTP 503: {str(e)}{error_detail}")
                        return None

            if attempt < max_retries - 1:
                logging.warning(f"HTTP error, retry {attempt + 1}/{max_retries}: {str(e)}{error_detail}")
                time.sleep(1)
                continue
            logging.error(f"HTTP error: {str(e)}{error_detail}")
            return None
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            if attempt < max_retries - 1:
                logging.warning(f"Request failed, retry {attempt + 1}/{max_retries}: {str(e)}")
                logging.debug(f"Traceback: {error_trace}")
                time.sleep(1)
                continue
            logging.error(f"Request failed: {str(e)}")
            logging.error(f"Traceback: {error_trace}")
            return None
    
    return None

def extract_tool_call_content(text: str) -> Optional[str]:
    """Extract content inside <tool_call>...</tool_call>."""
    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_answer_content(text: str) -> Optional[str]:
    """Extract text inside <answer>...</answer>, or prefix after <answer> if unclosed."""
    # Full <answer>...</answer>
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback: prefix after <answer>
    pattern_fallback = r'<answer>\s*(.*)'
    match_fallback = re.search(pattern_fallback, text, re.DOTALL)
    if match_fallback:
        content = match_fallback.group(1).strip()
        if content:
            return content
    
    return None

def build_messages_for_function_call(
    conversations: List[Dict],
    call_idx: int,
    generated_calls: List[str],
    generated_answers: List[str],
    system_prompt: str,
    tools_formatted: str,
    current_query: str
) -> List[Dict[str, str]]:
    """Build messages for the n-th function call (call_idx=0 is the first).
    Same layout as training: history in <chat_history> inside system.
    """
    messages = []
    
    # system
    system_content = system_prompt
    
    # ，
    if call_idx == 0:
        # toolssystem
        if tools_formatted:
            system_content += f"\n\n<tools>\n{tools_formatted}\n</tools>"
        messages.append({"role": "system", "content": system_content})
        # query
        messages.append({"role": "user", "content": current_query})
        return messages
    
    # n：，<chat_history>，system
    # call_idxfunction_callconversations
    target_call_pos = None
    call_count = 0
    for idx, conv in enumerate(conversations):
        if conv["from"] == "function_call":
            if call_count == call_idx:
                target_call_pos = idx
                break
            call_count += 1
    
    # ，，
    if target_call_pos is None:
        target_call_pos = len(conversations)
    
    # （）
    history_parts = []
    call_count = 0
    answer_count = 0
    
    # userobservation（call_idx-1function_callobservation）
    next_obs_idx = None
    if call_idx > 0:
        obs_count = 0
        for i, c in enumerate(conversations):
            if c["from"] == "function_call":
                if obs_count == call_idx - 1:
                    # observation
                    if i + 1 < len(conversations) and conversations[i + 1]["from"] == "observation":
                        next_obs_idx = i + 1
                    break
                obs_count += 1
    
    for idx, conv in enumerate(conversations):
        if idx >= target_call_pos:
            break
            
        if conv["from"] == "human":
            # query（）
            history_parts.append(f"User query: {conv['value']}")
        
        elif conv["from"] == "function_call":
            # call_idxfunction_call（LLM）
            if call_count < len(generated_calls) and call_count < call_idx:
                # null，（null）
                if generated_calls[call_count] is None:
                    break
                
                # function_call（LLM）
                history_parts.append(f"Tool call: {generated_calls[call_count]}")
                
                call_count += 1
        
        elif conv["from"] == "observation":
            # ：function call，observationhistory
            # userobservation（next_obs_idx）history
            # target_call_posobservationhistory
            # Clear observation content to match training (empty tool responses).
            if next_obs_idx is None or idx != next_obs_idx:
                history_parts.append("Tool response: ")
        
        elif conv["from"] == "gpt":
            # answer（LLM，）
            if answer_count < len(generated_answers):
                # null，（null）
                if generated_answers[answer_count] is None:
                    break
                
                # answer（LLM）
                history_parts.append(f"Assistant reply: {generated_answers[answer_count]}")
                answer_count += 1
    
    # <chat_history>，system
    # chat_historyquery
    current_query_in_history = False
    if history_parts:
        # "User query:"current_query
        for part in reversed(history_parts):
            if part.startswith("User query: "):
                if part[len("User query: "):] == current_query:
                    current_query_in_history = True
                break
        history_content = "\n".join(history_parts)
        system_content += f"\n\n<chat_history>\n{history_content}\n</chat_history>"
    
    # toolssystem
    if tools_formatted:
        system_content += f"\n\n<tools>\n{tools_formatted}\n</tools>"
    
    messages.append({"role": "system", "content": system_content})
    
    # user
    # ，observationrole=OBSERVATION，format_observation
    # format_observation：<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n
    # ，observationquery，
    
    # tool call，observation（call_idx > 0，observation）
    if call_idx > 0:
        # Detect observation slot after call_idx-1; content always emptied to match training.
        has_obs_slot = False
        obs_count = 0
        for idx, conv in enumerate(conversations):
            if conv["from"] == "function_call":
                if obs_count == call_idx - 1:
                    if idx + 1 < len(conversations) and conversations[idx + 1]["from"] == "observation":
                        has_obs_slot = True
                    break
                obs_count += 1
        
        # ：function callobservation
        # ：target_call_posobservation
        has_prev_observation = False
        if target_call_pos > 0:
            # target_call_pos-1，observationhuman
            for idx in range(target_call_pos - 1, -1, -1):
                if conversations[idx]["from"] == "observation":
                    has_prev_observation = True
                    break
                elif conversations[idx]["from"] == "human":
                    # human，multi-turn，observation
                    break
        
        # observation slot present → single-turn multi-hop, OBSERVATION format
        # no observation → multi-turn (next turn first function call), USER format
        # tool_response content is always empty (aligned with training)
        if has_prev_observation and has_obs_slot:
            # single-turn multi-hop：observationfunction_call，query
            observation_content = "<tool_response>\n</tool_response>"
            messages.append({"role": "user", "content": observation_content})
            # query，querychat_history
            return messages
        else:
            # multi-turn：turnfunction call，USER
            # query（chat_history）
            if not current_query_in_history:
                messages.append({"role": "user", "content": current_query})
            return messages
    
    # call_idx=0，query
    # chat_historyquery，user
    # ，queryuser
    if not current_query_in_history:
        messages.append({"role": "user", "content": current_query})
    
    return messages

def build_chat_history_for_function_call(
    conversations: List[Dict],
    call_idx: int,
    generated_calls: List[str],
    generated_answers: List[str]
) -> str:
    """Build chat history for the n-th function call (call_idx=0 is the first).
    chat_history includes only content before the call_idx-th function_call.
    Follows conversation order; uses generated_calls and generated_answers.
    """
    history_parts = []
    
    # call_idxfunction_callconversations
    target_call_pos = None
    call_count = 0
    for idx, conv in enumerate(conversations):
        if conv["from"] == "function_call":
            if call_count == call_idx:
                target_call_pos = idx
                break
            call_count += 1
    
    # ，，
    if target_call_pos is None:
        target_call_pos = len(conversations)
    
    #    call_count = 0
    answer_count = 0
    
    for idx, conv in enumerate(conversations):
        if idx >= target_call_pos:
            break
            
        if conv["from"] == "human":
            # query（）
            history_parts.append(f"User query: {conv['value']}")
        
        elif conv["from"] == "function_call":
            # call_idxfunction_call（LLM）
            if call_count < len(generated_calls) and call_count < call_idx:
                # null，（null）
                if generated_calls[call_count] is None:
                    break
                
                # function_call（LLM）
                history_parts.append(f"Tool call: {generated_calls[call_count]}")
                
                # tool call，observationchat_history，observation
                # observationhistory_parts
                
                call_count += 1
        
        elif conv["from"] == "observation":
            # observationfunction_call，
            pass
        
        elif conv["from"] == "gpt":
            # answer（LLM，）
            if answer_count < len(generated_answers):
                # null，（null）
                if generated_answers[answer_count] is None:
                    break
                
                # answer（LLM）
                history_parts.append(f"Assistant reply: {generated_answers[answer_count]}")
                answer_count += 1
    
    return "\n".join(history_parts)

def build_messages_for_answer(
    conversations: List[Dict],
    generated_calls: List[str],
    generated_answers: List[str],
    answer_idx: int,
    system_prompt: str,
    tools_formatted: str,
    current_query: str
) -> List[Dict[str, str]]:
    """Build messages for the n-th answer (answer_idx=0 is the first).
    Same layout as training: history in <chat_history> inside system.
    """
    messages = []
    
    # answer_idxanswerconversations
    target_answer_pos = None
    answer_count = 0
    for idx, conv in enumerate(conversations):
        if conv["from"] == "gpt":
            if answer_count == answer_idx:
                target_answer_pos = idx
                break
            answer_count += 1
    
    # ，，
    if target_answer_pos is None:
        target_answer_pos = len(conversations)
    
    # observation（answerobservation）
    last_observation = None
    for idx in range(target_answer_pos - 1, -1, -1):
        if conversations[idx]["from"] == "observation":
            last_observation = conversations[idx]
            break
    
    # （，answertool call）
    history_parts = []
    answer_count = 0
    call_count = 0
    
    for idx, conv in enumerate(conversations):
        if idx >= target_answer_pos:
            break
            
        if conv["from"] == "human":
            # query（）
            history_parts.append(f"User query: {conv['value']}")
        
        elif conv["from"] == "function_call":
            # answertool call（）
            if call_count < len(generated_calls):
                history_parts.append(f"Tool call: {generated_calls[call_count]}")
                call_count += 1
        
        elif conv["from"] == "observation":
            # observation（内容强制置空，与训练对齐），不包括最后一个 observation
            if conv != last_observation:
                history_parts.append("Tool response: ")
        
        elif conv["from"] == "gpt":
            # answer_idxanswer（LLM，）
            if answer_count < len(generated_answers) and answer_count < answer_idx:
                # null，（null）
                if generated_answers[answer_count] is None:
                    break
                
                # answer（LLM）
                history_parts.append(f"Assistant reply: {generated_answers[answer_count]}")
                answer_count += 1
    
    # system，（）
    system_content = system_prompt
    
    # <chat_history>，system
    if history_parts:
        history_content = "\n".join(history_parts)
        system_content += f"\n\n<chat_history>\n{history_content}\n</chat_history>"
    
    # system（tools，answer）
    messages.append({"role": "system", "content": system_content})
    
    # observation（）
    # ，observationformat_observation，：
    # <|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n
    # ，vLLMchat_templaterole<|im_start|>user<|im_end|>\n<|im_start|>assistant\n
    # <tool_response> content always empty to match training
    observation_content = "<tool_response>\n</tool_response>"
    
    # observationuser（format_observation）
    # ：observationroleOBSERVATION，format_observation<|im_start|>user
    # ，role=user，vLLM<|im_start|>user<|im_end|>\n<|im_start|>assistant\n
    messages.append({"role": "user", "content": observation_content})
    
    # ：，observationassistant（answer），query
    # format_observation<|im_start|>assistant\n，assistant
    
    return messages

def build_chat_history_for_answer(
    conversations: List[Dict],
    generated_calls: List[str],
    generated_answers: List[str],
    answer_idx: int
) -> str:
    """Build chat history for the n-th answer (answer_idx=0 is the first).
    chat_history includes only content before the answer_idx-th answer (queries, answers, observations; not function_call rows).
    Follows conversation order; uses generated_answers.
    """
    history_parts = []
    
    # answer_idxanswerconversations
    target_answer_pos = None
    answer_count = 0
    for idx, conv in enumerate(conversations):
        if conv["from"] == "gpt":
            if answer_count == answer_idx:
                target_answer_pos = idx
                break
            answer_count += 1
    
    # ，，
    if target_answer_pos is None:
        target_answer_pos = len(conversations)
    
    # ，query、answerobservation（function_call）
    answer_count = 0
    
    for idx, conv in enumerate(conversations):
        if idx >= target_answer_pos:
            break
            
        if conv["from"] == "human":
            # query（）
            history_parts.append(f"User query: {conv['value']}")
        
        elif conv["from"] == "function_call":
            # function_call，chat_history
            pass
        
        elif conv["from"] == "observation":
            # observation content always emptied to match training
            history_parts.append("Tool response: ")
        
        elif conv["from"] == "gpt":
            # answer_idxanswer（LLM，）
            if answer_count < len(generated_answers) and answer_count < answer_idx:
                # null，（null）
                if generated_answers[answer_count] is None:
                    break
                
                # answer（LLM）
                history_parts.append(f"Assistant reply: {generated_answers[answer_count]}")
                answer_count += 1
    
    return "\n".join(history_parts)

def generate_function_call(
    query: str,
    tools: str,
    system_prompt: str,
    conversations: List[Dict],
    call_idx: int,
    generated_calls: List[str],
    generated_answers: List[str],
    logger: logging.Logger,
    thinking: bool = True,
    mode: str = "api1",
    port: int = None,
    model: str = None,
    temperature: float = 0.6,
    top_p: float = 0.95
) -> Optional[str]:
    """Generate the n-th function call (retries are inside call_qwen_api)."""

    # Parse tools JSON for prompt
    try:
        tools_list = json.loads(tools)
        tools_formatted = json.dumps(tools_list, ensure_ascii=False, indent=2)
    except:
        tools_formatted = tools
    
    # query（human，multi-turnsingle-hop）
    current_query = query
    queries = [conv["value"] for conv in conversations if conv["from"] == "human"]
    if queries:
        current_query = queries[-1]
    
    # messages，role
    messages = build_messages_for_function_call(
        conversations, call_idx, generated_calls, generated_answers,
        system_prompt, tools_formatted, current_query
    )
    
    # API（）
    response = call_qwen_api(messages, temperature=temperature, max_retries=3, thinking=thinking, mode=mode, port=port, model=model, top_p=top_p)
    if response:
        tool_call_content = extract_tool_call_content(response)
        if tool_call_content:
            return tool_call_content
        else:
            logger.warning(f"  Could not extract tool_call; response[:200]={response[:200]!r}")
            logger.debug(f"  Full response: {response}")
            return None
    else:
        logger.warning(f"  API returned None (failure or unexpected format)")
        return None

def generate_answer(
    query: str,
    tools: str,
    system_prompt: str,
    conversations: List[Dict],
    generated_calls: List[str],
    generated_answers: List[str],
    answer_idx: int,
    logger: logging.Logger,
    thinking: bool = True,
    mode: str = "api1",
    port: int = None,
    model: str = None,
    temperature: float = 0.6,
    top_p: float = 0.95
) -> Optional[str]:
    """Generate the n-th assistant answer (retries are inside call_qwen_api)."""

    # Parse tools JSON for prompt
    try:
        tools_list = json.loads(tools)
        tools_formatted = json.dumps(tools_list, ensure_ascii=False, indent=2)
    except:
        tools_formatted = tools
    
    # query（human）
    current_query = query
    queries = [conv["value"] for conv in conversations if conv["from"] == "human"]
    if queries:
        current_query = queries[-1]
    
    # messages，role
    messages = build_messages_for_answer(
        conversations, generated_calls, generated_answers, answer_idx,
        system_prompt, tools_formatted, current_query
    )
    
    # API（）
    response = call_qwen_api(messages, temperature=temperature, max_retries=3, thinking=thinking, mode=mode, port=port, model=model, top_p=top_p)
    if response:
        answer_content = extract_answer_content(response)
        if answer_content:
            return answer_content
        else:
            response_lower = response.lower()
            has_answer_tag = '<answer>' in response_lower
            has_closing_tag = '</answer>' in response_lower
            logger.warning(f"  Could not extract answer")
            logger.warning(f"    len(response)={len(response)}")
            logger.warning(f"    has <answer>: {has_answer_tag}")
            logger.warning(f"    has </answer>: {has_closing_tag}")
            logger.warning(f"    response[:500]={response[:500]!r}")
            if len(response) > 500:
                logger.warning(f"    response[-200:]={response[-200:]!r}")
            return None
    return None

def save_intermediate_result(output_data: List[Dict], output_file: str, logger: logging.Logger, lock: Optional[threading.Lock] = None):
    """Save intermediate results (thread-safe)."""
    try:
        if lock:
            lock.acquire()
        try:
            temp_file = output_file + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            os.replace(temp_file, output_file)
            logger.debug("Checkpoint saved")
        finally:
            if lock:
                lock.release()
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {str(e)}")

def is_conversation_complete(result_item: Dict[str, Any], original_item: Dict[str, Any]) -> bool:
    """True if every function_call and answer is generated and non-None."""
    if not result_item:
        return False
    
    result_conversations = result_item.get("conversations", [])
    original_conversations = original_item.get("conversations", [])
    
    # resultfunction_callanswer
    result_fc_count = sum(1 for conv in result_conversations if conv["from"] == "function_call")
    result_answer_count = sum(1 for conv in result_conversations if conv["from"] == "gpt")
    
    original_fc_count = sum(1 for conv in original_conversations if conv["from"] == "function_call")
    original_answer_count = sum(1 for conv in original_conversations if conv["from"] == "gpt")

    if result_fc_count != original_fc_count or result_answer_count != original_answer_count:
        return False

    # All slots filled and non-null
    for conv in result_conversations:
        if conv["from"] == "function_call":
            if conv["value"] is None:
                return False
        elif conv["from"] == "gpt":
            if conv["value"] is None:
                return False
    
    return True

def is_conversation_fully_generated(result_item: Dict[str, Any], original_item: Dict[str, Any]) -> bool:
    """True if every function_call and answer slot has a value (including null).
    Used to decide whether to run phase 2 (null handling).
    """
    if not result_item:
        return False
    
    result_conversations = result_item.get("conversations", [])
    original_conversations = original_item.get("conversations", [])
    
    # resultfunction_callanswer
    result_fc_count = sum(1 for conv in result_conversations if conv["from"] == "function_call")
    result_answer_count = sum(1 for conv in result_conversations if conv["from"] == "gpt")
    
    original_fc_count = sum(1 for conv in original_conversations if conv["from"] == "function_call")
    original_answer_count = sum(1 for conv in original_conversations if conv["from"] == "gpt")
    
    # （，null）
    if result_fc_count != original_fc_count or result_answer_count != original_answer_count:
        return False
    
    # function_callanswer（None）
    # （null）
    return True

def has_null_values(result_item: Dict[str, Any]) -> bool:
    """True if any function_call value is null (ignores answer)."""
    if not result_item:
        return False
    
    result_conversations = result_item.get("conversations", [])
    
    for conv in result_conversations:
        # function_callnull，answernull（answer）
        if conv["from"] == "function_call":
            if conv.get("value") is None:
                return True
    
    return False

def process_conversation(
    item: Dict[str, Any],
    system_prompt: str,
    output_data: List[Dict],
    item_idx: int,
    total: int,
    output_file: str,
    logger: logging.Logger,
    thinking: bool = True,
    mode: str = "api1",
    port: int = None,
    model: str = None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    lock: Optional[threading.Lock] = None,
    skip_null: bool = False,
    process_null_only: bool = False
):
    """Process one conversation."""
    logger.info(f"[{item_idx + 1}/{total}] Processing conversation...")
    
    conversations = item.get("conversations", [])
    tools = item.get("tools", "[]")
    
    # query（）
    queries = [conv["value"] for conv in conversations if conv["from"] == "human"]
    if not queries:
        logger.warning(f"  No human query; skip")
        return
    
    # ，
    # ：（process_null_only=False），conversation（）
    # （process_null_only=True）null
    generated_calls = []
    generated_answers = []
    
    # conversationsanswer（GTanswer）
    for conv in conversations:
        if conv["from"] == "gpt":
            generated_answers.append(conv.get("value"))  # keep raw answer (may be None or empty string)
    
    should_load_existing = False
    if process_null_only:
        should_load_existing = True
    else:
        if lock:
            lock.acquire()
        try:
            if item_idx < len(output_data) and "conversations" in output_data[item_idx]:
                result_item = output_data[item_idx]
                result_conversations = result_item.get("conversations", [])
                result_fc_count = sum(1 for conv in result_conversations if conv["from"] == "function_call")
                result_answer_count = sum(1 for conv in result_conversations if conv["from"] == "gpt")
                original_fc_count = sum(1 for conv in conversations if conv["from"] == "function_call")
                original_answer_count = sum(1 for conv in conversations if conv["from"] == "gpt")
                if result_fc_count != original_fc_count or result_answer_count != original_answer_count:
                    should_load_existing = True
        finally:
            if lock:
                lock.release()

    if should_load_existing:
        if lock:
            lock.acquire()
        try:
            existing_conversations = None
            if item_idx < len(output_data) and "conversations" in output_data[item_idx]:
                existing_conversations = output_data[item_idx]["conversations"]
        finally:
            if lock:
                lock.release()

        if existing_conversations:
            logger.info(f"  Loading partial results...")
            
            # function_callanswer
            for conv in existing_conversations:
                if conv["from"] == "function_call":
                    generated_calls.append(conv.get("value"))  # may be None
                elif conv["from"] == "gpt":
                    generated_answers.append(conv.get("value"))  # may be None
            
            # answer，conversations
            # ，answerNone，conversations
            original_answer_idx = 0
            for conv in conversations:
                if conv["from"] == "gpt":
                    if original_answer_idx < len(generated_answers):
                        # answerNone，
                        if generated_answers[original_answer_idx] is None:
                            generated_answers[original_answer_idx] = conv.get("value")
                    else:
                        # answer，conversations
                        generated_answers.append(conv.get("value"))
                    original_answer_idx += 1
            
            logger.info(
                f"  Loaded {len(generated_calls)} function_call(s) "
                f"({sum(1 for c in generated_calls if c is not None)} non-null)"
            )
            logger.info(
                f"  Loaded {len(generated_answers)} answer(s) "
                f"({sum(1 for a in generated_answers if a is not None)} non-null)"
            )
    else:
        logger.info(f"  Phase 1: not loading prior output; generating from scratch")
    
    # ：function_callanswer
    items_to_generate = []
    call_count = 0
    answer_count = 0
    
    for idx, conv in enumerate(conversations):
        if conv["from"] == "function_call":
            # process_null_only
            if process_null_only:
                # ：function_callnull，answer
                if call_count < len(generated_calls) and generated_calls[call_count] is None:
                    items_to_generate.append(("function_call", idx, call_count))
            else:
                # ：，function_callanswer
                if call_count >= len(generated_calls) or generated_calls[call_count] is None:
                    items_to_generate.append(("function_call", idx, call_count))
            call_count += 1
        elif conv["from"] == "gpt":
            if process_null_only:
                # ：answer
                answer_count += 1
            else:
                # ：answer（answer）
                items_to_generate.append(("answer", idx, answer_count))
                answer_count += 1
    
    # generated_callsconversations
    # None，null，None，
    # None，null
    while len(generated_calls) < call_count:
        generated_calls.append(None)
    # generated_answersconversations，answer_count
    # （），，
    if len(generated_answers) != answer_count:
        logger.warning(f"  generated_answers length ({len(generated_answers)}) != answer_count ({answer_count}); unexpected")
    
    fc_count = sum(1 for t, _, _ in items_to_generate if t == 'function_call')
    ans_count = sum(1 for t, _, _ in items_to_generate if t == 'answer')
    logger.info(f"  Need to generate {len(items_to_generate)} items ({fc_count} function_call, {ans_count} answer)")
    
    #    call_idx = 0
    answer_idx = 0
    
    for gen_type, orig_idx, gen_idx in items_to_generate:
        if gen_type == "function_call":
            # function_call
            call_idx = gen_idx
            logger.info(f"  Generating function_call {call_idx + 1}...")
            current_query = queries[-1] if queries else ""
            
            call_content = generate_function_call(
                current_query, tools, system_prompt, conversations,
                call_idx, generated_calls, generated_answers, logger, thinking, mode, port, model, temperature, top_p
            )
            
            if call_content:
                generated_calls[call_idx] = call_content
                logger.info(f"    Generated: {call_content[:100]}...")
            else:
                logger.warning(f"    Generation failed; using null")
                generated_calls[call_idx] = None
                # skip_null
                if skip_null:
                    # ：null，
                    logger.info(f"    function_call is null; skipping rest of conversation")
                    # ，return
                else:
                    # ：null
                    logger.warning(f"    function_call is null; stopping this conversation, moving on")
                    # （nullfunction_call）
                    temp_result_conversations = []
                    temp_call_idx = 0
                    temp_answer_idx = 0
                    
                    for conv in conversations:
                        if conv["from"] == "human":
                            temp_result_conversations.append(conv)
                        elif conv["from"] == "function_call":
                            if temp_call_idx < len(generated_calls):
                                temp_result_conversations.append({
                                    "from": "function_call",
                                    "value": generated_calls[temp_call_idx] if generated_calls[temp_call_idx] is not None else None
                                })
                            else:
                                temp_result_conversations.append({
                                    "from": "function_call",
                                    "value": None
                                })
                            temp_call_idx += 1
                        elif conv["from"] == "observation":
                            temp_result_conversations.append(conv)
                        elif conv["from"] == "gpt":
                            # generated_answers（conversations）
                            if temp_answer_idx < len(generated_answers):
                                temp_result_conversations.append({
                                    "from": "gpt",
                                    "value": generated_answers[temp_answer_idx]  # keep raw answer (may be None or empty string)
                                })
                            else:
                                # generated_answers，conversations
                                temp_result_conversations.append({
                                    "from": "gpt",
                                    "value": conv.get("value")  # raw answer
                                })
                            temp_answer_idx += 1
                    
                    # （）
                    if lock:
                        lock.acquire()
                    try:
                        if item_idx < len(output_data):
                            output_data[item_idx] = {
                                "conversations": temp_result_conversations,
                                "system": item.get("system", system_prompt),
                                "tools": tools,
                                "properties": item.get("properties", {})
                            }
                        else:
                            output_data.append({
                                "conversations": temp_result_conversations,
                                "system": item.get("system", system_prompt),
                                "tools": tools,
                                "properties": item.get("properties", {})
                            })
                    finally:
                        if lock:
                            lock.release()
                    
                    #                    save_intermediate_result(output_data, output_file, logger, lock)
                    return  # stop this conversation; next item
            
            time.sleep(0.5)  # rate limit
            
        elif gen_type == "answer":
            # ：answer
            answer_idx = gen_idx
            logger.info(f"  Generating answer {answer_idx + 1}...")
            current_query = queries[-1] if queries else ""
            
            answer_content = generate_answer(
                current_query, tools, system_prompt, conversations,
                generated_calls, generated_answers, answer_idx, logger, thinking, mode, port, model, temperature, top_p
            )
            
            if answer_content:
                generated_answers[answer_idx] = answer_content
                logger.info(f"    Generated: {answer_content[:100]}...")
            else:
                logger.warning(f"    Generation failed; using null")
                generated_answers[answer_idx] = None
                if skip_null:
                    logger.info(f"    answer is null; skipping rest of conversation")
                else:
                    logger.warning(f"    answer is null; stopping conversation; next item")
                    temp_result_conversations = []
                    temp_call_idx = 0
                    temp_answer_idx = 0
                    for conv in conversations:
                        if conv["from"] == "human":
                            temp_result_conversations.append(conv)
                        elif conv["from"] == "function_call":
                            if temp_call_idx < len(generated_calls):
                                temp_result_conversations.append({
                                    "from": "function_call",
                                    "value": generated_calls[temp_call_idx] if generated_calls[temp_call_idx] is not None else None
                                })
                            else:
                                temp_result_conversations.append({"from": "function_call", "value": None})
                            temp_call_idx += 1
                        elif conv["from"] == "observation":
                            temp_result_conversations.append(conv)
                        elif conv["from"] == "gpt":
                            if temp_answer_idx < len(generated_answers):
                                temp_result_conversations.append({
                                    "from": "gpt",
                                    "value": generated_answers[temp_answer_idx]
                                })
                            else:
                                temp_result_conversations.append({"from": "gpt", "value": conv.get("value")})
                            temp_answer_idx += 1
                    if lock:
                        lock.acquire()
                    try:
                        if item_idx < len(output_data):
                            output_data[item_idx] = {
                                "conversations": temp_result_conversations,
                                "system": item.get("system", system_prompt),
                                "tools": tools,
                                "properties": item.get("properties", {})
                            }
                        else:
                            output_data.append({
                                "conversations": temp_result_conversations,
                                "system": item.get("system", system_prompt),
                                "tools": tools,
                                "properties": item.get("properties", {})
                            })
                    finally:
                        if lock:
                            lock.release()
                    save_intermediate_result(output_data, output_file, logger, lock)
                    return
            time.sleep(0.5)  # rate limit
        
        #        # result_conversations
        temp_result_conversations = []
        temp_call_idx = 0
        temp_answer_idx = 0
        
        for conv in conversations:
            if conv["from"] == "human":
                temp_result_conversations.append(conv)
            elif conv["from"] == "function_call":
                if temp_call_idx < len(generated_calls):
                    temp_result_conversations.append({
                        "from": "function_call",
                        "value": generated_calls[temp_call_idx] if generated_calls[temp_call_idx] is not None else None
                    })
                else:
                    temp_result_conversations.append({
                        "from": "function_call",
                        "value": None
                    })
                temp_call_idx += 1
            elif conv["from"] == "observation":
                temp_result_conversations.append(conv)
            elif conv["from"] == "gpt":
                # generated_answers（conversations）
                if temp_answer_idx < len(generated_answers):
                    temp_result_conversations.append({
                        "from": "gpt",
                        "value": generated_answers[temp_answer_idx]  # keep raw answer (may be None or empty string)
                    })
                else:
                    # generated_answers，conversations
                    temp_result_conversations.append({
                        "from": "gpt",
                        "value": conv.get("value")  # raw answer
                    })
                temp_answer_idx += 1
        
        # （）
        if lock:
            lock.acquire()
        try:
            if item_idx < len(output_data):
                output_data[item_idx] = {
                    "conversations": temp_result_conversations,
                    "system": item.get("system", system_prompt),
                    "tools": tools,
                    "properties": item.get("properties", {})
                }
            else:
                output_data.append({
                    "conversations": temp_result_conversations,
                    "system": item.get("system", system_prompt),
                    "tools": tools,
                    "properties": item.get("properties", {})
                })
        finally:
            if lock:
                lock.release()
        
        #        save_intermediate_result(output_data, output_file, logger, lock)
    
    logger.info(f"  Done.")

def process_single_file(input_file: str, output_file: str, system_prompt: str, logger: logging.Logger, thinking: bool, mode: str, batch_size: int, port: int = None, max_workers: int = 4, model: str = None, temperature: float = 0.6, top_p: float = 0.95):
    """Process one input file (parallel workers supported)."""
    logger.info("=" * 60)
    logger.info(f"Processing file: {os.path.basename(input_file)}")
    logger.info("=" * 60)
    
    #    logger.info(f"\nReading: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {str(e)}")
        logger.error(f"At line {e.lineno}, col {e.colno}, pos {e.pos}")
        
        #        backup_file = input_file + ".backup"
        backup_file2 = os.path.join(os.path.dirname(input_file), "other", os.path.basename(input_file) + ".backup")
        
        if os.path.exists(backup_file):
            logger.info(f"Found backup: {backup_file}")
            logger.info("Consider using the backup or fix JSON in the current file")
        elif os.path.exists(backup_file2):
            logger.info(f"Found backup: {backup_file2}")
            logger.info("Consider using the backup or fix JSON in the current file")
        else:
            logger.warning("No backup file found")
        
        #        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
            open_brackets = content.count('[')
            close_brackets = content.count(']')
            open_braces = content.count('{')
            close_braces = content.count('}')
            
            if open_brackets != close_brackets or open_braces != close_braces:
                logger.error(f"File may be truncated or incomplete:")
                logger.error(f"  '[' count {open_brackets}, ']' count {close_brackets} (missing {open_brackets - close_brackets} ']')")
                logger.error(f"  '{{' count {open_braces}, '}}' count {close_braces} (missing {open_braces - close_braces} '}}')")
        
        raise
    
    total_items = len(input_data)
    logger.info(f"Total items: {total_items}")
    
    # （）
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # （）
    output_data = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                output_data = json.load(f)
            logger.info(f"Loaded existing output: {len(output_data)} items")

            complete_count = 0
            for idx, result_item in enumerate(output_data):
                if idx < len(input_data):
                    if is_conversation_complete(result_item, input_data[idx]):
                        complete_count += 1
            
            logger.info(f"  Complete: {complete_count}")
            logger.info(f"  Resuming from item {complete_count + 1}...")
        except Exception as e:
            logger.warning(f"Could not read existing output; starting fresh: {str(e)}")
            output_data = []
    
    # ， output_data
    lock = threading.Lock()
    
    # ：input_data（gtfile）
    # 1. output_datainput_data，conversation，
    # 2. output_datainput_data，conversation（），
    # 3. output_datainput_data，function_callnull，
    #    ：answernull，answer
    # 4. output_datainput_data，function_callnull，
    
    has_unprocessed = len(output_data) < len(input_data)
    has_nulls = False
    if len(output_data) == len(input_data):
        # nullconversation
        for idx, item in enumerate(input_data):
            if idx < len(output_data):
                if has_null_values(output_data[idx]):
                    has_nulls = True
                    break
    
    # conversation（）
    phase1_items_to_process = []
    if has_unprocessed:
        for idx, item in enumerate(input_data):
            # output_dataconversation（）
            if idx >= len(output_data):
                phase1_items_to_process.append((idx, item))
    else:
        # ，conversation（）
        for idx, item in enumerate(input_data):
            if idx < len(output_data):
                # ：（），
                result_item = output_data[idx]
                if not is_conversation_complete(result_item, item):
                    # （），null
                    result_conversations = result_item.get("conversations", [])
                    original_conversations = item.get("conversations", [])
                    result_fc_count = sum(1 for conv in result_conversations if conv["from"] == "function_call")
                    result_answer_count = sum(1 for conv in result_conversations if conv["from"] == "gpt")
                    original_fc_count = sum(1 for conv in original_conversations if conv["from"] == "function_call")
                    original_answer_count = sum(1 for conv in original_conversations if conv["from"] == "gpt")
                    # ，，（）
                    if result_fc_count != original_fc_count or result_answer_count != original_answer_count:
                        phase1_items_to_process.append((idx, item))
    
    # conversation（null）
    phase2_items_to_process = []
    if has_nulls:
        for idx, item in enumerate(input_data):
            if idx < len(output_data):
                result_item = output_data[idx]
                # （）
                result_conversations = result_item.get("conversations", [])
                original_conversations = item.get("conversations", [])
                result_fc_count = sum(1 for conv in result_conversations if conv["from"] == "function_call")
                result_answer_count = sum(1 for conv in result_conversations if conv["from"] == "gpt")
                original_fc_count = sum(1 for conv in original_conversations if conv["from"] == "function_call")
                original_answer_count = sum(1 for conv in original_conversations if conv["from"] == "gpt")
                if result_fc_count == original_fc_count and result_answer_count == original_answer_count:
                    if not is_conversation_complete(result_item, item):
                        if has_null_values(result_item):
                            phase2_items_to_process.append((idx, item))

    should_run_phase1 = len(phase1_items_to_process) > 0
    should_run_phase2 = len(phase2_items_to_process) > 0
    
    #    logger.info("\n" + "=" * 60)
    logger.info("Phase selection:")
    logger.info(f"  input_data length: {len(input_data)}")
    logger.info(f"  output_data length: {len(output_data)}")
    logger.info(f"  Unprocessed conversations: {has_unprocessed} ({len(phase1_items_to_process)})")
    logger.info(f"  Conversations with null: {has_nulls} ({len(phase2_items_to_process)})")
    if should_run_phase1:
        logger.info("  Running phase 1 (conversations with no output rows yet)")
    elif should_run_phase2:
        logger.info("  Running phase 2 (conversations with nulls)")
    else:
        logger.info("  Nothing to do: all complete (no nulls)")
    logger.info("=" * 60)
    
    if should_run_phase1:
        # ========== ：conversation（output_data） ==========
        logger.info("\n" + "=" * 60)
        logger.info("Phase 1: conversations with no record in output_data (null stops current conversation)")
        logger.info("=" * 60)
        
        for idx, item in phase1_items_to_process:
            logger.info(f"[{idx + 1}/{total_items}] conversation has no record; phase 1")

        completed_count = [0]
        total_to_process = len(phase1_items_to_process)
        pbar = tqdm(
            total=total_to_process,
            desc="Phase 1",
            unit="item",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        def process_with_progress_phase1(idx, item):
            """Phase-1 worker with progress updates."""
            try:
                process_conversation(
                    item,
                    system_prompt,
                    output_data,
                    idx,
                    total_items,
                    output_file,
                    logger,
                    thinking,
                    mode,
                    port,
                    model,
                    temperature,
                    top_p,
                    lock,
                    skip_null=False,
                    process_null_only=False,
                )
                lock.acquire()
                try:
                    completed_count[0] += 1
                    pbar.update(1)
                finally:
                    lock.release()
            except Exception as e:
                logger.error(f"\nError on item {idx + 1}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                lock.acquire()
                try:
                    completed_count[0] += 1
                    pbar.update(1)
                finally:
                    lock.release()

        logger.info(
            f"\nPhase 1: processing {total_to_process} item(s) "
            f"(batch_size={batch_size}, max_workers={max_workers})..."
        )

        for batch_start in range(0, len(phase1_items_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(phase1_items_to_process))
            batch_items = phase1_items_to_process[batch_start:batch_end]

            logger.info(
                f"\nPhase 1 batch {batch_start // batch_size + 1} "
                f"(items {batch_start + 1}-{batch_end}), workers={max_workers}"
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_with_progress_phase1, idx, item): (idx, item)
                    for idx, item in batch_items
                }
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Task error: {str(e)}")

            if batch_end < len(phase1_items_to_process):
                time.sleep(1)

        pbar.close()
        logger.info(f"\nPhase 1 done: {len(phase1_items_to_process)} item(s).")
        logger.info(f"\nHint: run again for phase 2 if any null function_call remain.")
    elif should_run_phase2:
        # ========== ：function_call nullconversation（function_callnull） ==========
        logger.info("\n" + "=" * 60)
        logger.info("Phase 2: conversations with null function_call (regenerate null fields)")
        logger.info("Note: answers are not generated; null answers are ignored")
        logger.info("=" * 60)
        
        for idx, item in phase2_items_to_process:
            logger.info(f"[{idx + 1}/{total_items}] conversation has null function_call; phase 2")
    
    if should_run_phase2:
        completed_count = [0]
        total_to_process = len(phase2_items_to_process)
        pbar = tqdm(
            total=total_to_process,
            desc="Phase 2",
            unit="item",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        def process_with_progress_phase2(idx, item):
            """Phase-2 worker (null function_call only)."""
            try:
                process_conversation(
                    item,
                    system_prompt,
                    output_data,
                    idx,
                    total_items,
                    output_file,
                    logger,
                    thinking,
                    mode,
                    port,
                    model,
                    temperature,
                    top_p,
                    lock,
                    skip_null=False,
                    process_null_only=True,
                )
                lock.acquire()
                try:
                    completed_count[0] += 1
                    pbar.update(1)
                finally:
                    lock.release()
            except Exception as e:
                logger.error(f"\nError on item {idx + 1}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                lock.acquire()
                try:
                    completed_count[0] += 1
                    pbar.update(1)
                finally:
                    lock.release()

        logger.info(
            f"\nPhase 2: processing {total_to_process} item(s) "
            f"(batch_size={batch_size}, max_workers={max_workers})..."
        )

        for batch_start in range(0, len(phase2_items_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(phase2_items_to_process))
            batch_items = phase2_items_to_process[batch_start:batch_end]

            logger.info(
                f"\nPhase 2 batch {batch_start // batch_size + 1} "
                f"(items {batch_start + 1}-{batch_end}), workers={max_workers}"
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_with_progress_phase2, idx, item): (idx, item)
                    for idx, item in batch_items
                }
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Task error: {str(e)}")

            if batch_end < len(phase2_items_to_process):
                time.sleep(1)

        pbar.close()
        logger.info(f"\nPhase 2 done: {len(phase2_items_to_process)} item(s).")
    
    #    logger.info(f"\n" + "=" * 60)
    if should_run_phase1:
        logger.info(f"This run: phase 1 (no prior output rows)")
        logger.info(f"Items to process: {len(phase1_items_to_process)}")
    elif should_run_phase2:
        logger.info(f"This run: phase 2 (null handling)")
        logger.info(f"Items to process: {len(phase2_items_to_process)}")
    else:
        logger.info(f"This run: nothing to do")
        logger.info(f"All conversations complete (no nulls)")
    
    # output_data
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                final_output_data = json.load(f)
            logger.info(f"Result file has {len(final_output_data)}/{total_items} items")

            complete_count = 0
            null_count = 0
            for idx, result_item in enumerate(final_output_data):
                if idx < len(input_data):
                    if is_conversation_complete(result_item, input_data[idx]):
                        complete_count += 1
                    elif has_null_values(result_item):
                        null_count += 1
            
            logger.info(f"  Complete (no nulls): {complete_count}")
            if null_count > 0:
                logger.info(f"  With nulls (run phase 2 later): {null_count}")
        except Exception as e:
            logger.warning(f"Could not read final output: {str(e)}")
    
    logger.info(f"Saved to: {output_file}")
    logger.info("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description='Generate function calls and answers (backend selected via --mode).'
    )
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Input JSON file (if omitted, all JSON under data_notime_toollist are used)')
    parser.add_argument('--inputfile', type=str, default=None,
                        help='Input directory (default: test_set/data_notime_toollist)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output JSON path (default: under test_set/predictions)')
    parser.add_argument('--think', type=str, default='true', choices=['true', 'false'],
                        help='Enable Qwen-style thinking (default: true)')
    parser.add_argument(
        '--mode',
        type=str,
        default='api1',
        choices=['api1', 'api2', 'api3', 'api4', 'server', 'sft'],
        help=(
            'api1=SiliconFlow; api2=OpenAI official; api3=Anthropic Messages; '
            'api4=Gemini; server/sft=local vLLM. '
            'Model ids containing "gemini" auto-select api4.'
        ),
    )
    parser.add_argument('--port', type=int, default=None,
                        help='Custom port (only for sft/server)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model id (defaults per --mode)')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Sampling temperature (default: 0.6)')
    parser.add_argument('--top-p', type=float, default=0.95,
                        help='Top-p (default: 0.95; ineffective if temperature is 0)')
    parser.add_argument('--outputfile', type=str, default=None,
                        help='Output directory (default: predictions/)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for grouping (default: 64)')
    parser.add_argument('--max-workers', type=int, default=2,
                        help='Max threads per batch (default: 2)')

    args = parser.parse_args()

    thinking = args.think.lower() == 'true'
    mode = args.mode
    batch_size = args.batch_size
    port = args.port
    max_workers = args.max_workers
    model = args.model
    temperature = args.temperature
    top_p = args.top_p

    if port is not None and mode not in ['sft', 'server']:
        print(f"Warning: --port applies only to sft/server; mode={mode}, ignoring --port")
        port = None

    print("=" * 60)
    print("Generate function calls and answers")
    print("=" * 60)
    print(f"Batch size: {batch_size}")
    print(f"Max workers per batch: {max_workers}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir))

    if args.inputfile:
        input_dir = args.inputfile if os.path.isabs(args.inputfile) else os.path.abspath(args.inputfile)
    else:
        input_dir = os.path.join(base_dir, "test_set", "data_notime_toollist")

    if args.outputfile:
        output_dir = args.outputfile if os.path.isabs(args.outputfile) else os.path.abspath(args.outputfile)
    else:
        output_dir = os.path.join(base_dir, "predictions")

    print("\nLoading system prompt...")
    system_prompt = read_system_prompt()
    print(f"System prompt length: {len(system_prompt)} chars")

    input_files = []
    if args.input:
        input_file = args.input if os.path.isabs(args.input) else os.path.join(base_dir, args.input)
        if os.path.isfile(input_file):
            input_files.append(input_file)
        else:
            print(f"Error: input file not found: {input_file}")
            return
    else:
        if os.path.exists(input_dir):
            input_files = glob.glob(os.path.join(input_dir, "*.json"))
            input_files.sort()
            print(f"\nInput directory: {input_dir}")
            print(f"Found {len(input_files)} JSON file(s):")
            for f in input_files:
                print(f"  - {os.path.basename(f)}")
        else:
            print(f"Error: input directory not found: {input_dir}")
            return

    if not input_files:
        print("Error: no input files to process")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    output_dir_basename = os.path.basename(output_dir)
    suffix = "_toollist_1000"
    if "predictions" in output_dir_basename.lower():
        pred_idx = output_dir_basename.lower().find("predictions")
        if pred_idx != -1:
            suffix_part = output_dir_basename[pred_idx + len("predictions") :]
            if suffix_part:
                suffix = suffix_part if suffix_part.startswith("_") else "_" + suffix_part
    print(f"Output filename suffix from directory: {suffix}")

    print("\n" + "=" * 60)
    print("Scanning file status...")
    print("=" * 60)

    file_statuses = []

    for input_file in input_files:
        if args.output and len(input_files) == 1:
            output_file = args.output if os.path.isabs(args.output) else os.path.join(base_dir, args.output)
        else:
            input_basename = os.path.basename(input_file)
            output_basename = input_basename.replace("test_converted_", "").replace(".json", f"{suffix}.json")
            output_file = os.path.join(output_dir, output_basename)

        if not os.path.exists(output_file):
            status = "pending"
            priority = 2
        else:
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    input_data = json.load(f)
                with open(output_file, 'r', encoding='utf-8') as f:
                    output_data = json.load(f)

                if len(output_data) < len(input_data):
                    status = f"incomplete ({len(output_data)}/{len(input_data)})"
                    priority = 2
                elif len(output_data) == len(input_data):
                    has_incomplete_structure = False
                    has_nulls = False
                    complete_count = 0

                    for idx, item in enumerate(input_data):
                        if idx < len(output_data):
                            result_item = output_data[idx]
                            if not is_conversation_complete(result_item, item):
                                result_conversations = result_item.get("conversations", [])
                                original_conversations = item.get("conversations", [])
                                result_fc_count = sum(1 for conv in result_conversations if conv["from"] == "function_call")
                                result_answer_count = sum(1 for conv in result_conversations if conv["from"] == "gpt")
                                original_fc_count = sum(1 for conv in original_conversations if conv["from"] == "function_call")
                                original_answer_count = sum(1 for conv in original_conversations if conv["from"] == "gpt")

                                if result_fc_count != original_fc_count or result_answer_count != original_answer_count:
                                    has_incomplete_structure = True
                                    break
                                elif has_null_values(result_item):
                                    has_nulls = True
                            else:
                                complete_count += 1

                    if has_incomplete_structure:
                        status = f"bad structure ({complete_count}/{len(input_data)} ok)"
                        priority = 1
                    elif has_nulls:
                        null_count = sum(1 for idx, item in enumerate(input_data)
                                       if idx < len(output_data) and has_null_values(output_data[idx]))
                        status = f"has nulls ({complete_count}/{len(input_data)} ok, {null_count} null)"
                        priority = 3
                    else:
                        status = f"complete ({complete_count}/{len(input_data)})"
                        priority = 4
                else:
                    status = f"anomaly ({len(output_data)}/{len(input_data)})"
                    priority = 2
            except Exception as e:
                status = f"read error: {str(e)[:50]}"
                priority = 2

        file_statuses.append((input_file, output_file, status, priority))
        print(f"  {os.path.basename(input_file):50s} -> {status}")

    file_statuses.sort(key=lambda x: x[3])

    status_counts = {}
    for _, _, status, priority in file_statuses:
        if priority == 1:
            status_counts["bad_structure"] = status_counts.get("bad_structure", 0) + 1
        elif priority == 2:
            status_counts["pending"] = status_counts.get("pending", 0) + 1
        elif priority == 3:
            status_counts["has_nulls"] = status_counts.get("has_nulls", 0) + 1
        elif priority == 4:
            status_counts["complete"] = status_counts.get("complete", 0) + 1

    print("\nFile status summary:")
    for status_type, count in status_counts.items():
        print(f"  {status_type}: {count} file(s)")

    files_to_process = [(f, o, s) for f, o, s, p in file_statuses if p < 4]
    files_skipped = [(f, o, s) for f, o, s, p in file_statuses if p == 4]

    if files_skipped:
        print(f"\nSkipping {len(files_skipped)} already-complete file(s):")
        for f, o, s in files_skipped:
            print(f"  - {os.path.basename(f)}")

    if files_to_process:
        print(f"\nWill process {len(files_to_process)} file(s) (priority order):")
        for f, o, s in files_to_process:
            print(f"  - {os.path.basename(f)} ({s})")
    else:
        print("\nAll files are complete; nothing to do.")
        return

    print("=" * 60 + "\n")

    for input_file, output_file, status in files_to_process:
        if args.output and len(input_files) == 1:
            output_file = args.output if os.path.isabs(args.output) else os.path.join(base_dir, args.output)
        else:
            input_basename = os.path.basename(input_file)
            output_basename = input_basename.replace("test_converted_", "").replace(".json", f"{suffix}.json")
            output_file = os.path.join(output_dir, output_basename)

        log_file = os.path.join(output_dir, os.path.splitext(os.path.basename(output_file))[0] + ".log")

        logger = setup_logging(log_file)
        logger.info("=" * 60)
        logger.info("Run start")
        logger.info(f"Input: {input_file}")
        logger.info(f"Output: {output_file}")
        logger.info(f"Log: {log_file}")
        logger.info(f"Thinking: {'on' if thinking else 'off'}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Max workers: {max_workers}")
        mode_desc = {
            'api1': 'SiliconFlow API',
            'api2': 'OpenAI Chat Completions (platform.openai.com)',
            'api3': 'Anthropic Messages API',
            'api4': 'Gemini API',
            'server': 'Local vLLM server',
            'sft': 'SFT / LoRA server',
        }
        logger.info(f"Mode: {mode} ({mode_desc.get(mode, 'unknown')})")
        if model:
            logger.info(f"Model: {model}")
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Top-p: {top_p}")
        if port is not None and mode in ['sft', 'server']:
            logger.info(f"Port: {port}")

        try:
            process_single_file(input_file, output_file, system_prompt, logger, thinking, mode, batch_size, port, max_workers, model, temperature, top_p)
        except Exception as e:
            logger.error(f"Failed processing file: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            print(f"\nError on {os.path.basename(input_file)}: {str(e)}")
            continue

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)

if __name__ == "__main__":
    main()
