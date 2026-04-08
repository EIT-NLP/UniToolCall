#!/usr/bin/env python3
import argparse
import concurrent.futures
import datetime
import email.utils
import json
import os
import random
import re
import sys
import threading
import time
import ssl
import urllib.request

import certifi
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_TOOLSET_JSON = _REPO_ROOT / "tool_set" / "apis" / "toolset.json"
_DEFAULT_USAGE_STATS = _REPO_ROOT / "multi-hop_pipeline" / "db" / "usage_stats.json"


class ProviderRateLimitError(RuntimeError):
    """Raised when provider-side limits block requests in a non-retryable way."""


RATE_LIMIT_CONFIG = {
    "max_attempts": 6,
    "base_sleep": 5.0,
    "max_sleep": 120.0,
    "jitter": 0.15,
    "min_request_interval": 0.0,
}
_NEXT_REQUEST_TS = 0.0
_THROTTLE_LOCK = threading.Lock()


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_tool_name(tool):
    if isinstance(tool, str):
        return tool
    for key in ("name", "tool_name", "function_name"):
        if key in tool and tool[key]:
            return tool[key]
    return None


def get_tool_domain(tool):
    if isinstance(tool, str):
        return None
    for key in ("domain", "category", "tool_domain", "tool_category"):
        if key in tool and tool[key]:
            return tool[key]
    return None


def summarize_usage(usage_stats, tool_names, top_k=5, bottom_k=5):
    pairs = [(name, int(usage_stats.get(name, 0))) for name in tool_names]
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:top_k]
    bottom = sorted(pairs, key=lambda x: x[1])[:bottom_k]
    return {
        "top_usage": top,
        "low_usage": bottom,
        "total_tools": len(tool_names),
    }


def sample_tools(tools, usage_stats, domain_lock, rng):
    if domain_lock:
        in_domain = [t for t in tools if get_tool_domain(t) == domain_lock]
        if not in_domain:
            in_domain = tools
    else:
        in_domain = tools

    def usage_count(tool):
        name = get_tool_name(tool) or ""
        return int(usage_stats.get(name, 0))

    in_domain_sorted = sorted(in_domain, key=usage_count, reverse=True)
    core_n = min(6, max(4, len(in_domain_sorted)))
    core_n = min(core_n, 6)
    core_n = max(4, min(core_n, 6)) if len(in_domain_sorted) >= 4 else len(in_domain_sorted)
    core_tools = in_domain_sorted[:core_n]

    remaining = [t for t in in_domain_sorted if t not in core_tools]
    low_n = min(4, max(2, len(remaining))) if remaining else 0
    low_tools = sorted(remaining, key=usage_count)[:low_n]

    selected = core_tools + low_tools
    remaining = [t for t in in_domain_sorted if t not in selected]

    while len(selected) < 10 and remaining:
        pick = rng.choice(remaining)
        remaining.remove(pick)
        selected.append(pick)

    if len(selected) < 10:
        all_remaining = [t for t in tools if t not in selected]
        rng.shuffle(all_remaining)
        selected.extend(all_remaining[: 10 - len(selected)])

    return selected[:10]


def extract_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    candidates = []
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
        seg = (m.group(1) or "").strip()
        if seg:
            candidates.append(seg)

    def collect_balanced_objects(s):
        objs = []
        depth = 0
        start = None
        in_string = False
        escape = False
        for i, ch in enumerate(s):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        objs.append(s[start : i + 1])
                        start = None
        return objs

    candidates.extend(collect_balanced_objects(text))
    cleaned = re.sub(r"```(?:json)?|```", "", text, flags=re.IGNORECASE)
    if cleaned != text:
        candidates.extend(collect_balanced_objects(cleaned))

    seen = set()
    for cand in sorted(candidates, key=len, reverse=True):
        if cand in seen:
            continue
        seen.add(cand)
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON in response")


def load_api_key(api_key):
    if api_key:
        return api_key.strip()
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key.strip()
    raise ValueError("Missing API key. Set --api-key or environment variable OPENAI_API_KEY.")


def find_existing_episode_indices(output_dir):
    indices = set()
    if not os.path.isdir(output_dir):
        return indices
    pattern = re.compile(r"^episode_(\d{4})\.json$")
    for name in os.listdir(output_dir):
        m = pattern.match(name)
        if not m:
            continue
        try:
            indices.add(int(m.group(1)))
        except Exception:
            continue
    return indices


def list_existing_episode_paths(output_dir):
    items = []
    if not os.path.isdir(output_dir):
        return items
    pattern = re.compile(r"^episode_(\d{4})\.json$")
    for name in os.listdir(output_dir):
        m = pattern.match(name)
        if not m:
            continue
        try:
            idx = int(m.group(1))
        except Exception:
            continue
        items.append((idx, os.path.join(output_dir, name)))
    items.sort(key=lambda x: x[0])
    return [p for _, p in items]


def list_existing_scored_paths(scored_dir):
    items = []
    if not os.path.isdir(scored_dir):
        return items
    pattern = re.compile(r"^episode_(\d{4})\.json$")
    for name in os.listdir(scored_dir):
        m = pattern.match(name)
        if not m:
            continue
        try:
            idx = int(m.group(1))
        except Exception:
            continue
        items.append((idx, os.path.join(scored_dir, name)))
    items.sort(key=lambda x: x[0])
    return [p for _, p in items]


def build_eval_summary(scored_dir: str, summary_file: str):
    rows = []
    for p in list_existing_scored_paths(scored_dir):
        try:
            data = read_json(p)
        except Exception:
            continue
        rows.append(
            {
                "episode": os.path.basename(p),
                "episode_path": data.get("episode_path"),
                "total_score": data.get("total_score"),
                "min_score": data.get("min_score"),
                "query_avg": data.get("query_avg"),
                "trajectory_avg": data.get("trajectory_avg"),
                "anchor_score": data.get("anchor_score"),
                "passed": data.get("passed"),
            }
        )
    summary = {
        "count": len(rows),
        "passed_count": sum(1 for r in rows if r.get("passed") is True),
        "failed_count": sum(1 for r in rows if r.get("passed") is False),
        "items": rows,
    }
    write_json(summary_file, summary)
    return summary


def configure_rate_limit(
    max_attempts: int,
    base_sleep: float,
    max_sleep: float,
    jitter: float,
    min_request_interval: float,
):
    RATE_LIMIT_CONFIG["max_attempts"] = max(1, int(max_attempts))
    RATE_LIMIT_CONFIG["base_sleep"] = max(0.0, float(base_sleep))
    RATE_LIMIT_CONFIG["max_sleep"] = max(0.1, float(max_sleep))
    RATE_LIMIT_CONFIG["jitter"] = max(0.0, float(jitter))
    RATE_LIMIT_CONFIG["min_request_interval"] = max(0.0, float(min_request_interval))


def throttle_before_request():
    global _NEXT_REQUEST_TS
    with _THROTTLE_LOCK:
        interval = RATE_LIMIT_CONFIG["min_request_interval"]
        if interval <= 0:
            return
        now = time.time()
        if now < _NEXT_REQUEST_TS:
            wait_s = _NEXT_REQUEST_TS - now
            if wait_s > 0:
                print(f"Client throttle: sleeping {wait_s:.2f}s to respect request interval.")
                time.sleep(wait_s)
        _NEXT_REQUEST_TS = time.time() + interval


def call_openai_chat(
    api_base,
    api_key,
    model,
    messages,
    temperature=0.7,
    timeout=120,
    ssl_no_verify=False,
):
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    if ssl_no_verify:
        context = ssl._create_unverified_context()
    else:
        context = ssl.create_default_context(cafile=certifi.where())
    last_err = None
    max_attempts = int(RATE_LIMIT_CONFIG["max_attempts"])
    for attempt in range(max_attempts):
        try:
            throttle_before_request()
            with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
                body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            return parsed["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")
            except Exception:
                body = ""
            print(f"HTTPError {e.code}: {body[:500]}")
            last_err = e
            body_l = body.lower()
            is_rate_limited = (
                "rpm limit exceeded" in body_l
                or "rate limit" in body_l
                or "too many requests" in body_l
            )
            needs_verification = "identity verification" in body_l

            if e.code in (403, 429, 503) and is_rate_limited:
                if needs_verification:
                    raise ProviderRateLimitError(
                        "Provider rejected requests due to account verification requirements: "
                        "'RPM limit exceeded. Please complete identity verification to lift the restriction.' "
                        "This is not recoverable by retrying. Complete verification in the provider console, "
                        "or switch to a key/model/account without this restriction."
                    ) from e

                retry_after = None
                if hasattr(e, "headers") and e.headers:
                    ra_value = e.headers.get("Retry-After")
                    if ra_value:
                        try:
                            retry_after = int(ra_value)
                        except ValueError:
                            try:
                                dt = email.utils.parsedate_to_datetime(ra_value)
                                if dt.tzinfo is None:
                                    dt = dt.replace(tzinfo=datetime.timezone.utc)
                                retry_after = max(
                                    0,
                                    int(
                                        (
                                            dt - datetime.datetime.now(datetime.timezone.utc)
                                        ).total_seconds()
                                    ),
                                )
                            except Exception:
                                retry_after = None

                if attempt == max_attempts - 1:
                    break
                if retry_after is not None:
                    wait_s = retry_after
                else:
                    base = RATE_LIMIT_CONFIG["base_sleep"]
                    max_wait = RATE_LIMIT_CONFIG["max_sleep"]
                    jitter = RATE_LIMIT_CONFIG["jitter"]
                    wait_s = min(max_wait, base * (2**attempt))
                    if jitter > 0 and wait_s > 0:
                        low = max(0.0, 1.0 - jitter)
                        high = 1.0 + jitter
                        wait_s = wait_s * random.uniform(low, high)
                print(
                    f"Rate-limited (attempt {attempt + 1}/{max_attempts}). "
                    f"Sleeping {wait_s:.2f}s before retry."
                )
                time.sleep(wait_s)
                continue
            raise
        except Exception as e:
            last_err = e
            time.sleep(1 * (attempt + 1))
    if last_err:
        raise last_err
    raise RuntimeError("Unknown error in call_openai_chat")


def render_template(template, mapping):
    result = template
    for key, value in mapping.items():
        result = result.replace("{{" + key + "}}", value)
    return result


def build_user_input(
    base_prompt: str,
    anchors: list | None = None,
    force_anchor: bool = False,
    required_values: list | None = None,
) -> str:
    prompt = base_prompt
    prompt += "\n\nHard constraint: write the user message in English only."
    if force_anchor and anchors:
        anchor_list = ", ".join([str(a) for a in anchors if a is not None])
        prompt += (
            "\n\nHard constraint: the user message must explicitly include at least one of these anchors: "
            + anchor_list
        )
    if required_values:
        values_list = ", ".join([str(v) for v in required_values if v is not None])
        prompt += (
            "\n\nHard constraint: the user message must explicitly include all of these values: "
            + values_list
        )
    return prompt


def build_plan_input(base_prompt: str, force_user_only: bool = False) -> str:
    prompt = base_prompt + "\n\nHard constraint: output English only."
    if not force_user_only:
        return prompt
    return prompt + "\n\nHard constraint: args_source.field must be 'user' for all planned calls."


def extract_json_object(text: str) -> dict:
    cleaned = re.sub(r"```[a-zA-Z]*", "", text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        segment = cleaned[start : end + 1].strip()
        try:
            return json.loads(segment)
        except Exception:
            pass
    return json.loads(text)


def summarize_tool_params(tool: dict) -> str:
    schema = tool.get("inputSchema") or tool.get("input_schema") or tool.get("parameters")
    if not isinstance(schema, dict):
        return "(no schema)"
    props = schema.get("properties") or {}
    req = schema.get("required") or []
    items = []
    for k, v in props.items():
        typ = v.get("type") if isinstance(v, dict) else None
        enum = v.get("enum") if isinstance(v, dict) else None
        need = "required" if k in req else "optional"
        piece = f"{k}:{typ or 'any'}({need})"
        if enum:
            piece += f" enum={len(enum)}"
        items.append(piece)
    return ", ".join(items) if items else "(no properties)"


def build_tools_context(selected_tools: list) -> str:
    rows = []
    for t in selected_tools:
        name = t.get("name") or t.get("tool_name")
        domain = t.get("domain") or t.get("Domain")
        category = t.get("category") or t.get("Category")
        params = summarize_tool_params(t)
        rows.append(
            f"- name: {name}\n  domain: {domain}\n  category: {category}\n  params: {params}"
        )
    return "\n".join(rows)


def build_trajectory_text(tool_calls: list, tool_outputs: list) -> str:
    lines = ["<trajectory>"]
    for idx, call in enumerate(tool_calls, start=1):
        tool_name = call.get("tool_name")
        arguments = call.get("arguments", {})
        output = {}
        if idx - 1 < len(tool_outputs):
            output = tool_outputs[idx - 1].get("output", {})
        lines.append(f'  <call id="{idx}">')
        lines.append(f"    <tool_name>{tool_name}</tool_name>")
        lines.append(f"    <arguments>{json.dumps(arguments, ensure_ascii=False)}</arguments>")
        lines.append(f"    <result>{json.dumps(output, ensure_ascii=False)}</result>")
        lines.append("  </call>")
    lines.append("</trajectory>")
    return "\n".join(lines)


def evaluate_query_turn(
    provider,
    model,
    api_base,
    api_key,
    prompt_path,
    tools_context,
    query_text,
    multi_hop_flag,
    timeout,
    ssl_no_verify,
):
    tpl = read_text(prompt_path)
    prompt = (
        tpl.replace("{tools_context}", tools_context)
        .replace("{query_text}", query_text)
        .replace("{multi_hop_flag}", "true" if multi_hop_flag else "false")
    )
    if provider != "openai":
        raise ValueError("Only openai-compatible provider is supported in this script.")
    resp = call_openai_chat(
        api_base,
        api_key,
        model,
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        timeout=timeout,
        ssl_no_verify=ssl_no_verify,
    )
    return extract_json_object(resp)


def evaluate_trajectory_turn(
    provider,
    model,
    api_base,
    api_key,
    prompt_path,
    tools_context,
    query_text,
    trajectory_text,
    final_answer,
    multi_hop_flag,
    timeout,
    ssl_no_verify,
):
    tpl = read_text(prompt_path)
    prompt = (
        tpl.replace("{tools_context}", tools_context)
        .replace("{query_text}", query_text)
        .replace("{trajectory_text}", trajectory_text)
        .replace("{final_answer}", final_answer)
        .replace("{multi_hop_flag}", "true" if multi_hop_flag else "false")
    )
    if provider != "openai":
        raise ValueError("Only openai-compatible provider is supported in this script.")
    resp = call_openai_chat(
        api_base,
        api_key,
        model,
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        timeout=timeout,
        ssl_no_verify=ssl_no_verify,
    )
    return extract_json_object(resp)


def evaluate_anchor_episode(
    provider,
    model,
    api_base,
    api_key,
    prompt_path,
    episode_json,
    timeout,
    ssl_no_verify,
):
    tpl = read_text(prompt_path)
    prompt = tpl.replace("{episode_json}", json.dumps(episode_json, ensure_ascii=False))
    if provider != "openai":
        raise ValueError("Only openai-compatible provider is supported in this script.")
    resp = call_openai_chat(
        api_base,
        api_key,
        model,
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        timeout=timeout,
        ssl_no_verify=ssl_no_verify,
    )
    return extract_json_object(resp)


def extract_anchor_score(anchor_eval: dict) -> float | None:
    try:
        return float((anchor_eval.get("anchor_linkage") or {}).get("score"))
    except Exception:
        return None


def extract_query_scores(query_eval: dict) -> list:
    scores = []
    for key in ("toolfit", "clarity", "naturalness"):
        try:
            scores.append(float((query_eval.get(key) or {}).get("score")))
        except Exception:
            pass
    return scores


def extract_traj_scores(traj_eval: dict) -> list:
    scores = []
    for key in ("success", "grounding", "efficiency"):
        try:
            scores.append(float((traj_eval.get(key) or {}).get("score")))
        except Exception:
            pass
    return scores


def _collect_values(value):
    if isinstance(value, dict):
        values = []
        for item in value.values():
            values.extend(_collect_values(item))
        return values
    if isinstance(value, list):
        values = []
        for item in value:
            values.extend(_collect_values(item))
        return values
    return [value]


def _value_in_message(value, message):
    msg = message.lower()
    if isinstance(value, bool):
        if value:
            keywords = ("true", "yes", "activate", "activated", "on", "enable", "enabled")
        else:
            keywords = ("false", "no", "deactivate", "deactivated", "off", "disable", "disabled")
        return any(k in msg for k in keywords)
    if isinstance(value, (int, float)):
        candidates = {str(value)}
        if isinstance(value, float) and value.is_integer():
            candidates.add(str(int(value)))
        if isinstance(value, int):
            candidates.add(f"{value}.0")
        for cand in candidates:
            if re.search(rf"(?<![0-9.]){re.escape(cand)}(?![0-9.])", message):
                return True
        return False
    if isinstance(value, str):
        return value.lower() in msg
    return False


def validate_tool_calls_from_user(tool_calls, user_message):
    missing = []
    for call_index, call in enumerate(tool_calls):
        args = call.get("arguments", {})
        for value in _collect_values(args):
            if value is None or not _value_in_message(value, user_message):
                missing.append({"call_index": call_index, "value": value})
    return missing


def inject_missing_values(user_message: str, missing_values: list[dict]) -> str:
    if not missing_values:
        return user_message
    parts = []
    for item in missing_values:
        val = item.get("value")
        if isinstance(val, bool):
            parts.append("true" if val else "false")
        else:
            parts.append(str(val))
    unique = []
    for p in parts:
        if p not in unique:
            unique.append(p)
    if len(unique) == 1:
        suffix = f" Specifically: {unique[0]}."
    else:
        suffix = " Specifically: " + ", ".join(unique) + "."
    return (user_message or "").strip() + suffix


def contains_cjk(text: str) -> bool:
    if not text:
        return False
    return re.search(r"[\u3400-\u4DBF\u4E00-\u9FFF\u3000-\u303F]", text) is not None


def validate_anchor_present(user_message, anchors):
    if not anchors:
        return True
    msg = user_message.lower()
    for anchor in anchors:
        if anchor is None:
            continue
        if str(anchor).lower() in msg:
            return True
    return False


def plan_uses_user_args_only(turn_plan):
    for call in turn_plan.get("planned_calls", []):
        args_source = call.get("args_source", {})
        if args_source.get("field") != "user":
            return False
    return True


def generate_episode(args, tools, usage_stats, rng, episode_index):
    if args.lock_random_domain:
        domains = [get_tool_domain(t) for t in tools if get_tool_domain(t)]
        domain_lock = rng.choice(domains) if domains else None
    else:
        domain_lock = args.lock_domain

    selected_tools = sample_tools(tools, usage_stats, domain_lock, rng)
    selected_tool_names = [get_tool_name(t) for t in selected_tools if get_tool_name(t)]
    usage_summary = summarize_usage(usage_stats, selected_tool_names)

    episode_prompt = read_text(args.episode_plan_prompt)
    episode_input = render_template(
        episode_prompt,
        {
            "domain_lock": json.dumps(domain_lock),
            "selected_tools": json.dumps(selected_tools, ensure_ascii=False),
            "usage_stats_summary": json.dumps(usage_summary, ensure_ascii=False),
        },
    )

    plan = None
    plan_err = None
    for attempt in range(args.max_retries):
        content = call_openai_chat(
            args.api_base,
            args.api_key,
            args.model,
            [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": episode_input},
            ],
            temperature=args.temperature,
            timeout=args.timeout,
            ssl_no_verify=args.ssl_no_verify,
        )
        try:
            plan = extract_json(content)
            break
        except Exception as e:
            plan_err = e
            if attempt == args.max_retries - 1:
                break
            time.sleep(1)
    if plan is None:
        raise ValueError(f"Failed to generate episode plan JSON: {plan_err}")

    conversation_history = []
    global_state = {}
    conversations = []
    turn_strategies = []

    turn_outline = plan.get("turn_outline", [])
    num_turns = len(turn_outline) if turn_outline else rng.randint(2, 4)

    for turn_index in range(1, num_turns + 1):
        user_message = None
        turn_plan = None
        trajectory = None
        prev_anchors = []
        if conversations:
            prev_anchors = conversations[-1].get("turn_plan", {}).get("required_anchors", [])
        forced_values = None
        force_user_only_plan = False
        for attempt in range(args.max_retries):
            user_prompt = read_text(args.turn_user_prompt)
            user_input_base = render_template(
                user_prompt,
                {
                    "episode_level_plan": json.dumps(plan, ensure_ascii=False),
                    "selected_tools": json.dumps(selected_tools, ensure_ascii=False),
                    "conversation_history": json.dumps(conversation_history, ensure_ascii=False),
                    "global_state": json.dumps(global_state, ensure_ascii=False),
                    "turn_index": str(turn_index),
                },
            )

            user_message = None
            for user_attempt in range(args.max_retries):
                force_anchor = user_attempt >= max(1, args.max_retries // 2)
                user_input = build_user_input(
                    user_input_base,
                    prev_anchors,
                    force_anchor,
                    required_values=forced_values,
                )
                content = call_openai_chat(
                    args.api_base,
                    args.api_key,
                    args.model,
                    [
                        {"role": "system", "content": args.system_prompt},
                        {"role": "user", "content": user_input},
                    ],
                    temperature=args.temperature,
                    timeout=args.timeout,
                    ssl_no_verify=args.ssl_no_verify,
                )
                if content.strip():
                    user_message = content.strip()
                    if turn_index < 2 or validate_anchor_present(user_message, prev_anchors):
                        break

            if user_message and turn_index >= 2 and not validate_anchor_present(user_message, prev_anchors):
                # Last-resort: force anchor injection to avoid hard failure.
                for anchor in prev_anchors:
                    if anchor is not None:
                        user_message = f"About {anchor}, {user_message}"
                        break

            if not user_message:
                raise ValueError("Empty user message")
            if args.enforce_english and contains_cjk(user_message):
                if attempt == args.max_retries - 1:
                    raise ValueError("Failed to generate English-only user message.")
                continue

            plan_prompt = read_text(args.turn_plan_prompt)
            plan_input_base = render_template(
                plan_prompt,
                {
                    "user_message": user_message,
                    "conversation_history": json.dumps(conversation_history, ensure_ascii=False),
                    "global_state": json.dumps(global_state, ensure_ascii=False),
                    "selected_tools": json.dumps(selected_tools, ensure_ascii=False),
                },
            )
            plan_input = build_plan_input(plan_input_base, force_user_only_plan)

            turn_plan = None
            turn_plan_err = None
            for plan_attempt in range(args.max_retries):
                content = call_openai_chat(
                    args.api_base,
                    args.api_key,
                    args.model,
                    [
                        {"role": "system", "content": args.system_prompt},
                        {"role": "user", "content": plan_input},
                    ],
                    temperature=args.temperature,
                    timeout=args.timeout,
                    ssl_no_verify=args.ssl_no_verify,
                )
                try:
                    turn_plan = extract_json(content)
                    break
                except Exception as e:
                    turn_plan_err = e
                    if plan_attempt == args.max_retries - 1:
                        break
                    time.sleep(1)
            if turn_plan is None:
                if attempt == args.max_retries - 1:
                    raise ValueError(f"Failed to generate turn plan JSON: {turn_plan_err}")
                continue

            if not plan_uses_user_args_only(turn_plan):
                force_user_only_plan = True
                if attempt == args.max_retries - 1:
                    # Force-coerce args_source to user to avoid hard failure.
                    for call in turn_plan.get("planned_calls", []):
                        call["args_source"] = {
                            "field": "user",
                            "details": "Forced to user to satisfy strict parameter sourcing.",
                        }
                    break
                continue

            traj_prompt = read_text(args.turn_trajectory_prompt)
            traj_input = render_template(
                traj_prompt,
                {
                    "turn_plan": json.dumps(turn_plan, ensure_ascii=False),
                    "user_message": user_message,
                    "conversation_history": json.dumps(conversation_history, ensure_ascii=False),
                    "global_state": json.dumps(global_state, ensure_ascii=False),
                    "selected_tools": json.dumps(selected_tools, ensure_ascii=False),
                },
            )
            if args.enforce_english:
                traj_input += "\n\nHard constraint: output JSON in English only."

            trajectory = None
            trajectory_err = None
            for traj_attempt in range(args.max_retries):
                content = call_openai_chat(
                    args.api_base,
                    args.api_key,
                    args.model,
                    [
                        {"role": "system", "content": args.system_prompt},
                        {"role": "user", "content": traj_input},
                    ],
                    temperature=args.temperature,
                    timeout=args.timeout,
                    ssl_no_verify=args.ssl_no_verify,
                )
                try:
                    trajectory = extract_json(content)
                    break
                except Exception as e:
                    trajectory_err = e
                    if traj_attempt == args.max_retries - 1:
                        break
                    time.sleep(1)
            if trajectory is None:
                if attempt == args.max_retries - 1:
                    raise ValueError(f"Failed to generate trajectory JSON: {trajectory_err}")
                continue
            if args.enforce_english and contains_cjk(json.dumps(trajectory, ensure_ascii=False)):
                if attempt == args.max_retries - 1:
                    raise ValueError("Failed to generate English-only trajectory.")
                continue

            tool_calls = trajectory.get("tool_calls", [])
            missing_values = validate_tool_calls_from_user(tool_calls, user_message)
            if missing_values:
                forced_values = [item.get("value") for item in missing_values]
                if attempt == args.max_retries - 1:
                    user_message = inject_missing_values(user_message, missing_values)
                    break
                continue
            break

        if trajectory is None:
            raise ValueError("Failed to generate a valid trajectory for this turn.")

        turn_strategies.append(turn_plan.get("turn_strategy", ""))

        tool_calls = trajectory.get("tool_calls", [])
        tool_outputs = trajectory.get("tool_outputs", [])
        assistant_response = trajectory.get("assistant_response", "")

        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
        ]
        for item in tool_outputs:
            messages.append(
                {
                    "role": "tool",
                    "tool_name": item.get("tool_name"),
                    "content": json.dumps(item.get("output", {}), ensure_ascii=False),
                }
            )
        messages.append({"role": "assistant", "content": assistant_response})

        conversations.append(
            {
                "turn_index": turn_index,
                "turn_plan": turn_plan,
                "messages": messages,
            }
        )

        conversation_history.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_response},
            ]
        )

        if isinstance(tool_outputs, list):
            global_state.setdefault("tool_outputs", []).append(tool_outputs)

    episode = {
        "system": args.system_prompt,
        "selected_tools": selected_tools,
        "properties": {
            "turn": "multi-turn",
            "domain": plan.get("domain") or domain_lock,
            "num_turns": len(conversations),
            "num_tools": len(selected_tools),
            "turn_strategies": turn_strategies,
        },
        "episode_level_plan": plan,
        "conversations": conversations,
    }

    out_name = f"episode_{episode_index:04d}.json"
    out_path = os.path.join(args.output_dir, out_name)
    write_json(out_path, episode)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate multi-turn tool-trajectory episodes (planning + generation only).")
    parser.add_argument(
        "--tools-json",
        default=str(_DEFAULT_TOOLSET_JSON),
        help="Full tools JSON (default: ../tool_set/apis/toolset.json from repo root)",
    )
    parser.add_argument(
        "--usage-stats",
        default=str(_DEFAULT_USAGE_STATS),
        help="Tool usage counts JSON (default: multi-hop_pipeline/db/usage_stats.json)",
    )
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1, help="How many episode indices to submit per scheduling batch.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of worker threads for generation/evaluation when --count > 0.")
    parser.add_argument(
        "--append-count",
        action="store_true",
        help="Generate exactly --count new episodes starting after the max existing index in --output-dir.",
    )
    parser.add_argument("--output-dir", default="outputs/raw")
    parser.add_argument("--resume", action="store_true", help="Skip existing episode_XXXX.json files in output dir.")
    parser.add_argument("--lock-domain", default=None)
    parser.add_argument("--lock-random-domain", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=2)

    parser.add_argument("--episode-plan-prompt", default="prompts/episode_plan_prompt.md")
    parser.add_argument("--turn-user-prompt", default="prompts/turn_user_prompt.md")
    parser.add_argument("--turn-plan-prompt", default="prompts/turn_plan_prompt.md")
    parser.add_argument("--turn-trajectory-prompt", default="prompts/turn_trajectory_prompt.md")

    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", default=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=None, help="Override OPENAI_API_KEY for this run")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--ssl-no-verify", action="store_true")
    parser.add_argument(
        "--enforce-english",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require generated user/trajectory text to be English-only (reject CJK output).",
    )
    parser.add_argument("--min-request-interval", type=float, default=0.0, help="Client-side throttle interval in seconds between API requests.")
    parser.add_argument("--rate-limit-max-attempts", type=int, default=6, help="Max retry attempts for rate-limited requests.")
    parser.add_argument("--rate-limit-base-sleep", type=float, default=5.0, help="Base backoff seconds for rate-limit retries.")
    parser.add_argument("--rate-limit-max-sleep", type=float, default=120.0, help="Max backoff seconds for rate-limit retries.")
    parser.add_argument("--rate-limit-jitter", type=float, default=0.15, help="Jitter ratio applied to backoff waits (e.g. 0.15 = +-15%).")

    parser.add_argument("--eval-after", action="store_true", help="Run quality evaluation after generation")
    parser.add_argument("--eval-provider", default="openai")
    parser.add_argument("--eval-model", default=None)
    parser.add_argument("--eval-api-base", default=None)
    parser.add_argument("--eval-timeout", type=int, default=120)
    parser.add_argument("--eval-query-prompt", default="prompts/prompt_query_eval.md")
    parser.add_argument("--eval-trajectory-prompt", default="prompts/prompt_trajectory_eval.md")
    parser.add_argument("--eval-anchor-prompt", default="prompts/prompt_anchor_eval.md")
    parser.add_argument("--scored-dir", default="outputs/scored")
    parser.add_argument("--filtered-dir", default="outputs/filtered")
    parser.add_argument(
        "--eval-skip-scored",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip episodes that already have score files in --scored-dir.",
    )
    parser.add_argument(
        "--eval-summary-file",
        default=None,
        help="Path to merged evaluation summary JSON; default is <scored-dir>/score_summary.json.",
    )
    parser.add_argument("--min-score", type=float, default=4.0)
    parser.add_argument("--avg-threshold", type=float, default=8.0)

    return parser.parse_args()


def chunked(items, size):
    size = max(1, int(size))
    for i in range(0, len(items), size):
        yield items[i : i + size]


def process_episode_index(args, tools, usage_stats, episode_index, already_scored):
    rng = random.Random(args.seed + episode_index)
    out_path = generate_episode(args, tools, usage_stats, rng, episode_index)
    result = {
        "out_path": out_path,
        "generated": 1,
        "evaluated": 0,
        "eval_skipped": 0,
        "passed": 0,
    }
    if not args.eval_after:
        return result

    out_name = os.path.basename(out_path)
    if args.eval_skip_scored and out_name in already_scored:
        print(f"Skip scored {os.path.join(args.scored_dir, out_name)}")
        result["eval_skipped"] = 1
        return result

    passed = evaluate_episode_file(args, out_path)
    result["evaluated"] = 1
    if passed:
        result["passed"] = 1
    return result


def evaluate_episode_file(args, out_path):
    episode = read_json(out_path)
    eval_model = args.eval_model or args.model
    eval_api_base = args.eval_api_base or args.api_base

    tools_context = build_tools_context(episode.get("selected_tools", []))

    per_turn = []
    all_query_scores = []
    all_traj_scores = []
    min_scores = []

    for turn in episode.get("conversations", []):
        messages = turn.get("messages", [])
        user_msg = ""
        tool_calls = []
        tool_outputs = []
        assistant_final = ""

        for msg in messages:
            role = msg.get("role")
            if role == "user":
                user_msg = msg.get("content", "")
            elif role == "assistant" and msg.get("tool_calls"):
                tool_calls = msg.get("tool_calls", [])
            elif role == "tool":
                output = {}
                try:
                    output = json.loads(msg.get("content", "{}"))
                except Exception:
                    output = {}
                tool_outputs.append({"tool_name": msg.get("tool_name"), "output": output})
            elif role == "assistant" and msg.get("content"):
                assistant_final = msg.get("content", "")

        traj_text = build_trajectory_text(tool_calls, tool_outputs)
        multi_hop = len(tool_calls) >= 2

        query_eval = evaluate_query_turn(
            args.eval_provider,
            eval_model,
            eval_api_base,
            args.api_key,
            args.eval_query_prompt,
            tools_context,
            user_msg,
            multi_hop,
            args.eval_timeout,
            args.ssl_no_verify,
        )
        traj_eval = evaluate_trajectory_turn(
            args.eval_provider,
            eval_model,
            eval_api_base,
            args.api_key,
            args.eval_trajectory_prompt,
            tools_context,
            user_msg,
            traj_text,
            assistant_final,
            multi_hop,
            args.eval_timeout,
            args.ssl_no_verify,
        )

        q_scores = extract_query_scores(query_eval)
        t_scores = extract_traj_scores(traj_eval)
        all_query_scores.extend(q_scores)
        all_traj_scores.extend(t_scores)
        min_scores.extend(q_scores + t_scores)

        per_turn.append(
            {
                "turn_index": turn.get("turn_index"),
                "query_evaluation": query_eval,
                "trajectory_evaluation": traj_eval,
            }
        )

    anchor_eval = evaluate_anchor_episode(
        args.eval_provider,
        eval_model,
        eval_api_base,
        args.api_key,
        args.eval_anchor_prompt,
        episode,
        args.eval_timeout,
        args.ssl_no_verify,
    )
    anchor_score = extract_anchor_score(anchor_eval)
    if anchor_score is not None:
        min_scores.append(anchor_score)

    query_avg = sum(all_query_scores) / len(all_query_scores) if all_query_scores else None
    traj_avg = sum(all_traj_scores) / len(all_traj_scores) if all_traj_scores else None
    if query_avg is None or traj_avg is None or anchor_score is None:
        total_score = None
    else:
        total_score = 0.4 * query_avg + 0.4 * traj_avg + 0.2 * anchor_score

    min_score = min(min_scores) if min_scores else None
    passed = (
        min_score is not None
        and total_score is not None
        and min_score >= args.min_score
        and total_score >= args.avg_threshold
    )

    scored = {
        "episode_path": out_path,
        "query_avg": query_avg,
        "trajectory_avg": traj_avg,
        "anchor_score": anchor_score,
        "total_score": total_score,
        "min_score": min_score,
        "passed": passed,
        "per_turn": per_turn,
        "anchor_evaluation": anchor_eval,
    }

    scored_path = os.path.join(args.scored_dir, os.path.basename(out_path))
    write_json(scored_path, scored)
    if passed:
        filtered_path = os.path.join(args.filtered_dir, os.path.basename(out_path))
        write_json(filtered_path, episode)

    print(
        f"{'PASS' if passed else 'FAIL'} total={total_score if total_score is not None else 'NA'} "
        f"min={min_score if min_score is not None else 'NA'} file={os.path.basename(out_path)}"
    )
    return passed


def main():
    args = parse_args()
    args.batch_size = max(1, int(args.batch_size))
    args.num_workers = max(1, int(args.num_workers))
    if args.provider != "openai":
        raise ValueError("Only openai-compatible provider is supported in this script.")
    args.api_key = load_api_key(args.api_key)
    configure_rate_limit(
        args.rate_limit_max_attempts,
        args.rate_limit_base_sleep,
        args.rate_limit_max_sleep,
        args.rate_limit_jitter,
        args.min_request_interval,
    )

    tools = read_json(args.tools_json)
    if isinstance(tools, dict):
        tools = [{"id": key, **value} for key, value in tools.items()]

    usage_stats = read_json(args.usage_stats)
    if isinstance(usage_stats, dict) and "tool_usage" in usage_stats:
        usage_stats = usage_stats["tool_usage"]

    os.makedirs(args.output_dir, exist_ok=True)
    if args.eval_after:
        os.makedirs(args.scored_dir, exist_ok=True)
        os.makedirs(args.filtered_dir, exist_ok=True)
    if args.eval_after and not args.eval_summary_file:
        args.eval_summary_file = os.path.join(args.scored_dir, "score_summary.json")
    rng = random.Random(args.seed)
    discovered_indices = find_existing_episode_indices(args.output_dir)
    existing_indices = discovered_indices if args.resume else set()
    existing_scored = {
        os.path.basename(p) for p in list_existing_scored_paths(args.scored_dir)
    } if args.eval_after and args.eval_skip_scored else set()

    generated_count = 0
    skipped_count = 0
    evaluated_count = 0
    eval_skipped_count = 0
    passed_count = 0
    stopped_by_rate_limit = False

    if args.eval_after and args.count == 0:
        for path in list_existing_episode_paths(args.output_dir):
            out_name = os.path.basename(path)
            if args.eval_skip_scored and out_name in existing_scored:
                print(f"Skip scored {os.path.join(args.scored_dir, out_name)}")
                eval_skipped_count += 1
                continue
            try:
                passed = evaluate_episode_file(args, path)
                evaluated_count += 1
                if passed:
                    passed_count += 1
            except ProviderRateLimitError as e:
                stopped_by_rate_limit = True
                print(f"Stopped during evaluation at {os.path.basename(path)}: {e}", file=sys.stderr)
                print(
                    "Progress is preserved. Re-run evaluation after account verification or with a different API key/provider.",
                    file=sys.stderr,
                )
                break
        summary = build_eval_summary(args.scored_dir, args.eval_summary_file)
        print(
            f"Wrote summary {args.eval_summary_file} "
            f"(count={summary['count']}, passed={summary['passed_count']}, failed={summary['failed_count']})"
        )
        print(
            f"Done. generated={generated_count}, skipped={skipped_count}, requested={args.count}, "
            f"evaluated={evaluated_count}, eval_skipped={eval_skipped_count}, passed={passed_count}"
            + (", stopped_by_rate_limit=true" if stopped_by_rate_limit else "")
        )
        return

    if args.append_count:
        start_index = (max(discovered_indices) if discovered_indices else 0) + 1
        episode_indices = range(start_index, start_index + args.count)
    else:
        episode_indices = range(1, args.count + 1)

    pending_episode_indices = []
    for episode_index in episode_indices:
        if episode_index in existing_indices:
            out_path = os.path.join(args.output_dir, f"episode_{episode_index:04d}.json")
            print(f"Skip existing {out_path}")
            skipped_count += 1
            continue
        pending_episode_indices.append(episode_index)

    if args.num_workers <= 1:
        for episode_index in pending_episode_indices:
            try:
                out_path = generate_episode(args, tools, usage_stats, rng, episode_index)
                generated_count += 1
                print(f"Wrote {out_path}")
                if args.eval_after:
                    out_name = os.path.basename(out_path)
                    if args.eval_skip_scored and out_name in existing_scored:
                        print(f"Skip scored {os.path.join(args.scored_dir, out_name)}")
                        eval_skipped_count += 1
                        continue
                    passed = evaluate_episode_file(args, out_path)
                    evaluated_count += 1
                    if passed:
                        passed_count += 1
            except ProviderRateLimitError as e:
                stopped_by_rate_limit = True
                print(f"Stopped at episode_{episode_index:04d}: {e}", file=sys.stderr)
                print(
                    "Progress is preserved. Re-run with --resume after account verification or with a different API key/provider.",
                    file=sys.stderr,
                )
                break
    else:
        for batch in chunked(pending_episode_indices, args.batch_size):
            if stopped_by_rate_limit:
                break
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = {
                    executor.submit(
                        process_episode_index,
                        args,
                        tools,
                        usage_stats,
                        episode_index,
                        existing_scored,
                    ): episode_index
                    for episode_index in batch
                }
                for future in concurrent.futures.as_completed(futures):
                    episode_index = futures[future]
                    try:
                        item = future.result()
                        out_path = item["out_path"]
                        generated_count += item["generated"]
                        evaluated_count += item["evaluated"]
                        eval_skipped_count += item["eval_skipped"]
                        passed_count += item["passed"]
                        print(f"Wrote {out_path}")
                    except ProviderRateLimitError as e:
                        stopped_by_rate_limit = True
                        print(f"Stopped at episode_{episode_index:04d}: {e}", file=sys.stderr)
                        print(
                            "Progress is preserved. Re-run with --resume after account verification or with a different API key/provider.",
                            file=sys.stderr,
                        )
                        for f in futures:
                            f.cancel()
                        break

    if args.eval_after:
        summary = build_eval_summary(args.scored_dir, args.eval_summary_file)
        print(
            f"Wrote summary {args.eval_summary_file} "
            f"(count={summary['count']}, passed={summary['passed_count']}, failed={summary['failed_count']})"
        )
    print(
        f"Done. generated={generated_count}, skipped={skipped_count}, requested={args.count}, "
        f"evaluated={evaluated_count}, eval_skipped={eval_skipped_count}, passed={passed_count}"
        + (", stopped_by_rate_limit=true" if stopped_by_rate_limit else "")
    )


if __name__ == "__main__":
    main()
