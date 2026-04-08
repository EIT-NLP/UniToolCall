#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
import sys
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
from uni_toolcall.secrets import get_google_api_key, get_openai_compatible_key

from core import (
    sample_for_generation,
    OUT_DIR,
    parse_sections,
    count_calls,
    load_usage_stats,
    save_usage_stats,
    parse_trajectory,
    build_tools_context_detailed,
    build_query_trajectory_xml,
    render_prompt1,
)
from standardize_merge import detect_strategy

ROOT = Path(__file__).resolve().parent
_DEFAULT_TOOLSET_JSON = _REPO_ROOT / "tool_set" / "apis" / "toolset.json"


def get_gemini_key() -> Optional[str]:
    return get_google_api_key()


def get_siliconflow_key() -> Optional[str]:
    return get_openai_compatible_key()


def http_post(url: str, headers: dict, payload: dict, timeout: int = 60):
    try:
        import requests  # type: ignore
        import warnings
        # Suppress SSL warnings when verification is disabled
        warnings.filterwarnings('ignore', message='Unverified HTTPS request')
        # Disable SSL verification as a workaround for certificate issues
        r = requests.post(url, headers=headers, json=payload, timeout=timeout, verify=False)
        r.raise_for_status()
        return r.json()
    except ImportError:
        # requests not installed, fallback to urllib
        pass
    except requests.exceptions.Timeout as timeout_err:
        # Determine timeout type for better error message
        timeout_type = "read timeout" if isinstance(timeout_err, requests.exceptions.ReadTimeout) else "connection timeout" if isinstance(timeout_err, requests.exceptions.ConnectTimeout) else "timeout"
        raise RuntimeError(
            f"Request {timeout_type} after {timeout} seconds for URL: {url}\n"
            f"Error: {timeout_err}\n"
            f"Possible causes:\n"
            f"  1. Model is taking too long to respond - try increasing --timeout (current: {timeout}s)\n"
            f"     Recommended: --timeout 300 (5 min) or --timeout 600 (10 min)\n"
            f"  2. Prompt/tool context is too large - reduce --num-tools-max (current default: 5)\n"
            f"     Try: --num-tools-max 3 to reduce context size\n"
            f"  3. Network connection is slow - check your internet speed\n"
            f"  4. API server is overloaded - try again later\n"
            f"  5. Model response is very long - consider using a faster model\n"
            f"\nQuick fix: Add --timeout 300 or --timeout 600 to your command"
        ) from timeout_err
    except requests.exceptions.RequestException as req_err:
        # For other requests errors, fallback to urllib
        pass
    except Exception as e:
        # For other exceptions, fallback to urllib
        pass
    
    # Fallback to urllib
    import urllib.request, urllib.error
    import json as _json
    import ssl
    # Create SSL context that doesn't verify certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    req = urllib.request.Request(url, data=_json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_context) as resp:
            data = resp.read().decode("utf-8")
            return json.loads(data)
    except urllib.error.HTTPError as he:
        # HTTPError is a subclass of URLError; handle it first so we can surface response body.
        msg = he.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTPError {he.code}: {msg}") from he
    except urllib.error.URLError as url_err:
        # DNS resolution or network error
        error_msg = str(url_err)
        # Check for timeout in urllib errors
        if "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
            raise RuntimeError(
                f"Request timeout after {timeout} seconds for URL: {url}\n"
                f"Possible causes:\n"
                f"  1. Model is taking too long to respond - try increasing --timeout (current: {timeout}s)\n"
                f"  2. Network connection is slow - check your internet speed\n"
                f"  3. API server is overloaded - try again later\n"
                f"  4. Prompt is too long - consider reducing tool context size\n"
                f"Error details: {error_msg}"
            ) from url_err
        elif "nodename nor servname provided" in error_msg or "Name or service not known" in error_msg:
            raise RuntimeError(
                f"DNS resolution failed for URL: {url}\n"
                f"Possible causes:\n"
                f"  1. Network connection issue - check your internet connection\n"
                f"  2. DNS server problem - try: ping api.siliconflow.cn\n"
                f"  3. Incorrect API base URL - check --api-base parameter\n"
                f"  4. Firewall/proxy blocking - check network settings\n"
                f"Error details: {error_msg}"
            ) from url_err
        else:
            raise RuntimeError(f"Network error: {error_msg}") from url_err


def call_gemini(model: str, prompt: str, timeout: int = 60) -> str:
    api_key = get_gemini_key()
    if not api_key:
        raise RuntimeError("Missing Gemini API key. Set GOOGLE_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.6},
    }
    data = http_post(url, headers={"Content-Type": "application/json"}, payload=payload, timeout=timeout)
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        raise RuntimeError(f"Unexpected Gemini response: {json.dumps(data)[:500]}")


def call_openai_compatible(model: str, prompt: str, api_base: str = "https://api.siliconflow.cn", timeout: int = 60) -> str:
    api_key = get_siliconflow_key()
    if not api_key:
        raise RuntimeError("Missing SiliconFlow API key. Set SILICONFLOW_API_KEY or OPENAI_API_KEY")
    url = api_base.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = http_post(url, headers=headers, payload=payload, timeout=timeout)
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected OpenAI-compatible response: {json.dumps(data)[:500]}")


def extract_json_object(text: str):
    """Best-effort JSON extraction from model output.

    Returns:
    - dict/list/str/etc. if JSON parses successfully
    - raw text (str) if parsing fails
    """
    raw = text or ""
    cleaned = re.sub(r"```[a-zA-Z]*", "", raw)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        seg = cleaned[start : end + 1].strip()
        try:
            return json.loads(seg)
        except Exception:
            pass
    try:
        return json.loads(cleaned.strip())
    except Exception:
        return raw.strip()


_INVALID_ENTITY_RE = re.compile(
    r'&(?!#\d+;|#x[0-9a-fA-F]+;|(?i:amp|lt|gt|quot|apos);)'
)


def _trim_to_query_trajectory(text: str) -> str:
    """Best-effort extraction of the <query_trajectory>...</query_trajectory> block."""
    if not text:
        return text
    low = text.lower()
    start = low.find("<query_trajectory")
    if start != -1:
        text = text[start:]
        low = text.lower()
    end = low.rfind("</query_trajectory>")
    if end != -1:
        end += len("</query_trajectory>")
        text = text[:end]
    elif "<query_trajectory" in low:
        text = text.rstrip() + "\n</query_trajectory>"
    return text


def _clean_xml_output(text: str) -> str:
    """
    Clean model output to remove markdown code blocks, trim stray content,
    escape invalid entities, and validate XML structure.
    """
    if not text:
        return text

    # Remove markdown code block markers (```xml, ```, etc.)
    text = re.sub(r'^```xml\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)

    # Fix common XML tag errors
    text = re.sub(r'</tables>', '</rows>', text)

    # Extract XML declaration if present; otherwise add a default one.
    decl_match = re.search(r'<\?xml[^>]+\?>', text)
    decl = decl_match.group(0) if decl_match else '<?xml version="1.0" encoding="UTF-8"?>'
    if decl_match:
        text = text[decl_match.end():]
    text = re.sub(r'<\?xml[^>]+\?>', '', text)

    # Keep only the main query_trajectory block and strip whitespace
    text = _trim_to_query_trajectory(text)
    text = text.strip()

    # Re-add declaration at the top
    if not text.startswith("<query_trajectory"):
        # If we never found the main tag, keep best-effort original body
        text = text.lstrip()
    text = decl + "\n" + text

    # Escape stray ampersands / unsupported HTML entities
    text = _INVALID_ENTITY_RE.sub("&amp;", text)

    # Final validation pass – raise if still invalid so the caller can retry.
    try:
        ET.fromstring(text)
    except ET.ParseError as e:
        snippet = text[:400].replace("\n", "\\n")
        raise RuntimeError(f"Generated XML is invalid after cleaning: {e}. Snippet: {snippet}") from e

    return text


def _llm_call(provider: str, model: str, prompt: str, api_base: str, timeout: int) -> str:
    # Basic pacing + backoff to avoid RPM limit (stepwise can generate many requests per sample).
    # No extra CLI flags to keep UX simple.
    max_retries = 5
    backoff_seconds = [5, 10, 20, 40, 60]

    for attempt in range(max_retries + 1):
        try:
            if provider == "gemini":
                return call_gemini(model, prompt, timeout=timeout)
            return call_openai_compatible(model, prompt, api_base=api_base, timeout=timeout)
        except RuntimeError as e:
            msg = str(e)
            low = msg.lower()
            is_rpm = ("rpm limit exceeded" in low) or ("rate limit" in low) or ("http error 429" in low) or ("httperror 429" in low)
            if not is_rpm or attempt >= max_retries:
                raise
            wait = backoff_seconds[min(attempt, len(backoff_seconds) - 1)]
            print(f"[WARN] Rate limit hit. Sleeping {wait}s then retrying... ({attempt+1}/{max_retries})", file=sys.stderr)
            time.sleep(wait)
            continue


def _history_to_text(history: list[dict]) -> str:
    if not history:
        return "(none)"
    lines = []
    for i, h in enumerate(history, start=1):
        lines.append(f"[CALL {i}] Tool: {h.get('tool_name','')}")
        args = h.get("arguments") or {}
        if isinstance(args, dict) and args:
            lines.append("Arguments:")
            for k, v in args.items():
                lines.append(f"- {k}: {v}")
        lines.append("Result:")
        lines.append(f"- {h.get('result_summary','')}")
        for it in (h.get("result_list_items") or [])[:10]:
            lines.append(f"- {it}")
        lines.append("")
    return "\n".join(lines).strip()


def _looks_specific_value(s: str) -> bool:
    s = (s or "").strip()
    if len(s) < 3:
        return False
    has_numbers = bool(re.search(r"\d", s))
    has_special = bool(re.search(r"[_\-\.]", s))
    return has_numbers or has_special or len(s) > 12


def _extract_specific_candidates(text: str, max_n: int = 12) -> list[str]:
    """Extract candidate identifiers/codes from previous tool results to encourage serial dependencies."""
    if not text:
        return []
    # Tokens containing digits and/or special characters are usually good dependency anchors.
    candidates = re.findall(r"\b(?:0x)?[A-Za-z0-9][A-Za-z0-9_\-\.]*\d+[A-Za-z0-9_\-\.]*\b", text)
    out = []
    seen = set()
    for c in candidates:
        c = c.strip()
        if not _looks_specific_value(c):
            continue
        low = c.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(c)
        if len(out) >= max_n:
            break
    return out


def _args_have_serial_dependency(args: dict, query_text: str, history_text: str) -> bool:
    """Check whether any argument value comes from previous results and is not present in query text."""
    if not isinstance(args, dict):
        return False
    q = (query_text or "").lower()
    prev = (history_text or "").lower()

    for _, v in args.items():
        if v is None:
            continue
        val = str(v).strip()
        if not _looks_specific_value(val):
            continue
        low = val.lower()
        if not low:
            continue
        if low in q:
            continue
        # Exact or substring match against previous tool outputs
        if low in prev:
            return True
    return False


def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_stepwise_bundle(prompts_dir: Path) -> dict:
    """Load stepwise templates from a single bundled file to avoid prompt-file proliferation."""
    bundle = _load_prompt(prompts_dir / "stepwise_serial_bundle.md")
    markers = {
        "QUERY_ONLY_JSON": "===== TEMPLATE: QUERY_ONLY_JSON =====",
        "AGENT_NEXT_CALL_JSON": "===== TEMPLATE: AGENT_NEXT_CALL_JSON =====",
        "TOOL_EXECUTE_JSON": "===== TEMPLATE: TOOL_EXECUTE_JSON =====",
        "FINAL_ANSWER_JSON": "===== TEMPLATE: FINAL_ANSWER_JSON =====",
    }
    # Find marker positions
    pos = {}
    for k, m in markers.items():
        i = bundle.find(m)
        if i == -1:
            raise RuntimeError(f"Missing marker in stepwise bundle: {m}")
        pos[k] = i
    # Slice by sorted positions
    ordered = sorted(pos.items(), key=lambda kv: kv[1])
    out = {}
    for idx, (k, start) in enumerate(ordered):
        start_body = start + len(markers[k])
        end = ordered[idx + 1][1] if idx + 1 < len(ordered) else len(bundle)
        out[k] = bundle[start_body:end].strip()
    return out


def _render(tpl: str, **kwargs) -> str:
    out = tpl
    for k, v in kwargs.items():
        out = out.replace("{" + k + "}", "" if v is None else str(v))
    return out


def _selected_tool_by_name(selected_tools: list[dict], name: str) -> dict | None:
    if not name:
        return None
    for t in selected_tools:
        n = t.get("name") or t.get("tool_name")
        if n == name or (isinstance(n, str) and isinstance(name, str) and n.lower() == name.lower()):
            return t
    return None


def _required_params(tool: dict) -> list[str]:
    schema = tool.get("inputSchema") or tool.get("input_schema") or tool.get("parameters")
    if isinstance(schema, dict):
        req = schema.get("required") or []
        return [str(x) for x in req if x is not None]
    return []


def _validate_args_required(tool: dict, args: dict) -> tuple[bool, list[str]]:
    req = _required_params(tool)
    missing = []
    for r in req:
        if r not in (args or {}):
            missing.append(r)
    return (len(missing) == 0, missing)


def generate_stepwise_serial_xml(
    provider: str,
    model: str,
    api_base: str,
    timeout: int,
    selected_tools: list[dict],
    target_calls: int | None,
    max_calls: int = 5,
    enforce_serial: bool = True,
) -> str:
    prompts_dir = Path(__file__).resolve().parent / "prompts"
    tpls = _load_stepwise_bundle(prompts_dir)
    tpl_query = tpls["QUERY_ONLY_JSON"]
    tpl_agent = tpls["AGENT_NEXT_CALL_JSON"]
    tpl_exec = tpls["TOOL_EXECUTE_JSON"]
    tpl_final = tpls["FINAL_ANSWER_JSON"]

    tools_ctx = build_tools_context_detailed(selected_tools, include_schema=True, include_description=False)

    # 1) Generate query
    q_prompt = _render(
        tpl_query,
        tools_context_detailed=tools_ctx,
        target_calls=str(target_calls or "2-5"),
        target_strategy="serial",
    )
    q_resp = _llm_call(provider, model, q_prompt, api_base, timeout)
    q_parsed = extract_json_object(q_resp)
    if isinstance(q_parsed, dict):
        query_text = str(q_parsed.get("query") or "").strip()
    elif isinstance(q_parsed, str):
        query_text = q_parsed.strip()
    else:
        query_text = str(q_parsed).strip()
    if not query_text:
        query_text = str(q_resp).strip()

    # 2) Stepwise loop: agent decides ONE call, then tool is "executed" (simulated) and appended to history
    history: list[dict] = []
    desired_calls = int(target_calls) if isinstance(target_calls, int) else None

    while len(history) < max_calls:
        step_index = len(history) + 1
        history_text = _history_to_text(history)

        # Agent may need a couple retries to satisfy the serial dependency constraint after step 1
        a_obj = {}
        for agent_try in range(3):
            a_prompt = _render(
                tpl_agent,
                query_text=query_text,
                tools_context_detailed=tools_ctx,
                history_text=history_text,
                target_calls=str(desired_calls or "2-5"),
                step_index=str(step_index),
            )
            if step_index >= 2:
                candidates = _extract_specific_candidates(history_text, max_n=10)
                if candidates:
                    a_prompt = (
                        a_prompt
                        + "\n\nCRITICAL: To satisfy SERIAL dependency, you MUST use at least ONE of these values in an argument:\n- "
                        + "\n- ".join(candidates)
                        + "\n"
                    )
            a_resp = _llm_call(provider, model, a_prompt, api_base, timeout)
            a_obj = extract_json_object(a_resp)
            if not isinstance(a_obj, dict):
                # Model didn't follow JSON object format; retry
                a_obj = {}
                continue

            if bool(a_obj.get("done")):
                break
            tool_name_tmp = (a_obj.get("tool_name") or "").strip()
            args_tmp = a_obj.get("arguments") or {}
            if not isinstance(args_tmp, dict):
                args_tmp = {}
            if step_index >= 2 and not _args_have_serial_dependency(args_tmp, query_text, history_text):
                # retry to enforce dependency
                continue
            if _selected_tool_by_name(selected_tools, tool_name_tmp) is None:
                continue
            # looks acceptable
            break

        if bool(a_obj.get("done")):
            # Ensure minimum 2 calls overall for dataset constraints
            if len(history) >= 2:
                break
            # otherwise ignore early stop and continue

        tool_name = (a_obj.get("tool_name") or "").strip()
        args = a_obj.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}

        tool_def = _selected_tool_by_name(selected_tools, tool_name)
        if tool_def is None:
            # Fallback: stop if model refuses to pick a valid tool
            break

        ok, missing = _validate_args_required(tool_def, args)
        if not ok:
            # Let the agent retry by providing an empty call (will be re-generated next loop)
            # In practice this tends to self-correct on the next iteration due to schema context.
            # To avoid infinite loops, we just fill missing keys with empty strings.
            for m in missing:
                args[m] = ""

        tool_block = build_tools_context_detailed([tool_def], include_schema=True, include_description=False)
        e_prompt = _render(
            tpl_exec,
            query_text=query_text,
            tool_block=tool_block,
            arguments_json=json.dumps(args, ensure_ascii=False),
        )
        e_resp = _llm_call(provider, model, e_prompt, api_base, timeout)
        e_obj = extract_json_object(e_resp)
        if not isinstance(e_obj, dict):
            e_obj = {}
        summary = str(e_obj.get("summary") or "").strip()
        list_title = str(e_obj.get("list_title") or "Key Output").strip()
        list_items = e_obj.get("list_items") or []
        if not isinstance(list_items, list):
            list_items = []

        history.append(
            {
                "tool_name": tool_name,
                "arguments": args,
                "result_summary": summary,
                "result_list_title": list_title,
                "result_list_items": [str(x) for x in list_items if str(x).strip()][:10],
            }
        )

        # Optional: stop exactly at desired_calls if provided
        if desired_calls is not None and len(history) >= desired_calls and len(history) >= 2:
            break

    # 3) Final answer
    history_text = _history_to_text(history)
    f_prompt = _render(tpl_final, query_text=query_text, history_text=history_text)
    f_resp = _llm_call(provider, model, f_prompt, api_base, timeout)
    f_obj = extract_json_object(f_resp)
    if not isinstance(f_obj, dict):
        f_obj = {}
    final_answer = str(f_obj.get("final_answer") or "").strip()
    if not final_answer:
        final_answer = str(f_resp).strip()

    xml_text = build_query_trajectory_xml(
        selected_tools=selected_tools,
        query_text=query_text,
        trajectory_calls=history,
        final_answer=final_answer,
        planning_text=None,
    )

    if enforce_serial:
        try:
            secs = parse_sections(xml_text)
            used = parse_trajectory(secs.get("TRAJECTORY", ""))
            strat = detect_strategy(used, secs.get("QUERY", ""))
            if strat != "serial":
                # Let caller decide to retry; here we just return the XML as-is.
                return xml_text
        except Exception:
            return xml_text

    return xml_text


# Thread-safe stats update lock
_stats_lock = threading.Lock()


def _update_stats_after_generation(text: str):
    """Thread-safe stats update after a sample is generated."""
    try:
        sections = parse_sections(text)
        traj_text = sections.get("TRAJECTORY", "")
        query_text = sections.get("QUERY", "")
        calls = count_calls(traj_text)
        if 2 <= calls <= 5:
            with _stats_lock:
                stats = load_usage_stats()
                hist = stats.setdefault("call_count_hist", {})
                hist[str(calls)] = hist.get(str(calls), 0) + 1
                # Update tool usage by actually used tool name occurrences
                used = parse_trajectory(traj_text)
                counts = stats.setdefault("tool_usage", {})
                for c in used:
                    tool_name = c.get("tool")
                    if tool_name:
                        counts[tool_name] = counts.get(tool_name, 0) + 1
                # Update strategy histogram (serial/parallel)
                strategy = detect_strategy(used, query_text)
                strategy_hist = stats.setdefault("strategy_hist", {})
                strategy_hist[strategy] = strategy_hist.get(strategy, 0) + 1
                save_usage_stats(stats)
    except Exception:
        pass


def _generate_one_sample(args, idx: int, tools_json: str):
    """Generate a single sample. Returns (idx, stem, success, error_msg)."""
    try:
        stem, prompt, selected_tools, meta = sample_for_generation(
            tools_json,
            args.num_tools_min,
            args.num_tools_max,
            args.lock_domain,
            args.lock_category,
            args.lock_random,
            args.lock_random_domain,
            args.lock_random_category,
            balance_tools=not (args.no_balance or args.no_balance_tools),
            balance_calls=not (args.no_balance or args.no_balance_calls),
            balance_strategy=not (args.no_balance or args.no_balance_strategy),
        )

        target_strategy = meta.get("target_strategy")
        if args.force_strategy in ("serial", "parallel"):
            target_strategy = args.force_strategy
            # Re-render one-shot prompt with forced strategy so the instruction is consistent.
            prompt = render_prompt1(selected_tools, meta.get("target_calls"), target_strategy)

        # Generation with optional retries to enforce target_strategy and XML validation
        cleaned_text = ""
        last_error: Exception | None = None
        strategy_retries = args.max_retries if args.enforce_target_strategy else 0
        max_attempts = max(strategy_retries + 1, 2)  # Always allow at least one retry for XML fixes
        for attempt in range(max_attempts):
            if args.stepwise_serial and target_strategy == "serial":
                raw_text = generate_stepwise_serial_xml(
                    provider=args.provider,
                    model=args.model,
                    api_base=args.api_base,
                    timeout=args.timeout,
                    selected_tools=selected_tools,
                    target_calls=meta.get("target_calls"),
                    max_calls=5,
                    enforce_serial=False,  # enforcement handled below if enabled
                )
            else:
                raw_text = _llm_call(args.provider, args.model, prompt, args.api_base, args.timeout)

            try:
                candidate_text = _clean_xml_output(raw_text)
            except Exception as err:
                last_error = err
                continue

            if args.enforce_target_strategy and target_strategy in ("serial", "parallel"):
                try:
                    sections = parse_sections(candidate_text)
                    used = parse_trajectory(sections.get("TRAJECTORY", ""))
                    strategy = detect_strategy(used, sections.get("QUERY", ""))
                    if strategy != target_strategy:
                        last_error = RuntimeError(f"Strategy mismatch: expected {target_strategy}, got {strategy}")
                        continue
                except Exception as err:
                    last_error = err
                    continue

            cleaned_text = candidate_text
            break

        if not cleaned_text:
            raise RuntimeError(f"Failed to generate valid XML after {max_attempts} attempts: {last_error}")
        
        # Save raw output
        out_path = OUT_DIR / "raw" / f"{stem}_{idx:02d}.xml"
        out_path.write_text(cleaned_text, encoding="utf-8")
        print(f"Generated: {out_path}")
        
        # Update stats
        _update_stats_after_generation(cleaned_text)
        
        return (idx, stem, True, None)
    except Exception as e:
        return (idx, None, False, str(e))


def main():
    ap = argparse.ArgumentParser(description="Generate synchronized query+trajectory via Gemini or SiliconFlow")
    ap.add_argument(
        "--tools-json",
        default=str(_DEFAULT_TOOLSET_JSON),
        help="Path to tools JSON (default: tool_set/apis/toolset.json)",
    )
    ap.add_argument("--num-tools-min", type=int, default=3)
    ap.add_argument("--num-tools-max", type=int, default=5)
    ap.add_argument("--lock-domain", default=None, help="Lock to a specific domain")
    ap.add_argument("--lock-category", default=None, help="Lock to a specific category")
    ap.add_argument("--lock-random", action="store_true", help="Randomly lock to either a domain or category")
    ap.add_argument("--lock-random-domain", action="store_true", help="Randomly lock to a domain (not category)")
    ap.add_argument("--lock-random-category", action="store_true", help="Randomly lock to a category (not domain)")
    ap.add_argument("--count", type=int, default=1, help="Number of samples to generate")
    # Balancing switches
    ap.add_argument("--no-balance", action="store_true", help="Disable all balancing (tools, calls, and strategy)")
    ap.add_argument("--no-balance-tools", action="store_true", help="Disable tool sampling balancing only")
    ap.add_argument("--no-balance-calls", action="store_true", help="Disable call-count balancing only")
    ap.add_argument("--no-balance-strategy", action="store_true", help="Disable strategy (serial/parallel) balancing only")

    prov = ap.add_argument_group("provider")
    prov.add_argument("--provider", choices=["gemini", "openai"], required=True)
    prov.add_argument("--model", required=True, help="Provider model id")
    prov.add_argument("--api-base", default="https://api.siliconflow.cn", help="OpenAI-compatible API base (for provider=openai)")
    prov.add_argument("--timeout", type=int, default=180, help="HTTP timeout seconds (default 180, increase for thinking models)")

    gen = ap.add_argument_group("generation")
    gen.add_argument("--stepwise-serial", action="store_true", help="If target_strategy=serial, generate via ToolACE-style stepwise loop (one tool call per iteration)")
    gen.add_argument("--enforce-target-strategy", action="store_true", help="Retry generation until detect_strategy matches target_strategy (best-effort)")
    gen.add_argument("--max-retries", type=int, default=2, help="Max retries when enforce-target-strategy is enabled (default 2)")
    gen.add_argument("--force-strategy", choices=["serial", "parallel"], default=None, help="Override strategy selection and force generating only this strategy")
    
    perf = ap.add_argument_group("performance")
    perf.add_argument("--batch", type=int, default=64, help="Number of samples to generate in parallel (default 1, serial)")
    perf.add_argument("--max-workers", type=int, default=2, help="Maximum number of worker threads for parallel generation (default 2)")

    args = ap.parse_args()

    OUT_DIR.joinpath("raw").mkdir(parents=True, exist_ok=True)

    # Determine if we should use parallel generation
    use_parallel = args.batch > 1 and args.count > 1
    
    if use_parallel:
        # Parallel generation with batching
        max_workers = min(args.max_workers, args.count)
        print(f"Using parallel generation: batch={args.batch}, max_workers={max_workers}")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = []
            for idx in range(1, args.count + 1):
                future = executor.submit(_generate_one_sample, args, idx, args.tools_json)
                futures.append(future)
            
            # Process results as they complete
            completed = 0
            errors = []
            for future in as_completed(futures):
                idx, stem, success, error_msg = future.result()
                completed += 1
                if not success:
                    errors.append((idx, error_msg))
                    print(f"  ✗ Failed sample {idx}: {error_msg}")
                print(f"Progress: {completed}/{args.count}")
            
            if errors:
                print(f"\n⚠️  {len(errors)} samples failed:")
                for idx, msg in errors:
                    print(f"  Sample {idx}: {msg}")
    else:
        # Serial generation (original behavior)
        for idx in range(1, args.count + 1):
            _idx, _stem, success, error_msg = _generate_one_sample(args, idx, args.tools_json)
            if not success:
                print(f"  ✗ Failed sample {idx}: {error_msg}")


if __name__ == "__main__":
    main()
