#!/usr/bin/env python3
import json
import random
import re
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

# Paths relative to this directory
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
DB_DIR = ROOT / "db"
PROMPTS_DIR = ROOT / "prompts"
MANIFESTS_DIR = OUT_DIR / "manifests"
MANIFEST_PATH = MANIFESTS_DIR / "pipeline_manifest.json"


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dirs():
    for p in [
        OUT_DIR / "raw",
        # filtered directory intentionally unused to avoid file redundancy
        OUT_DIR / "scored",
        OUT_DIR / "augmented",
        OUT_DIR / "standardized",
        DB_DIR,
        MANIFESTS_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def load_tools_json(path: Path) -> list:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "tools" in data:
        return data["tools"]
    if isinstance(data, dict):
        vals = list(data.values())
        if vals and isinstance(vals[0], dict) and ("name" in vals[0] or "tool_name" in vals[0]):
            return vals
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported tools JSON format")


def tool_signature(tool: dict) -> str:
    name = tool.get("name") or tool.get("tool_name")
    domain = tool.get("domain") or tool.get("Domain")
    category = tool.get("category") or tool.get("Category")
    return f"{domain or 'NA'}::{category or 'NA'}::{name}"


def load_usage_stats() -> dict:
    p = DB_DIR / "usage_stats.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"tool_usage": {}, "samples": 0, "call_count_hist": {}, "strategy_hist": {}}


def save_usage_stats(stats: dict):
    (DB_DIR / "usage_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def update_usage_stats_with_selection(selected_tools: list):
    """Deprecated: previously incremented counts for sampled tools.
    Kept for compatibility but now no-ops. Usage balancing is based on actually used tools after generation.
    """
    stats = load_usage_stats()
    stats["samples"] = int(stats.get("samples", 0)) + 1
    save_usage_stats(stats)


def weighted_sample_tools(tools: list, k: int, lock_domain: str = None, lock_category: str = None, balance_tools: bool = True) -> list:
    def matches(t):
        d = t.get("domain") or t.get("Domain")
        c = t.get("category") or t.get("Category")
        ok = True
        if lock_domain:
            ok = ok and (str(d).lower() == str(lock_domain).lower())
        if lock_category:
            ok = ok and (str(c).lower() == str(lock_category).lower())
        return ok

    pool = [t for t in tools if matches(t)]
    if len(pool) < k:
        raise ValueError(f"Not enough tools after filtering: need {k}, have {len(pool)}")

    if not balance_tools:
        import random as _r
        return _r.sample(pool, k)
    else:
        stats = load_usage_stats().get("tool_usage", {})
        weights = []
        for t in pool:
            sig = tool_signature(t)
            name = t.get("name") or t.get("tool_name")
            cnt = stats.get(sig, stats.get(name, 0))
            weights.append(1.0 / (cnt + 1.0))

        chosen = []
        pool_local = pool.copy()
        weights_local = weights.copy()
        for _ in range(k):
            total = sum(weights_local)
            r = random.random() * total
            acc = 0.0
            idx = 0
            for i, w in enumerate(weights_local):
                acc += w
                if acc >= r:
                    idx = i
                    break
            chosen.append(pool_local[idx])
            pool_local.pop(idx)
            weights_local.pop(idx)
        return chosen


def summarize_tool_params(t: dict) -> str:
    schema = t.get("inputSchema") or t.get("input_schema") or t.get("parameters")
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
        rows.append(f"- name: {name}\n  domain: {domain}\n  category: {category}\n  params: {params}")
    return "\n".join(rows)


def build_tools_context_detailed(selected_tools: list, include_schema: bool = True, include_description: bool = True) -> str:
    """Render a detailed tool context for stepwise prompting.

    We intentionally only include the *selected* tools to keep context small.
    """
    rows = []
    for t in selected_tools:
        name = t.get("name") or t.get("tool_name") or ""
        domain = t.get("domain") or t.get("Domain") or ""
        category = t.get("category") or t.get("Category") or ""
        desc = (t.get("description") or "").strip()
        schema = t.get("inputSchema") or t.get("input_schema") or t.get("parameters")

        lines = [f"TOOL: {name}"]
        if domain or category:
            lines.append(f"domain: {domain}")
            lines.append(f"category: {category}")
        if include_description and desc:
            # Keep description as-is (may be multilingual), but it's still useful context.
            lines.append(f"description: {desc}")
        if include_schema and isinstance(schema, dict):
            lines.append("inputSchema: " + json.dumps(schema, ensure_ascii=False))
        else:
            lines.append(f"params_summary: {summarize_tool_params(t)}")
        rows.append("\n".join(lines))
    return "\n\n".join(rows)


def build_query_trajectory_xml(
    selected_tools: list,
    query_text: str,
    trajectory_calls: list,
    final_answer: str,
    planning_text: str | None = None,
) -> str:
    """Build a valid <query_trajectory> XML string that parse_sections() can parse reliably."""
    root = ET.Element("query_trajectory")

    st = ET.SubElement(root, "selected_tools")
    for t in selected_tools:
        tool_el = ET.SubElement(st, "tool")
        name = t.get("name") or t.get("tool_name") or ""
        domain = t.get("domain") or t.get("Domain") or ""
        category = t.get("category") or t.get("Category") or ""
        params = summarize_tool_params(t)
        ET.SubElement(tool_el, "name").text = str(name)
        ET.SubElement(tool_el, "domain").text = str(domain)
        ET.SubElement(tool_el, "category").text = str(category)
        ET.SubElement(tool_el, "params").text = str(params)

    if planning_text is not None:
        ET.SubElement(root, "planning").text = str(planning_text)

    ET.SubElement(root, "query").text = (query_text or "").strip()

    traj_el = ET.SubElement(root, "trajectory")
    for i, c in enumerate(trajectory_calls, start=1):
        call_el = ET.SubElement(traj_el, "call", {"id": str(i)})
        ET.SubElement(call_el, "tool_name").text = str(c.get("tool_name") or "")

        args_el = ET.SubElement(call_el, "arguments")
        args = c.get("arguments") or {}
        if isinstance(args, dict):
            for k, v in args.items():
                arg_el = ET.SubElement(args_el, "argument", {"name": str(k)})
                # Handle array/list values: convert to JSON string
                if isinstance(v, (list, tuple)):
                    import json as _json
                    arg_el.text = _json.dumps(v, ensure_ascii=False)
                else:
                    arg_el.text = "" if v is None else str(v)

        res_el = ET.SubElement(call_el, "result")
        ET.SubElement(res_el, "summary").text = str(c.get("result_summary") or "")
        data_el = ET.SubElement(res_el, "data")

        list_items = c.get("result_list_items")
        if isinstance(list_items, list) and list_items:
            lst = ET.SubElement(data_el, "list")
            title = c.get("result_list_title") or "Key Output"
            ET.SubElement(lst, "title").text = str(title)
            for it in list_items[:10]:
                ET.SubElement(lst, "item").text = str(it)

    ET.SubElement(root, "final_answer").text = (final_answer or "").strip()

    xml_body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_body


def render_prompt1(selected_tools: list, target_calls: int | None = None, target_strategy: str | None = None) -> str:
    template = (PROMPTS_DIR / "query_trajectory_generation.md").read_text(encoding="utf-8")
    calls_text = str(target_calls) if target_calls else "2-5"
    
    # Build strategy guidance text
    if target_strategy == "serial":
        strategy_text = "**IMPORTANT**: Please generate a SERIAL trajectory where later tool calls depend on results from earlier calls (e.g., first call returns an ID, and later calls use that ID)."
    elif target_strategy == "parallel":
        strategy_text = "**IMPORTANT**: Please generate a PARALLEL trajectory where tool calls are independent and do not depend on each other's results (all parameters should come from the query text, not from previous tool outputs)."
    else:
        strategy_text = "**IMPORTANT**: Please generate either a SERIAL trajectory (later calls depend on earlier results) or a PARALLEL trajectory (calls are independent). Choose based on what makes the most sense for the query."
    
    return (
        template
        .replace("{tools_context}", build_tools_context(selected_tools))
        .replace("{target_calls}", calls_text)
        .replace("{strategy_guidance}", strategy_text)
    )


def write_prompt_with_context(stem: str, prompt_text: str, selected_tools: list, meta: dict):
    ensure_dirs()
    prompt_path = OUT_DIR / "prompts" / f"{stem}_query_trajectory_generation.md"
    ctx_path = OUT_DIR / "prompts" / f"{stem}_context.json"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    ctx = {"selected_tools": selected_tools, "meta": meta}
    ctx_path.write_text(json.dumps(ctx, ensure_ascii=False, indent=2), encoding="utf-8")
    return prompt_path, ctx_path


CALL_HEADER_RE = re.compile(r"^\[CALL\s+(\d+)\]\s+Tool:\s*(.+)$")


def parse_sections(text: str) -> dict:
    """Parse sections from either XML or Markdown format."""
    sections = {"SELECTED_TOOLS": "", "QUERY": "", "TRAJECTORY": "", "FINAL_ANSWER": ""}
    
    # Try XML parsing first
    text_stripped = text.strip()
    if text_stripped.startswith("<?xml") or text_stripped.startswith("<query_trajectory"):
        try:
            root = ET.fromstring(text_stripped)
            
            # Parse selected_tools
            selected_tools_elem = root.find("selected_tools")
            if selected_tools_elem is not None:
                tools_lines = []
                for tool in selected_tools_elem.findall("tool"):
                    name = tool.find("name")
                    domain = tool.find("domain")
                    category = tool.find("category")
                    params = tool.find("params")
                    tools_lines.append(f"- name: {name.text if name is not None else ''}")
                    tools_lines.append(f"  domain: {domain.text if domain is not None else ''}")
                    tools_lines.append(f"  category: {category.text if category is not None else ''}")
                    tools_lines.append(f"  params: {params.text if params is not None else ''}")
                sections["SELECTED_TOOLS"] = "\n".join(tools_lines)
            
            # Parse query
            query_elem = root.find("query")
            if query_elem is not None:
                sections["QUERY"] = (query_elem.text or "").strip()
            
            # Parse trajectory
            traj_elem = root.find("trajectory")
            if traj_elem is not None:
                traj_lines = []
                for call in traj_elem.findall("call"):
                    call_id = call.get("id", "")
                    tool_name = call.find("tool_name")
                    traj_lines.append(f"[CALL {call_id}] Tool: {tool_name.text if tool_name is not None else ''}")
                    
                    traj_lines.append("Arguments:")
                    args_elem = call.find("arguments")
                    if args_elem is not None:
                        for arg in args_elem.findall("argument"):
                            arg_name = arg.get("name", "")
                            arg_value = arg.text or ""
                            traj_lines.append(f"- {arg_name}: {arg_value}")
                    
                    traj_lines.append("Result:")
                    result_elem = call.find("result")
                    if result_elem is not None:
                        summary = result_elem.find("summary")
                        if summary is not None and summary.text:
                            traj_lines.append(f"- {summary.text}")
                        # Convert structured data back to readable format
                        data_elem = result_elem.find("data")
                        if data_elem is not None:
                            for child in data_elem:
                                traj_lines.append(f"- {child.tag}")
                                traj_lines.append(_format_xml_data_element(child, indent=2))
                    traj_lines.append("")
                sections["TRAJECTORY"] = "\n".join(traj_lines)
            
            # Parse final_answer
            final_elem = root.find("final_answer")
            if final_elem is not None:
                sections["FINAL_ANSWER"] = (final_elem.text or "").strip()
            
            return sections
        except ET.ParseError:
            pass  # Fall back to Markdown parsing
    
    # Markdown parsing (backward compatibility)
    current = None
    for line in text.splitlines():
        if line.strip().startswith("## "):
            title = line.strip()[3:].strip()
            if title in sections:
                current = title
                continue
        if current:
            sections[current] += (line + "\n")
    for k in sections:
        sections[k] = sections[k].strip()
    return sections


def _format_xml_data_element(elem, indent=0):
    """Helper to format XML data elements back to readable text."""
    lines = []
    prefix = "  " * indent
    
    if elem.tag == "chart_data":
        title = elem.find("title")
        if title is not None and title.text:
            lines.append(f"{prefix}- title: {title.text}")
        items = elem.find("items")
        if items is not None:
            lines.append(f"{prefix}- items/series:")
            for item in items.findall("item"):
                name = item.get("name", "")
                value = item.text or ""
                lines.append(f"{prefix}  - {name}: {value}")
        total = elem.find("total")
        if total is not None and total.text:
            lines.append(f"{prefix}- total: {total.text}")
        unit = elem.find("unit")
        if unit is not None and unit.text:
            lines.append(f"{prefix}- unit: {unit.text}")
    
    elif elem.tag == "table":
        title = elem.find("title")
        if title is not None and title.text:
            lines.append(f"{prefix}- title: {title.text}")
        headers = elem.find("headers")
        if headers is not None:
            header_texts = [h.text for h in headers.findall("header") if h.text]
            lines.append(f"{prefix}- columns: {header_texts}")
        rows = elem.find("rows")
        if rows is not None:
            lines.append(f"{prefix}- rows:")
            for row in rows.findall("row"):
                cells = [c.text or "" for c in row.findall("cell")]
                if len(cells) >= 2:
                    lines.append(f"{prefix}  - {cells[0]}: {cells[1]}")
    
    elif elem.tag == "list":
        title = elem.find("title")
        if title is not None and title.text:
            lines.append(f"{prefix}- title: {title.text}")
        lines.append(f"{prefix}- items:")
        for item in elem.findall("item"):
            if item.text:
                lines.append(f"{prefix}  - {item.text}")
    
    elif elem.tag == "kpis":
        for kpi in elem.findall("kpi"):
            name = kpi.get("name", "")
            value = kpi.text or ""
            lines.append(f"{prefix}- {name}: {value}")
    
    return "\n".join(lines)


def count_calls(trajectory_text: str) -> int:
    return sum(1 for line in trajectory_text.splitlines() if CALL_HEADER_RE.match(line.strip()))


def load_manifest() -> dict:
    ensure_dirs()
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"basic_filter": {}, "qc": {}}


def save_manifest(m: dict):
    ensure_dirs()
    MANIFEST_PATH.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


def find_context_for_stem(stem: str) -> Path | None:
    ctx = OUT_DIR / "prompts" / f"{stem}_context.json"
    return ctx if ctx.exists() else None


def _choose_target_calls(call_hist: dict, min_calls: int = 2, max_calls: int = 5) -> int:
    """Pick a desired total call count in [min,max] favoring underrepresented counts."""
    candidates = list(range(min_calls, max_calls + 1))
    usages = [(int(call_hist.get(str(c), 0)), c) for c in candidates]
    min_use = min(u for u, _ in usages)
    least = [c for u, c in usages if u == min_use]
    return random.choice(least)


def _choose_target_strategy(strategy_hist: dict) -> str | None:
    """Pick a desired strategy type (serial/parallel) favoring underrepresented counts.
    Returns 'serial', 'parallel', or None (no preference) if both are balanced."""
    serial_count = int(strategy_hist.get("serial", 0))
    parallel_count = int(strategy_hist.get("parallel", 0))
    
    if serial_count == parallel_count:
        return None  # Balanced, no preference
    elif serial_count < parallel_count:
        return "serial"  # Favor serial to balance
    else:
        return "parallel"  # Favor parallel to balance


def _collect_field_values(tools: list, key_primary: str, key_alt: str) -> list:
    vals = []
    for t in tools:
        v = t.get(key_primary) or t.get(key_alt)
        if v is not None:
            vals.append(str(v))
    return sorted(set(vals))


def _pool_size(tools: list, k: int, lock_domain: str | None, lock_category: str | None) -> int:
    def matches(t):
        d = t.get("domain") or t.get("Domain")
        c = t.get("category") or t.get("Category")
        if lock_domain and str(d).lower() != str(lock_domain).lower():
            return False
        if lock_category and str(c).lower() != str(lock_category).lower():
            return False
        return True
    return sum(1 for t in tools if matches(t))


def cmd_sample(tools_json: str, num_min: int, num_max: int, lock_domain: str | None, lock_category: str | None, lock_random: bool = False, balance_tools: bool = True, balance_calls: bool = True, balance_strategy: bool = True):
    tools = load_tools_json(Path(tools_json))
    k = random.randint(num_min, num_max)

    strategy = {"type": None, "value": None, "enabled": False}
    use_lock_domain, use_lock_category = lock_domain, lock_category
    if not lock_domain and not lock_category and lock_random:
        # randomly decide lock type and pick a feasible value; fallback if not enough pool
        strategy["enabled"] = True
        candidates = ["domain", "category"]
        random.shuffle(candidates)
        chosen_lock = None
        chosen_value = None
        for field in candidates:
            values = _collect_field_values(tools, field, field.capitalize())
            random.shuffle(values)
            ok = False
            for v in values[: min(20, len(values))]:
                if field == "domain":
                    if _pool_size(tools, k, v, None) >= k:
                        chosen_lock = "domain"
                        chosen_value = v
                        ok = True
                        break
                else:
                    if _pool_size(tools, k, None, v) >= k:
                        chosen_lock = "category"
                        chosen_value = v
                        ok = True
                        break
            if ok:
                break
        if chosen_lock == "domain":
            use_lock_domain = chosen_value
        elif chosen_lock == "category":
            use_lock_category = chosen_value
        strategy["type"] = chosen_lock
        strategy["value"] = chosen_value

    selected = weighted_sample_tools(tools, k, use_lock_domain, use_lock_category, balance_tools=balance_tools)
    # Call-count balancing (soft suggestion)
    target_calls = None
    if balance_calls:
        stats = load_usage_stats()
        target_calls = _choose_target_calls(stats.get("call_count_hist", {}), 2, 5)
    
    # Strategy balancing (serial/parallel)
    target_strategy = None
    if balance_strategy:
        stats = load_usage_stats()
        target_strategy = _choose_target_strategy(stats.get("strategy_hist", {}))
    
    meta = {
        "num_tools": k,
        "lock_domain": use_lock_domain,
        "lock_category": use_lock_category,
        "lock_random": bool(lock_random),
        "lock_strategy": strategy,
        "target_calls": target_calls,
        "target_strategy": target_strategy,
        "timestamp": now_ts(),
    }
    stem = meta["timestamp"]
    prompt = render_prompt1(selected, target_calls, target_strategy)
    p_path, c_path = write_prompt_with_context(stem, prompt, selected, meta)
    # Note: usage balancing now updated after generation using actually used tools
    return p_path, c_path


def sample_for_generation(tools_json: str, num_min: int, num_max: int, lock_domain: str | None, lock_category: str | None, lock_random: bool = False, lock_random_domain: bool = False, lock_random_category: bool = False, balance_tools: bool = True, balance_calls: bool = True, balance_strategy: bool = True):
    """Sample tools and build Prompt1 in-memory without writing files.
    Returns: (stem, prompt_text, selected_tools, meta)
    """
    tools = load_tools_json(Path(tools_json))
    # Reuse sampling logic via cmd_sample but avoid file IO
    k = random.randint(num_min, num_max)

    # lock-random selection (same logic as cmd_sample)
    strategy = {"type": None, "value": None, "enabled": False}
    use_lock_domain, use_lock_category = lock_domain, lock_category
    
    # Priority: specific lock > lock_random_domain/category > lock_random
    if not lock_domain and not lock_category:
        if lock_random_domain:
            # Only consider domain
            strategy["enabled"] = True
            values = _collect_field_values(tools, "domain", "Domain")
            random.shuffle(values)
            for v in values[: min(20, len(values))]:
                if _pool_size(tools, k, v, None) >= k:
                    use_lock_domain = v
                    strategy["type"] = "domain"
                    strategy["value"] = v
                    break
        elif lock_random_category:
            # Only consider category
            strategy["enabled"] = True
            values = _collect_field_values(tools, "category", "Category")
            random.shuffle(values)
            for v in values[: min(20, len(values))]:
                if _pool_size(tools, k, None, v) >= k:
                    use_lock_category = v
                    strategy["type"] = "category"
                    strategy["value"] = v
                    break
        elif lock_random:
            # Original logic: randomly choose between domain and category
            strategy["enabled"] = True
            candidates = ["domain", "category"]
            random.shuffle(candidates)
            chosen_lock = None
            chosen_value = None
            for field in candidates:
                values = _collect_field_values(tools, field, field.capitalize())
                random.shuffle(values)
                ok = False
                for v in values[: min(20, len(values))]:
                    if field == "domain":
                        if _pool_size(tools, k, v, None) >= k:
                            chosen_lock = "domain"; chosen_value = v; ok = True; break
                    else:
                        if _pool_size(tools, k, None, v) >= k:
                            chosen_lock = "category"; chosen_value = v; ok = True; break
                if ok:
                    break
            if chosen_lock == "domain":
                use_lock_domain = chosen_value
            elif chosen_lock == "category":
                use_lock_category = chosen_value
            strategy["type"] = chosen_lock
            strategy["value"] = chosen_value

    selected = weighted_sample_tools(tools, k, use_lock_domain, use_lock_category, balance_tools=balance_tools)
    target_calls = None
    if balance_calls:
        stats = load_usage_stats()
        target_calls = _choose_target_calls(stats.get("call_count_hist", {}), 2, 5)
    
    target_strategy = None
    if balance_strategy:
        stats = load_usage_stats()
        target_strategy = _choose_target_strategy(stats.get("strategy_hist", {}))
    
    meta = {
        "num_tools": k,
        "lock_domain": use_lock_domain,
        "lock_category": use_lock_category,
        "lock_random": bool(lock_random),
        "lock_strategy": strategy,
        "target_calls": target_calls,
        "target_strategy": target_strategy,
        "timestamp": now_ts(),
    }
    stem = meta["timestamp"]
    prompt = render_prompt1(selected, target_calls, target_strategy)
    # Note: usage balancing now updated after generation using actually used tools
    return stem, prompt, selected, meta


def cmd_basic_filter(inputs_dir: str, min_calls: int = 2, max_calls: int = 5):
    """Check raw outputs and record nonconforming filenames to manifest, without copying files.
    Returns: dict with passed and failed lists.
    """
    ensure_dirs()
    in_dir = Path(inputs_dir)
    stats = load_usage_stats()
    call_hist = stats.setdefault("call_count_hist", {})
    files = list(in_dir.glob("*.md")) + list(in_dir.glob("*.txt")) + list(in_dir.glob("*.xml"))
    passed, failed = [], []
    for p in sorted(set(files)):
        text = p.read_text(encoding="utf-8")
        sections = parse_sections(text)
        calls = count_calls(sections.get("TRAJECTORY", ""))
        call_hist[str(calls)] = call_hist.get(str(calls), 0) + 1
        if min_calls <= calls <= max_calls:
            passed.append(p.name)
        else:
            failed.append(p.name)
    save_usage_stats(stats)
    manifest = load_manifest()
    manifest["basic_filter"] = {
        "min_calls": min_calls,
        "max_calls": max_calls,
        "checked_files": sorted([p.name for p in set(files)]),
        "passed": passed,
        "failed": failed,
    }
    save_manifest(manifest)
    return {"passed": passed, "failed": failed}




def render_augment_prompt(query_text: str, trajectory_text: str, tools_context: str) -> str:
    tpl = (PROMPTS_DIR / "query_augment.md").read_text(encoding="utf-8")
    return (
        tpl.replace("{query_text}", query_text)
           .replace("{trajectory_text}", trajectory_text)
           .replace("{tools_context}", tools_context)
    )


def parse_query_and_traj(sample_text: str):
    secs = parse_sections(sample_text)
    return secs.get("QUERY", ""), secs.get("TRAJECTORY", "")


def cmd_make_augment(inputs_dir: str):
    ensure_dirs()
    in_dir = Path(inputs_dir)
    out_dir = OUT_DIR / "augment_prompts"
    files = list(in_dir.glob("*.md")) + list(in_dir.glob("*.txt")) + list(in_dir.glob("*.xml"))
    out = []
    for p in sorted(set(files)):
        stem = p.stem
        raw = p.read_text(encoding="utf-8")
        query_text, traj_text = parse_query_and_traj(raw)
        ctx_path = find_context_for_stem(stem)
        tools_context = ""
        if ctx_path:
            ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
            tools_context = build_tools_context(ctx.get("selected_tools", []))
        prompt = render_augment_prompt(query_text, traj_text, tools_context)
        out_path = out_dir / f"{stem}_query_augment.md"
        out_path.write_text(prompt, encoding="utf-8")
        out.append(out_path)
    return out


def parse_selected_tools(text: str) -> list:
    """Parse SELECTED_TOOLS from plain text.
    Robust to missing blank lines by detecting lines that start a new tool with "- name:".
    Expects keys: name, domain, category, params (lowercase).
    """
    tools = []
    block = parse_sections(text).get("SELECTED_TOOLS", "")
    if not block.strip():
        return tools
    current = {"name": None, "domain": None, "category": None, "params_summary": None}
    def flush():
        nonlocal current
        if current.get("name"):
            tools.append({
                "name": current.get("name"),
                "domain": current.get("domain"),
                "category": current.get("category"),
                "params_summary": current.get("params_summary"),
            })
        current = {"name": None, "domain": None, "category": None, "params_summary": None}

    for raw in block.splitlines():
        s = raw.strip()
        low = s.lower()
        if not s:
            # blank line: consider boundary but keep accumulating until next name
            continue
        if low.startswith("- name:") or low.startswith("name:"):
            # new tool begins; flush previous
            if current.get("name"):
                flush()
            current["name"] = s.split(":", 1)[1].strip()
            continue
        if low.startswith("domain:"):
            current["domain"] = s.split(":", 1)[1].strip()
        elif low.startswith("category:"):
            current["category"] = s.split(":", 1)[1].strip()
        elif low.startswith("params:"):
            current["params_summary"] = s.split(":", 1)[1].strip()
    # flush last
    if current.get("name"):
        flush()
    return tools


def parse_trajectory(traj_text: str) -> list:
    calls = []
    lines = traj_text.splitlines()
    i = 0
    current = None
    mode = None
    while i < len(lines):
        line = lines[i]
        m = CALL_HEADER_RE.match(line.strip())
        if m:
            if current:
                calls.append(current)
            current = {"index": int(m.group(1)), "tool": m.group(2).strip(), "arguments": {}, "result": []}
            mode = None
            i += 1
            continue
        if current:
            s = line.strip()
            if s.lower().startswith("arguments:"):
                mode = "args"
            elif s.lower().startswith("result:"):
                mode = "res"
            elif s.startswith("-"):
                content = s[1:].strip()
                if mode == "args":
                    if ":" in content:
                        k, v = content.split(":", 1)
                        current["arguments"][k.strip()] = v.strip()
                elif mode == "res":
                    current["result"].append(content)
        i += 1
    if current:
        calls.append(current)
    for c in calls:
        c["result"] = "\n".join(c["result"]).strip()
    return sorted(calls, key=lambda x: x.get("index", 0))
