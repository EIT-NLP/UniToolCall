#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

# Import core parsing functions
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import parse_sections, parse_trajectory, parse_selected_tools

ROOT = Path(__file__).resolve().parent
# Pipeline root (multi-hop_pipeline/): outputs/, prompts/, db/, optional original_data/
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
# UniToolCall repo root (parent of multi-hop_pipeline/)
UNI_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TOOLSET_JSON = UNI_ROOT / "tool_set" / "apis" / "toolset.json"
DEFAULT_SYSTEM_PROMPT_MD = UNI_ROOT / "test_set" / "prompt_and_format" / "system_prompt.md"
PROJECT_REL_PREFIXES = (
    "outputs/",
    "original_data/",
    "prompts/",
    "db/",
)


def resolve_project_path(path_str: str) -> Path:
    """Resolve common project-relative paths from any working directory."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    if path_str.startswith("multi-hop_pipeline/"):
        return p
    if path_str.startswith(PROJECT_REL_PREFIXES):
        return PIPELINE_ROOT / p
    return p


def load_tools_db(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "tools" in data:
        data = data["tools"]
    elif isinstance(data, dict):
        data = list(data.values())
    if not isinstance(data, list):
        raise ValueError("Unsupported tools json format")
    return data


def build_tools_index(tools: List[dict]) -> Dict[str, dict]:
    idx = {}
    for t in tools:
        name = t.get("name") or t.get("tool_name")
        if name:
            idx[name] = t
            idx[name.lower()] = t
    return idx


def _extract_bullets(text: str) -> list:
    items = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("•") or s.startswith("-") or s.startswith("*"):
            items.append(s.lstrip("•-* ").strip())
        else:
            # simple numbered list
            if len(s) > 2 and (s[0].isdigit() and s[1] in ")."):
                items.append(s[2:].strip())
    return items


def _build_params_table(args: dict) -> dict | None:
    if not args:
        return None
    rows = []
    for k, v in args.items():
        rows.append({"parameter": str(k), "value": v if isinstance(v, (int, float)) else str(v)})
    return {"type": "table", "title": "Input Parameters", "headers": ["parameter", "value"], "rows": rows}


def _obs_value_from_result(result_text: str, arguments: dict) -> str:
    """Build observation.value stringified JSON array per dataset_prompt:
    - first element is a text block with a fixed prefix
    - followed by structured blocks (list/table/kpis) when available
    """
    prefix = "Result analysis completed."
    text = f"{prefix}\n\n{(result_text or '').strip()}".strip()
    arr = [{"type": "text", "text": text}]
    bullets = _extract_bullets(result_text or "")
    if bullets:
        arr.append({"type": "list", "title": "Key Findings", "items": bullets})
    tbl = _build_params_table(arguments or {})
    if tbl:
        arr.append(tbl)
    return json.dumps(arr, ensure_ascii=False)


def detect_strategy(trajectory: List[dict], query_text: str = "") -> str:
    """
    Detect if trajectory is parallel or serial by checking parameter dependencies.
    Returns 'serial' if any later call depends on earlier results, 'parallel' otherwise.
    
    Strategy:
    - Extract specific IDs, codes, and structured values from previous results
    - Check if later arguments use these specific values
    - **Key improvement**: Only consider it a dependency if the value is NOT in the original query
      (if it's in the query, the tool likely got it from the query, not from previous results)
    - Ignore common domain/generic words that appear in natural language
    """
    if len(trajectory) <= 1:
        return "parallel"
    
    import re
    
    # Common generic/domain words that should NOT be considered as dependencies
    # These are words that commonly appear in both results and queries independently
    GENERIC_WORDS = {
        "query", "search", "data", "info", "result", "value", "item", "list", "table",
        "user", "users", "name", "type", "status", "active", "total", "count", "number",
        "title", "description", "content", "text", "message", "error", "success",
        "true", "false", "null", "none", "yes", "no",
        # Domain-specific generic words
        "educational", "education", "learning", "student", "students", "teacher", "course",
        "book", "books", "design", "layout", "theme", "app", "application", "product",
        "technology", "tech", "digital", "online", "interactive", "modern", "new",
        "post", "comment", "comments", "feedback", "engagement", "market", "analysis",
        "bundle", "bundles", "nft", "asset", "assets", "sale", "price", "available",
    }
    
    # Normalize query text for searching
    query_lower = (query_text or "").lower()
    
    # Collect structured values from previous results (IDs, codes, specific identifiers)
    previous_values = {}  # {value: source_index}
    
    for i, call in enumerate(trajectory):
        # Check current call's arguments against previous results
        if i > 0:
            args = call.get("arguments") or {}
            
            # Check each argument value
            for arg_key, arg_value in args.items():
                # Convert to string for comparison
                arg_str = str(arg_value).strip()
                arg_lower = arg_str.lower()
                
                # Skip very short values, booleans, empty
                if len(arg_str) < 3 or arg_lower in ["true", "false", "null", "none", ""]:
                    continue
                
                # Skip if it's a generic word
                if arg_lower in GENERIC_WORDS:
                    continue
                
                # Check if this argument value appears in previous results
                # Support both exact match and substring match (e.g., "789456" matches "post_id_789456")
                found_in_previous = False
                for prev_val in previous_values:
                    # Exact match or arg is substring of previous value or vice versa
                    if (arg_lower == prev_val or 
                        arg_lower in prev_val or 
                        prev_val in arg_lower):
                        found_in_previous = True
                        break
                
                if found_in_previous:
                    # Found a potential dependency
                    # Additional check: only consider it a dependency if it's not a pure common word
                    # and looks like a specific value (has numbers, special chars, or is long enough)
                    has_numbers = bool(re.search(r'\d', arg_str))
                    has_special = bool(re.search(r'[_\-\.]', arg_str))
                    is_specific = has_numbers or has_special or len(arg_str) > 12
                    
                    if is_specific:
                        # **Critical check**: Is this value already in the original query?
                        # If yes, then it's likely from the query, not from previous tool outputs
                        if arg_lower in query_lower:
                            # Value is in query, so NOT a dependency on previous results
                            continue
                        
                        # Value is NOT in query but IS in previous results -> real dependency!
                        return "serial"
        
        # Extract structured values from current call's result
        result_text = call.get("result") or ""
        
        # Extract potential IDs, codes, and structured values
        # Pattern 1: IDs with numbers (e.g., "user123", "id_456", "0x1234...")
        id_patterns = re.findall(r'\b(?:[a-zA-Z]*_)?(?:0x)?[a-zA-Z0-9]*\d+[a-zA-Z0-9]*\b', result_text)
        
        # Pattern 2: Quoted strings (these are often specific values)
        quoted_patterns = re.findall(r'"([^"]+)"', result_text)
        
        # Pattern 3: Key-value pairs (e.g., "id: 12345", "Author: Kim Young-ha")
        # Improved: capture values after colons, including multi-word values
        kv_patterns = re.findall(r':\s*([a-zA-Z0-9_\-\.\s]+?)(?:[,<]|\n|$)', result_text)
        
        # Pattern 4: Extract from structured data (e.g., <item>Author: Kim Young-ha, Title: The Silent Sea</item>)
        # Extract text content from XML-like structures
        item_patterns = re.findall(r'<item>([^<]+)</item>', result_text)
        cell_patterns = re.findall(r'<cell>([^<]+)</cell>', result_text)
        
        # Pattern 5: Extract specific values that might be used as parameters
        # Look for patterns like "Name: Value" or "Key: Value" in text
        name_value_patterns = re.findall(r'(?:Author|Title|Name|ID|Code|Key|Value|Item)\s*:\s*([a-zA-Z0-9_\-\.\s]+?)(?:[,;]|\n|$)', result_text, re.IGNORECASE)
        
        all_candidates = id_patterns + quoted_patterns + kv_patterns + item_patterns + cell_patterns + name_value_patterns
        
        for candidate in all_candidates:
            candidate_lower = candidate.lower().strip()
            
            # Clean up: remove common prefixes and extra whitespace
            # Handle patterns like "Author: Kim Young-ha" -> extract "Kim Young-ha"
            if ':' in candidate_lower:
                parts = candidate_lower.split(':', 1)
                if len(parts) > 1:
                    candidate_lower = parts[1].strip()
            
            # Split by comma and process each part
            parts = [p.strip() for p in candidate_lower.split(',')]
            
            for part in parts:
                if len(part) < 3:
                    continue
                
                # Skip if it's a generic word
                if part in GENERIC_WORDS:
                    continue
                
                # Check if it looks like a specific value
                has_numbers = bool(re.search(r'\d', part))
                has_special = bool(re.search(r'[_\-\.]', part))
                is_long = len(part) > 8
                
                # More lenient: accept if it has special chars, numbers, or is reasonably long
                if has_numbers or has_special or is_long:
                    previous_values[part] = i
    
    return "parallel"


def make_conversations(human_text: str, trajectory: List[dict], final_answer: str) -> List[dict]:
    conv = []
    conv.append({"from": "human", "value": human_text})
    for call in trajectory:
        name = call.get("tool")
        args = call.get("arguments") or {}
        value_fc = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
        conv.append({"from": "function_call", "value": value_fc})
        # observation: as per dataset_prompt, first item must be type=text with a fixed prefix
        result_text = call.get("result") or ""
        conv.append({"from": "observation", "value": _obs_value_from_result(result_text, args)})
    conv.append({"from": "gpt", "value": final_answer or ""})
    return conv


def load_augmented_evaluation_results(aug_scored_dir: Path) -> Dict[str, dict]:
    """Load augmented evaluation results to filter only passing variants."""
    results = {}
    if not aug_scored_dir.exists():
        return results
    
    for p in aug_scored_dir.glob("*.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        stem = p.stem
        results[stem] = data
    
    return results


def load_raw_evaluation_results(scored_dir: Path) -> Dict[str, dict]:
    """Load raw evaluation results from outputs/scored/ to filter passing samples."""
    results = {}
    if not scored_dir.exists():
        return results
    
    for p in scored_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            stem = p.stem
            results[stem] = data
        except Exception:
            continue
    
    return results


def main():
    ap = argparse.ArgumentParser(description="Merge augmented samples into final standardized JSON format")
    ap.add_argument("--augmented-dir", default="outputs/augmented", help="Directory of merged augmented JSONs")
    ap.add_argument("--augmented-scored-dir", default="outputs/augmented_scored", help="Directory of augmented evaluation results")
    ap.add_argument("--raw-dir", default="outputs/raw", help="Directory of raw XML files (used with --use-raw)")
    ap.add_argument("--scored-dir", default="outputs/scored", help="Directory of raw evaluation results (used with --use-raw)")
    ap.add_argument("--use-raw", action="store_true", help="Skip augment: directly process raw XML files from --raw-dir using --scored-dir evaluation results")
    ap.add_argument(
        "--tools-json",
        default=str(DEFAULT_TOOLSET_JSON),
        help="Path to full tools database JSON (default: tool_set/apis/toolset.json)",
    )
    ap.add_argument(
        "--system-file",
        default=str(DEFAULT_SYSTEM_PROMPT_MD),
        help="Path to system prompt file (default: test_set/prompt_and_format/system_prompt.md)",
    )
    ap.add_argument("--output", default="outputs/standardized/final_dataset.json", help="Output JSON file path")
    ap.add_argument("--include-original", action="store_true", help="Include original (non-augmented) queries in output")
    args = ap.parse_args()

    tools_db = load_tools_db(resolve_project_path(args.tools_json))
    tools_idx = build_tools_index(tools_db)
    system_text = resolve_project_path(args.system_file).read_text(encoding="utf-8")
    
    items = []
    
    if args.use_raw:
        # Process raw XML files directly, skipping augment
        raw_dir = resolve_project_path(args.raw_dir)
        scored_dir = resolve_project_path(args.scored_dir)
        
        # Load evaluation results to filter passing samples
        eval_results = load_raw_evaluation_results(scored_dir)
        
        for p in sorted(raw_dir.glob("*.xml")):
            stem = p.stem
            # Check if this sample passed quality evaluation
            eval_data = eval_results.get(stem, {})
            quality_check = eval_data.get("quality_check", {})
            if not quality_check.get("passed", False):
                print(f"  ✗ Skipped {stem} (failed quality check)")
                continue
            
            try:
                # Parse raw XML
                text = p.read_text(encoding="utf-8", errors="ignore")
                secs = parse_sections(text)
                query_text = secs.get("QUERY", "").strip()
                traj_text = secs.get("TRAJECTORY", "")
                final_answer = secs.get("FINAL_ANSWER", "").strip()
                selected_tools = parse_selected_tools(text)
                traj = parse_trajectory(traj_text)
                
                if not query_text or not traj:
                    print(f"  ✗ Skipped {stem} (missing query or trajectory)")
                    continue
                
                # Derive actually-used tool names from trajectory and match schemas
                used_names = []
                for c in traj:
                    nm = c.get("tool")
                    if nm and nm not in used_names:
                        used_names.append(nm)
                used_tools_full = []
                for nm in used_names:
                    t = tools_idx.get(nm) or tools_idx.get(str(nm).lower())
                    if t:
                        used_tools_full.append(t)
                
                # Detect strategy
                strategy = detect_strategy(traj, query_text)
                
                # Create conversation
                conversations = make_conversations(query_text, traj, final_answer)
                properties = {
                    "turn_1": "multi-hop",
                    "strategy": strategy,
                }
                
                item = {
                    "conversations": conversations,
                    "system": system_text,
                    "tools": json.dumps(used_tools_full, ensure_ascii=False),
                    "properties": properties,
                }
                items.append(item)
                print(f"  ✓ Added {stem}")
            except Exception as e:
                print(f"  ✗ Error processing {stem}: {e}")
                continue
    else:
        # Original logic: process augmented samples
        aug_dir = resolve_project_path(args.augmented_dir)
        aug_scored_dir = resolve_project_path(args.augmented_scored_dir)
        
        # Load evaluation results to filter passing augmented querys
        aug_eval_results = load_augmented_evaluation_results(aug_scored_dir)
        
        for p in sorted(aug_dir.glob("*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            selected_tools = data.get("selected_tools") or []
            traj = data.get("trajectory") or []
            final_answer = data.get("final_answer") or ""
            base_query = data.get("query") or ""
            aug = (data.get("augmented_queries") or {})

            # derive actually-used tool names from trajectory and match schemas
            used_names = []
            for c in traj:
                nm = c.get("tool")
                if nm and nm not in used_names:
                    used_names.append(nm)
            used_tools_full = []
            for nm in used_names:
                t = tools_idx.get(nm) or tools_idx.get(str(nm).lower())
                if t:
                    used_tools_full.append(t)

            # Get evaluation results for this file
            stem = p.stem
            eval_result = aug_eval_results.get(stem, {})
            aug_evaluations = eval_result.get("augmented_evaluations", {})
            
            # Detect strategy once for this trajectory, using base_query as reference
            # (augmented queries have same entities, just different phrasing)
            strategy = detect_strategy(traj, base_query)
            
            # helpers
            def add_item(human_text: str, augmented: str = None):
                conversations = make_conversations(human_text, traj, final_answer)
                properties = {
                    "turn_1": "multi-hop",
                    "strategy": strategy,
                }
                if augmented:
                    properties["augmented"] = augmented
                
                item = {
                    "conversations": conversations,
                    "system": system_text,
                    "tools": json.dumps(used_tools_full, ensure_ascii=False),
                    "properties": properties,
                }
                items.append(item)

            # base (original query) - only if --include-original flag is set
            if base_query and args.include_original:
                add_item(base_query)
            
            # augmented variations - only include those that passed quality evaluation
            for aug_type in ["noisy_overspecified", "ambiguous_simplified"]:
                aug_data = aug.get(aug_type)
                if not isinstance(aug_data, dict) or not aug_data.get("query"):
                    continue
                
                # Check if this variant passed quality evaluation
                aug_eval = aug_evaluations.get(aug_type, {})
                if aug_eval.get("quality_check", {}).get("passed", False):
                    aug_query = aug_data["query"]
                    # Map aug_type to simpler names for properties field
                    aug_label = "overspecified" if aug_type == "noisy_overspecified" else "simplified"
                    add_item(aug_query, augmented=aug_label)
                    print(f"  ✓ Added {aug_type} variant")
                else:
                    print(f"  ✗ Skipped {aug_type} variant (failed quality check)")

    out_path = resolve_project_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote standardized dataset: {out_path} (items={len(items)})")


if __name__ == "__main__":
    main()
