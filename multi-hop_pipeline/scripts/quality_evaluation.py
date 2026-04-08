#!/usr/bin/env python3
import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from core import (
    OUT_DIR,
    PROMPTS_DIR,
    MANIFESTS_DIR,
    load_manifest,
    save_manifest,
    parse_sections,
    count_calls,
    ensure_dirs,
)
from generate_via_api import call_gemini, call_openai_compatible


def extract_json_object(text: str) -> dict:
    """Extract the first JSON object from model output."""
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


def determine_multi_hop(call_count: int) -> bool:
    """Determine if a sample is multi-hop based on call count."""
    return call_count >= 2


def attempt_xml_fix(text: str) -> tuple[bool, str, list[str]]:
    """
    Attempt to validate and fix common XML formatting errors.
    
    Returns:
        (is_valid, corrected_text, error_messages)
    """
    errors = []
    corrected = text
    
    # Check if it's XML format
    text_stripped = text.strip()
    if not (text_stripped.startswith("<?xml") or text_stripped.startswith("<query_trajectory")):
        return (True, text, [])  # Not XML format, skip validation
    
    # Try to parse the XML
    try:
        ET.fromstring(text_stripped)
        return (True, text, [])  # Valid XML
    except ET.ParseError as e:
        error_msg = str(e)
        errors.append(f"Original XML error: {error_msg}")
        
        # Attempt common fixes
        
        # Fix 1: Mismatched closing tags (e.g., </author> instead of </argument>)
        # Pattern: <tag name="X">Value</wrongtag>
        # This is complex, so we'll use a heuristic
        
        lines = corrected.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            original_line = line
            
            # Fix 1: Remove illegal XML tags with hyphens or spaces (e.g., <Chain-of-Thought Planning>)
            # These tags are invalid in XML and should be removed, keeping only the content
            # Pattern matches: <tag-with-hyphens>, <tag with spaces>, </tag-with-hyphens>
            illegal_opening_pattern = r'<([\w-]+\s+[\w-]+)(\s+[^>]*)?>|<([\w-]+-[^>]+)(\s+[^>]*)?>'
            illegal_closing_pattern = r'</([\w-]+\s+[\w-]+)>|</([\w-]+-[^>]+)>'
            
            # Remove illegal opening tags (keep content if any)
            line = re.sub(illegal_opening_pattern, '', line)
            # Remove illegal closing tags
            line = re.sub(illegal_closing_pattern, '', line)
            
            if original_line != line:
                errors.append(f"Line {i+1}: Removed illegal XML tag with hyphens/spaces (e.g., <Chain-of-Thought Planning>)")
            
            # Fix 2: Mismatched closing tags (e.g., </author> instead of </argument>)
            # Pattern: <tag name="X">Value</wrongtag>
            match = re.search(r'<(\w+)(\s+[^>]*)?>([^<]*)</(\w+)>', line)
            if match:
                opening_tag = match.group(1)
                attributes = match.group(2) or ""
                content = match.group(3)
                closing_tag = match.group(4)
                
                if opening_tag != closing_tag:
                    # Found mismatched tags, fix it
                    line = line.replace(
                        f'<{opening_tag}{attributes}>{content}</{closing_tag}>',
                        f'<{opening_tag}{attributes}>{content}</{opening_tag}>'
                    )
                    errors.append(f"Line {i+1}: Fixed mismatched tag <{opening_tag}> ... </{closing_tag}> → </{opening_tag}>")
            
            fixed_lines.append(line)
        
        # Fix 3: Check for unclosed <call> tags before </trajectory>
        corrected = '\n'.join(fixed_lines)
        
        # Find all <call> tags
        call_pattern = r'<call\s+id="(\d+)"[^>]*>'
        call_matches = list(re.finditer(call_pattern, corrected))
        
        if call_matches:
            # Check each call to see if it's properly closed
            for i, call_match in enumerate(call_matches):
                call_start = call_match.end()
                # Find the next <call> or </trajectory>
                next_call_match = call_matches[i+1] if i+1 < len(call_matches) else None
                trajectory_close_match = re.search(r'</trajectory>', corrected[call_start:])
                
                if next_call_match:
                    call_end = next_call_match.start()
                elif trajectory_close_match:
                    call_end = call_start + trajectory_close_match.start()
                else:
                    continue
                
                call_content = corrected[call_start:call_end]
                
                # Check if this call is closed
                if '</call>' not in call_content:
                    # Find the last </result> in this call's content
                    result_close_matches = list(re.finditer(r'</result>', call_content))
                    if result_close_matches:
                        last_result_close = result_close_matches[-1]
                        insert_pos = call_start + last_result_close.end()
                        # Insert </call> after </result>
                        corrected = corrected[:insert_pos] + '\n    </call>' + corrected[insert_pos:]
                        errors.append(f"Fixed unclosed <call id=\"{call_match.group(1)}\"> tag")
        
        fixed_lines = corrected.split('\n')
        
        corrected = '\n'.join(fixed_lines)
        
        # Try parsing again
        try:
            ET.fromstring(corrected.strip())
            return (True, corrected, errors)  # Successfully fixed
        except ET.ParseError as e2:
            errors.append(f"After fix attempt: {str(e2)}")
            return (False, corrected, errors)  # Could not fix
    except Exception as e:
        errors.append(f"Unexpected error: {str(e)}")
        return (False, text, errors)


def evaluate_query(
    provider: str,
    model: str,
    api_base: str,
    tools_context: str,
    query_text: str,
    multi_hop_flag: bool,
    timeout: int = 120
) -> dict:
    """Evaluate query quality using 3 metrics: toolfit, clarity, naturalness."""
    tpl = (PROMPTS_DIR / "prompt_query_eval.md").read_text(encoding="utf-8")
    prompt = (
        tpl.replace("{tools_context}", tools_context)
           .replace("{query_text}", query_text)
           .replace("{multi_hop_flag}", "true" if multi_hop_flag else "false")
    )
    
    if provider == "gemini":
        resp = call_gemini(model, prompt, timeout=timeout)
    else:
        resp = call_openai_compatible(model, prompt, api_base=api_base, timeout=timeout)
    
    return extract_json_object(resp)


def evaluate_trajectory(
    provider: str,
    model: str,
    api_base: str,
    tools_context: str,
    query_text: str,
    trajectory_text: str,
    final_answer: str,
    multi_hop_flag: bool,
    timeout: int = 120
) -> dict:
    """Evaluate trajectory quality using 3 metrics: success, grounding, efficiency."""
    tpl = (PROMPTS_DIR / "prompt_trajectory_eval.md").read_text(encoding="utf-8")
    prompt = (
        tpl.replace("{tools_context}", tools_context)
           .replace("{query_text}", query_text)
           .replace("{trajectory_text}", trajectory_text)
           .replace("{final_answer}", final_answer)
           .replace("{multi_hop_flag}", "true" if multi_hop_flag else "false")
    )
    
    if provider == "gemini":
        resp = call_gemini(model, prompt, timeout=timeout)
    else:
        resp = call_openai_compatible(model, prompt, api_base=api_base, timeout=timeout)
    
    return extract_json_object(resp)


def extract_scores(query_eval: dict, traj_eval: dict) -> dict:
    """Extract all 6 scores and compute average."""
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
    
    # Compute average (only for non-None scores)
    valid_scores = [s for s in scores.values() if s is not None]
    average = sum(valid_scores) / len(valid_scores) if valid_scores else None
    scores["average"] = average
    
    # Find minimum score
    min_score = min(valid_scores) if valid_scores else None
    scores["min_score"] = min_score
    
    return scores


def check_quality_threshold(scores: dict, min_threshold: float = 4.0, avg_threshold: float = 8.0) -> tuple:
    """
    Check if sample passes quality thresholds.
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
    ap = argparse.ArgumentParser(
        description="Two-stage quality evaluation: basic filter + 6-metric scoring"
    )
    ap.add_argument("--inputs", required=True, help="Directory of raw model outputs (md/txt/xml)")
    ap.add_argument("--min-calls", type=int, default=2, help="Minimum call count (default 2)")
    ap.add_argument("--max-calls", type=int, default=5, help="Maximum call count (default 5)")
    
    ap.add_argument("--provider", choices=["gemini", "openai"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-base", default="https://api.siliconflow.cn", help="OpenAI-compatible base")
    
    ap.add_argument("--min-threshold", type=float, default=4.0, help="Minimum score threshold (default 4.0)")
    ap.add_argument("--avg-threshold", type=float, default=8.0, help="Average score threshold (default 8.0)")
    ap.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds (default 120)")
    
    args = ap.parse_args()
    
    ensure_dirs()
    raw_dir = Path(args.inputs)
    scored_dir = OUT_DIR / "scored"
    scored_dir.mkdir(parents=True, exist_ok=True)
    
    # Stage 1: Basic filter (XML validation + call count)
    print("\n=== STAGE 1: Basic Filter (XML Validation + Call Count) ===")
    files = list(raw_dir.glob("*.md")) + list(raw_dir.glob("*.txt")) + list(raw_dir.glob("*.xml"))
    basic_passed = []
    basic_failed = []
    xml_fixed_count = 0
    
    for p in sorted(set(files)):
        text = p.read_text(encoding="utf-8")
        
        # Step 1: XML validation and auto-fix
        is_valid, corrected_text, xml_errors = attempt_xml_fix(text)
        
        if not is_valid:
            # XML parsing failed and could not be fixed
            print(f"  ❌ {p.name}: XML format error (could not auto-fix)")
            for err in xml_errors:
                print(f"     - {err}")
            basic_failed.append({
                "filename": p.name, 
                "call_count": 0, 
                "reason": f"XML format error: {xml_errors[0] if xml_errors else 'unknown'}"
            })
            continue
        
        if xml_errors:
            # XML was fixed
            print(f"  🔧 {p.name}: XML format auto-corrected")
            for err in xml_errors:
                print(f"     - {err}")
            # Save the corrected version
            p.write_text(corrected_text, encoding="utf-8")
            text = corrected_text
            xml_fixed_count += 1
        
        # Step 2: Parse sections and count calls
        sections = parse_sections(text)
        calls = count_calls(sections.get("TRAJECTORY", ""))
        
        if calls == 0:
            # If call count is 0, it might be a parsing issue
            print(f"  ⚠️  {p.name}: Trajectory parsing returned 0 calls (possible XML structure issue)")
            basic_failed.append({
                "filename": p.name, 
                "call_count": 0, 
                "reason": "Trajectory parsing failed (0 calls detected)"
            })
            continue
        
        if args.min_calls <= calls <= args.max_calls:
            basic_passed.append(p.name)
        else:
            basic_failed.append({"filename": p.name, "call_count": calls, "reason": f"Call count {calls} not in [{args.min_calls}, {args.max_calls}]"})
    
    print(f"\nBasic filter — passed: {len(basic_passed)}, failed: {len(basic_failed)}")
    if xml_fixed_count > 0:
        print(f"  ✅ Auto-fixed XML errors in {xml_fixed_count} file(s)")
    
    # Stage 2: Quality scoring (6 metrics)
    print("\n=== STAGE 2: Quality Scoring (6 Metrics) ===")
    qc_passed = []
    qc_failed = []
    all_scores = {}
    
    for filename in sorted(basic_passed):
        p = raw_dir / filename
        print(f"\nEvaluating: {filename}")
        
        try:
            text = p.read_text(encoding="utf-8")
            sections = parse_sections(text)
            
            tools_context = sections.get("SELECTED_TOOLS", "")
            query_text = sections.get("QUERY", "")
            trajectory_text = sections.get("TRAJECTORY", "")
            final_answer = sections.get("FINAL_ANSWER", "")
            
            call_count = count_calls(trajectory_text)
            multi_hop = determine_multi_hop(call_count)
            
            # Evaluate query
            print(f"  Evaluating query...")
            query_eval = evaluate_query(
                args.provider, args.model, args.api_base,
                tools_context, query_text, multi_hop, args.timeout
            )
            
            # Evaluate trajectory
            print(f"  Evaluating trajectory...")
            traj_eval = evaluate_trajectory(
                args.provider, args.model, args.api_base,
                tools_context, query_text, trajectory_text, final_answer, multi_hop, args.timeout
            )
            
            # Combine evaluations
            combined = {
                "query_evaluation": query_eval,
                "trajectory_evaluation": traj_eval,
                "call_count": call_count,
                "multi_hop": multi_hop,
            }
            
            # Extract scores
            scores = extract_scores(query_eval, traj_eval)
            combined["scores"] = scores
            
            # Check threshold
            passed, reason = check_quality_threshold(scores, args.min_threshold, args.avg_threshold)
            combined["quality_check"] = {"passed": passed, "reason": reason}
            
            # Save detailed evaluation
            (scored_dir / f"{p.stem}.json").write_text(
                json.dumps(combined, ensure_ascii=False, indent=2), 
                encoding="utf-8"
            )
            
            # Track results
            all_scores[filename] = scores
            
            if passed:
                qc_passed.append(filename)
                print(f"  ✓ PASSED - Avg: {scores['average']:.2f}, Min: {scores['min_score']}")
            else:
                qc_failed.append({"filename": filename, "reason": reason, "scores": scores})
                print(f"  ✗ FAILED - {reason}")
        
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            qc_failed.append({"filename": filename, "reason": f"Evaluation error: {str(e)}", "scores": {}})
    
    print(f"\nQuality scoring — passed: {len(qc_passed)}, failed: {len(qc_failed)}")
    
    # Save manifest
    manifest = load_manifest()
    manifest["basic_filter"] = {
        "min_calls": args.min_calls,
        "max_calls": args.max_calls,
        "total_files": len(files),
        "passed": basic_passed,
        "failed": [f["filename"] for f in basic_failed],
        "failed_details": basic_failed,
    }
    manifest["quality_evaluation"] = {
        "min_threshold": args.min_threshold,
        "avg_threshold": args.avg_threshold,
        "evaluated": len(basic_passed),
        "passed": qc_passed,
        "failed": [f["filename"] for f in qc_failed],
        "failed_details": qc_failed,
    }
    save_manifest(manifest)
    print(f"\nManifest saved: {MANIFESTS_DIR / 'pipeline_manifest.json'}")
    
    # Save scores summary CSV
    summaries = OUT_DIR / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    csv_path = summaries / "quality_scores_summary.csv"
    
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("filename,toolfit,clarity,naturalness,success,grounding,efficiency,average,min_score,status\n")
        for filename in sorted(all_scores.keys()):
            scores = all_scores[filename]
            status = "PASSED" if filename in qc_passed else "FAILED"
            row = [
                filename,
                scores.get("toolfit", ""),
                scores.get("clarity", ""),
                scores.get("naturalness", ""),
                scores.get("success", ""),
                scores.get("grounding", ""),
                scores.get("efficiency", ""),
                f"{scores.get('average', ''):.2f}" if scores.get('average') is not None else "",
                scores.get("min_score", ""),
                status,
            ]
            f.write(",".join(map(str, row)) + "\n")
    
    print(f"Scores summary saved: {csv_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("QUALITY EVALUATION SUMMARY")
    print("="*60)
    print(f"Total files: {len(files)}")
    print(f"Basic filter passed: {len(basic_passed)}")
    print(f"Basic filter failed: {len(basic_failed)}")
    print(f"Quality evaluation passed: {len(qc_passed)}")
    print(f"Quality evaluation failed: {len(qc_failed)}")
    print(f"Final high-quality samples: {len(qc_passed)}")
    print("="*60)
    
    # Print failed reasons summary
    if qc_failed:
        print("\nFailed reasons breakdown:")
        reason_counts = {}
        for item in qc_failed:
            reason = item["reason"]
            if "Min score" in reason:
                key = "Min score too low"
            elif "Average score" in reason:
                key = "Average score too low"
            else:
                key = reason
            reason_counts[key] = reason_counts.get(key, 0) + 1
        
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  - {reason}: {count}")


if __name__ == "__main__":
    main()
