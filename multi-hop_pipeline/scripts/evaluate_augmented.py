#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

from core import (
    OUT_DIR,
    PROMPTS_DIR,
    MANIFESTS_DIR,
    load_manifest,
    save_manifest,
    count_calls,
    ensure_dirs,
)
from quality_evaluation import (
    evaluate_query,
    evaluate_trajectory,
    extract_scores,
    check_quality_threshold,
    determine_multi_hop,
)


def load_original_scores() -> Dict[str, dict]:
    """Load original query scores from scored/ directory."""
    scored_dir = OUT_DIR / "scored"
    scores_map = {}
    
    if not scored_dir.exists():
        return scores_map
    
    for p in scored_dir.glob("*.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        scores = data.get("scores", {})
        scores_map[p.stem] = scores
    
    return scores_map


def reconstruct_trajectory_text(trajectory: List[dict]) -> str:
    """Reconstruct trajectory text from parsed structure."""
    lines = []
    for call in trajectory:
        call_id = call.get("id", "")
        tool_name = call.get("tool", "")
        arguments = call.get("arguments", {})
        result = call.get("result", "")
        
        lines.append(f"<call id=\"{call_id}\">")
        lines.append(f"  <tool_name>{tool_name}</tool_name>")
        lines.append(f"  <arguments>")
        for k, v in arguments.items():
            lines.append(f"    <argument name=\"{k}\">{v}</argument>")
        lines.append(f"  </arguments>")
        lines.append(f"  <result>")
        lines.append(f"    <summary>{result}</summary>")
        lines.append(f"  </result>")
        lines.append(f"</call>")
    
    return "\n".join(lines)


def reconstruct_tools_context(selected_tools: List[dict]) -> str:
    """Reconstruct tools context from parsed structure."""
    lines = []
    for tool in selected_tools:
        name = tool.get("name", "")
        domain = tool.get("domain", "")
        category = tool.get("category", "")
        params = tool.get("params", "")
        
        lines.append(f"<tool>")
        lines.append(f"  <name>{name}</name>")
        lines.append(f"  <domain>{domain}</domain>")
        lines.append(f"  <category>{category}</category>")
        lines.append(f"  <params>{params}</params>")
        lines.append(f"</tool>")
    
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate augmented querys and compare with original scores"
    )
    ap.add_argument("--augmented-dir", default="outputs/augmented", help="Directory of augmented JSONs")
    
    ap.add_argument("--provider", choices=["gemini", "openai"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-base", default="https://api.siliconflow.cn", help="OpenAI-compatible base")
    
    ap.add_argument("--min-threshold", type=float, default=4.0, help="Minimum score threshold (default 4.0)")
    ap.add_argument("--avg-threshold", type=float, default=8.0, help="Average score threshold (default 8.0)")
    ap.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds (default 120)")
    
    args = ap.parse_args()
    
    ensure_dirs()
    aug_dir = Path(args.augmented_dir)
    
    # Load original scores for comparison
    original_scores = load_original_scores()
    
    # Output directories
    aug_scored_dir = OUT_DIR / "augmented_scored"
    aug_scored_dir.mkdir(parents=True, exist_ok=True)
    
    summaries_dir = OUT_DIR / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n=== Augmented Query Quality Evaluation ===\n")
    
    # Results tracking
    all_results = []
    passed_files = []
    failed_files = []
    
    for p in sorted(aug_dir.glob("*.json")):
        print(f"\nProcessing: {p.name}")
        
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            
            # Extract components
            selected_tools = data.get("selected_tools", [])
            original_query = data.get("query", "")
            trajectory = data.get("trajectory", [])
            final_answer = data.get("final_answer", "")
            augmented_queries = data.get("augmented_queries", {})
            
            # Reconstruct texts for evaluation
            tools_context = reconstruct_tools_context(selected_tools)
            trajectory_text = reconstruct_trajectory_text(trajectory)
            call_count = len(trajectory)
            multi_hop = determine_multi_hop(call_count)
            
            # Get original scores
            stem = p.stem
            orig_scores = original_scores.get(stem, {})
            
            # Evaluate each augmented variant
            results = {
                "filename": p.name,
                "original_query": original_query,
                "original_scores": orig_scores,
                "augmented_evaluations": {},
                "call_count": call_count,
                "multi_hop": multi_hop,
            }
            
            for aug_type, aug_data in augmented_queries.items():
                if not isinstance(aug_data, dict) or not aug_data.get("query"):
                    continue
                
                aug_query = aug_data.get("query", "")
                aug_how = aug_data.get("how", "")
                
                print(f"  Evaluating {aug_type}...")
                
                # Evaluate augmented query
                print(f"    Query evaluation...")
                query_eval = evaluate_query(
                    args.provider, args.model, args.api_base,
                    tools_context, aug_query, multi_hop, args.timeout
                )
                
                # Evaluate trajectory consistency with augmented query
                print(f"    Trajectory evaluation...")
                traj_eval = evaluate_trajectory(
                    args.provider, args.model, args.api_base,
                    tools_context, aug_query, trajectory_text, final_answer, multi_hop, args.timeout
                )
                
                # Extract scores
                scores = extract_scores(query_eval, traj_eval)
                
                # Check threshold
                passed, reason = check_quality_threshold(scores, args.min_threshold, args.avg_threshold)
                
                # Store results
                results["augmented_evaluations"][aug_type] = {
                    "query": aug_query,
                    "how": aug_how,
                    "query_evaluation": query_eval,
                    "trajectory_evaluation": traj_eval,
                    "scores": scores,
                    "quality_check": {"passed": passed, "reason": reason},
                }
                
                if passed:
                    print(f"    ✓ PASSED - Avg: {scores['average']:.2f}, Min: {scores['min_score']}")
                else:
                    print(f"    ✗ FAILED - {reason}")
            
            # Save detailed results
            (aug_scored_dir / f"{stem}.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            
            all_results.append(results)
            
            # Track pass/fail
            any_passed = any(
                v.get("quality_check", {}).get("passed", False)
                for v in results["augmented_evaluations"].values()
            )
            
            if any_passed:
                passed_files.append(p.name)
            else:
                failed_files.append(p.name)
        
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed_files.append(p.name)
    
    # Generate comparison CSV
    csv_path = summaries_dir / "augmented_scores_comparison.csv"
    
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("filename,aug_type,toolfit,clarity,naturalness,success,grounding,efficiency,average,min_score,status,")
        f.write("orig_toolfit,orig_clarity,orig_naturalness,orig_success,orig_grounding,orig_efficiency,orig_average,orig_min_score,")
        f.write("toolfit_delta,clarity_delta,naturalness_delta,success_delta,grounding_delta,efficiency_delta,average_delta\n")
        
        for result in all_results:
            filename = result["filename"]
            orig_scores = result["original_scores"]
            
            for aug_type, aug_eval in result["augmented_evaluations"].items():
                scores = aug_eval["scores"]
                status = "PASSED" if aug_eval["quality_check"]["passed"] else "FAILED"
                
                # Calculate deltas
                def delta(key):
                    aug_val = scores.get(key)
                    orig_val = orig_scores.get(key)
                    if aug_val is not None and orig_val is not None:
                        return f"{aug_val - orig_val:+.2f}"
                    return ""
                
                row = [
                    filename,
                    aug_type,
                    scores.get("toolfit", ""),
                    scores.get("clarity", ""),
                    scores.get("naturalness", ""),
                    scores.get("success", ""),
                    scores.get("grounding", ""),
                    scores.get("efficiency", ""),
                    f"{scores.get('average', ''):.2f}" if scores.get('average') is not None else "",
                    scores.get("min_score", ""),
                    status,
                    orig_scores.get("toolfit", ""),
                    orig_scores.get("clarity", ""),
                    orig_scores.get("naturalness", ""),
                    orig_scores.get("success", ""),
                    orig_scores.get("grounding", ""),
                    orig_scores.get("efficiency", ""),
                    f"{orig_scores.get('average', ''):.2f}" if orig_scores.get('average') is not None else "",
                    orig_scores.get("min_score", ""),
                    delta("toolfit"),
                    delta("clarity"),
                    delta("naturalness"),
                    delta("success"),
                    delta("grounding"),
                    delta("efficiency"),
                    delta("average"),
                ]
                f.write(",".join(map(str, row)) + "\n")
    
    print(f"\n✓ Comparison CSV saved: {csv_path}")
    
    # Update manifest
    manifest = load_manifest()
    manifest["augmented_evaluation"] = {
        "min_threshold": args.min_threshold,
        "avg_threshold": args.avg_threshold,
        "total_files": len(list(aug_dir.glob("*.json"))),
        "passed": passed_files,
        "failed": failed_files,
    }
    save_manifest(manifest)
    print(f"✓ Manifest updated: {MANIFESTS_DIR / 'pipeline_manifest.json'}")
    
    # Print summary
    print("\n" + "="*60)
    print("AUGMENTED QUERY EVALUATION SUMMARY")
    print("="*60)
    print(f"Total augmented files evaluated: {len(all_results)}")
    print(f"Files with at least one passing variant: {len(passed_files)}")
    print(f"Files with all variants failing: {len(failed_files)}")
    
    # Count by type
    type_stats = {}
    for result in all_results:
        for aug_type, aug_eval in result["augmented_evaluations"].items():
            if aug_type not in type_stats:
                type_stats[aug_type] = {"passed": 0, "failed": 0}
            if aug_eval["quality_check"]["passed"]:
                type_stats[aug_type]["passed"] += 1
            else:
                type_stats[aug_type]["failed"] += 1
    
    print("\nBy augmentation type:")
    for aug_type, stats in sorted(type_stats.items()):
        total = stats["passed"] + stats["failed"]
        print(f"  {aug_type}:")
        print(f"    Passed: {stats['passed']}/{total} ({stats['passed']/total*100:.1f}%)")
        print(f"    Failed: {stats['failed']}/{total} ({stats['failed']/total*100:.1f}%)")
    
    print("="*60)


if __name__ == "__main__":
    main()

