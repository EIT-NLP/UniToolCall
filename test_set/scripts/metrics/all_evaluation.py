#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from data_evaluation import evaluate_conversation  # type: ignore
from datetime import datetime
from collections import defaultdict

# Default dirs: predictions and eval outputs (this script lives under test_set/scripts/metrics/)
_TEST_SET_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_PREDICTIONS_DIR = _TEST_SET_ROOT / "predictions"
_DEFAULT_EVAL_RESULTS_DIR = _TEST_SET_ROOT / "eval_results"

# Mapping from dataset names to metric file stems (no extension)
# Legacy filename mapping when names differ
DATA_TO_METRICS_MAP: Dict[str, List[str]] = {
    "test_converted_alpaca": ["alpaca_qwen"],
    "test_converted_ACEBench": ["ACEBench_qwen"],
    "test_converted_APIBank_level3": ["APIBank_level3_qwen"],
    "test_converted_ComplexFunc": ["ComplexFunc_qwen"],
    "test_converted_HammerBench_single": ["HammerBench_single_qwen"],
    "test_converted_Seal-Tools": ["Seal-Tools_qwen"],
    "test_converted_bfcl": ["bfcl_qwen"],
}


def extract_turn_info(gt_item: Dict) -> Tuple[str, List[Tuple[str, int]]]:
    """Return (turn_type, [(hop_type, turn_index), ...]) from GT properties or conversations."""
    properties = gt_item.get("properties", {})
    
    turn_keys = [k for k in properties.keys() if k.startswith("turn_")]
    if not turn_keys:
        conversations = gt_item.get("conversations", [])
        if not conversations:
            return "single-turn", []
        
        # Identify turn boundaries
        turns = identify_turns_from_conversations(conversations)
        
        if not turns:
            return "single-turn", []
        
        turn_count = len(turns)
        turn_type = "multi-turn" if turn_count >= 2 else "single-turn"
        
        turn_info = []
        for idx, (start_idx, end_idx) in enumerate(turns, start=1):
            turn_convs = conversations[start_idx:end_idx + 1]
            function_call_count = sum(1 for conv in turn_convs if conv.get("from") == "function_call")
            hop_type = "multi-hop" if function_call_count >= 2 else "single-hop"
            turn_info.append((hop_type, idx))
        
        return turn_type, turn_info
    
    def get_turn_number(key: str) -> int:
        try:
            return int(key.replace("turn_", ""))
        except:
            return 0
    
    turn_keys.sort(key=get_turn_number)
    
    turn_count = len(turn_keys)
    turn_type = "multi-turn" if turn_count >= 2 else "single-turn"
    
    turn_info = []
    for turn_key in turn_keys:
        hop_type = properties.get(turn_key, "single-hop")
        if hop_type not in ["single-hop", "multi-hop"]:
            hop_type = "single-hop"  # default
        turn_index = get_turn_number(turn_key)
        turn_info.append((hop_type, turn_index))
    
    return turn_type, turn_info


def identify_turns_from_conversations(conversations: List[Dict]) -> List[Tuple[int, int]]:
    """
    Identify turn boundaries from conversations
    Returns list of (start_idx, end_idx) inclusive per turn
    Turn boundary: from each human message until the next human (or EOF)
    """
    turns = []
    turn_start = None
    
    for idx, conv in enumerate(conversations):
        if conv.get("from") == "human":
            if turn_start is not None:
                turns.append((turn_start, idx - 1))
            turn_start = idx

    if turn_start is not None:
        turns.append((turn_start, len(conversations) - 1))
    
    return turns


def calculate_turn_level_metrics(
    pred_item: Dict, 
    gt_item: Dict,
    turn_index: int
) -> Tuple[float, float, float, float, bool]:
    """
    Compute strict/flexible metrics for one turn
    
    Args:
        pred_item: Prediction item (conversations)
        gt_item: Ground-truth item (conversations)
        turn_index: Turn index (1-based, turn_1, turn_2, ...)
    
    Returns:
        strict_precision_at_1, flexible_precision_at_1, strict_parameter_accuracy,
        flexible_parameter_accuracy, has_matched
    """
    # Get conversations
    pred_conversations = pred_item.get("conversations", [])
    gt_conversations = gt_item.get("conversations", [])
    
    # Identify turn boundaries
    pred_turns = identify_turns_from_conversations(pred_conversations)
    gt_turns = identify_turns_from_conversations(gt_conversations)
    
    # Validate turn index (1-based to 0-based)
    turn_idx_0 = turn_index - 1
    if turn_idx_0 >= len(pred_turns) or turn_idx_0 >= len(gt_turns):
        return 0.0, 0.0, 0.0, 0.0, False
    
    # Slice range for this turn
    pred_start, pred_end = pred_turns[turn_idx_0]
    gt_start, gt_end = gt_turns[turn_idx_0]
    
    # Extract function calls in this turn
    from data_evaluation import (
        extract_function_calls, match_function_calls_unordered, 
        parse_tools_from_conversation, normalize_tool_name,
        evaluate_function_call_flexible
    )
    
    pred_turn_convs = pred_conversations[pred_start:pred_end + 1]
    gt_turn_convs = gt_conversations[gt_start:gt_end + 1]
    
    pred_turn_calls = extract_function_calls(pred_turn_convs)
    gt_turn_calls = extract_function_calls(gt_turn_convs)
    
    # If no function calls, return zeros
    if len(pred_turn_calls) == 0:
        return 0.0, 0.0, 0.0, 0.0, False
    
    # Load tool definitions
    tools_def = parse_tools_from_conversation(gt_item)
    
    # Match function calls
    matches = match_function_calls_unordered(pred_turn_calls, gt_turn_calls, tools_def)
    
    # Strict metrics
    tool_name_matches = []
    
    # Flexible metrics
    all_pred_tool_name_matches = []
    all_matched_flexible_matches = []
    
    # Parameter accuracy (denominator: all predicted calls)
    all_pred_tool_name_and_parameter_matches = []
    all_pred_tool_name_and_flexible_matches = []
    
    for match in matches:
        eval_result = match["eval_result"]
        # Count tool_name_match for flexible precision
        if match["pred_index"] >= 0:
            all_pred_tool_name_matches.append(eval_result["tool_name_match"])
            
            # Count name+param matches among predicted calls
            tool_name_match = eval_result["tool_name_match"]
            if tool_name_match and match["gt_index"] >= 0:
                all_pred_tool_name_and_parameter_matches.append(eval_result["parameter_match"])
                # Flexible match for accuracy / fixed denominator
                pred_call = pred_turn_calls[match["pred_index"]]
                gt_call = gt_turn_calls[match["gt_index"]]
                tool_name = gt_call.get("tool_name") or pred_call.get("tool_name")
                tool_defaults = None
                if tool_name:
                    for tool_name_in_def, defaults in tools_def.items():
                        if normalize_tool_name(tool_name) == normalize_tool_name(tool_name_in_def):
                            tool_defaults = defaults
                            break
                flexible_match = evaluate_function_call_flexible(pred_call, gt_call, tool_defaults, rouge_threshold=0.7)
                all_pred_tool_name_and_flexible_matches.append(flexible_match)
                # Also used for flexible accuracy
                all_matched_flexible_matches.append(flexible_match)
            else:
                all_pred_tool_name_and_parameter_matches.append(False)
                all_pred_tool_name_and_flexible_matches.append(False)
        
        # Only matched pred calls (gt_index >= 0)
        if match["gt_index"] >= 0:
            tool_name_matches.append(eval_result["tool_name_match"])
    
    # Strict precision: all preds matched with correct names
    matched_pred_count = sum(1 for m in matches if m["pred_index"] >= 0 and m["gt_index"] >= 0)
    if matched_pred_count == len(pred_turn_calls) and len(tool_name_matches) == len(pred_turn_calls):
        strict_precision_at_1 = 1.0 if all(tool_name_matches) else 0.0
    else:
        strict_precision_at_1 = 0.0
    
    # Flexible precision: fraction of preds with matching tool names
    flexible_precision_at_1 = (
        sum(all_pred_tool_name_matches) / len(all_pred_tool_name_matches)
        if len(all_pred_tool_name_matches) > 0
        else 0.0
    )
    
    # Parameter accuracy (denominator: all predicted calls)
    strict_parameter_accuracy = (
        sum(all_pred_tool_name_and_parameter_matches) / len(all_pred_tool_name_and_parameter_matches)
        if len(all_pred_tool_name_and_parameter_matches) > 0
        else 0.0
    )
    
    flexible_parameter_accuracy = (
        sum(all_pred_tool_name_and_flexible_matches) / len(all_pred_tool_name_and_flexible_matches)
        if len(all_pred_tool_name_and_flexible_matches) > 0
        else 0.0
    )
    
    # Returns metrics and whether any call matched
    has_matched = len(all_matched_flexible_matches) > 0
    return strict_precision_at_1, flexible_precision_at_1, \
           strict_parameter_accuracy, flexible_parameter_accuracy, has_matched


def extract_dataset_name(filename: str) -> str:
    """
    Extract dataset name from filename
    Supported patterns:
    - ACEBench_toollist1_1000.json -> ACEBench
    - test_converted_ACEBench.json -> ACEBench
    - ACEBench_qwen.json -> ACEBench
    - ACEBench_gt_qwen.json -> ACEBench
    - ACEBench_gt_kimi.json -> ACEBench
    - HammerBench_single_gt_kimi.json -> HammerBench_single
    - APIBank_level3_gt_kimi.json -> APIBank_level3
    - Seal-Tools_gt_qwen.json -> Seal-Tools
    - alpaca_toollist1_1000.json -> alpaca
    """
    # Strip extension
    stem = Path(filename).stem
    
    # If stem starts with test_converted_, strip prefix
    if stem.startswith("test_converted_"):
        return stem.replace("test_converted_", "")
    
    # If _gt_ present, take part before _gt_
    if "_gt_" in stem:
        return stem.split("_gt_")[0]
    
    # If _toollist present, take part before it
    if "_toollist" in stem:
        return stem.split("_toollist")[0]
    
    # If _qwen present (not _gt_qwen), take part before it
    if "_qwen" in stem:
        return stem.split("_qwen")[0]
    
    # If underscores, try first segment (keep hyphens e.g. Seal-Tools)
    # Strip numeric suffix like _1000
    # Remove numeric suffix
    stem = re.sub(r'_\d+$', '', stem)
    
    # If underscores remain, take first segment
    if "_" in stem:
        parts = stem.split("_")
        # Return first segment if it looks like a dataset name
        # Otherwise return whole stem
        if len(parts) > 1:
            return parts[0]
    
    return stem


def find_matching_gt_file(pred_filename: str, gt_dir: Path) -> Optional[Path]:
    """
    Find GT file for a prediction filename
    Several GT naming patterns
    """
    dataset_name = extract_dataset_name(pred_filename)
    
    # Candidate GT filenames
    possible_gt_names = [
        f"test_converted_{dataset_name}.json",  # Standard
        f"{dataset_name}.json",  # Direct
    ]
    
    for gt_name in possible_gt_names:
        gt_path = gt_dir / gt_name
        if gt_path.exists():
            return gt_path
    
    return None


def calculate_dimension_stats(conversation_results: List[Dict], gt_data: List[Dict], pred_data: List[Dict]) -> Dict:
    """Aggregate metrics for single/multi hop and single/multi turn."""
    # Init dimension buckets
    # Hop dims use per-turn metrics
    # Turn dims use per-conversation metrics
    dimension_stats = {
        "single-hop": {"turn_metrics": []},  # Per-turn metric list
        "multi-hop": {"turn_metrics": []},
        "single-turn": {"conversations": [], "conversations_with_matched": []},
        "multi-turn": {"conversations": [], "conversations_with_matched": []}
    }
    
    # Classify each conversation by turn info
    for idx, conv_result in enumerate(conversation_results):
        if idx >= len(gt_data) or idx >= len(pred_data):
            continue
        
        gt_item = gt_data[idx]
        pred_item = pred_data[idx]
        turn_type, turn_info_list = extract_turn_info(gt_item)
        
        # Turn-type stats at conversation level (once per conv)
        # Still record conv-level if no turn list
        dimension_stats[turn_type]["conversations"].append(conv_result)
        if conv_result.get("function_call_level_stats", {}).get("matched_function_calls", 0) > 0:
            dimension_stats[turn_type]["conversations_with_matched"].append(conv_result)
        
        # Skip per-turn if no turn info
        if not turn_info_list:
            continue
        
        # Hop-type stats per turn
        # Compute per-turn metrics
        for hop_type, turn_index in turn_info_list:
            # Strict/flexible for this turn
            strict_prec, flexible_prec, strict_param_acc, flexible_param_acc, has_matched = (
                calculate_turn_level_metrics(pred_item, gt_item, turn_index)
            )
            dimension_stats[hop_type]["turn_metrics"].append({
                "strict_precision_at_1": strict_prec,
                "flexible_precision_at_1": flexible_prec,
                "strict_parameter_accuracy": strict_param_acc,
                "flexible_parameter_accuracy": flexible_param_acc,
                "has_matched": has_matched
            })
    
    # Aggregate per dimension
    dimension_metrics = {}
    
    # Hop dim: mean over turns (equal weight)
    for hop_dimension in ["single-hop", "multi-hop"]:
        turn_metrics = dimension_stats[hop_dimension]["turn_metrics"]
        total_turns = len(turn_metrics)
        
        if total_turns == 0:
            dimension_metrics[hop_dimension] = {
                "total_turns": 0,
                "strict_precision_at_1": 0.0,
                "flexible_precision_at_1_unweighted": 0.0,
                "strict_parameter_accuracy": 0.0,
                "flexible_parameter_accuracy": 0.0
            }
            continue
        
        # Mean over turns
        strict_precision_at_1 = sum(t["strict_precision_at_1"] for t in turn_metrics) / total_turns
        flexible_precision_at_1_unweighted = sum(t["flexible_precision_at_1"] for t in turn_metrics) / total_turns
        strict_parameter_accuracy = sum(t["strict_parameter_accuracy"] for t in turn_metrics) / total_turns
        flexible_parameter_accuracy = sum(t["flexible_parameter_accuracy"] for t in turn_metrics) / total_turns
        
        turns_with_matched = [t for t in turn_metrics if t.get("has_matched", False)]
        
        dimension_metrics[hop_dimension] = {
            "total_turns": total_turns,
            "matched_turns": len(turns_with_matched),
            "strict_precision_at_1": strict_precision_at_1,
            "flexible_precision_at_1_unweighted": flexible_precision_at_1_unweighted,
            "strict_parameter_accuracy": strict_parameter_accuracy,
            "flexible_parameter_accuracy": flexible_parameter_accuracy
        }
    
    # Turn dim: conversation-level metrics
    for turn_dimension in ["single-turn", "multi-turn"]:
        conversations = dimension_stats[turn_dimension]["conversations"]
        conversations_with_matched = dimension_stats[turn_dimension]["conversations_with_matched"]
        
        total_conversations = len(conversations)
        
        if total_conversations == 0:
            dimension_metrics[turn_dimension] = {
                "total_conversations": 0,
                "strict_precision_at_1": 0.0,
                "flexible_precision_at_1_unweighted": 0.0,
                "strict_parameter_accuracy": 0.0,
                "flexible_parameter_accuracy": 0.0
            }
            continue
        
        # Strict precision (conversation)
        strict_precision_at_1 = (
            sum(r["precision_at_1"] for r in conversations) / total_conversations
        )
        
        # Flexible precision averaged over convs
        flexible_precision_at_1_unweighted = (
            sum(r.get("function_call_precision_at_1_unweighted", 
                      r.get("function_call_level_stats", {}).get("function_call_precision_at_1_unweighted", 0.0)) 
                for r in conversations) / total_conversations
        )
        
        # Parameter accuracy averaged over convs
        strict_parameter_accuracy = (
            sum(r.get("strict_parameter_accuracy_fixed_denominator",
                      r.get("function_call_level_stats", {}).get("strict_parameter_accuracy_fixed_denominator", 0.0))
                for r in conversations) / total_conversations
        )
        
        flexible_parameter_accuracy = (
            sum(r.get("flexible_parameter_accuracy_fixed_denominator",
                      r.get("function_call_level_stats", {}).get("flexible_parameter_accuracy_fixed_denominator", 0.0))
                for r in conversations) / total_conversations
        )
        
        dimension_metrics[turn_dimension] = {
            "total_conversations": total_conversations,
            "matched_conversations": len(conversations_with_matched),
            "strict_precision_at_1": strict_precision_at_1,
            "flexible_precision_at_1_unweighted": flexible_precision_at_1_unweighted,
            "strict_parameter_accuracy": strict_parameter_accuracy,
            "flexible_parameter_accuracy": flexible_parameter_accuracy
        }
    
    return dimension_metrics


def evaluate_dataset_pair(gt_path: Path, pred_path: Path) -> Tuple[Dict, Dict]:
    """Evaluate one pred/GT file pair"""
    with gt_path.open("r", encoding="utf-8") as f:
        gt_data = json.load(f)

    with pred_path.open("r", encoding="utf-8") as f:
        pred_data = json.load(f)

    if not isinstance(gt_data, list) or not isinstance(pred_data, list):
        raise ValueError("Input must be a list of conversations")

    gt_len = len(gt_data)
    pred_len = len(pred_data)
    min_len = min(gt_len, pred_len)

    if gt_len != pred_len:
        print(
            f"  Warning: {pred_path.name} vs {gt_path.name} length mismatch "
            f"(pred={pred_len}, gt={gt_len}); evaluating first {min_len} items only"
        )

    conversation_results: List[Dict] = []

    for idx in range(min_len):
        conv_result = evaluate_conversation(pred_data[idx], gt_data[idx], idx + 1)
        conversation_results.append(conv_result)

        if (idx + 1) % 500 == 0:
            print(f"    evaluated {idx + 1}/{min_len} conversations")

    # Dimension stats for turns
    # Dimension stats use first min_len rows
    dimension_metrics = calculate_dimension_stats(conversation_results, gt_data[:min_len], pred_data[:min_len])

    summary = {
        "pred_count": pred_len,
        "gt_count": gt_len,
        "evaluated_count": min_len,
        "dimension_metrics": dimension_metrics
    }

    result = {
        "summary": summary,
        "conversation_results": conversation_results,
    }

    return result, summary


def main() -> None:
    """CLI: batch or single-file eval"""
    parser = argparse.ArgumentParser(description='Batch-evaluate tool-call predictions')
    parser.add_argument('--inputfile', type=str, 
                       default=str(_DEFAULT_PREDICTIONS_DIR),
                       help='Folder with prediction JSON (default test_set/predictions)')
    parser.add_argument('--outputfile', type=str,
                       default=str(_DEFAULT_EVAL_RESULTS_DIR),
                       help='Output folder for results (default test_set/eval_results)')
    parser.add_argument('--gtfile', type=str,
                       default='/home/yijuan_liang/10.12Tool_Set/test_set/data_notime',
                       help='Ground-truth folder')
    parser.add_argument('--use-mapping', action='store_true',
                       help='Use legacy filename mapping')
    parser.add_argument('--input', type=str,
                       help='Single prediction file (use with --gt)')
    parser.add_argument('--gt', type=str,
                       help='Single GT file (use with --input)')
    
    args = parser.parse_args()
    
    # Validate single-file args
    if (args.input and not args.gt) or (args.gt and not args.input):
        parser.error("--input and --gt are both required for single-file mode")
    
    # Single-file branch
    if args.input and args.gt:
        pred_path = Path(args.input)
        gt_path = Path(args.gt)
        
        if not pred_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {pred_path}")
        if not gt_path.exists():
            raise FileNotFoundError(f"GT file not found: {gt_path}")
        
        # Resolve output path
        if args.outputfile:
            output_dir = Path(args.outputfile)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_stem = pred_path.stem
            output_path = output_dir / f"{output_stem}_result.json"
        else:
            # Default output dir: pred file dir
            output_dir = pred_path.parent
            output_stem = pred_path.stem
            output_path = output_dir / f"{output_stem}_result.json"
        
        print("=" * 60)
        print("Single-file evaluation")
        print("=" * 60)
        print(f"Prediction: {pred_path}")
        print(f"GT file: {gt_path}")
        print(f"Output: {output_path}")
        
        try:
            result, summary = evaluate_dataset_pair(gt_path, pred_path)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Error: evaluation failed - {exc}")
            return
        
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print("Done: evaluated {evaluated_count} items".format(**summary))
        print(f"Saved: {output_path}")
        
        # Write single-file markdown report
        dataset_name = extract_dataset_name(pred_path.name)
        file_pairs = [(gt_path, pred_path, output_path, dataset_name)]
        generate_markdown_report(file_pairs, output_dir)
        return
    
    # Batch mode
    # Path objects
    PREDICTIONS_DIR = Path(args.inputfile)
    OUTPUT_DIR = Path(args.outputfile)
    DATA_DIR = Path(args.gtfile)
    
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"GT directory not found: {DATA_DIR}")
    if not PREDICTIONS_DIR.exists():
        raise FileNotFoundError(f"Predictions directory not found: {PREDICTIONS_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Batch eval preds vs GT")
    print("=" * 60)
    print(f"Input folder: {PREDICTIONS_DIR}")
    print(f"GT folder: {DATA_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")

    # Branch on legacy mapping
    if args.use_mapping:
        # Legacy mapping table
        file_pairs = []
        for data_stem, metrics_list in DATA_TO_METRICS_MAP.items():
            gt_path = DATA_DIR / f"{data_stem}.json"
            if not gt_path.exists():
                continue
            for metrics_stem in metrics_list:
                pred_path = PREDICTIONS_DIR / f"{metrics_stem}.json"
                if pred_path.exists():
                    output_path = OUTPUT_DIR / f"{metrics_stem}_result.json"
                    file_pairs.append((gt_path, pred_path, output_path, metrics_stem))
    else:
        # Auto-match by dataset name
        pred_files = [f for f in PREDICTIONS_DIR.glob("*.json") if f.is_file()]
        file_pairs = []
        for pred_path in pred_files:
            # Smart GT lookup
            gt_path = find_matching_gt_file(pred_path.name, DATA_DIR)
            if gt_path:
                # Output name: stem + _result
                output_stem = pred_path.stem
                output_path = OUTPUT_DIR / f"{output_stem}_result.json"
                dataset_name = extract_dataset_name(pred_path.name)
                file_pairs.append((gt_path, pred_path, output_path, dataset_name))
            else:
                print(
                    f"\nWarning: no GT for {pred_path.name} "
                    f"(dataset={extract_dataset_name(pred_path.name)}), skip"
                )

    if not file_pairs:
        print("\nWarning: no file pairs")
        return

    print(f"\n{len(file_pairs)} file pair(s) to process\n")

    for gt_path, pred_path, output_path, dataset_name in file_pairs:
        print("\n" + "-" * 60)
        print(f"Dataset: {dataset_name}")
        print(f"Prediction: {pred_path}")
        print(f"GT file: {gt_path}")

        try:
            result, summary = evaluate_dataset_pair(gt_path, pred_path)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  Error: evaluation failed - {exc}")
            continue

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print("  Done: evaluated {evaluated_count} items".format(**summary))
        print(f"  Saved: {output_path}")

    # Write markdown summary
    generate_markdown_report(file_pairs, OUTPUT_DIR)


def aggregate_dimension_metrics(all_file_metrics: List[Tuple[str, Dict]]) -> Dict:
    """
    Aggregate per-file dimension metrics
    Four dims: single/multi hop/turn
    """
    # Init summary
    # Hop uses turns; turn uses convs
    aggregated = {
        "single-hop": {"total_turns": 0, "metrics_sum": defaultdict(float)},
        "multi-hop": {"total_turns": 0, "metrics_sum": defaultdict(float)},
        "single-turn": {"total_conversations": 0, "metrics_sum": defaultdict(float)},
        "multi-turn": {"total_conversations": 0, "metrics_sum": defaultdict(float)}
    }
    
    # Sum per file
    for dataset_name, dimension_metrics in all_file_metrics:
        for dimension in ["single-hop", "multi-hop", "single-turn", "multi-turn"]:
            metrics = dimension_metrics.get(dimension, {})
            
            # Hop uses turns; turn uses convs
            if dimension in ["single-hop", "multi-hop"]:
                total_count = metrics.get("total_turns", 0)
            else:
                total_count = metrics.get("total_conversations", 0)
            
            if total_count == 0:
                continue
            
            # Update totals
            if dimension in ["single-hop", "multi-hop"]:
                aggregated[dimension]["total_turns"] += total_count
            else:
                aggregated[dimension]["total_conversations"] += total_count
            
            # Weighted sum
            aggregated[dimension]["metrics_sum"]["strict_precision_at_1"] += (
                metrics.get("strict_precision_at_1", 0.0) * total_count
            )
            aggregated[dimension]["metrics_sum"]["flexible_precision_at_1_unweighted"] += (
                metrics.get("flexible_precision_at_1_unweighted", 0.0) * total_count
            )
            aggregated[dimension]["metrics_sum"]["strict_parameter_accuracy"] += (
                metrics.get("strict_parameter_accuracy", 0.0) * total_count
            )
            aggregated[dimension]["metrics_sum"]["flexible_parameter_accuracy"] += (
                metrics.get("flexible_parameter_accuracy", 0.0) * total_count
            )
    
    # Compute mean
    averages = {}
    for dimension in ["single-hop", "multi-hop", "single-turn", "multi-turn"]:
        agg = aggregated[dimension]
        
        # Get totals
        if dimension in ["single-hop", "multi-hop"]:
            total_count = agg["total_turns"]
            count_key = "total_turns"
        else:
            total_count = agg["total_conversations"]
            count_key = "total_conversations"
        
        if total_count == 0:
            averages[dimension] = {
                count_key: 0,
                "strict_precision_at_1": 0.0,
                "flexible_precision_at_1_unweighted": 0.0,
                "strict_parameter_accuracy": 0.0,
                "flexible_parameter_accuracy": 0.0
            }
            continue
        
        averages[dimension] = {
            count_key: total_count,
            "strict_precision_at_1": agg["metrics_sum"]["strict_precision_at_1"] / total_count,
            "flexible_precision_at_1_unweighted": agg["metrics_sum"]["flexible_precision_at_1_unweighted"] / total_count,
            "strict_parameter_accuracy": agg["metrics_sum"]["strict_parameter_accuracy"] / total_count,
            "flexible_parameter_accuracy": agg["metrics_sum"]["flexible_parameter_accuracy"] / total_count
        }
    
    return averages


def generate_markdown_report(file_pairs: List[Tuple], output_dir: Path):
    """
    Write markdown summary
    Markdown name follows output folder suffix
    """
    # Load all eval JSONs
    all_file_metrics = []
    
    for gt_path, pred_path, output_path, dataset_name in file_pairs:
        if not output_path.exists():
            continue
        
        with output_path.open("r", encoding="utf-8") as f:
            result = json.load(f)
        
        dimension_metrics = result.get("summary", {}).get("dimension_metrics", {})
        if dimension_metrics:
            all_file_metrics.append((dataset_name, dimension_metrics))
    
    if not all_file_metrics:
        print("\nWarning: no eval results")
        return

    averages = aggregate_dimension_metrics(all_file_metrics)
    
    # Markdown filename
    output_dir_name = output_dir.name
    md_filename = f"{output_dir_name}.md"
    md_path = output_dir / md_filename
    
    # Markdown body
    md_content = []
    md_content.append("# Evaluation summary report\n\n")
    md_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Per file
    md_content.append("## Per file\n\n")
    
    for dataset_name, dimension_metrics in all_file_metrics:
        md_content.append(f"### {dataset_name}\n\n")
        
        # Table
        md_content.append("| Dimension | Conversations / turns | Strict Precision | Flexible Precision | Strict Parameter Accuracy | Flexible Parameter Accuracy |\n")
        md_content.append("|------|------------------------|------------------|-------------------|---------------------------|-----------------------------|\n")
        
        # Four dimensions
        for dimension in ["single-hop", "multi-hop", "single-turn", "multi-turn"]:
            metrics = dimension_metrics.get(dimension, {})
            # Hop uses turns; turn uses convs
            if dimension in ["single-hop", "multi-hop"]:
                total_count = metrics.get("total_turns", 0)
            else:
                total_count = metrics.get("total_conversations", 0)
            
            if total_count == 0:
                continue
            
            # Labels by dimension
            # Turn dim shows conv count; hop shows turn count
            if dimension in ["single-turn", "multi-turn"]:
                count_label = f"{total_count} (conversations)"
            else:  # single-hop, multi-hop
                count_label = f"{total_count} (turns)"
            
            md_content.append(
                f"| {dimension} | {count_label} | "
                f"{metrics.get('strict_precision_at_1', 0.0):.4f} | "
                f"{metrics.get('flexible_precision_at_1_unweighted', 0.0):.4f} | "
                f"{metrics.get('strict_parameter_accuracy', 0.0):.4f} | "
                f"{metrics.get('flexible_parameter_accuracy', 0.0):.4f} |\n"
            )
        
        md_content.append("\n")
    
    # Average across files
    md_content.append("## Average across files\n\n")
    md_content.append("| Dimension | Conversations / turns | Strict Precision | Flexible Precision | Strict Parameter Accuracy | Flexible Parameter Accuracy |\n")
    md_content.append("|------|------------------------|------------------|-------------------|---------------------------|-----------------------------|\n")

    for dimension in ["single-hop", "multi-hop", "single-turn", "multi-turn"]:
        metrics = averages.get(dimension, {})
        # Hop uses turns; turn uses convs
        if dimension in ["single-hop", "multi-hop"]:
            total_count = metrics.get("total_turns", 0)
        else:
            total_count = metrics.get("total_conversations", 0)
        
        if total_count == 0:
            continue
        
        # Labels by dimension
        # Turn dim shows conv count; hop shows turn count
        if dimension in ["single-turn", "multi-turn"]:
            count_label = f"{total_count} (conversations)"
        else:  # single-hop, multi-hop
            count_label = f"{total_count} (turns)"
        
        md_content.append(
            f"| {dimension} | {count_label} | "
            f"{metrics.get('strict_precision_at_1', 0.0):.4f} | "
            f"{metrics.get('flexible_precision_at_1_unweighted', 0.0):.4f} | "
            f"{metrics.get('strict_parameter_accuracy', 0.0):.4f} | "
            f"{metrics.get('flexible_parameter_accuracy', 0.0):.4f} |\n"
        )
    
    # Save
    with md_path.open("w", encoding="utf-8") as f:
        f.write(''.join(md_content))
    
    print(f"\nSummary saved to: {md_path}")


if __name__ == "__main__":
    main()


