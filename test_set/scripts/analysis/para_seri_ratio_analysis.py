#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

# Path to detect_para_seri_ratio (external LLaMA-Factory dataset helpers)
SCRIPT_DIR = Path(__file__).resolve().parent
DETECT_RATIO_DIR = SCRIPT_DIR.parent / "metrics"
# LLaMA-Factory may sit next to the workspace root
_CANDIDATE_LLAMA_PATHS = [
    Path("/code/LLaMA-Factory/data/dataset/01_26"),  # common layout
    SCRIPT_DIR.parents[3] / "LLaMA-Factory" / "data" / "dataset" / "01_26",  # ascend from test_set/scripts/analysis
    SCRIPT_DIR.parents[2] / "LLaMA-Factory" / "data" / "dataset" / "01_26",
]
LLAMA_DETECT_DIR = None
for p in _CANDIDATE_LLAMA_PATHS:
    if p.exists() and (p / "detect_para_seri_ratio.py").exists():
        LLAMA_DETECT_DIR = p
        break
if LLAMA_DETECT_DIR is None:
    LLAMA_DETECT_DIR = Path("/code/LLaMA-Factory/data/dataset/01_26")  # fallback
sys.path.insert(0, str(LLAMA_DETECT_DIR))

try:
    from detect_para_seri_ratio import detect_item_strategy  # type: ignore
except ImportError:
    # If LLaMA-Factory path is missing, detect_item_strategy stays None
    detect_item_strategy = None

# Import evaluation helpers from test_set/scripts/metrics
sys.path.insert(0, str(DETECT_RATIO_DIR))
from data_evaluation import evaluate_conversation  # type: ignore
from all_evaluation import (  # type: ignore
    extract_dataset_name,
    calculate_dimension_stats,
    extract_turn_info,
    identify_turns_from_conversations,
    calculate_turn_level_metrics,
    find_matching_gt_file,
)


def get_strategy_from_detect(gt_item: Dict) -> Optional[str]:
    """Serial/parallel strategy for one item using detect_para_seri_ratio. Returns serial, parallel, or None."""
    if detect_item_strategy is None:
        raise ImportError(
            "Cannot import detect_item_strategy; ensure LLaMA-Factory/data/dataset/01_26/detect_para_seri_ratio.py exists"
        )
    return detect_item_strategy(gt_item)


def evaluate_by_strategy_subset(
    gt_data: List[Dict],
    pred_data: List[Dict],
    strategy_filter: Optional[str],
) -> Tuple[Dict, Dict]:
    """Filter by serial/parallel strategy and evaluate."""
    min_len = min(len(gt_data), len(pred_data))
    filtered_indices: List[int] = []

    for idx in range(min_len):
        gt_item = gt_data[idx]
        strategy = get_strategy_from_detect(gt_item)

        if strategy_filter is None:
            filtered_indices.append(idx)
        elif strategy_filter == "all_multi_hop":
            if strategy in ("serial", "parallel"):
                filtered_indices.append(idx)
        elif strategy_filter in ("serial", "parallel"):
            if strategy == strategy_filter:
                filtered_indices.append(idx)

    if not filtered_indices:
        return {
            "summary": {
                "evaluated_count": 0,
                "strategy_filter": strategy_filter or "all",
                "filtered_count": 0,
                "dimension_metrics": {},
            },
            "conversation_results": [],
        }, {
            "evaluated_count": 0,
            "strategy_filter": strategy_filter or "all",
            "filtered_count": 0,
            "dimension_metrics": {},
        }

    conversation_results: List[Dict] = []
    filtered_gt = []
    filtered_pred = []

    for idx in filtered_indices:
        conv_result = evaluate_conversation(pred_data[idx], gt_data[idx], idx + 1)
        conversation_results.append(conv_result)
        filtered_gt.append(gt_data[idx])
        filtered_pred.append(pred_data[idx])

    dimension_metrics = calculate_dimension_stats(
        conversation_results, filtered_gt, filtered_pred
    )

    summary = {
        "evaluated_count": len(filtered_indices),
        "strategy_filter": strategy_filter or "all",
        "filtered_count": len(filtered_indices),
        "dimension_metrics": dimension_metrics,
    }

    result = {
        "summary": summary,
        "conversation_results": conversation_results,
    }

    return result, summary


def evaluate_by_ratio_sampling(
    gt_data: List[Dict],
    pred_data: List[Dict],
    parallel_ratio: int,
    seed: int = 42,
) -> Tuple[Dict, Dict]:
    """Random sample serial:parallel = 1:N (excludes single-hop)."""
    min_len = min(len(gt_data), len(pred_data))
    serial_indices: List[int] = []
    parallel_indices: List[int] = []

    for idx in range(min_len):
        strategy = get_strategy_from_detect(gt_data[idx])
        if strategy == "serial":
            serial_indices.append(idx)
        elif strategy == "parallel":
            parallel_indices.append(idx)
    serial_count = len(serial_indices)
    parallel_count = len(parallel_indices)

    serial_sampled = min(serial_count, parallel_count // parallel_ratio)
    if serial_sampled == 0:
        return {
            "summary": {
                "evaluated_count": 0,
                "strategy_filter": f"ratio_1_{parallel_ratio}",
                "filtered_count": 0,
                "dimension_metrics": {},
            },
            "conversation_results": [],
        }, {
            "evaluated_count": 0,
            "strategy_filter": f"ratio_1_{parallel_ratio}",
            "filtered_count": 0,
            "dimension_metrics": {},
        }

    rng = random.Random(seed)
    sampled_serial = rng.sample(serial_indices, serial_sampled)
    sampled_parallel = rng.sample(parallel_indices, parallel_ratio * serial_sampled)
    filtered_indices = sorted(sampled_serial + sampled_parallel)

    conversation_results: List[Dict] = []
    filtered_gt = []
    filtered_pred = []

    for idx in filtered_indices:
        conv_result = evaluate_conversation(pred_data[idx], gt_data[idx], idx + 1)
        conversation_results.append(conv_result)
        filtered_gt.append(gt_data[idx])
        filtered_pred.append(pred_data[idx])

    dimension_metrics = calculate_dimension_stats(
        conversation_results, filtered_gt, filtered_pred
    )

    summary = {
        "evaluated_count": len(filtered_indices),
        "strategy_filter": f"ratio_1_{parallel_ratio}",
        "filtered_count": len(filtered_indices),
        "dimension_metrics": dimension_metrics,
    }

    result = {
        "summary": summary,
        "conversation_results": conversation_results,
    }

    return result, summary


# Ratio grid: serial fixed at 1, parallel N in 1..10
RATIO_PARALLEL_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def generate_para_seri_md_report(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path,
    pred_dir_name: str,
    seed: int = 42,
) -> None:
    """Write Markdown report for serial/parallel ratio analysis."""
    md_content = []
    md_content.append("# Serial / parallel ratio report\n\n")
    md_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    md_content.append(f"Predictions dir: {pred_dir_name}\n\n")
    md_content.append(
        "Notes: Serial/parallel labels follow detect_para_seri_ratio.py (trajectory and dependency). "
        "Single-hop excluded. Ratio subsets use random sampling (seed={}).\n\n".format(seed)
    )

    strategy_labels = {
        "all_multi_hop": "All multi-hop (serial + parallel)",
        "serial_only": "Serial only",
        "parallel_only": "Parallel only",
        "all": "All data",
    }
    for n in RATIO_PARALLEL_VALUES:
        strategy_labels[f"ratio_1_{n}"] = f"Serial:parallel = 1:{n} (random)"

    all_strategy_keys = ["all_multi_hop", "serial_only", "parallel_only"] + [
        f"ratio_1_{n}" for n in RATIO_PARALLEL_VALUES
    ]

    md_content.append("## Per dataset\n\n")

    for dataset_name, strategy_results in sorted(all_results.items()):
        md_content.append(f"### {dataset_name}\n\n")

        md_content.append(
            "| Subset | Count | Strict Precision@1 | Strict Accuracy | "
            "Flexible Precision@1 | Flexible Accuracy | "
            "Strict Param Acc (Fixed) | Flexible Param Acc (Fixed) |\n"
        )
        md_content.append(
            "|----------|--------|---------------------|-----------------|"
            "----------------------|-------------------|"
            "------------------------|---------------------------|\n"
        )

        for strategy_key in all_strategy_keys:
            summary = strategy_results.get(strategy_key, {})
            dim_metrics = summary.get("dimension_metrics", {})
            evaluated_count = summary.get("evaluated_count", 0)

            if evaluated_count == 0:
                label = strategy_labels.get(strategy_key, strategy_key)
                md_content.append(
                    f"| {label} | 0 | - | - | - | - | - | - |\n"
                )
                continue

            metrics = dim_metrics.get("multi-hop", dim_metrics.get("single-hop", {}))
            if not metrics:
                for dim in ["multi-hop", "single-hop", "single-turn", "multi-turn"]:
                    if dim in dim_metrics and dim_metrics[dim].get("total_turns", 0) + dim_metrics[dim].get("total_conversations", 0) > 0:
                        metrics = dim_metrics[dim]
                        break

            if not metrics:
                md_content.append(
                    f"| {strategy_labels.get(strategy_key, strategy_key)} | {evaluated_count} | - | - | - | - | - | - |\n"
                )
                continue

            label = strategy_labels.get(strategy_key, strategy_key)
            total = metrics.get("total_turns", metrics.get("total_conversations", 0))
            md_content.append(
                f"| {label} | {total} | "
                f"{metrics.get('strict_precision_at_1', 0.0):.4f} | "
                f"{metrics.get('strict_accuracy', 0.0):.4f} | "
                f"{metrics.get('flexible_precision_at_1_unweighted', 0.0):.4f} | "
                f"{metrics.get('flexible_accuracy_unweighted', 0.0):.4f} | "
                f"{metrics.get('strict_parameter_accuracy_fixed_denominator', 0.0):.4f} | "
                f"{metrics.get('flexible_parameter_accuracy_fixed_denominator', 0.0):.4f} |\n"
            )

        md_content.append("\n")

    md_content.append("## Aggregate (sample-weighted)\n\n")

    aggregated = {k: {"total": 0, "metrics_sum": defaultdict(float)} for k in all_strategy_keys}

    for dataset_name, strategy_results in all_results.items():
        for strategy_key in all_strategy_keys:
            summary = strategy_results.get(strategy_key, {})
            dim_metrics = summary.get("dimension_metrics", {})
            metrics = dim_metrics.get("multi-hop", dim_metrics.get("single-hop", {}))
            if not metrics:
                continue

            total = metrics.get("total_turns", metrics.get("total_conversations", 0))
            if total == 0:
                continue

            aggregated[strategy_key]["total"] += total
            for k in [
                "strict_precision_at_1",
                "strict_accuracy",
                "flexible_precision_at_1_unweighted",
                "flexible_accuracy_unweighted",
                "strict_parameter_accuracy_fixed_denominator",
                "flexible_parameter_accuracy_fixed_denominator",
            ]:
                aggregated[strategy_key]["metrics_sum"][k] += (
                    metrics.get(k, 0.0) * total
                )

    md_content.append(
        "| Subset | Total | Strict Precision@1 | Strict Accuracy | "
        "Flexible Precision@1 | Flexible Accuracy | "
        "Strict Param Acc (Fixed) | Flexible Param Acc (Fixed) |\n"
    )
    md_content.append(
        "|----------|----------|---------------------|-----------------|"
        "----------------------|-------------------|"
        "------------------------|---------------------------|\n"
    )

    for strategy_key in all_strategy_keys:
        agg = aggregated[strategy_key]
        total = agg["total"]
        if total == 0:
            md_content.append(
                f"| {strategy_labels.get(strategy_key, strategy_key)} | 0 | - | - | - | - | - | - |\n"
            )
            continue

        label = strategy_labels.get(strategy_key, strategy_key)
        ms = agg["metrics_sum"]
        md_content.append(
            f"| {label} | {total} | "
            f"{ms['strict_precision_at_1'] / total:.4f} | "
            f"{ms['strict_accuracy'] / total:.4f} | "
            f"{ms['flexible_precision_at_1_unweighted'] / total:.4f} | "
            f"{ms['flexible_accuracy_unweighted'] / total:.4f} | "
            f"{ms['strict_parameter_accuracy_fixed_denominator'] / total:.4f} | "
            f"{ms['flexible_parameter_accuracy_fixed_denominator'] / total:.4f} |\n"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("".join(md_content))

    print(f"\nReport saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serial/parallel ratio analysis: evaluate subsets and write Markdown report"
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default="/code/10.12Tool_Set/test_set/3_1/metrics_toollist1_qwen3_8B_pipeline_mix",
        help="Directory with prediction JSON files",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="/code/10.12Tool_Set/test_set/data/data_toollist",
        help="Ground truth directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/code/10.12Tool_Set/test_set/3_1",
        help="Output directory for Markdown report",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for ratio sampling",
    )

    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir)

    if not pred_dir.exists():
        print(f"Error: prediction directory not found: {pred_dir}")
        sys.exit(1)
    if not gt_dir.exists():
        print(f"Error: GT directory not found: {gt_dir}")
        sys.exit(1)

    if detect_item_strategy is None:
        print("Error: cannot import detect_item_strategy")
        print(f"Ensure {LLAMA_DETECT_DIR / 'detect_para_seri_ratio.py'} exists")
        sys.exit(1)

    pred_files = [f for f in pred_dir.glob("*.json") if f.is_file()]

    if not pred_files:
        print(f"Warning: no JSON files under {pred_dir}")
        sys.exit(1)

    all_results: Dict[str, Dict[str, Dict]] = {}
    file_pairs: List[Tuple[Path, Path, str]] = []

    for pred_path in sorted(pred_files):
        gt_path = find_matching_gt_file(pred_path.name, gt_dir)
        if not gt_path:
            print(f"Warning: no GT file for {pred_path.name}, skip")
            continue

        dataset_name = extract_dataset_name(pred_path.name)
        file_pairs.append((gt_path, pred_path, dataset_name))

    print("=" * 70)
    print("Serial / parallel ratio analysis")
    print("=" * 70)
    print(f"Pred dir: {pred_dir}")
    print(f"GT dir: {gt_dir}")
    print(f"Output dir: {output_dir}")
    print(f"{len(file_pairs)} file pairs\n")

    for gt_path, pred_path, dataset_name in file_pairs:
        print(f"\nDataset: {dataset_name}")

        with gt_path.open("r", encoding="utf-8") as f:
            gt_data = json.load(f)
        with pred_path.open("r", encoding="utf-8") as f:
            pred_data = json.load(f)

        min_len = min(len(gt_data), len(pred_data))
        gt_data = gt_data[:min_len]
        pred_data = pred_data[:min_len]

        dataset_results = {}

        # Three fixed subsets
        for strategy_key, strategy_filter in [
            ("all_multi_hop", "all_multi_hop"),
            ("serial_only", "serial"),
            ("parallel_only", "parallel"),
        ]:
            try:
                result, summary = evaluate_by_strategy_subset(
                    gt_data, pred_data, strategy_filter
                )
                dataset_results[strategy_key] = summary
                print(f"  {strategy_key}: {summary['evaluated_count']} items")
            except Exception as e:
                print(f"  {strategy_key}: error - {e}")
                dataset_results[strategy_key] = {
                    "evaluated_count": 0,
                    "dimension_metrics": {},
                }

        # Ratio sampling serial:parallel = 1:N
        for n in RATIO_PARALLEL_VALUES:
            strategy_key = f"ratio_1_{n}"
            try:
                result, summary = evaluate_by_ratio_sampling(
                    gt_data, pred_data, parallel_ratio=n, seed=args.seed
                )
                dataset_results[strategy_key] = summary
                print(f"  {strategy_key}: {summary['evaluated_count']} items")
            except Exception as e:
                print(f"  {strategy_key}: error - {e}")
                dataset_results[strategy_key] = {
                    "evaluated_count": 0,
                    "dimension_metrics": {},
                }

        all_results[dataset_name] = dataset_results

    pred_dir_name = pred_dir.name
    md_filename = f"{pred_dir_name}_para_seri_ratio_analysis.md"
    md_path = output_dir / md_filename

    generate_para_seri_md_report(all_results, md_path, pred_dir_name, seed=args.seed)

    print("\n" + "=" * 70)
    print("Done")
    print("=" * 70)


if __name__ == "__main__":
    main()
