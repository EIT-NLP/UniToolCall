#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLLIST_SCRIPT_DIR = REPO_ROOT / "train_set" / "scripts" / "toollist"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(TOOLLIST_SCRIPT_DIR))

from uni_toolcall.secrets import get_openai_compatible_key  # noqa: E402
import toollist_settings as ts  # noqa: E402

# Paths relative to UniToolCall root (edit as needed)
INPUT_PATH = REPO_ROOT / "single-hop_pipeline" / "data" / "2.8_data_set_2.json"
OUTPUT_DIR_RAW = REPO_ROOT / "single-hop_pipeline" / "data"
OUTPUT_DIR_PROCESSED = REPO_ROOT / "single-hop_pipeline" / "processed_llamafactory"
FIELDS_TO_KEEP = ["conversations", "system", "tools"]
EMPTY_ANSWER_TAG = "<answer></answer>"

# SiliconFlow-compatible embeddings API (keys via env)
SILICONFLOW_EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"


def apply_processed_format(conversations: list) -> list:
    """Clear observation values; set gpt value to EMPTY_ANSWER_TAG."""
    result = []
    for conv in conversations:
        conv = conv.copy()
        if conv.get("from") == "observation":
            conv["value"] = ""
        elif conv.get("from") == "gpt":
            conv["value"] = EMPTY_ANSWER_TAG
        result.append(conv)
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build single_hop_toollist.jsonl")
    parser.add_argument("--mode", choices=["api", "server"], default="api")
    parser.add_argument("--server-url", type=str, default=ts.EMBED_SERVER_URL)
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API key for mode=api (default: env SILICONFLOW_API_KEY / OPENAI_API_KEY)",
    )
    args = parser.parse_args()

    mode = args.mode
    server_url = args.server_url or ts.EMBED_SERVER_URL

    ts.EMBED_URL = SILICONFLOW_EMBED_URL
    ts.API_KEY = (args.api_key or "").strip() or (get_openai_compatible_key() or "")
    if mode == "api" and not ts.API_KEY:
        raise SystemExit(
            "mode=api requires an API key (--api-key or SILICONFLOW_API_KEY / OPENAI_API_KEY)"
        )

    print("Loading tool pool...")
    tool_pool = ts.load_tool_pool()

    print("Loading tool-pool embedding cache...")
    cache = {}
    for cf in ts.EMBED_CACHE_DIR.glob("*.embeddings.json"):
        fc = ts.load_embedding_cache(cf)
        cache.update(fc)
    print(f"Tool-pool embedding cache entries: {len(cache)}\n")

    print("Loading GT tool embedding cache...")
    gt_cache = ts.load_embedding_cache(ts.GT_CACHE_FILE)
    print(f"GT tool embedding cache entries: {len(gt_cache)}\n")

    tool_embeddings = {}

    print("Reading input and collecting GT tools...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON array")

    all_gt_tool_names = set()
    for item in data:
        convs = item.get("conversations", [])
        all_gt_tool_names.update(ts.extract_gt_tools(convs))

    gt_tool_keys = set()
    for name in all_gt_tool_names:
        res = ts.find_tool_by_name(tool_pool, name)
        if res:
            gt_tool_keys.add(res[0])
    print(f"Collected {len(gt_tool_keys)} GT tools\n")

    print("Building Faiss index...")
    faiss_index, index_to_tool_key = ts.build_faiss_index_from_tool_pool(
        tool_pool,
        cache,
        tool_embeddings,
        mode,
        server_url,
        gt_cache=gt_cache,
        gt_tool_keys=gt_tool_keys,
        gt_cache_file=ts.GT_CACHE_FILE,
    )
    if faiss_index is not None:
        print(f"Faiss index size: {len(index_to_tool_key)} tools\n")
    else:
        print("Faiss unavailable; using NumPy fallback\n")

    raw_items = []

    for item in ts.tqdm(data, desc="rows", unit="row"):
        conversations = item.get("conversations", [])
        gt_tools = ts.extract_gt_tools(conversations)

        if not gt_tools or len(gt_tools) == 20:
            raw = item.copy()
        else:
            toollist = ts.build_toollist_setting(
                gt_tools,
                tool_pool,
                tool_embeddings,
                cache,
                gt_cache,
                mode,
                server_url,
                faiss_index,
                index_to_tool_key,
            )
            raw = item.copy()
            raw["tools"] = json.dumps(toollist, ensure_ascii=False)
        raw_items.append(raw)

    OUTPUT_DIR_RAW.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_PROCESSED.mkdir(parents=True, exist_ok=True)

    out_raw = OUTPUT_DIR_RAW / "single_hop_toollist.jsonl"
    out_proc = OUTPUT_DIR_PROCESSED / "single_hop_toollist.jsonl"

    def write_jsonl(path, items, apply_processed=False):
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                convs = it.get("conversations", [])
                if apply_processed:
                    convs = apply_processed_format(convs)
                out_it = {"conversations": convs, "system": it.get("system", ""), "tools": it.get("tools", "[]")}
                f.write(json.dumps(out_it, ensure_ascii=False) + "\n")

    print(f"Writing {out_raw} ...")
    write_jsonl(out_raw, raw_items, apply_processed=False)
    print(f"Writing {out_proc} ...")
    write_jsonl(out_proc, raw_items, apply_processed=True)

    print(f"\nDone. Processed {len(data)} rows.")
    print(f"  - raw: {out_raw}")
    print(f"  - processed: {out_proc}")


if __name__ == "__main__":
    main()
