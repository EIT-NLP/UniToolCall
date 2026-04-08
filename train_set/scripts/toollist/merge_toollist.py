#!/usr/bin/env python3

import json
import os
from pathlib import Path
from typing import List, Dict, Any


def process_conversations(conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize conversations: clear observation values; set gpt to <answer></answer>."""
    processed = []
    for conv in conversations:
        from_type = conv.get("from", "")

        if from_type == "observation":
            processed.append({
                "from": "observation",
                "value": ""
            })
        elif from_type == "gpt":
            processed.append({
                "from": "gpt",
                "value": "<answer></answer>"
            })
        else:
            processed.append(conv)

    return processed


def process_data_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Process one record: normalize conversations and drop properties."""
    processed_item = item.copy()

    if "conversations" in processed_item:
        processed_item["conversations"] = process_conversations(processed_item["conversations"])

    if "properties" in processed_item:
        del processed_item["properties"]

    return processed_item


def merge_json_files(input_dir: str, output_file: str, sample_output_file: str = None):
    """Merge all *.json in input_dir into one JSONL; optional first-10 sample JSON."""
    input_path = Path(input_dir)
    output_path = Path(output_file)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_path.glob("*.json"))

    if not json_files:
        print(f"Warning: no JSON files in {input_dir}")
        return

    print(f"Found {len(json_files)} JSON files")

    total_items = 0
    sample_items = []

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for json_file in json_files:
            print(f"Processing: {json_file.name}")

            try:
                with open(json_file, 'r', encoding='utf-8') as infile:
                    data = json.load(infile)

                if not isinstance(data, list):
                    data = [data]

                for item in data:
                    processed_item = process_data_item(item)
                    outfile.write(json.dumps(processed_item, ensure_ascii=False) + '\n')

                    if len(sample_items) < 10:
                        sample_items.append(processed_item)

                    total_items += 1

            except Exception as e:
                print(f"Error processing {json_file.name}: {e}")
                continue

    if sample_output_file and sample_items:
        sample_path = Path(sample_output_file)
        sample_path.parent.mkdir(parents=True, exist_ok=True)

        with open(sample_path, 'w', encoding='utf-8') as sample_file:
            json.dump(sample_items, sample_file, ensure_ascii=False, indent=2)

        print(f"First 10 rows saved to: {sample_path}")

    print(f"Done. Wrote {total_items} rows to {output_path}")


def main():
    base_dir = Path("/home/yijuan_liang/10.12Tool_Set/train_set/data")
    output_base_dir = Path("/home/yijuan_liang/LLaMA-Factory/data/dataset/01_30")

    input_dir = base_dir / "data_toollist"

    output_file = output_base_dir / "toollist1.jsonl"

    sample_file = output_base_dir / "toollist1_sample.json"

    if not input_dir.exists():
        print(f"Error: input directory not found: {input_dir}")
        return

    print("=" * 60)
    print("Merge data_toollist")
    print("=" * 60)
    merge_json_files(str(input_dir), str(output_file), str(sample_file))

    print("\n" + "=" * 60)
    print("All done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
