#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path


def load_map(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def names_from_map(obj: dict) -> set:
    names: set[str] = set()
    for v in obj.values():
        if isinstance(v, dict):
            n = v.get("name")
            if isinstance(n, str) and n:
                names.add(n)
    return names


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    fc = load_map(root / "fc_tool_set_dedup_by_name.json")
    cor9 = load_map(root / "apis" / "corpus_9.15_translated.json")
    cort = load_map(root / "apis" / "corpus_tool_set_translated.json")

    fc_names = names_from_map(fc)
    cor9_names = names_from_map(cor9)
    cort_names = names_from_map(cort)

    inter_9 = sorted(fc_names & cor9_names)
    inter_t = sorted(fc_names & cort_names)

    print(f"fc unique names: {len(fc_names)}")
    print(f"corpus_9.15_translated unique names: {len(cor9_names)}")
    print(f"corpus_tool_set_translated unique names: {len(cort_names)}")
    print(f"overlap with corpus_9.15_translated: {len(inter_9)}")
    print(f"overlap with corpus_tool_set_translated: {len(inter_t)}")

    def sample(lst: list[str], k: int = 10) -> list[str]:
        return lst[:k]

    print("examples overlap (9.15):", sample(inter_9, 10))
    print("examples overlap (tool_set):", sample(inter_t, 10))


if __name__ == "__main__":
    main()


