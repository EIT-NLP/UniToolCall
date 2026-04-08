#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import shutil
from typing import Dict, List, Any, Set
from datetime import datetime


class ConversationCleaner:
    """Removes conversations that fail validation checks."""

    def __init__(self):
        self.stats = {
            "total_files": 0,
            "total_conversations": 0,
            "removed_conversations": 0,
            "kept_conversations": 0,
            "removed_by_type": {
                "tool_not_found": 0,
                "invalid_flow": 0,
                "invalid_function_call_json": 0,
                "invalid_json": 0,
                "incomplete_flow": 0
            },
            "files_modified": []
        }

    def check_conversation_issues(self, item: Dict[str, Any]) -> List[str]:
        """Return issue type strings for this item; empty list means keep."""

        issues = []

        if "conversations" not in item or not isinstance(item["conversations"], list):
            return issues

        conversations = item["conversations"]

        try:
            tools_str = item.get("tools", "[]")
            tools = json.loads(tools_str) if isinstance(tools_str, str) else tools_str
            tools_dict = {tool["name"]: tool for tool in tools} if isinstance(tools, list) else {}
        except Exception:
            tools_dict = {}

        for i, msg in enumerate(conversations):
            role = msg.get("from")

            if role == "function_call":
                if i + 1 < len(conversations):
                    next_role = conversations[i + 1].get("from")
                    if next_role != "observation":
                        if "invalid_flow" not in issues:
                            issues.append("invalid_flow")
                else:
                    if "incomplete_flow" not in issues:
                        issues.append("incomplete_flow")

            elif role == "observation":
                if i + 1 < len(conversations):
                    next_role = conversations[i + 1].get("from")
                    if next_role not in {"gpt", "function_call"}:
                        if "invalid_flow" not in issues:
                            issues.append("invalid_flow")

            if role == "function_call":
                value = msg.get("value", "")

                try:
                    call_data = json.loads(value)
                except json.JSONDecodeError:
                    if "invalid_json" not in issues:
                        issues.append("invalid_json")
                    continue

                if not isinstance(call_data, dict) or "name" not in call_data:
                    if "invalid_function_call_json" not in issues:
                        issues.append("invalid_function_call_json")
                    continue

                tool_name = call_data.get("name")

                if tool_name and tool_name not in tools_dict:
                    if "tool_not_found" not in issues:
                        issues.append("tool_not_found")

                arguments = call_data.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        json.loads(arguments)
                    except Exception:
                        if "invalid_function_call_json" not in issues:
                            issues.append("invalid_function_call_json")

        return issues

    def process_file(self, file_path: str, backup_dir: str) -> Dict[str, Any]:
        """Process one JSON file."""
        print(f"\nProcessing file: {file_path}")

        file_stats = {
            "file_name": file_path,
            "original_count": 0,
            "removed_count": 0,
            "kept_count": 0
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  Error: cannot read file - {e}")
            return file_stats

        if not isinstance(data, list):
            data = [data]

        file_stats["original_count"] = len(data)
        print(f"  Original conversations: {len(data)}")

        cleaned_data = []
        removed_count = 0

        for idx, item in enumerate(data):
            issues = self.check_conversation_issues(item)

            if issues:
                removed_count += 1
                for issue_type in issues:
                    self.stats["removed_by_type"][issue_type] += 1
            else:
                cleaned_data.append(item)

        file_stats["removed_count"] = removed_count
        file_stats["kept_count"] = len(cleaned_data)

        print(f"  Removed conversations: {removed_count}")
        print(f"  Kept conversations: {len(cleaned_data)}")

        if removed_count > 0:
            rel_path = os.path.relpath(file_path, os.path.dirname(os.path.dirname(file_path)))
            backup_path = os.path.join(backup_dir, rel_path)
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)

            shutil.copy2(file_path, backup_path)
            print(f"  Backup saved: {backup_path}")

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

            print(f"  Cleaned data saved")
            self.stats["files_modified"].append(file_path)
        else:
            print(f"  No changes needed")

        return file_stats

    def clean_directory(self, input_dir: str, backup_dir: str):
        """Clean all JSON files under input_dir."""
        print("=" * 80)
        print("Remove conversations with validation issues")
        print("=" * 80)
        print(f"Input directory: {input_dir}")
        print(f"Backup directory: {backup_dir}")
        print()
        print("Conversations will be removed if they match:")
        print("  - tool_not_found: called tool not in the tool list")
        print("  - invalid_flow: message order is invalid")
        print("  - invalid_function_call_json: function_call JSON cannot be parsed")
        print("  - invalid_json: function_call value is not valid JSON")
        print("  - incomplete_flow: conversation ends mid-flow")

        os.makedirs(backup_dir, exist_ok=True)

        files = []
        for root, dirs, filenames in os.walk(input_dir):
            for filename in filenames:
                if filename.endswith('.json'):
                    files.append(os.path.join(root, filename))

        print(f"\nFound {len(files)} file(s) to process")

        self.stats["total_files"] = len(files)

        for file_path in files:
            file_stats = self.process_file(file_path, backup_dir)
            self.stats["total_conversations"] += file_stats["original_count"]
            self.stats["removed_conversations"] += file_stats["removed_count"]
            self.stats["kept_conversations"] += file_stats["kept_count"]

        self.generate_report(backup_dir)

    def generate_report(self, output_dir: str):
        """Write a short cleaning report."""
        print("\n" + "=" * 80)
        print("Cleaning finished")
        print("=" * 80)

        print(f"\nTotal files: {self.stats['total_files']}")
        print(f"Modified files: {len(self.stats['files_modified'])}")
        print(f"Total conversations: {self.stats['total_conversations']}")
        print(f"Removed conversations: {self.stats['removed_conversations']}")
        print(f"Kept conversations: {self.stats['kept_conversations']}")

        if self.stats['total_conversations'] > 0:
            removal_rate = self.stats['removed_conversations'] / self.stats['total_conversations'] * 100
            print(f"Removal rate: {removal_rate:.2f}%")

        print("\nCounts by issue type:")
        for issue_type, count in sorted(self.stats['removed_by_type'].items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  - {issue_type}: {count}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"cleaning_report_{timestamp}.json")

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

        print(f"\nDetailed report saved to: {report_file}")


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Remove conversations that fail validation checks')
    parser.add_argument('--input_dir', type=str,
                       default='/home/yijuan_liang/10.12Tool_Set/train_set/data/data_nonull',
                       help='Input data directory')
    parser.add_argument('--backup_dir', type=str,
                       default='/home/yijuan_liang/10.12Tool_Set/train_set/data/data_nonull_bak',
                       help='Backup directory')

    args = parser.parse_args()

    print("\nWarning: this will modify files in place (after backup)")
    print(f"Input directory: {args.input_dir}")
    print(f"Backup directory: {args.backup_dir}")

    response = input("\nContinue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled")
        return

    cleaner = ConversationCleaner()
    cleaner.clean_directory(args.input_dir, args.backup_dir)


if __name__ == "__main__":
    main()
