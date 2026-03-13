#!/usr/bin/env python3
import argparse
from pathlib import Path

from adaptive_evidence_vqa.data.base import load_jsonl, save_jsonl
from adaptive_evidence_vqa.data.tvqa import load_subtitles_map, normalize_tvqa_record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize raw TVQA annotations into the project JSONL schema.")
    parser.add_argument("--qa-path", required=True, help="Path to raw TVQA question JSONL.")
    parser.add_argument("--subtitles-path", required=True, help="Path to raw TVQA subtitles JSONL.")
    parser.add_argument("--output-path", required=True, help="Path to output normalized JSONL.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick debugging.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    qa_records = load_jsonl(args.qa_path)
    subtitles_map = load_subtitles_map(args.subtitles_path)

    if args.limit is not None:
        qa_records = qa_records[: args.limit]

    normalized = [
        normalize_tvqa_record(record, subtitles_map=subtitles_map, dataset_name="tvqa")
        for record in qa_records
    ]
    save_jsonl(normalized, args.output_path)
    print(f"Wrote {len(normalized)} normalized TVQA records to {Path(args.output_path)}")


if __name__ == "__main__":
    main()
