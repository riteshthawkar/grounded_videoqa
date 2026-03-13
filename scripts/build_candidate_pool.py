#!/usr/bin/env python3
import argparse
from pathlib import Path

from adaptive_evidence_vqa.data.base import load_jsonl, save_jsonl
from adaptive_evidence_vqa.data.candidates import build_candidate_record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build candidate evidence pools from normalized VideoQA JSONL.")
    parser.add_argument("--input-path", required=True, help="Path to normalized JSONL.")
    parser.add_argument("--output-path", required=True, help="Path to output candidate-pool JSONL.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for debugging.")
    parser.add_argument("--subtitle-chunk-seconds", type=float, default=6.0)
    parser.add_argument("--subtitle-gap-seconds", type=float, default=1.5)
    parser.add_argument("--frame-stride-seconds", type=float, default=2.0)
    parser.add_argument("--segment-window-seconds", type=float, default=4.0)
    parser.add_argument("--segment-stride-seconds", type=float, default=2.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = load_jsonl(args.input_path)
    if args.limit is not None:
        records = records[: args.limit]

    candidate_records = [
        build_candidate_record(
            record,
            subtitle_chunk_seconds=args.subtitle_chunk_seconds,
            subtitle_gap_seconds=args.subtitle_gap_seconds,
            frame_stride_seconds=args.frame_stride_seconds,
            segment_window_seconds=args.segment_window_seconds,
            segment_stride_seconds=args.segment_stride_seconds,
        )
        for record in records
    ]
    save_jsonl(candidate_records, args.output_path)
    print(f"Wrote {len(candidate_records)} candidate-pool records to {Path(args.output_path)}")


if __name__ == "__main__":
    main()
