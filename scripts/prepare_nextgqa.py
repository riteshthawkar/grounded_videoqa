#!/usr/bin/env python3
import argparse
from pathlib import Path

from adaptive_evidence_vqa.data.base import save_jsonl
from adaptive_evidence_vqa.data.nextqa import (
    build_duration_map,
    load_nextqa_csv,
    load_optional_json,
    normalize_nextgqa_record,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize raw NExT-GQA annotations into the project JSONL schema.")
    parser.add_argument("--qa-path", required=True, help="Path to raw NExT-GQA CSV.")
    parser.add_argument("--output-path", required=True, help="Path to output normalized JSONL.")
    parser.add_argument("--dataset-name", default="nextgqa", help="Dataset name prefix used in example ids.")
    parser.add_argument("--gsub-path", help="Optional path to grounded-span JSON, e.g. gsub_val.json.")
    parser.add_argument(
        "--frame-times-path",
        help="Optional path to frame timestamp JSON, e.g. frame2time_val.json or upbd_val.json.",
    )
    parser.add_argument(
        "--video-map-path",
        help="Optional path to map_vid_vidorID.json so video ids resolve to relative paths under --video-root.",
    )
    parser.add_argument(
        "--video-root",
        help="Optional root directory containing source videos. If provided, durations are probed for videos missing clip-span metadata.",
    )
    parser.add_argument("--ffprobe-bin", default="ffprobe", help="ffprobe executable name or path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick debugging.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    qa_records = load_nextqa_csv(args.qa_path)
    if args.limit is not None:
        qa_records = qa_records[: args.limit]

    grounding_map = load_optional_json(args.gsub_path)
    frame_times_map = load_optional_json(args.frame_times_path)
    video_map = load_optional_json(args.video_map_path)
    duration_map = (
        build_duration_map(
            qa_records,
            args.video_root,
            video_map=video_map,
            ffprobe_bin=args.ffprobe_bin,
        )
        if args.video_root
        else None
    )

    normalized = [
        normalize_nextgqa_record(
            record,
            dataset_name=args.dataset_name,
            grounding_map=grounding_map,
            frame_times_map=frame_times_map,
            video_map=video_map,
            duration_map=duration_map,
        )
        for record in qa_records
    ]
    save_jsonl(normalized, args.output_path)

    records_with_clip_span = sum(1 for record in normalized if record["metadata"].get("clip_span") is not None)
    records_with_grounding = sum(1 for record in normalized if record["metadata"].get("temporal_grounding_spans"))
    print(f"Wrote {len(normalized)} normalized NExT-GQA records to {Path(args.output_path)}")
    print(f"Records with clip span metadata: {records_with_clip_span}")
    print(f"Records with grounding spans: {records_with_grounding}")
    if records_with_clip_span < len(normalized):
        print(
            "Warning: some records are missing clip-span metadata. "
            "Candidate generation for those records may produce no visual evidence."
        )


if __name__ == "__main__":
    main()
