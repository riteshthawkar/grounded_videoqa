import argparse
from pathlib import Path

from adaptive_evidence_vqa.data.base import load_jsonl, save_jsonl
from adaptive_evidence_vqa.data.visual import (
    build_video_index,
    materialize_visual_evidence,
    resolve_video_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize real frame and segment artifacts for candidate pools.")
    parser.add_argument("--input-path", required=True, help="Path to candidate-pool JSONL.")
    parser.add_argument("--video-root", required=True, help="Root directory containing source videos.")
    parser.add_argument("--output-path", required=True, help="Path to enriched output JSONL.")
    parser.add_argument("--frames-dir", required=True, help="Directory for extracted frame images.")
    parser.add_argument("--segments-dir", help="Directory for extracted segment clips.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of records.")
    parser.add_argument(
        "--extract-segments",
        action="store_true",
        help="If set, extract individual segment clips; otherwise keep segment source paths pointing to the full video.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing visual artifacts.")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable name or path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input_path)
    if args.limit is not None:
        records = records[: args.limit]

    video_index = build_video_index(args.video_root)
    enriched_records = []
    missing_videos = []

    for record in records:
        video_path = resolve_video_path(
            video_id=record["video_id"],
            video_root=args.video_root,
            video_index=video_index,
        )
        if video_path is None:
            metadata = dict(record.get("metadata", {}))
            metadata["visual_materialization_status"] = "video_missing"
            record["metadata"] = metadata
            enriched_records.append(record)
            missing_videos.append(record["video_id"])
            continue

        enriched_records.append(
            materialize_visual_evidence(
                record=record,
                video_path=video_path,
                frames_root=args.frames_dir,
                segments_root=args.segments_dir,
                extract_segments=args.extract_segments,
                ffmpeg_bin=args.ffmpeg_bin,
                overwrite=args.overwrite,
            )
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(enriched_records, output_path)

    print(f"Wrote {len(enriched_records)} enriched records to {output_path}")
    if missing_videos:
        unique_missing = sorted(set(missing_videos))
        print(f"Missing videos for {len(unique_missing)} records")
        for video_id in unique_missing[:10]:
            print(f"  missing: {video_id}")


if __name__ == "__main__":
    main()
