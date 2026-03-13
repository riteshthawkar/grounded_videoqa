import argparse
from pathlib import Path

import numpy as np

from adaptive_evidence_vqa.data.base import load_jsonl, save_jsonl
from adaptive_evidence_vqa.retrieval.visual_features import (
    aggregate_segment_embeddings,
    clip_output_to_numpy,
    feature_artifact_path,
    l2_normalize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CLIP features for frame and segment evidence.")
    parser.add_argument("--input-path", required=True, help="Path to enriched candidate-pool JSONL.")
    parser.add_argument("--output-path", required=True, help="Path to output JSONL with feature metadata.")
    parser.add_argument("--feature-dir", required=True, help="Directory to store per-example feature files.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of records.")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32", help="CLIP model name.")
    parser.add_argument("--batch-size", type=int, default=16, help="Image batch size.")
    parser.add_argument("--device", help="Torch device override, e.g. cpu or mps.")
    return parser.parse_args()


def resolve_device(torch_module, requested_device: str | None) -> str:
    if requested_device:
        return requested_device
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def batched(items: list[str], batch_size: int) -> list[list[str]]:
    return [
        items[index : index + batch_size]
        for index in range(0, len(items), batch_size)
    ]


def main() -> None:
    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise ImportError(
            "extract_clip_features.py requires the vision dependencies. "
            "Run `conda env update -f environment.yml --prune` first."
        ) from exc

    args = parse_args()
    records = load_jsonl(args.input_path)
    if args.limit is not None:
        records = records[: args.limit]

    device = resolve_device(torch, args.device)
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(args.model_name)

    updated_records = []
    for record in records:
        frame_paths = [frame.get("source_path") for frame in record.get("frames", [])]
        if not frame_paths or any(path is None for path in frame_paths):
            metadata = dict(record.get("metadata", {}))
            metadata["visual_feature_status"] = "missing_frame_artifacts"
            record["metadata"] = metadata
            updated_records.append(record)
            continue

        frame_embeddings = []
        for batch_paths in batched(frame_paths, args.batch_size):
            images = []
            for path in batch_paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            with torch.no_grad():
                image_features = model.get_image_features(pixel_values=pixel_values)
            frame_embeddings.append(clip_output_to_numpy(image_features))

        stacked_frame_embeddings = l2_normalize(np.vstack(frame_embeddings))
        frame_times = np.asarray(
            [float(frame["time"]) for frame in record.get("frames", [])],
            dtype=np.float32,
        )
        segment_embeddings = aggregate_segment_embeddings(
            frame_times=frame_times,
            frame_embeddings=stacked_frame_embeddings,
            segments=record.get("segments", []),
        )

        feature_path = feature_artifact_path(args.feature_dir, record["example_id"])
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            feature_path,
            frame_embeddings=stacked_frame_embeddings,
            frame_times=frame_times,
            segment_embeddings=segment_embeddings,
        )

        for index, frame in enumerate(record.get("frames", [])):
            metadata = dict(frame.get("metadata", {}))
            metadata["feature_index"] = index
            metadata["feature_kind"] = "clip_image"
            metadata["visual_feature_path"] = str(feature_path)
            frame["metadata"] = metadata

        for index, segment in enumerate(record.get("segments", [])):
            metadata = dict(segment.get("metadata", {}))
            metadata["feature_index"] = index
            metadata["feature_kind"] = "clip_segment_mean"
            metadata["visual_feature_path"] = str(feature_path)
            segment["metadata"] = metadata

        metadata = dict(record.get("metadata", {}))
        metadata["visual_feature_status"] = "ok"
        metadata["visual_feature_path"] = str(feature_path)
        metadata["visual_encoder"] = args.model_name
        metadata["visual_embedding_dim"] = int(stacked_frame_embeddings.shape[1])
        record["metadata"] = metadata
        updated_records.append(record)

    save_jsonl(updated_records, args.output_path)
    print(f"Wrote feature-enriched records to {args.output_path}")


if __name__ == "__main__":
    main()
