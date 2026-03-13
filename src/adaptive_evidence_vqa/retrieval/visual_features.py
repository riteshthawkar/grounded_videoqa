from pathlib import Path

import numpy as np


def clip_output_to_numpy(output) -> np.ndarray:
    tensor = output
    for attribute in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, attribute, None)
        if value is not None:
            tensor = value
            break

    if not hasattr(tensor, "detach"):
        raise TypeError(
            "Expected a tensor-like CLIP output or an object exposing "
            "`image_embeds`, `text_embeds`, or `pooler_output`."
        )
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors.astype(np.float32, copy=False)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return (vectors / norms).astype(np.float32, copy=False)


def select_segment_frame_indices(
    frame_times: np.ndarray,
    start_time: float,
    end_time: float,
) -> np.ndarray:
    indices = np.where((frame_times >= start_time) & (frame_times <= end_time))[0]
    if indices.size > 0:
        return indices

    if frame_times.size == 0:
        return np.zeros(0, dtype=np.int64)

    midpoint = (start_time + end_time) / 2.0
    nearest = int(np.argmin(np.abs(frame_times - midpoint)))
    return np.asarray([nearest], dtype=np.int64)


def aggregate_segment_embeddings(
    frame_times: np.ndarray,
    frame_embeddings: np.ndarray,
    segments: list[dict],
) -> np.ndarray:
    if not segments:
        width = frame_embeddings.shape[1] if frame_embeddings.ndim == 2 and frame_embeddings.size else 0
        return np.zeros((0, width), dtype=np.float32)

    aggregated = []
    for segment in segments:
        indices = select_segment_frame_indices(
            frame_times=frame_times,
            start_time=float(segment["start"]),
            end_time=float(segment["end"]),
        )
        if indices.size == 0:
            width = frame_embeddings.shape[1] if frame_embeddings.ndim == 2 and frame_embeddings.size else 0
            aggregated.append(np.zeros(width, dtype=np.float32))
            continue
        aggregated.append(frame_embeddings[indices].mean(axis=0))

    return l2_normalize(np.vstack(aggregated)) if aggregated else np.zeros((0, 0), dtype=np.float32)


def feature_artifact_path(output_dir: str | Path, example_id: str) -> Path:
    safe_name = example_id.replace("/", "_").replace(":", "_")
    return Path(output_dir) / f"{safe_name}.npz"
