import csv
from pathlib import Path

from adaptive_evidence_vqa.data.base import load_json
from adaptive_evidence_vqa.data.visual import build_video_index, probe_video_duration, resolve_video_path


def load_nextqa_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_optional_json(path: str | Path | None) -> dict | None:
    if path is None:
        return None
    return load_json(path)


def resolve_answer_index(options: list[str], answer: str) -> int:
    stripped_answer = answer.strip()
    for index, option in enumerate(options):
        if option.strip() == stripped_answer:
            return index

    normalized_answer = stripped_answer.casefold()
    for index, option in enumerate(options):
        if option.strip().casefold() == normalized_answer:
            return index

    raise ValueError(f"Could not resolve answer `{answer}` against options: {options}")


def normalize_grounding_spans(spans: list[list[float]] | list[tuple[float, float]]) -> list[list[float]]:
    normalized: list[list[float]] = []
    for start_time, end_time in spans:
        start = float(start_time)
        end = float(end_time)
        if end < start:
            start, end = end, start
        normalized.append([start, end])
    return normalized


def grounding_span_hull(spans: list[list[float]]) -> list[float] | None:
    if not spans:
        return None
    return [
        min(span[0] for span in spans),
        max(span[1] for span in spans),
    ]


def build_duration_map(
    qa_records: list[dict[str, str]],
    video_root: str | Path,
    *,
    video_map: dict[str, str] | None = None,
    ffprobe_bin: str = "ffprobe",
) -> dict[str, float]:
    durations: dict[str, float] = {}
    video_index = build_video_index(video_root)
    for record in qa_records:
        raw_video_id = str(record.get("video_id") or record.get("video"))
        if raw_video_id in durations:
            continue

        mapped_video_id = video_map.get(raw_video_id, raw_video_id) if video_map else raw_video_id
        video_path = resolve_video_path(
            video_id=mapped_video_id,
            video_root=video_root,
            video_index=video_index,
        )
        if video_path is None:
            continue
        durations[raw_video_id] = probe_video_duration(video_path, ffprobe_bin=ffprobe_bin)
    return durations


def normalize_nextgqa_record(
    record: dict[str, str],
    *,
    dataset_name: str = "nextgqa",
    grounding_map: dict | None = None,
    frame_times_map: dict | None = None,
    video_map: dict[str, str] | None = None,
    duration_map: dict[str, float] | None = None,
) -> dict:
    raw_video_id = str(record.get("video_id") or record.get("video"))
    qid = str(record["qid"])
    options = [record[f"a{index}"] for index in range(5)]
    answer_index = resolve_answer_index(options, str(record["answer"]))
    mapped_video_id = video_map.get(raw_video_id, raw_video_id) if video_map else raw_video_id

    temporal_grounding_spans: list[list[float]] = []
    video_duration = None
    fps = None
    if grounding_map is not None:
        video_grounding = grounding_map.get(raw_video_id) or grounding_map.get(mapped_video_id)
        if isinstance(video_grounding, dict):
            video_duration = float(video_grounding["duration"]) if "duration" in video_grounding else None
            fps = float(video_grounding["fps"]) if "fps" in video_grounding else None
            location_map = video_grounding.get("location", {})
            if qid in location_map:
                temporal_grounding_spans = normalize_grounding_spans(location_map[qid])

    if duration_map is not None and raw_video_id in duration_map:
        video_duration = float(duration_map[raw_video_id])

    frame_timestamps: list[float] = []
    if frame_times_map is not None:
        raw_times = frame_times_map.get(raw_video_id) or frame_times_map.get(mapped_video_id) or []
        frame_timestamps = [float(value) for value in raw_times]

    clip_span = None
    if frame_timestamps:
        clip_span = [min(frame_timestamps), max(frame_timestamps)]
    elif video_duration is not None:
        clip_span = [0.0, float(video_duration)]

    metadata = {
        "dataset": dataset_name,
        "qid": int(qid) if qid.isdigit() else qid,
        "question_type": record.get("type"),
        "raw_video_id": raw_video_id,
        "frame_count": int(record["frame_count"]) if record.get("frame_count") else None,
        "width": int(record["width"]) if record.get("width") else None,
        "height": int(record["height"]) if record.get("height") else None,
    }
    if mapped_video_id != raw_video_id:
        metadata["mapped_video_id"] = mapped_video_id
    if temporal_grounding_spans:
        metadata["temporal_grounding_spans"] = temporal_grounding_spans
    if frame_timestamps:
        metadata["frame_timestamps"] = frame_timestamps
    if clip_span is not None:
        metadata["clip_span"] = clip_span
    if video_duration is not None:
        metadata["video_duration"] = float(video_duration)
    if fps is not None:
        metadata["fps"] = float(fps)

    return {
        "example_id": f"{dataset_name}:{raw_video_id}:{qid}",
        "video_id": mapped_video_id,
        "question": record["question"],
        "options": options,
        "answer_index": answer_index,
        "temporal_grounding": grounding_span_hull(temporal_grounding_spans),
        "subtitles": [],
        "frames": [],
        "segments": [],
        "metadata": metadata,
    }
