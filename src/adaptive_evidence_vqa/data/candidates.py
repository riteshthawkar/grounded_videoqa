from copy import deepcopy


def infer_clip_span(record: dict) -> tuple[float, float]:
    subtitles = record.get("subtitles", [])
    if subtitles:
        starts = [float(item["start"]) for item in subtitles]
        ends = [float(item["end"]) for item in subtitles]
        return min(starts), max(ends)

    metadata = record.get("metadata", {})
    clip_span = metadata.get("clip_span")
    if isinstance(clip_span, (list, tuple)) and len(clip_span) == 2:
        return float(clip_span[0]), float(clip_span[1])

    frame_timestamps = metadata.get("frame_timestamps")
    if isinstance(frame_timestamps, list) and frame_timestamps:
        normalized_times = [float(value) for value in frame_timestamps]
        return min(normalized_times), max(normalized_times)

    video_duration = metadata.get("video_duration")
    if video_duration is not None:
        return 0.0, float(video_duration)

    temporal_grounding = record.get("temporal_grounding")
    if temporal_grounding:
        return float(temporal_grounding[0]), float(temporal_grounding[1])

    return 0.0, 0.0


def overlapping_subtitle_text(subtitles: list[dict], start: float, end: float) -> str:
    texts = [
        item["text"].strip()
        for item in subtitles
        if float(item["end"]) >= start and float(item["start"]) <= end
    ]
    return " ".join(text for text in texts if text)


def nearest_subtitle_text(subtitles: list[dict], time_point: float, max_distance: float = 2.0) -> str:
    best_text = ""
    best_distance = None
    for item in subtitles:
        start = float(item["start"])
        end = float(item["end"])
        if start <= time_point <= end:
            return item["text"].strip()
        distance = min(abs(time_point - start), abs(time_point - end))
        if distance <= max_distance and (best_distance is None or distance < best_distance):
            best_distance = distance
            best_text = item["text"].strip()
    return best_text


def chunk_subtitles(
    subtitles: list[dict],
    max_chunk_seconds: float = 6.0,
    max_gap_seconds: float = 1.5,
    max_chars: int = 240,
) -> list[dict]:
    if not subtitles:
        return []

    ordered = sorted(subtitles, key=lambda item: (float(item["start"]), float(item["end"])))
    chunks: list[dict] = []
    current: list[dict] = []

    def flush() -> None:
        if not current:
            return
        text = " ".join(item["text"].strip() for item in current if item["text"].strip())
        chunks.append(
            {
                "text": text,
                "start": float(current[0]["start"]),
                "end": float(current[-1]["end"]),
            }
        )

    for item in ordered:
        if not current:
            current.append(item)
            continue

        current_start = float(current[0]["start"])
        current_end = float(current[-1]["end"])
        proposed_end = float(item["end"])
        gap = float(item["start"]) - current_end
        proposed_text = " ".join(part["text"].strip() for part in current + [item] if part["text"].strip())
        duration = proposed_end - current_start

        if gap > max_gap_seconds or duration > max_chunk_seconds or len(proposed_text) > max_chars:
            flush()
            current = [item]
        else:
            current.append(item)

    flush()
    return chunks


def generate_segment_candidates(
    subtitles: list[dict],
    clip_start: float,
    clip_end: float,
    window_seconds: float = 4.0,
    stride_seconds: float = 2.0,
) -> list[dict]:
    if clip_end <= clip_start:
        return []

    candidates = []
    cursor = clip_start
    while cursor < clip_end:
        end = min(cursor + window_seconds, clip_end)
        text = overlapping_subtitle_text(subtitles, cursor, end)
        candidates.append(
            {
                "text": text,
                "start": round(cursor, 3),
                "end": round(end, 3),
            }
        )
        if end >= clip_end:
            break
        cursor += stride_seconds
    return candidates


def generate_frame_candidates(
    subtitles: list[dict],
    clip_start: float,
    clip_end: float,
    stride_seconds: float = 2.0,
    frame_timestamps: list[float] | None = None,
) -> list[dict]:
    if frame_timestamps:
        return [
            {
                "text": nearest_subtitle_text(subtitles, float(time_point)),
                "time": round(float(time_point), 3),
            }
            for time_point in sorted(set(frame_timestamps))
        ]

    if clip_end <= clip_start:
        return []

    frames = []
    cursor = clip_start
    while cursor <= clip_end:
        frames.append(
            {
                "text": nearest_subtitle_text(subtitles, cursor),
                "time": round(cursor, 3),
            }
        )
        cursor += stride_seconds
    return frames


def build_candidate_record(
    record: dict,
    subtitle_chunk_seconds: float = 6.0,
    subtitle_gap_seconds: float = 1.5,
    frame_stride_seconds: float = 2.0,
    segment_window_seconds: float = 4.0,
    segment_stride_seconds: float = 2.0,
) -> dict:
    candidate_record = deepcopy(record)
    raw_subtitles = record.get("subtitles", [])
    clip_start, clip_end = infer_clip_span(record)

    candidate_record["subtitles"] = chunk_subtitles(
        raw_subtitles,
        max_chunk_seconds=subtitle_chunk_seconds,
        max_gap_seconds=subtitle_gap_seconds,
    )
    candidate_record["frames"] = generate_frame_candidates(
        raw_subtitles,
        clip_start=clip_start,
        clip_end=clip_end,
        stride_seconds=frame_stride_seconds,
        frame_timestamps=record.get("metadata", {}).get("frame_timestamps"),
    )
    candidate_record["segments"] = generate_segment_candidates(
        raw_subtitles,
        clip_start=clip_start,
        clip_end=clip_end,
        window_seconds=segment_window_seconds,
        stride_seconds=segment_stride_seconds,
    )
    metadata = dict(candidate_record.get("metadata", {}))
    metadata["raw_subtitle_count"] = len(raw_subtitles)
    metadata["clip_span"] = [clip_start, clip_end]
    metadata["candidate_pool_version"] = "v1"
    candidate_record["metadata"] = metadata
    return candidate_record
