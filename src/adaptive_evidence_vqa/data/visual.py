import re
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm")
SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")
H264_SAFE_PAD_FILTER = "pad=ceil(iw/2)*2:ceil(ih/2)*2"


def sanitize_identifier(value: str) -> str:
    return SAFE_ID_RE.sub("_", value).strip("_") or "item"


def build_video_index(
    video_root: str | Path,
    extensions: tuple[str, ...] = VIDEO_EXTENSIONS,
) -> dict[str, str]:
    root = Path(video_root)
    index: dict[str, str] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        relative_no_suffix = str(path.relative_to(root).with_suffix(""))
        index.setdefault(relative_no_suffix, str(path))
        index.setdefault(path.stem, str(path))
    return index


def resolve_video_path(
    video_id: str,
    video_root: str | Path,
    video_index: dict[str, str] | None = None,
    extensions: tuple[str, ...] = VIDEO_EXTENSIONS,
) -> Path | None:
    candidate = Path(video_id)
    if candidate.is_file():
        return candidate

    root = Path(video_root)
    if video_index is not None:
        indexed = video_index.get(video_id)
        if indexed:
            return Path(indexed)

    if candidate.suffix.lower() in extensions:
        direct = root / candidate
        if direct.is_file():
            return direct

    for extension in extensions:
        direct = root / f"{video_id}{extension}"
        if direct.is_file():
            return direct

    for path in root.rglob(f"{video_id}.*"):
        if path.is_file() and path.suffix.lower() in extensions:
            return path

    return None


def ensure_ffmpeg(ffmpeg_bin: str = "ffmpeg") -> None:
    if shutil.which(ffmpeg_bin) is None:
        raise FileNotFoundError(
            f"Could not find `{ffmpeg_bin}` on PATH. Install ffmpeg before materializing visual evidence."
        )


def ensure_ffprobe(ffprobe_bin: str = "ffprobe") -> None:
    if shutil.which(ffprobe_bin) is None:
        raise FileNotFoundError(
            f"Could not find `{ffprobe_bin}` on PATH. Install ffprobe before probing video durations."
        )


def probe_video_duration(
    video_path: str | Path,
    ffprobe_bin: str = "ffprobe",
) -> float:
    ensure_ffprobe(ffprobe_bin)
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video_path),
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def extract_frame_image(
    video_path: str | Path,
    time_point: float,
    output_path: str | Path,
    ffmpeg_bin: str = "ffmpeg",
    video_duration: float | None = None,
    overwrite: bool = False,
) -> Path:
    ensure_ffmpeg(ffmpeg_bin)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        return output

    resolved_duration = video_duration
    if resolved_duration is None:
        try:
            resolved_duration = probe_video_duration(video_path)
        except Exception:
            resolved_duration = None

    candidate_times: list[float] = []

    def add_candidate(value: float) -> None:
        candidate = max(0.0, float(value))
        if resolved_duration is not None:
            safe_upper_bound = max(0.0, float(resolved_duration) - 0.1)
            candidate = min(candidate, safe_upper_bound)
        if all(abs(candidate - existing) > 1e-3 for existing in candidate_times):
            candidate_times.append(candidate)

    add_candidate(time_point)
    add_candidate(time_point - 0.25)
    add_candidate(time_point - 0.5)
    if resolved_duration is not None:
        add_candidate(resolved_duration - 0.25)
        add_candidate(resolved_duration - 0.5)
        add_candidate(resolved_duration - 1.0)
    add_candidate(0.0)

    for candidate_time in candidate_times:
        for accurate_seek in (False, True):
            if output.exists():
                output.unlink()

            if accurate_seek:
                command = [
                    ffmpeg_bin,
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(video_path),
                    "-ss",
                    f"{candidate_time:.3f}",
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(output),
                ]
            else:
                command = [
                    ffmpeg_bin,
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{candidate_time:.3f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(output),
                ]

            subprocess.run(command, check=True)
            if output.exists() and output.stat().st_size > 0:
                return output

    raise FileNotFoundError(
        f"ffmpeg completed without writing a frame image for {video_path} at time {time_point:.3f}."
    )


def extract_segment_clip(
    video_path: str | Path,
    start_time: float,
    end_time: float,
    output_path: str | Path,
    ffmpeg_bin: str = "ffmpeg",
    overwrite: bool = False,
) -> Path:
    ensure_ffmpeg(ffmpeg_bin)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        return output

    duration = max(0.0, end_time - start_time)
    command = [
        ffmpeg_bin,
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_time:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
        "-an",
        "-vf",
        H264_SAFE_PAD_FILTER,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)
    return output


def frame_artifact_path(
    frames_root: str | Path,
    example_id: str,
    index: int,
    time_point: float,
) -> Path:
    safe_example_id = sanitize_identifier(example_id)
    return Path(frames_root) / safe_example_id / f"frame_{index:04d}_{time_point:.3f}.jpg"


def segment_artifact_path(
    segments_root: str | Path,
    example_id: str,
    index: int,
    start_time: float,
    end_time: float,
) -> Path:
    safe_example_id = sanitize_identifier(example_id)
    return Path(segments_root) / safe_example_id / f"segment_{index:04d}_{start_time:.3f}_{end_time:.3f}.mp4"


def materialize_visual_evidence(
    record: dict,
    video_path: str | Path,
    frames_root: str | Path,
    segments_root: str | Path | None = None,
    extract_segments: bool = False,
    ffmpeg_bin: str = "ffmpeg",
    overwrite: bool = False,
) -> dict:
    enriched = deepcopy(record)
    video_source_path = str(video_path)
    record_video_duration = enriched.get("metadata", {}).get("video_duration")

    for index, frame in enumerate(enriched.get("frames", [])):
        output_path = frame_artifact_path(
            frames_root=frames_root,
            example_id=enriched["example_id"],
            index=index,
            time_point=float(frame["time"]),
        )
        frame_image = extract_frame_image(
            video_path=video_path,
            time_point=float(frame["time"]),
            output_path=output_path,
            ffmpeg_bin=ffmpeg_bin,
            video_duration=float(record_video_duration) if record_video_duration is not None else None,
            overwrite=overwrite,
        )
        metadata = dict(frame.get("metadata", {}))
        metadata["video_source_path"] = video_source_path
        metadata["artifact_type"] = "frame_image"
        frame["source_path"] = str(frame_image)
        frame["metadata"] = metadata

    for index, segment in enumerate(enriched.get("segments", [])):
        metadata = dict(segment.get("metadata", {}))
        metadata["video_source_path"] = video_source_path
        metadata["artifact_type"] = "segment_reference"
        source_path = video_source_path
        if extract_segments:
            if segments_root is None:
                raise ValueError("segments_root is required when extract_segments is enabled.")
            output_path = segment_artifact_path(
                segments_root=segments_root,
                example_id=enriched["example_id"],
                index=index,
                start_time=float(segment["start"]),
                end_time=float(segment["end"]),
            )
            clip_path = extract_segment_clip(
                video_path=video_path,
                start_time=float(segment["start"]),
                end_time=float(segment["end"]),
                output_path=output_path,
                ffmpeg_bin=ffmpeg_bin,
                overwrite=overwrite,
            )
            source_path = str(clip_path)
            metadata["artifact_type"] = "segment_clip"

        segment["source_path"] = source_path
        segment["metadata"] = metadata

    record_metadata = dict(enriched.get("metadata", {}))
    record_metadata["video_source_path"] = video_source_path
    record_metadata["visual_materialization_status"] = "ok"
    record_metadata["frame_artifacts_root"] = str(Path(frames_root))
    if segments_root is not None:
        record_metadata["segment_artifacts_root"] = str(Path(segments_root))
    record_metadata["segments_extracted"] = extract_segments
    enriched["metadata"] = record_metadata
    return enriched
