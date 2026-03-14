import shutil
import subprocess
from pathlib import Path

import pytest

from adaptive_evidence_vqa.data.visual import build_video_index, materialize_visual_evidence, resolve_video_path


pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for visual tests")


def create_test_video(path: Path, *, size: str = "64x64") -> None:
    command = ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"color=c=blue:s={size}:d=1.5"]
    if path.suffix == ".mkv":
        command.extend(["-c:v", "ffv1"])
    else:
        command.extend(["-pix_fmt", "yuv420p"])
    command.append(str(path))
    subprocess.run(command, check=True)


def test_resolve_video_path_uses_index(tmp_path: Path) -> None:
    video_root = tmp_path / "videos"
    video_root.mkdir()
    video_path = video_root / "clip_001.mp4"
    create_test_video(video_path)

    index = build_video_index(video_root)
    resolved = resolve_video_path("clip_001", video_root=video_root, video_index=index)

    assert resolved == video_path


def test_materialize_visual_evidence_extracts_frame_and_segment_paths(tmp_path: Path) -> None:
    video_root = tmp_path / "videos"
    video_root.mkdir()
    video_path = video_root / "clip_002.mp4"
    create_test_video(video_path)

    record = {
        "example_id": "tvqa:visual-test",
        "video_id": "clip_002",
        "question": "What happens?",
        "options": ["A", "B"],
        "answer_index": 0,
        "subtitles": [],
        "frames": [{"text": "", "time": 0.5}],
        "segments": [{"text": "", "start": 0.2, "end": 0.8}],
        "metadata": {},
    }

    enriched = materialize_visual_evidence(
        record=record,
        video_path=video_path,
        frames_root=tmp_path / "frames",
        segments_root=tmp_path / "segments",
        extract_segments=True,
    )

    assert Path(enriched["frames"][0]["source_path"]).is_file()
    assert Path(enriched["segments"][0]["source_path"]).is_file()
    assert enriched["metadata"]["visual_materialization_status"] == "ok"


def test_materialize_visual_evidence_handles_odd_sized_video_segments(tmp_path: Path) -> None:
    video_root = tmp_path / "videos"
    video_root.mkdir()
    video_path = video_root / "clip_odd.mkv"
    create_test_video(video_path, size="63x65")

    record = {
        "example_id": "nextgqa:odd-test",
        "video_id": "clip_odd",
        "question": "What happens?",
        "options": ["A", "B"],
        "answer_index": 0,
        "subtitles": [],
        "frames": [{"text": "", "time": 0.5}],
        "segments": [{"text": "", "start": 0.2, "end": 0.8}],
        "metadata": {},
    }

    enriched = materialize_visual_evidence(
        record=record,
        video_path=video_path,
        frames_root=tmp_path / "frames",
        segments_root=tmp_path / "segments",
        extract_segments=True,
    )

    assert Path(enriched["segments"][0]["source_path"]).is_file()
