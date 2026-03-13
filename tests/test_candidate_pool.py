from adaptive_evidence_vqa.data.candidates import build_candidate_record, chunk_subtitles


def test_chunk_subtitles_merges_close_lines() -> None:
    chunks = chunk_subtitles(
        [
            {"text": "Hello there.", "start": 0.0, "end": 1.0},
            {"text": "General Kenobi.", "start": 1.2, "end": 2.0},
            {"text": "New scene.", "start": 6.0, "end": 7.0},
        ],
        max_chunk_seconds=4.0,
        max_gap_seconds=1.0,
    )
    assert len(chunks) == 2
    assert chunks[0]["text"] == "Hello there. General Kenobi."


def test_build_candidate_record_generates_frames_and_segments() -> None:
    record = {
        "example_id": "tvqa:0",
        "video_id": "grey_s03e20_seg02_clip_14",
        "question": "Where is Meredith when George approaches her?",
        "options": ["Cafeteria", "Hallway", "Car", "Patients room", "Outside"],
        "answer_index": 4,
        "temporal_grounding": [76.01, 84.2],
        "subtitles": [
            {"text": "Meredith is outside.", "start": 76.0, "end": 77.0},
            {"text": "George approaches her.", "start": 78.0, "end": 79.0},
        ],
        "frames": [],
        "segments": [],
        "metadata": {"dataset": "tvqa", "qid": 0},
    }

    candidate_record = build_candidate_record(record, segment_window_seconds=4.0, segment_stride_seconds=2.0)

    assert candidate_record["subtitles"]
    assert candidate_record["frames"]
    assert candidate_record["segments"]
    assert candidate_record["metadata"]["clip_span"] == [76.0, 79.0]


def test_build_candidate_record_uses_metadata_frame_timestamps_without_grounding_leakage() -> None:
    record = {
        "example_id": "nextgqa:4882821564:1",
        "video_id": "0001/4882821564",
        "question": "Why did the boy move to the sofa?",
        "options": ["Share", "Approach", "Unwrap", "Play", "Gesture"],
        "answer_index": 2,
        "temporal_grounding": [12.0, 34.0],
        "subtitles": [],
        "frames": [],
        "segments": [],
        "metadata": {
            "dataset": "nextgqa",
            "qid": 1,
            "frame_timestamps": [0.5, 14.5, 31.5, 60.0, 88.0],
            "clip_span": [0.5, 88.0],
            "temporal_grounding_spans": [[12.0, 16.0], [31.5, 34.0]],
        },
    }

    candidate_record = build_candidate_record(
        record,
        segment_window_seconds=10.0,
        segment_stride_seconds=10.0,
    )

    assert [frame["time"] for frame in candidate_record["frames"]] == [0.5, 14.5, 31.5, 60.0, 88.0]
    assert candidate_record["segments"][0]["start"] == 0.5
    assert candidate_record["segments"][-1]["end"] == 88.0
    assert candidate_record["metadata"]["clip_span"] == [0.5, 88.0]
