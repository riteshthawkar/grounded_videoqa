from adaptive_evidence_vqa.data.nextqa import normalize_nextgqa_record
from adaptive_evidence_vqa.data.tvqa import load_subtitles_map, normalize_tvqa_record


def test_normalize_tvqa_record_from_official_style_fields(tmp_path) -> None:
    subtitle_path = tmp_path / "subs.jsonl"
    subtitle_path.write_text(
        (
            '{"vid_name":"grey_s03e20_seg02_clip_14","sub":'
            '[{"text":"Meredith is outside.","start":76.0,"end":77.0}]}\n'
        ),
        encoding="utf-8",
    )
    subtitles_map = load_subtitles_map(subtitle_path)

    normalized = normalize_tvqa_record(
        {
            "qid": 0,
            "vid_name": "grey_s03e20_seg02_clip_14",
            "show_name": "grey",
            "ts": "76.01-84.2",
            "q": "Where is Meredith when George approaches her?",
            "a0": "Cafeteria",
            "a1": "Hallway",
            "a2": "Car",
            "a3": "Patients room",
            "a4": "Outside",
            "answer_idx": 4,
        },
        subtitles_map=subtitles_map,
        dataset_name="tvqa",
    )

    assert normalized["example_id"] == "tvqa:0"
    assert normalized["answer_index"] == 4
    assert normalized["temporal_grounding"] == [76.01, 84.2]
    assert normalized["subtitles"][0]["text"] == "Meredith is outside."


def test_normalize_nextgqa_record_with_grounding_and_frame_times() -> None:
    normalized = normalize_nextgqa_record(
        {
            "video_id": "4882821564",
            "frame_count": "2697",
            "width": "640",
            "height": "480",
            "question": "why did the boy pick up one present from the group of them and move to the sofa",
            "answer": "unwrap it",
            "qid": "1",
            "type": "CW",
            "a0": "share with the girl",
            "a1": "approach lady sitting there",
            "a2": "unwrap it",
            "a3": "playing with toy train",
            "a4": "gesture something",
        },
        dataset_name="nextgqa",
        grounding_map={
            "4882821564": {
                "duration": 89.5,
                "fps": 29.97,
                "location": {
                    "1": [[12.0, 16.0], [31.5, 34.0]],
                },
            }
        },
        frame_times_map={
            "4882821564": [0.5, 14.5, 31.5, 60.0, 88.0],
        },
        video_map={"4882821564": "0001/4882821564"},
    )

    assert normalized["example_id"] == "nextgqa:4882821564:1"
    assert normalized["video_id"] == "0001/4882821564"
    assert normalized["answer_index"] == 2
    assert normalized["temporal_grounding"] == [12.0, 34.0]
    assert normalized["metadata"]["frame_timestamps"] == [0.5, 14.5, 31.5, 60.0, 88.0]
    assert normalized["metadata"]["clip_span"] == [0.5, 88.0]
    assert normalized["metadata"]["temporal_grounding_spans"] == [[12.0, 16.0], [31.5, 34.0]]
