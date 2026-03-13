from pathlib import Path

from adaptive_evidence_vqa.data.base import load_json, load_jsonl
from adaptive_evidence_vqa.schemas import AnswerOption, EvidenceItem, Modality, QuestionExample


def subtitle_item(
    example_id: str,
    index: int,
    text: str,
    start: float,
    end: float,
    source_path: str | None = None,
    metadata: dict[str, object] | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"{example_id}:subtitle:{index}",
        modality=Modality.SUBTITLE,
        text=text,
        start_time=start,
        end_time=end,
        source_path=source_path,
        acquisition_cost=1.0,
        metadata=metadata or {},
    )


def frame_item(
    example_id: str,
    index: int,
    description: str,
    time: float,
    source_path: str | None = None,
    metadata: dict[str, object] | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"{example_id}:frame:{index}",
        modality=Modality.FRAME,
        text=description,
        start_time=time,
        end_time=time,
        source_path=source_path,
        acquisition_cost=1.0,
        metadata=metadata or {},
    )


def segment_item(
    example_id: str,
    index: int,
    description: str,
    start: float,
    end: float,
    source_path: str | None = None,
    metadata: dict[str, object] | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"{example_id}:segment:{index}",
        modality=Modality.SEGMENT,
        text=description,
        start_time=start,
        end_time=end,
        source_path=source_path,
        acquisition_cost=1.5,
        metadata=metadata or {},
    )


def parse_time_span(value: str | list[float] | tuple[float, float] | None) -> tuple[float, float] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        if "-" not in value:
            return None
        start, end = value.split("-", maxsplit=1)
        return float(start), float(end)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    return None


def load_subtitles_map(path: str | Path) -> dict[str, list[dict]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        subtitle_records = load_jsonl(path)
        mapping: dict[str, list[dict]] = {}
        for record in subtitle_records:
            mapping[record["vid_name"]] = [
                {
                    "text": item["text"],
                    "start": float(item["start"]),
                    "end": float(item["end"]),
                }
                for item in record.get("sub", [])
            ]
        return mapping

    raw = load_json(path)
    mapping = {}
    for vid_name, payload in raw.items():
        if "sub" in payload:
            mapping[vid_name] = [
                {
                    "text": item["text"],
                    "start": float(item["start"]),
                    "end": float(item["end"]),
                }
                for item in payload.get("sub", [])
            ]
            continue

        sub_text = payload.get("sub_text", [])
        sub_time = payload.get("sub_time", [])
        mapping[vid_name] = [
            {
                "text": text,
                "start": float(time_pair[0]),
                "end": float(time_pair[1]),
            }
            for text, time_pair in zip(sub_text, sub_time, strict=False)
        ]
    return mapping


def normalize_tvqa_record(
    record: dict,
    subtitles_map: dict[str, list[dict]] | None = None,
    dataset_name: str = "tvqa",
) -> dict:
    qid = record["qid"]
    vid_name = record["vid_name"]
    options = [record[f"a{index}"] for index in range(5)]
    answer_index = int(record["answer_idx"]) if "answer_idx" in record else None
    temporal_grounding = parse_time_span(record.get("ts"))
    subtitles = subtitles_map.get(vid_name, []) if subtitles_map else []

    metadata = {
        "dataset": dataset_name,
        "qid": qid,
        "show_name": record.get("show_name"),
    }
    if "bbox" in record:
        metadata["bbox_grounding"] = record["bbox"]

    normalized = {
        "example_id": f"{dataset_name}:{qid}",
        "video_id": vid_name,
        "question": record["q"],
        "options": options,
        "answer_index": answer_index,
        "temporal_grounding": list(temporal_grounding) if temporal_grounding else None,
        "subtitles": subtitles,
        "frames": [],
        "segments": [],
        "metadata": metadata,
    }
    return normalized


def parse_tvqa_like_record(record: dict) -> QuestionExample:
    options = tuple(
        AnswerOption(index=index, text=text)
        for index, text in enumerate(record["options"])
    )
    subtitles = tuple(
        subtitle_item(
            example_id=record["example_id"],
            index=index,
            text=item["text"],
            start=float(item["start"]),
            end=float(item["end"]),
            source_path=item.get("source_path"),
            metadata=item.get("metadata", {}),
        )
        for index, item in enumerate(record.get("subtitles", []))
    )
    frames = tuple(
        frame_item(
            example_id=record["example_id"],
            index=index,
            description=item["text"],
            time=float(item["time"]),
            source_path=item.get("source_path"),
            metadata=item.get("metadata", {}),
        )
        for index, item in enumerate(record.get("frames", []))
    )
    segments = tuple(
        segment_item(
            example_id=record["example_id"],
            index=index,
            description=item["text"],
            start=float(item["start"]),
            end=float(item["end"]),
            source_path=item.get("source_path"),
            metadata=item.get("metadata", {}),
        )
        for index, item in enumerate(record.get("segments", []))
    )
    return QuestionExample(
        example_id=record["example_id"],
        video_id=record["video_id"],
        question=record["question"],
        options=options,
        answer_index=record.get("answer_index"),
        temporal_grounding=tuple(record["temporal_grounding"]) if record.get("temporal_grounding") else None,
        subtitles=subtitles,
        frames=frames,
        segments=segments,
        metadata=record.get("metadata", {}),
    )
