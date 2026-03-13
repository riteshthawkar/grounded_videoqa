from adaptive_evidence_vqa.schemas import EvidenceItem, Modality


def serialize_evidence_item(item: EvidenceItem) -> dict:
    return {
        "evidence_id": item.evidence_id,
        "modality": item.modality.value,
        "text": item.text,
        "start_time": item.start_time,
        "end_time": item.end_time,
        "source_path": item.source_path,
        "retrieval_score": item.retrieval_score,
        "acquisition_cost": item.acquisition_cost,
        "metadata": dict(item.metadata),
    }


def serialize_evidence(items: tuple[EvidenceItem, ...]) -> list[dict]:
    return [serialize_evidence_item(item) for item in items]


def parse_evidence_record(record: dict) -> EvidenceItem:
    start_time = record.get("start_time")
    end_time = record.get("end_time")
    if start_time is None and "time" in record:
        start_time = record["time"]
    if end_time is None and "time" in record:
        end_time = record["time"]

    return EvidenceItem(
        evidence_id=record["evidence_id"],
        modality=Modality(record["modality"]),
        text=record.get("text", ""),
        start_time=float(start_time) if start_time is not None else None,
        end_time=float(end_time) if end_time is not None else None,
        source_path=record.get("source_path"),
        retrieval_score=float(record.get("retrieval_score", 0.0)),
        acquisition_cost=float(record.get("acquisition_cost", 1.0)),
        metadata=dict(record.get("metadata", {})),
    )
