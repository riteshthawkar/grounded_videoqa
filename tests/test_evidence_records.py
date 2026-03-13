from adaptive_evidence_vqa.data.evidence_records import parse_evidence_record, serialize_evidence_item
from adaptive_evidence_vqa.schemas import EvidenceItem, Modality


def test_evidence_record_round_trip_preserves_visual_metadata() -> None:
    item = EvidenceItem(
        evidence_id="toy:frame:0",
        modality=Modality.FRAME,
        text="person opens the door",
        start_time=12.0,
        end_time=12.0,
        source_path="/tmp/frame0.jpg",
        retrieval_score=1.2,
        acquisition_cost=1.0,
        metadata={
            "visual_feature_path": "/tmp/features.npz",
            "feature_index": 3,
            "feature_kind": "clip_image",
        },
    )

    parsed = parse_evidence_record(serialize_evidence_item(item))

    assert parsed.source_path == item.source_path
    assert parsed.metadata["visual_feature_path"] == "/tmp/features.npz"
    assert parsed.metadata["feature_index"] == 3
    assert parsed.retrieval_score == 1.2
