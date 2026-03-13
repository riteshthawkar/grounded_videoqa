import math

from adaptive_evidence_vqa.eval.metrics import (
    comprehensiveness,
    evidence_jaccard,
    max_temporal_iou_for_items,
    max_temporal_iou_for_target_spans,
    modality_agreement,
    sufficiency,
    temporal_target_spans,
    temporal_interval_iou_for_items,
    temporal_iou,
)
from adaptive_evidence_vqa.schemas import EvidenceItem, ModelPrediction, Modality


def test_temporal_iou() -> None:
    assert temporal_iou(0.0, 4.0, 2.0, 6.0) == 2.0 / 6.0


def test_sufficiency_and_comprehensiveness_are_bounded() -> None:
    full = ModelPrediction(predicted_index=0, option_scores=(3.0, 1.0), confidence=0.88)
    subset = ModelPrediction(predicted_index=0, option_scores=(2.0, 1.0), confidence=0.73)
    reduced = ModelPrediction(predicted_index=1, option_scores=(1.0, 2.0), confidence=0.73)

    assert 0.0 < sufficiency(full, subset, gold_index=0) <= 1.0
    assert comprehensiveness(full, reduced, gold_index=0) > 0.0


def test_max_temporal_iou_for_items() -> None:
    items = (
        EvidenceItem(
            evidence_id="a",
            modality=Modality.SUBTITLE,
            text="first",
            start_time=0.0,
            end_time=1.0,
        ),
        EvidenceItem(
            evidence_id="b",
            modality=Modality.SEGMENT,
            text="second",
            start_time=2.0,
            end_time=5.0,
        ),
    )

    assert max_temporal_iou_for_items(items, gold_start=3.0, gold_end=4.0) == 1.0 / 3.0


def test_evidence_jaccard_and_modality_agreement() -> None:
    items_a = (
        EvidenceItem(evidence_id="a", modality=Modality.SUBTITLE, text="first"),
        EvidenceItem(evidence_id="b", modality=Modality.FRAME, text="second"),
    )
    items_b = (
        EvidenceItem(evidence_id="a", modality=Modality.SUBTITLE, text="first"),
        EvidenceItem(evidence_id="c", modality=Modality.SEGMENT, text="third"),
    )

    assert evidence_jaccard(items_a, items_b) == 1.0 / 3.0
    assert 0.0 < modality_agreement(items_a, items_b) < 1.0


def test_temporal_interval_iou_for_item_sets() -> None:
    items_a = (
        EvidenceItem(
            evidence_id="a",
            modality=Modality.SEGMENT,
            text="first",
            start_time=0.0,
            end_time=2.0,
        ),
        EvidenceItem(
            evidence_id="b",
            modality=Modality.SEGMENT,
            text="second",
            start_time=3.0,
            end_time=5.0,
        ),
    )
    items_b = (
        EvidenceItem(
            evidence_id="c",
            modality=Modality.SEGMENT,
            text="third",
            start_time=1.0,
            end_time=4.0,
        ),
    )

    assert temporal_interval_iou_for_items(items_a, items_b) == 2.0 / 5.0


def test_temporal_target_spans_prefers_full_metadata_spans() -> None:
    spans = temporal_target_spans(
        temporal_grounding=(10.0, 30.0),
        metadata={"temporal_grounding_spans": [[12.0, 16.0], [20.0, 24.0]]},
    )
    items = (
        EvidenceItem(
            evidence_id="b",
            modality=Modality.SEGMENT,
            text="second",
            start_time=20.0,
            end_time=24.0,
        ),
    )

    assert spans == [(12.0, 16.0), (20.0, 24.0)]
    assert max_temporal_iou_for_target_spans(items, spans) == 1.0


def test_empty_evidence_returns_nan_for_pairwise_metrics() -> None:
    empty: tuple[EvidenceItem, ...] = ()
    assert math.isnan(evidence_jaccard(empty, empty))
    assert math.isnan(modality_agreement(empty, empty))
    assert math.isnan(temporal_interval_iou_for_items(empty, empty))
