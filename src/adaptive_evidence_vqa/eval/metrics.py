from adaptive_evidence_vqa.schemas import EvidenceItem, ModelPrediction
from adaptive_evidence_vqa.utils import softmax


def accuracy(predicted_index: int, gold_index: int) -> float:
    return 1.0 if predicted_index == gold_index else 0.0


def evidence_cost(items: tuple[EvidenceItem, ...]) -> float:
    return sum(item.acquisition_cost for item in items)


def temporal_iou(
    a_start: float,
    a_end: float,
    b_start: float,
    b_end: float,
) -> float:
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return intersection / union


def max_temporal_iou_for_items(
    items: tuple[EvidenceItem, ...],
    gold_start: float,
    gold_end: float,
) -> float:
    best = 0.0
    for item in items:
        if item.start_time is None and item.end_time is None:
            continue
        item_start = item.start_time if item.start_time is not None else item.end_time
        item_end = item.end_time if item.end_time is not None else item.start_time
        if item_start is None or item_end is None:
            continue
        best = max(best, temporal_iou(item_start, item_end, gold_start, gold_end))
    return best


def temporal_target_spans(
    temporal_grounding: tuple[float, float] | list[float] | None = None,
    metadata: dict | None = None,
) -> list[tuple[float, float]]:
    if metadata is not None:
        raw_spans = metadata.get("temporal_grounding_spans")
        if isinstance(raw_spans, list) and raw_spans:
            normalized = []
            for span in raw_spans:
                if isinstance(span, (list, tuple)) and len(span) == 2:
                    start = float(span[0])
                    end = float(span[1])
                    if end < start:
                        start, end = end, start
                    normalized.append((start, end))
            if normalized:
                return normalized

    if temporal_grounding is None:
        return []
    return [(float(temporal_grounding[0]), float(temporal_grounding[1]))]


def max_temporal_iou_for_target_spans(
    items: tuple[EvidenceItem, ...],
    target_spans: list[tuple[float, float]],
) -> float:
    if not target_spans:
        return 0.0
    return max(
        max_temporal_iou_for_items(items, gold_start, gold_end)
        for gold_start, gold_end in target_spans
    )


def evidence_jaccard(
    items_a: tuple[EvidenceItem, ...],
    items_b: tuple[EvidenceItem, ...],
) -> float:
    ids_a = {item.evidence_id for item in items_a}
    ids_b = {item.evidence_id for item in items_b}
    if not ids_a and not ids_b:
        return float("nan")
    return len(ids_a & ids_b) / max(len(ids_a | ids_b), 1)


def modality_counts(items: tuple[EvidenceItem, ...]) -> dict[str, int]:
    counts = {"subtitle": 0, "frame": 0, "segment": 0}
    for item in items:
        counts[item.modality.value] += 1
    return counts


def modality_agreement(
    items_a: tuple[EvidenceItem, ...],
    items_b: tuple[EvidenceItem, ...],
) -> float:
    counts_a = modality_counts(items_a)
    counts_b = modality_counts(items_b)
    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())
    if total_a == 0 and total_b == 0:
        return float("nan")

    proportions_a = {
        key: (value / total_a if total_a > 0 else 0.0)
        for key, value in counts_a.items()
    }
    proportions_b = {
        key: (value / total_b if total_b > 0 else 0.0)
        for key, value in counts_b.items()
    }
    l1_distance = sum(abs(proportions_a[key] - proportions_b[key]) for key in counts_a)
    return max(0.0, 1.0 - (0.5 * l1_distance))


def _normalized_intervals(items: tuple[EvidenceItem, ...]) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for item in items:
        if item.start_time is None and item.end_time is None:
            continue
        start_time = item.start_time if item.start_time is not None else item.end_time
        end_time = item.end_time if item.end_time is not None else item.start_time
        if start_time is None or end_time is None:
            continue
        start, end = sorted((float(start_time), float(end_time)))
        intervals.append((start, end))
    return sorted(intervals)


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []

    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
            continue
        merged.append((start, end))
    return merged


def _interval_length(intervals: list[tuple[float, float]]) -> float:
    return sum(max(0.0, end - start) for start, end in intervals)


def _interval_intersection_length(
    intervals_a: list[tuple[float, float]],
    intervals_b: list[tuple[float, float]],
) -> float:
    intersection = 0.0
    index_a = 0
    index_b = 0
    while index_a < len(intervals_a) and index_b < len(intervals_b):
        start_a, end_a = intervals_a[index_a]
        start_b, end_b = intervals_b[index_b]
        overlap_start = max(start_a, start_b)
        overlap_end = min(end_a, end_b)
        if overlap_end > overlap_start:
            intersection += overlap_end - overlap_start
        if end_a <= end_b:
            index_a += 1
        else:
            index_b += 1
    return intersection


def temporal_interval_iou_for_items(
    items_a: tuple[EvidenceItem, ...],
    items_b: tuple[EvidenceItem, ...],
) -> float:
    merged_a = _merge_intervals(_normalized_intervals(items_a))
    merged_b = _merge_intervals(_normalized_intervals(items_b))
    if not merged_a and not merged_b:
        return float("nan")
    if not merged_a or not merged_b:
        return 0.0

    intersection = _interval_intersection_length(merged_a, merged_b)
    union = _interval_length(merged_a) + _interval_length(merged_b) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def gold_probability(prediction: ModelPrediction, gold_index: int) -> float:
    return softmax(list(prediction.option_scores))[gold_index]


def sufficiency(full_prediction: ModelPrediction, subset_prediction: ModelPrediction, gold_index: int) -> float:
    ratio = gold_probability(subset_prediction, gold_index) / max(
        gold_probability(full_prediction, gold_index),
        1e-8,
    )
    return min(1.0, ratio)


def comprehensiveness(full_prediction: ModelPrediction, reduced_prediction: ModelPrediction, gold_index: int) -> float:
    return max(0.0, gold_probability(full_prediction, gold_index) - gold_probability(reduced_prediction, gold_index))
