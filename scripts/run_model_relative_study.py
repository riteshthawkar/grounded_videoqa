import argparse
import json
import math
from pathlib import Path

from adaptive_evidence_vqa.data.base import save_jsonl
from adaptive_evidence_vqa.data.evidence_records import serialize_evidence
from adaptive_evidence_vqa.data.normalized import load_normalized_examples
from adaptive_evidence_vqa.eval.metrics import (
    accuracy,
    evidence_cost,
    evidence_jaccard,
    max_temporal_iou_for_target_spans,
    modality_agreement,
    modality_counts,
    sufficiency,
    temporal_target_spans,
    temporal_interval_iou_for_items,
)
from adaptive_evidence_vqa.models.answerer import Answerer, build_answerer
from adaptive_evidence_vqa.models.oracle import ORACLE_MODES, MinimalEvidenceOracle, OracleConfig
from adaptive_evidence_vqa.retrieval.base import FixedBudgetRetriever, RetrievalAllocation, build_named_retriever
from adaptive_evidence_vqa.schemas import EvidenceItem, ModelPrediction, QuestionExample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare model-relative minimal-evidence subsets across two answerers."
    )
    parser.add_argument("--input-path", required=True, help="Path to candidate-pool JSONL.")
    parser.add_argument("--output-dir", required=True, help="Directory where summary and per-example JSONL are stored.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on the number of examples.")

    parser.add_argument("--answerer-a", choices=("lexical", "linear", "frozen_multimodal"), required=True)
    parser.add_argument("--answerer-a-model-dir", help="Model directory for answerer A when using `linear`.")
    parser.add_argument(
        "--answerer-a-model-name",
        default="openai/clip-vit-base-patch32",
        help="Model name for answerer A when using `frozen_multimodal`.",
    )
    parser.add_argument("--answerer-a-device", help="Optional device override for answerer A.")
    parser.add_argument("--answerer-a-label", default="answerer_a", help="Human-readable label for answerer A.")

    parser.add_argument("--answerer-b", choices=("lexical", "linear", "frozen_multimodal"), required=True)
    parser.add_argument("--answerer-b-model-dir", help="Model directory for answerer B when using `linear`.")
    parser.add_argument(
        "--answerer-b-model-name",
        default="openai/clip-vit-base-patch32",
        help="Model name for answerer B when using `frozen_multimodal`.",
    )
    parser.add_argument("--answerer-b-device", help="Optional device override for answerer B.")
    parser.add_argument("--answerer-b-label", default="answerer_b", help="Human-readable label for answerer B.")

    parser.add_argument(
        "--retriever",
        choices=("lexical", "bm25", "hybrid_clip"),
        default="bm25",
        help="Retriever used to build the seed pool.",
    )
    parser.add_argument(
        "--visual-model-name",
        default="openai/clip-vit-base-patch32",
        help="Visual-text encoder used when `--retriever hybrid_clip` is selected.",
    )
    parser.add_argument("--visual-device", help="Optional device override for the hybrid retriever.")
    parser.add_argument("--subtitle-k", type=int, default=2, help="Number of subtitle items in the seed pool.")
    parser.add_argument("--frame-k", type=int, default=2, help="Number of frame items in the seed pool.")
    parser.add_argument("--segment-k", type=int, default=2, help="Number of segment items in the seed pool.")

    parser.add_argument(
        "--oracle-mode",
        choices=ORACLE_MODES,
        default="correctness_plus_sufficiency_plus_grounding",
        help="Constraint bundle used for both answerers when constructing minimal subsets.",
    )
    parser.add_argument(
        "--oracle-min-sufficiency",
        type=float,
        default=0.8,
        help="Minimum sufficiency ratio used by modes that include sufficiency.",
    )
    parser.add_argument(
        "--oracle-min-temporal-iou",
        type=float,
        default=0.1,
        help="Minimum temporal IoU used by modes that include grounding.",
    )
    return parser.parse_args()


def average(values: list[float]) -> float:
    finite = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not finite:
        return 0.0
    return sum(finite) / len(finite)


def build_named_answerer(args: argparse.Namespace, prefix: str) -> tuple[str, Answerer, dict[str, str | None]]:
    name = getattr(args, prefix)
    label = getattr(args, f"{prefix}_label")
    model_dir = getattr(args, f"{prefix}_model_dir")
    model_name = getattr(args, f"{prefix}_model_name")
    device = getattr(args, f"{prefix}_device")
    answerer = build_answerer(
        name,
        model_dir,
        model_name=model_name,
        device=device,
    )
    metadata = {
        "name": name,
        "model_dir": model_dir,
        "model_name": model_name if name == "frozen_multimodal" else None,
        "device": device,
    }
    return label, answerer, metadata


def subset_prediction_metrics(
    example: QuestionExample,
    answerer: Answerer,
    full_prediction: ModelPrediction,
    subset: tuple[EvidenceItem, ...],
) -> tuple[dict, ModelPrediction]:
    prediction = answerer.predict(example, subset)
    metrics = {
        "predicted_index": prediction.predicted_index,
        "confidence": prediction.confidence,
        "subset_count": len(subset),
        "subset_cost": evidence_cost(subset),
        "subset_modality_counts": modality_counts(subset),
        "subset": serialize_evidence(subset),
    }
    if example.answer_index is not None:
        metrics["correct"] = accuracy(prediction.predicted_index, example.answer_index)
        metrics["full_pool_sufficiency"] = sufficiency(full_prediction, prediction, example.answer_index)
    if example.temporal_grounding is not None:
        metrics["subset_temporal_iou"] = max_temporal_iou_for_target_spans(
            subset,
            temporal_target_spans(example.temporal_grounding, example.metadata),
        )
    return metrics, prediction


def cross_subset_metrics(
    example: QuestionExample,
    answerer: Answerer,
    full_prediction: ModelPrediction,
    subset: tuple[EvidenceItem, ...],
) -> dict:
    prediction = answerer.predict(example, subset)
    metrics = {
        "predicted_index": prediction.predicted_index,
        "confidence": prediction.confidence,
    }
    if example.answer_index is not None:
        metrics["correct"] = accuracy(prediction.predicted_index, example.answer_index)
        metrics["full_pool_sufficiency"] = sufficiency(full_prediction, prediction, example.answer_index)
    return metrics


def pairwise_summary(
    subset_a: tuple[EvidenceItem, ...],
    subset_b: tuple[EvidenceItem, ...],
) -> dict:
    return {
        "evidence_jaccard": evidence_jaccard(subset_a, subset_b),
        "modality_agreement": modality_agreement(subset_a, subset_b),
        "temporal_agreement": temporal_interval_iou_for_items(subset_a, subset_b),
        "subset_count_gap": abs(len(subset_a) - len(subset_b)),
        "subset_cost_gap": abs(evidence_cost(subset_a) - evidence_cost(subset_b)),
    }


def main() -> None:
    args = parse_args()
    examples = load_normalized_examples(args.input_path)
    if args.limit is not None:
        examples = examples[: args.limit]

    label_a, answerer_a, answerer_a_metadata = build_named_answerer(args, "answerer_a")
    label_b, answerer_b, answerer_b_metadata = build_named_answerer(args, "answerer_b")
    if label_a == label_b:
        raise ValueError("Answerer labels must be distinct.")

    allocation = RetrievalAllocation(
        subtitle=args.subtitle_k,
        frame=args.frame_k,
        segment=args.segment_k,
    )
    oracle_config = OracleConfig.from_mode(
        args.oracle_mode,
        min_sufficiency=args.oracle_min_sufficiency,
        min_temporal_iou=args.oracle_min_temporal_iou,
    )
    retriever = FixedBudgetRetriever(
        build_named_retriever(
            args.retriever,
            visual_model_name=args.visual_model_name,
            visual_device=args.visual_device,
        )
    )
    oracle_a = MinimalEvidenceOracle(answerer_a, config=oracle_config)
    oracle_b = MinimalEvidenceOracle(answerer_b, config=oracle_config)

    per_example_records: list[dict] = []
    for example in examples:
        seed_evidence = retriever.retrieve(example, allocation)

        full_prediction_a = answerer_a.predict(example, example.evidence_pool)
        full_prediction_b = answerer_b.predict(example, example.evidence_pool)
        seed_prediction_a = answerer_a.predict(example, seed_evidence)
        seed_prediction_b = answerer_b.predict(example, seed_evidence)

        oracle_valid_a = oracle_a.seed_satisfies_constraints(example, seed_evidence)
        oracle_valid_b = oracle_b.seed_satisfies_constraints(example, seed_evidence)
        subset_a = oracle_a.minimal_subset(example, seed_evidence)
        subset_b = oracle_b.minimal_subset(example, seed_evidence)

        own_metrics_a, _ = subset_prediction_metrics(example, answerer_a, full_prediction_a, subset_a)
        own_metrics_b, _ = subset_prediction_metrics(example, answerer_b, full_prediction_b, subset_b)
        cross_metrics_a = cross_subset_metrics(example, answerer_a, full_prediction_a, subset_b)
        cross_metrics_b = cross_subset_metrics(example, answerer_b, full_prediction_b, subset_a)

        if example.answer_index is not None:
            own_metrics_a["transfer_sufficiency_gap"] = (
                own_metrics_a["full_pool_sufficiency"] - cross_metrics_a["full_pool_sufficiency"]
            )
            own_metrics_b["transfer_sufficiency_gap"] = (
                own_metrics_b["full_pool_sufficiency"] - cross_metrics_b["full_pool_sufficiency"]
            )
            own_metrics_a["transfer_accuracy_gap"] = own_metrics_a["correct"] - cross_metrics_a["correct"]
            own_metrics_b["transfer_accuracy_gap"] = own_metrics_b["correct"] - cross_metrics_b["correct"]

        own_metrics_a["transfer_confidence_gap"] = (
            own_metrics_a["confidence"] - cross_metrics_a["confidence"]
        )
        own_metrics_b["transfer_confidence_gap"] = (
            own_metrics_b["confidence"] - cross_metrics_b["confidence"]
        )

        per_example_records.append(
            {
                "example_id": example.example_id,
                "video_id": example.video_id,
                "answer_index": example.answer_index,
                "temporal_grounding": list(example.temporal_grounding) if example.temporal_grounding else None,
                "seed_evidence": serialize_evidence(seed_evidence),
                "seed_evidence_count": len(seed_evidence),
                "seed_evidence_cost": evidence_cost(seed_evidence),
                "seed_modality_counts": modality_counts(seed_evidence),
                "answerer_a": {
                    "label": label_a,
                    "metadata": answerer_a_metadata,
                    "oracle_valid": oracle_valid_a,
                    "seed_prediction": {
                        "predicted_index": seed_prediction_a.predicted_index,
                        "confidence": seed_prediction_a.confidence,
                        "option_scores": list(seed_prediction_a.option_scores),
                    },
                    "full_prediction": {
                        "predicted_index": full_prediction_a.predicted_index,
                        "confidence": full_prediction_a.confidence,
                        "option_scores": list(full_prediction_a.option_scores),
                    },
                    "own_subset": own_metrics_a,
                    "cross_subset": cross_metrics_a,
                },
                "answerer_b": {
                    "label": label_b,
                    "metadata": answerer_b_metadata,
                    "oracle_valid": oracle_valid_b,
                    "seed_prediction": {
                        "predicted_index": seed_prediction_b.predicted_index,
                        "confidence": seed_prediction_b.confidence,
                        "option_scores": list(seed_prediction_b.option_scores),
                    },
                    "full_prediction": {
                        "predicted_index": full_prediction_b.predicted_index,
                        "confidence": full_prediction_b.confidence,
                        "option_scores": list(full_prediction_b.option_scores),
                    },
                    "own_subset": own_metrics_b,
                    "cross_subset": cross_metrics_b,
                },
                "pair_metrics": pairwise_summary(subset_a, subset_b),
            }
        )

    def answerer_metrics(key: str) -> dict[str, float]:
        return {
            "oracle_valid_rate": average(
                [1.0 if record[key]["oracle_valid"] else 0.0 for record in per_example_records]
            ),
            "own_subset_accuracy": average(
                [record[key]["own_subset"]["correct"] for record in per_example_records if "correct" in record[key]["own_subset"]]
            ),
            "cross_subset_accuracy": average(
                [record[key]["cross_subset"]["correct"] for record in per_example_records if "correct" in record[key]["cross_subset"]]
            ),
            "own_subset_count": average([record[key]["own_subset"]["subset_count"] for record in per_example_records]),
            "own_subset_cost": average([record[key]["own_subset"]["subset_cost"] for record in per_example_records]),
            "own_subset_sufficiency": average(
                [
                    record[key]["own_subset"]["full_pool_sufficiency"]
                    for record in per_example_records
                    if "full_pool_sufficiency" in record[key]["own_subset"]
                ]
            ),
            "cross_subset_sufficiency": average(
                [
                    record[key]["cross_subset"]["full_pool_sufficiency"]
                    for record in per_example_records
                    if "full_pool_sufficiency" in record[key]["cross_subset"]
                ]
            ),
            "transfer_sufficiency_gap": average(
                [
                    record[key]["own_subset"]["transfer_sufficiency_gap"]
                    for record in per_example_records
                    if "transfer_sufficiency_gap" in record[key]["own_subset"]
                ]
            ),
            "transfer_accuracy_gap": average(
                [
                    record[key]["own_subset"]["transfer_accuracy_gap"]
                    for record in per_example_records
                    if "transfer_accuracy_gap" in record[key]["own_subset"]
                ]
            ),
            "transfer_confidence_gap": average(
                [
                    record[key]["own_subset"]["transfer_confidence_gap"]
                    for record in per_example_records
                    if "transfer_confidence_gap" in record[key]["own_subset"]
                ]
            ),
            "own_subset_temporal_iou": average(
                [
                    record[key]["own_subset"]["subset_temporal_iou"]
                    for record in per_example_records
                    if "subset_temporal_iou" in record[key]["own_subset"]
                ]
            ),
        }

    summary = {
        "input_path": args.input_path,
        "num_examples": len(per_example_records),
        "answerers": {
            "answerer_a": {
                "label": label_a,
                **answerer_a_metadata,
            },
            "answerer_b": {
                "label": label_b,
                **answerer_b_metadata,
            },
        },
        "retriever": args.retriever,
        "visual_model_name": args.visual_model_name if args.retriever == "hybrid_clip" else None,
        "allocation": allocation.to_dict(),
        "oracle_config": oracle_config.to_dict(),
        "metrics": {
            "answerer_a": answerer_metrics("answerer_a"),
            "answerer_b": answerer_metrics("answerer_b"),
            "pair": {
                "evidence_jaccard": average([record["pair_metrics"]["evidence_jaccard"] for record in per_example_records]),
                "modality_agreement": average([record["pair_metrics"]["modality_agreement"] for record in per_example_records]),
                "temporal_agreement": average([record["pair_metrics"]["temporal_agreement"] for record in per_example_records]),
                "subset_count_gap": average([record["pair_metrics"]["subset_count_gap"] for record in per_example_records]),
                "subset_cost_gap": average([record["pair_metrics"]["subset_cost_gap"] for record in per_example_records]),
                "both_oracle_valid_rate": average(
                    [
                        1.0
                        if record["answerer_a"]["oracle_valid"] and record["answerer_b"]["oracle_valid"]
                        else 0.0
                        for record in per_example_records
                    ]
                ),
            },
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    examples_path = output_dir / "per_example.jsonl"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_jsonl(per_example_records, examples_path)

    print(f"Wrote model-relative study summary to {summary_path}")
    print(f"Wrote {len(per_example_records)} per-example records to {examples_path}")


if __name__ == "__main__":
    main()
