import argparse
import json
from pathlib import Path

from adaptive_evidence_vqa.data.base import save_jsonl
from adaptive_evidence_vqa.data.evidence_records import serialize_evidence
from adaptive_evidence_vqa.data.normalized import load_normalized_examples
from adaptive_evidence_vqa.eval.metrics import (
    accuracy,
    comprehensiveness,
    evidence_cost,
    max_temporal_iou_for_target_spans,
    sufficiency,
    temporal_target_spans,
)
from adaptive_evidence_vqa.models.answerer import Answerer, build_answerer
from adaptive_evidence_vqa.models.oracle import ORACLE_MODES, MinimalEvidenceOracle, OracleConfig
from adaptive_evidence_vqa.retrieval.base import (
    FixedBudgetRetriever,
    RetrievalAllocation,
    build_named_retriever,
)
from adaptive_evidence_vqa.schemas import EvidenceItem, QuestionExample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fixed-allocation baseline on candidate pools.")
    parser.add_argument("--input-path", required=True, help="Path to candidate-pool JSONL.")
    parser.add_argument("--summary-output", required=True, help="Path to summary JSON output.")
    parser.add_argument("--predictions-output", help="Optional JSONL output with per-example predictions.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of examples.")
    parser.add_argument(
        "--answerer",
        choices=("lexical", "linear", "frozen_multimodal"),
        default="lexical",
        help="Answerer used to score the retrieved evidence.",
    )
    parser.add_argument(
        "--answerer-model-dir",
        help="Model directory for the linear answerer.",
    )
    parser.add_argument(
        "--answerer-model-name",
        default="openai/clip-vit-base-patch32",
        help="Model name used by the frozen multimodal answerer.",
    )
    parser.add_argument(
        "--answerer-device",
        help="Optional device override for the frozen multimodal answerer, e.g. cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--retriever",
        choices=("lexical", "bm25", "hybrid_clip"),
        default="bm25",
        help="Retriever used to rank candidate evidence.",
    )
    parser.add_argument(
        "--visual-model-name",
        default="openai/clip-vit-base-patch32",
        help="Visual-text encoder used when `--retriever hybrid_clip` is selected.",
    )
    parser.add_argument(
        "--visual-device",
        help="Optional device override for the visual-text retriever, e.g. cpu or mps.",
    )
    parser.add_argument("--subtitle-k", type=int, default=2, help="Number of subtitle items to retrieve.")
    parser.add_argument("--frame-k", type=int, default=2, help="Number of frame items to retrieve.")
    parser.add_argument("--segment-k", type=int, default=2, help="Number of segment items to retrieve.")
    parser.add_argument(
        "--oracle-mode",
        choices=ORACLE_MODES,
        default="correctness_only",
        help="Constraint bundle used when pruning minimal-evidence subsets.",
    )
    parser.add_argument(
        "--oracle-min-sufficiency",
        type=float,
        default=0.8,
        help="Minimum sufficiency ratio required when pruning oracle evidence.",
    )
    parser.add_argument(
        "--oracle-min-temporal-iou",
        type=float,
        default=0.1,
        help="Minimum temporal IoU required when pruning oracle evidence on grounded examples.",
    )
    return parser.parse_args()


def remaining_evidence(
    example: QuestionExample,
    selected_items: tuple[EvidenceItem, ...],
) -> tuple[EvidenceItem, ...]:
    selected_ids = {item.evidence_id for item in selected_items}
    return tuple(item for item in example.evidence_pool if item.evidence_id not in selected_ids)


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def build_example_result(
    example: QuestionExample,
    selected_evidence: tuple[EvidenceItem, ...],
    answerer: Answerer,
    oracle: MinimalEvidenceOracle,
) -> dict:
    prediction = answerer.predict(example, selected_evidence)
    gold_index = example.answer_index
    remaining = remaining_evidence(example, selected_evidence)
    oracle_valid = oracle.seed_satisfies_constraints(example, selected_evidence)
    oracle_subset = oracle.minimal_subset(example, selected_evidence)

    result = {
        "example_id": example.example_id,
        "video_id": example.video_id,
        "predicted_index": prediction.predicted_index,
        "prediction_confidence": prediction.confidence,
        "selected_evidence": serialize_evidence(selected_evidence),
        "selected_evidence_count": len(selected_evidence),
        "selected_evidence_cost": evidence_cost(selected_evidence),
        "oracle_valid": oracle_valid,
        "oracle_subset": serialize_evidence(oracle_subset),
        "oracle_subset_count": len(oracle_subset),
        "oracle_subset_cost": evidence_cost(oracle_subset),
    }

    if gold_index is None:
        return result

    full_prediction = answerer.predict(example, example.evidence_pool)
    reduced_prediction = answerer.predict(example, remaining)
    result.update(
        {
            "gold_index": gold_index,
            "correct": accuracy(prediction.predicted_index, gold_index),
            "full_pool_confidence": full_prediction.confidence,
            "sufficiency": sufficiency(full_prediction, prediction, gold_index),
            "comprehensiveness": comprehensiveness(full_prediction, reduced_prediction, gold_index),
        }
    )

    if example.temporal_grounding is not None:
        target_spans = temporal_target_spans(example.temporal_grounding, example.metadata)
        result["selected_temporal_iou"] = max_temporal_iou_for_target_spans(
            selected_evidence,
            target_spans,
        )
        result["oracle_temporal_iou"] = max_temporal_iou_for_target_spans(
            oracle_subset,
            target_spans,
        )

    return result


def main() -> None:
    args = parse_args()
    examples = load_normalized_examples(args.input_path)
    if args.limit is not None:
        examples = examples[: args.limit]

    allocation = RetrievalAllocation(
        subtitle=args.subtitle_k,
        frame=args.frame_k,
        segment=args.segment_k,
    )
    answerer = build_answerer(
        args.answerer,
        args.answerer_model_dir,
        model_name=args.answerer_model_name,
        device=args.answerer_device,
    )
    oracle_config = OracleConfig.from_mode(
        args.oracle_mode,
        min_sufficiency=args.oracle_min_sufficiency,
        min_temporal_iou=args.oracle_min_temporal_iou,
    )
    oracle = MinimalEvidenceOracle(answerer, config=oracle_config)
    retriever = FixedBudgetRetriever(
        build_named_retriever(
            args.retriever,
            visual_model_name=args.visual_model_name,
            visual_device=args.visual_device,
        )
    )

    results = [
        build_example_result(
            example,
            retriever.retrieve(example, allocation),
            answerer,
            oracle,
        )
        for example in examples
    ]

    summary = {
        "input_path": str(args.input_path),
        "num_examples": len(results),
        "answerer": args.answerer,
        "answerer_model_name": args.answerer_model_name if args.answerer == "frozen_multimodal" else None,
        "retriever": args.retriever,
        "visual_model_name": args.visual_model_name if args.retriever == "hybrid_clip" else None,
        "allocation": allocation.to_dict(),
        "oracle_config": oracle_config.to_dict(),
        "metrics": {
            "accuracy": average([result["correct"] for result in results if "correct" in result]),
            "selected_evidence_cost": average(
                [result["selected_evidence_cost"] for result in results]
            ),
            "oracle_subset_cost": average([result["oracle_subset_cost"] for result in results]),
            "selected_evidence_count": average(
                [result["selected_evidence_count"] for result in results]
            ),
            "oracle_valid_rate": average([1.0 if result["oracle_valid"] else 0.0 for result in results]),
            "oracle_subset_count": average([result["oracle_subset_count"] for result in results]),
            "sufficiency": average([result["sufficiency"] for result in results if "sufficiency" in result]),
            "comprehensiveness": average(
                [result["comprehensiveness"] for result in results if "comprehensiveness" in result]
            ),
            "selected_temporal_iou": average(
                [result["selected_temporal_iou"] for result in results if "selected_temporal_iou" in result]
            ),
            "oracle_temporal_iou": average(
                [result["oracle_temporal_iou"] for result in results if "oracle_temporal_iou" in result]
            ),
        },
    }

    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.predictions_output:
        save_jsonl(results, args.predictions_output)

    print(f"Wrote summary to {summary_path}")
    if args.predictions_output:
        print(f"Wrote {len(results)} per-example records to {args.predictions_output}")


if __name__ == "__main__":
    main()
