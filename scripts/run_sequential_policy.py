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
from adaptive_evidence_vqa.models.policy import SequentialPolicy, build_policy
from adaptive_evidence_vqa.retrieval.base import (
    CandidatePoolBuilder,
    RetrievalAllocation,
    build_named_retriever,
)
from adaptive_evidence_vqa.schemas import EvidenceItem, QuestionExample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sequential evidence-acquisition policy on candidate pools.")
    parser.add_argument("--input-path", required=True, help="Path to candidate-pool JSONL.")
    parser.add_argument("--summary-output", required=True, help="Path to summary JSON output.")
    parser.add_argument("--predictions-output", help="Optional JSONL output with per-example predictions.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of examples.")
    parser.add_argument("--answerer", choices=("lexical", "linear", "frozen_multimodal"), default="lexical", help="Answerer used to score acquired evidence.")
    parser.add_argument("--answerer-model-dir", help="Model directory for the linear answerer.")
    parser.add_argument(
        "--answerer-model-name",
        default="openai/clip-vit-base-patch32",
        help="Model name used by the frozen multimodal answerer.",
    )
    parser.add_argument(
        "--answerer-device",
        help="Optional device override for the frozen multimodal answerer, e.g. cpu, cuda, or mps.",
    )
    parser.add_argument("--policy", choices=("keyword", "linear"), default="keyword", help="Sequential policy to evaluate.")
    parser.add_argument("--policy-model-dir", help="Model directory for the trainable sequential policy.")
    parser.add_argument("--max-items", type=int, default=6, help="Maximum number of acquisitions before forced stop.")
    parser.add_argument(
        "--retriever",
        choices=("lexical", "bm25", "hybrid_clip"),
        default="bm25",
        help="Retriever used to construct the candidate pool seen by the sequential policy.",
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
    return parser.parse_args()


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def remaining_evidence(
    example: QuestionExample,
    selected_items: tuple[EvidenceItem, ...],
) -> tuple[EvidenceItem, ...]:
    selected_ids = {item.evidence_id for item in selected_items}
    return tuple(item for item in example.evidence_pool if item.evidence_id not in selected_ids)


def build_example_result(
    example: QuestionExample,
    answerer: Answerer,
    policy: SequentialPolicy,
    pool_builder: CandidatePoolBuilder,
    allocation: RetrievalAllocation,
    max_items: int,
) -> dict:
    candidate_pool = pool_builder.build(example, top_k_per_modality=allocation)
    trace = policy.run(example, candidate_pool, max_items=max_items)
    selected_evidence = tuple(
        step.selected_item
        for step in trace.steps
        if step.selected_item is not None
    )
    prediction = trace.final_prediction

    result = {
        "example_id": example.example_id,
        "video_id": example.video_id,
        "predicted_index": prediction.predicted_index,
        "prediction_confidence": prediction.confidence,
        "trace": [
            {
                "step_index": step.step_index,
                "action": step.action,
                "selected_evidence_id": step.selected_item.evidence_id if step.selected_item else None,
                "confidence_after_step": step.confidence_after_step,
            }
            for step in trace.steps
        ],
        "selected_evidence": serialize_evidence(selected_evidence),
        "selected_evidence_count": len(selected_evidence),
        "selected_evidence_cost": evidence_cost(selected_evidence),
    }

    if example.answer_index is None:
        return result

    full_prediction = answerer.predict(example, example.evidence_pool)
    reduced_prediction = answerer.predict(example, remaining_evidence(example, selected_evidence))
    result.update(
        {
            "gold_index": example.answer_index,
            "correct": accuracy(prediction.predicted_index, example.answer_index),
            "full_pool_confidence": full_prediction.confidence,
            "sufficiency": sufficiency(full_prediction, prediction, example.answer_index),
            "comprehensiveness": comprehensiveness(full_prediction, reduced_prediction, example.answer_index),
        }
    )

    if example.temporal_grounding is not None:
        result["selected_temporal_iou"] = max_temporal_iou_for_target_spans(
            selected_evidence,
            temporal_target_spans(example.temporal_grounding, example.metadata),
        )

    return result


def main() -> None:
    args = parse_args()
    answerer = build_answerer(
        args.answerer,
        args.answerer_model_dir,
        model_name=args.answerer_model_name,
        device=args.answerer_device,
    )
    policy = build_policy(args.policy, answerer=answerer, model_dir=args.policy_model_dir)
    allocation = RetrievalAllocation(
        subtitle=args.subtitle_k,
        frame=args.frame_k,
        segment=args.segment_k,
    )
    pool_builder = CandidatePoolBuilder(
        build_named_retriever(
            args.retriever,
            visual_model_name=args.visual_model_name,
            visual_device=args.visual_device,
        )
    )
    examples = load_normalized_examples(args.input_path)
    if args.limit is not None:
        examples = examples[: args.limit]

    results = [
        build_example_result(
            example=example,
            answerer=answerer,
            policy=policy,
            pool_builder=pool_builder,
            allocation=allocation,
            max_items=args.max_items,
        )
        for example in examples
    ]

    summary = {
        "input_path": args.input_path,
        "num_examples": len(results),
        "answerer": args.answerer,
        "answerer_model_name": args.answerer_model_name if args.answerer == "frozen_multimodal" else None,
        "policy": args.policy,
        "policy_model_dir": args.policy_model_dir if args.policy == "linear" else None,
        "retriever": args.retriever,
        "visual_model_name": args.visual_model_name if args.retriever == "hybrid_clip" else None,
        "allocation": allocation.to_dict(),
        "max_items": args.max_items,
        "metrics": {
            "accuracy": average([result["correct"] for result in results if "correct" in result]),
            "selected_evidence_cost": average([result["selected_evidence_cost"] for result in results]),
            "selected_evidence_count": average([result["selected_evidence_count"] for result in results]),
            "sufficiency": average([result["sufficiency"] for result in results if "sufficiency" in result]),
            "comprehensiveness": average(
                [result["comprehensiveness"] for result in results if "comprehensiveness" in result]
            ),
            "selected_temporal_iou": average(
                [result["selected_temporal_iou"] for result in results if "selected_temporal_iou" in result]
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
