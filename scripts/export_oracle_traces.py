import argparse
import json

from adaptive_evidence_vqa.data.base import save_jsonl
from adaptive_evidence_vqa.data.evidence_records import serialize_evidence
from adaptive_evidence_vqa.data.normalized import load_normalized_examples
from adaptive_evidence_vqa.eval.metrics import modality_counts
from adaptive_evidence_vqa.models.answerer import build_answerer
from adaptive_evidence_vqa.models.oracle import ORACLE_MODES, MinimalEvidenceOracle, OracleConfig
from adaptive_evidence_vqa.retrieval.base import FixedBudgetRetriever, RetrievalAllocation, build_named_retriever
from adaptive_evidence_vqa.schemas import AcquisitionTrace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export oracle acquisition traces from retrieved candidate pools.")
    parser.add_argument("--input-path", required=True, help="Path to candidate-pool JSONL.")
    parser.add_argument("--output-path", required=True, help="Path to JSONL output.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of examples.")
    parser.add_argument(
        "--answerer",
        choices=("lexical", "linear", "frozen_multimodal"),
        default="lexical",
        help="Answerer used inside the oracle reduction loop.",
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
        help="Retriever used to rank candidate evidence before oracle reduction.",
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
    parser.add_argument(
        "--include-invalid-traces",
        action="store_true",
        help="Include traces whose retrieved seed evidence does not satisfy the oracle constraints.",
    )
    return parser.parse_args()


def serialize_trace(trace: AcquisitionTrace) -> list[dict]:
    return [
        {
            "step_index": step.step_index,
            "action": step.action,
            "selected_evidence_id": step.selected_item.evidence_id if step.selected_item else None,
            "confidence_after_step": step.confidence_after_step,
        }
        for step in trace.steps
    ]

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

    records = []
    skipped_invalid = 0
    for example in examples:
        seed_evidence = retriever.retrieve(example, allocation)
        seed_prediction = answerer.predict(example, seed_evidence)
        oracle_valid = oracle.seed_satisfies_constraints(example, seed_evidence)
        if not oracle_valid and not args.include_invalid_traces:
            skipped_invalid += 1
            continue

        trace = oracle.acquisition_trace(example, seed_evidence)
        oracle_subset = tuple(
            step.selected_item
            for step in trace.steps
            if step.selected_item is not None
        )
        records.append(
            {
                "example_id": example.example_id,
                "video_id": example.video_id,
                "question": example.question,
                "options": [option.text for option in example.options],
                "answer_index": example.answer_index,
                "temporal_grounding": list(example.temporal_grounding) if example.temporal_grounding else None,
                "answerer": args.answerer,
                "answerer_model_name": args.answerer_model_name if args.answerer == "frozen_multimodal" else None,
                "retriever": args.retriever,
                "visual_model_name": args.visual_model_name if args.retriever == "hybrid_clip" else None,
                "allocation": allocation.to_dict(),
                "oracle_config": oracle_config.to_dict(),
                "oracle_valid": oracle_valid,
                "seed_evidence": serialize_evidence(seed_evidence),
                "seed_modality_counts": modality_counts(seed_evidence),
                "seed_prediction": {
                    "predicted_index": seed_prediction.predicted_index,
                    "confidence": seed_prediction.confidence,
                    "option_scores": list(seed_prediction.option_scores),
                },
                "oracle_subset": serialize_evidence(oracle_subset),
                "oracle_modality_counts": modality_counts(oracle_subset),
                "trace": serialize_trace(trace),
                "final_prediction": {
                    "predicted_index": trace.final_prediction.predicted_index,
                    "confidence": trace.final_prediction.confidence,
                    "option_scores": list(trace.final_prediction.option_scores),
                },
            }
        )

    save_jsonl(records, args.output_path)
    print(f"Wrote {len(records)} oracle trace records to {args.output_path}")
    print(
        json.dumps(
            {
                "num_input_examples": len(examples),
                "num_exported_examples": len(records),
                "num_skipped_invalid_examples": skipped_invalid,
                "answerer": args.answerer,
                "answerer_model_name": args.answerer_model_name if args.answerer == "frozen_multimodal" else None,
                "retriever": args.retriever,
                "visual_model_name": args.visual_model_name if args.retriever == "hybrid_clip" else None,
                "allocation": allocation.to_dict(),
                "oracle_config": oracle_config.to_dict(),
                "include_invalid_traces": args.include_invalid_traces,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
