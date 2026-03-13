import argparse
import json
from pathlib import Path

from adaptive_evidence_vqa.data.base import save_jsonl
from adaptive_evidence_vqa.data.normalized import load_normalized_examples
from adaptive_evidence_vqa.eval.metrics import accuracy, evidence_cost, max_temporal_iou_for_items
from adaptive_evidence_vqa.models.answerer import (
    ExampleWithEvidence,
    LinearAnswererConfig,
    TrainableLinearAnswerer,
)
from adaptive_evidence_vqa.retrieval.base import FixedBudgetRetriever, RetrievalAllocation, build_named_retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fixed-budget evidence-conditioned answerer.")
    parser.add_argument("--train-path", required=True, help="Path to training candidate-pool JSONL.")
    parser.add_argument("--validation-path", help="Optional path to validation candidate-pool JSONL.")
    parser.add_argument("--model-dir", required=True, help="Directory to store the trained model.")
    parser.add_argument("--train-limit", type=int, default=None, help="Optional limit on training examples.")
    parser.add_argument("--validation-limit", type=int, default=None, help="Optional limit on validation examples.")
    parser.add_argument(
        "--retriever",
        choices=("lexical", "bm25", "hybrid_clip"),
        default="bm25",
        help="Retriever used to construct the fixed evidence budget.",
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
    parser.add_argument("--text-feature-dim", type=int, default=4096, help="Hashed text feature dimension.")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.2, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization strength.")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument("--train-predictions-output", help="Optional JSONL path for train predictions.")
    parser.add_argument("--validation-predictions-output", help="Optional JSONL path for validation predictions.")
    return parser.parse_args()


def load_dataset(
    path: str,
    retriever_name: str,
    allocation: RetrievalAllocation,
    visual_model_name: str,
    visual_device: str | None,
    limit: int | None = None,
) -> list[ExampleWithEvidence]:
    examples = load_normalized_examples(path)
    if limit is not None:
        examples = examples[:limit]

    retriever = FixedBudgetRetriever(
        build_named_retriever(
            retriever_name,
            visual_model_name=visual_model_name,
            visual_device=visual_device,
        )
    )
    dataset = [
        (example, retriever.retrieve(example, allocation))
        for example in examples
        if example.answer_index is not None
    ]
    if not dataset:
        raise ValueError(f"No labeled examples found in {path}.")
    return dataset


def serialize_predictions(
    model: TrainableLinearAnswerer,
    dataset: list[ExampleWithEvidence],
) -> list[dict]:
    records = []
    for example, evidence in dataset:
        prediction = model.predict(example, evidence)
        record = {
            "example_id": example.example_id,
            "video_id": example.video_id,
            "gold_index": example.answer_index,
            "predicted_index": prediction.predicted_index,
            "confidence": prediction.confidence,
            "correct": accuracy(prediction.predicted_index, example.answer_index),
            "selected_evidence_cost": evidence_cost(evidence),
            "option_scores": list(prediction.option_scores),
        }
        if example.temporal_grounding is not None:
            record["selected_temporal_iou"] = max_temporal_iou_for_items(
                evidence,
                example.temporal_grounding[0],
                example.temporal_grounding[1],
            )
        records.append(record)
    return records


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def dataset_metrics(
    model: TrainableLinearAnswerer,
    dataset: list[ExampleWithEvidence],
) -> dict[str, float]:
    predictions = serialize_predictions(model, dataset)
    return {
        "accuracy": average([record["correct"] for record in predictions]),
        "confidence": average([record["confidence"] for record in predictions]),
        "selected_evidence_cost": average([record["selected_evidence_cost"] for record in predictions]),
        "selected_temporal_iou": average(
            [record["selected_temporal_iou"] for record in predictions if "selected_temporal_iou" in record]
        ),
    }


def main() -> None:
    args = parse_args()
    allocation = RetrievalAllocation(
        subtitle=args.subtitle_k,
        frame=args.frame_k,
        segment=args.segment_k,
    )
    train_dataset = load_dataset(
        args.train_path,
        retriever_name=args.retriever,
        allocation=allocation,
        visual_model_name=args.visual_model_name,
        visual_device=args.visual_device,
        limit=args.train_limit,
    )
    validation_dataset = (
        load_dataset(
            args.validation_path,
            retriever_name=args.retriever,
            allocation=allocation,
            visual_model_name=args.visual_model_name,
            visual_device=args.visual_device,
            limit=args.validation_limit,
        )
        if args.validation_path
        else None
    )

    config = LinearAnswererConfig(
        text_feature_dim=args.text_feature_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
    )
    model = TrainableLinearAnswerer.fit(
        train_examples=train_dataset,
        validation_examples=validation_dataset,
        config=config,
    )
    model.save(args.model_dir)

    train_metrics = dataset_metrics(model, train_dataset)
    validation_metrics = dataset_metrics(model, validation_dataset) if validation_dataset else {}
    summary = {
        "model_dir": str(args.model_dir),
        "train_path": args.train_path,
        "validation_path": args.validation_path,
        "retriever": args.retriever,
        "visual_model_name": args.visual_model_name if args.retriever == "hybrid_clip" else None,
        "allocation": allocation.to_dict(),
        "config": {
            "text_feature_dim": args.text_feature_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
        },
        "num_train_examples": len(train_dataset),
        "num_validation_examples": len(validation_dataset or []),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "history": model.history,
    }

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.train_predictions_output:
        save_jsonl(serialize_predictions(model, train_dataset), args.train_predictions_output)
    if args.validation_predictions_output and validation_dataset:
        save_jsonl(serialize_predictions(model, validation_dataset), args.validation_predictions_output)

    print(f"Wrote trained model to {model_dir}")
    print(json.dumps({"train_metrics": train_metrics, "validation_metrics": validation_metrics}, indent=2))


if __name__ == "__main__":
    main()
