import argparse
import json
from pathlib import Path

from adaptive_evidence_vqa.data.base import load_jsonl
from adaptive_evidence_vqa.data.evidence_records import parse_evidence_record
from adaptive_evidence_vqa.models.answerer import build_answerer
from adaptive_evidence_vqa.models.policy import (
    PolicyTrainingState,
    SequentialPolicyConfig,
    TrainableSequentialPolicy,
)
from adaptive_evidence_vqa.schemas import AnswerOption, EvidenceItem, QuestionExample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a sequential evidence-acquisition policy from oracle traces.")
    parser.add_argument("--train-traces-path", required=True, help="Path to training oracle-trace JSONL.")
    parser.add_argument("--validation-traces-path", help="Optional path to validation oracle-trace JSONL.")
    parser.add_argument("--model-dir", required=True, help="Directory to store the trained policy.")
    parser.add_argument("--answerer", choices=("lexical", "linear", "frozen_multimodal"), default="lexical", help="Answerer used to featurize policy states.")
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
    parser.add_argument("--train-limit", type=int, default=None, help="Optional limit on training trace records.")
    parser.add_argument("--validation-limit", type=int, default=None, help="Optional limit on validation trace records.")
    parser.add_argument("--text-feature-dim", type=int, default=4096, help="Hashed text feature dimension.")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.2, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization strength.")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument(
        "--min-items-before-stop",
        type=int,
        default=1,
        help="Minimum number of acquired items required before the policy may emit stop.",
    )
    return parser.parse_args()

def parse_trace_record(record: dict) -> tuple[QuestionExample, tuple[EvidenceItem, ...], list[dict]]:
    temporal_grounding = record.get("temporal_grounding")
    example = QuestionExample(
        example_id=record["example_id"],
        video_id=record["video_id"],
        question=record["question"],
        options=tuple(
            AnswerOption(index=index, text=text)
            for index, text in enumerate(record["options"])
        ),
        answer_index=record.get("answer_index"),
        temporal_grounding=tuple(temporal_grounding) if temporal_grounding is not None else None,
    )
    seed_evidence = tuple(parse_evidence_record(item) for item in record.get("seed_evidence", []))
    return example, seed_evidence, list(record.get("trace", []))


def build_states_from_record(record: dict) -> list[PolicyTrainingState]:
    example, seed_evidence, trace_steps = parse_trace_record(record)
    by_id = {item.evidence_id: item for item in seed_evidence}
    ordered_by_modality = {
        "subtitle": tuple(item for item in seed_evidence if item.modality.value == "subtitle"),
        "frame": tuple(item for item in seed_evidence if item.modality.value == "frame"),
        "segment": tuple(item for item in seed_evidence if item.modality.value == "segment"),
    }

    acquired: list[EvidenceItem] = []
    used_ids: set[str] = set()
    states: list[PolicyTrainingState] = []
    max_steps = max(len(trace_steps), 1)

    for step_index, step in enumerate(trace_steps):
        states.append(
            PolicyTrainingState(
                example=example,
                acquired=tuple(acquired),
                remaining_subtitles=tuple(
                    item for item in ordered_by_modality["subtitle"] if item.evidence_id not in used_ids
                ),
                remaining_frames=tuple(
                    item for item in ordered_by_modality["frame"] if item.evidence_id not in used_ids
                ),
                remaining_segments=tuple(
                    item for item in ordered_by_modality["segment"] if item.evidence_id not in used_ids
                ),
                gold_action=step["action"],
                step_index=step_index,
                max_steps=max_steps,
            )
        )

        selected_id = step.get("selected_evidence_id")
        if selected_id is not None:
            used_ids.add(selected_id)
            acquired.append(by_id[selected_id])

    return states


def load_policy_states(path: str, limit: int | None = None) -> list[PolicyTrainingState]:
    records = load_jsonl(path)
    if limit is not None:
        records = records[:limit]

    states: list[PolicyTrainingState] = []
    for record in records:
        if record.get("oracle_valid") is False:
            continue
        states.extend(build_states_from_record(record))
    if not states:
        raise ValueError(f"No policy states found in {path}.")
    return states


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def state_metrics(
    model: TrainableSequentialPolicy,
    states: list[PolicyTrainingState],
) -> dict[str, float]:
    correct = []
    stop_rate = []
    for state in states:
        action, probabilities = model.predict_action(
            example=state.example,
            acquired=state.acquired,
            remaining_by_modality=state.remaining_by_modality(),
            step_index=state.step_index,
            max_steps=state.max_steps,
        )
        correct.append(1.0 if action == state.gold_action else 0.0)
        stop_rate.append(probabilities.get("stop", 0.0))
    return {
        "action_accuracy": average(correct),
        "mean_stop_probability": average(stop_rate),
    }


def main() -> None:
    args = parse_args()
    answerer = build_answerer(
        args.answerer,
        args.answerer_model_dir,
        model_name=args.answerer_model_name,
        device=args.answerer_device,
    )
    train_states = load_policy_states(args.train_traces_path, limit=args.train_limit)
    validation_states = (
        load_policy_states(args.validation_traces_path, limit=args.validation_limit)
        if args.validation_traces_path
        else None
    )

    config = SequentialPolicyConfig(
        text_feature_dim=args.text_feature_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        min_items_before_stop=args.min_items_before_stop,
    )
    model = TrainableSequentialPolicy.fit(
        train_states=train_states,
        validation_states=validation_states,
        answerer=answerer,
        config=config,
    )
    model.save(args.model_dir)

    train_metrics = state_metrics(model, train_states)
    validation_metrics = state_metrics(model, validation_states) if validation_states else {}
    summary = {
        "model_dir": str(args.model_dir),
        "train_traces_path": args.train_traces_path,
        "validation_traces_path": args.validation_traces_path,
        "answerer": args.answerer,
        "answerer_model_name": args.answerer_model_name if args.answerer == "frozen_multimodal" else None,
        "num_train_states": len(train_states),
        "num_validation_states": len(validation_states or []),
        "config": {
            "text_feature_dim": args.text_feature_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
            "min_items_before_stop": args.min_items_before_stop,
        },
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "history": model.history,
    }

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote trained policy to {model_dir}")
    print(json.dumps({"train_metrics": train_metrics, "validation_metrics": validation_metrics}, indent=2))


if __name__ == "__main__":
    main()
