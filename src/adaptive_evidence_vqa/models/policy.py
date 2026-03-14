import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from adaptive_evidence_vqa.models.answerer import Answerer
from adaptive_evidence_vqa.schemas import (
    AcquisitionStep,
    AcquisitionTrace,
    EvidenceItem,
    ModelPrediction,
    QuestionExample,
)
from adaptive_evidence_vqa.utils import normalize_text

KEYWORD_MODALITY_MAP = {
    "subtitle": {"who", "say", "said", "tell", "told", "conversation", "ask"},
    "frame": {"color", "wearing", "look", "holding", "room", "object", "see"},
    "segment": {"before", "after", "happen", "happens", "doing", "does", "leave"},
}

POLICY_ACTIONS = (
    "acquire_subtitle",
    "acquire_frame",
    "acquire_segment",
    "stop",
)
ACTION_TO_INDEX = {action: index for index, action in enumerate(POLICY_ACTIONS)}
INDEX_TO_ACTION = {index: action for action, index in ACTION_TO_INDEX.items()}
ACTION_TO_MODALITY = {
    "acquire_subtitle": "subtitle",
    "acquire_frame": "frame",
    "acquire_segment": "segment",
}
MODALITY_TO_ACTION = {modality: action for action, modality in ACTION_TO_MODALITY.items()}


def available_actions(
    remaining_by_modality: dict[str, tuple[EvidenceItem, ...]],
    *,
    acquired_count: int = 0,
    min_items_before_stop: int = 0,
) -> tuple[str, ...]:
    actions: list[str] = []
    for modality in ("subtitle", "frame", "segment"):
        if remaining_by_modality.get(modality):
            actions.append(MODALITY_TO_ACTION[modality])
    if acquired_count >= min_items_before_stop or not actions:
        actions.append("stop")
    return tuple(actions)


def action_mask(
    remaining_by_modality: dict[str, tuple[EvidenceItem, ...]],
    *,
    acquired_count: int = 0,
    min_items_before_stop: int = 0,
) -> np.ndarray:
    mask = np.zeros(len(POLICY_ACTIONS), dtype=np.float32)
    for action in available_actions(
        remaining_by_modality,
        acquired_count=acquired_count,
        min_items_before_stop=min_items_before_stop,
    ):
        mask[ACTION_TO_INDEX[action]] = 1.0
    return mask


def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if logits.size == 0:
        return np.zeros(0, dtype=np.float32)
    masked_logits = np.where(mask > 0, logits, -1e9)
    shifted = masked_logits - np.max(masked_logits)
    exps = np.exp(shifted) * mask
    total = float(np.sum(exps))
    if total <= 0:
        return mask / max(float(np.sum(mask)), 1.0)
    return exps / total


@dataclass(slots=True)
class PolicyTrainingState:
    example: QuestionExample
    acquired: tuple[EvidenceItem, ...]
    remaining_subtitles: tuple[EvidenceItem, ...] = ()
    remaining_frames: tuple[EvidenceItem, ...] = ()
    remaining_segments: tuple[EvidenceItem, ...] = ()
    gold_action: str = "stop"
    step_index: int = 0
    max_steps: int = 6

    def remaining_by_modality(self) -> dict[str, tuple[EvidenceItem, ...]]:
        return {
            "subtitle": self.remaining_subtitles,
            "frame": self.remaining_frames,
            "segment": self.remaining_segments,
        }


@dataclass(slots=True)
class SequentialPolicyConfig:
    text_feature_dim: int = 4096
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.2
    weight_decay: float = 1e-4
    patience: int = 4
    seed: int = 13
    min_items_before_stop: int = 1


@dataclass(slots=True)
class HashedPolicyFeatureExtractor:
    text_feature_dim: int = 4096
    numeric_feature_names: tuple[str, ...] = field(
        default_factory=lambda: (
            "step_fraction",
            "acquired_count",
            "acquired_cost",
            "current_confidence",
            "current_max_score",
            "acquired_subtitle_count",
            "acquired_frame_count",
            "acquired_segment_count",
            "remaining_subtitle_count",
            "remaining_frame_count",
            "remaining_segment_count",
            "remaining_subtitle_top_score",
            "remaining_frame_top_score",
            "remaining_segment_top_score",
        )
    )

    @property
    def total_dim(self) -> int:
        return self.text_feature_dim + len(self.numeric_feature_names)

    def _hash_index(self, key: str) -> int:
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") % self.text_feature_dim

    def transform(
        self,
        example: QuestionExample,
        acquired: tuple[EvidenceItem, ...],
        remaining_by_modality: dict[str, tuple[EvidenceItem, ...]],
        current_prediction: ModelPrediction,
        step_index: int,
        max_steps: int,
    ) -> np.ndarray:
        vector = np.zeros(self.total_dim, dtype=np.float32)
        query_text = " ".join([example.question] + [option.text for option in example.options])
        query_tokens = tuple(token for token in normalize_text(query_text).split() if token)

        for token in set(query_tokens):
            vector[self._hash_index(f"query:{token}")] += 1.0

        acquired_counts = {"subtitle": 0.0, "frame": 0.0, "segment": 0.0}
        acquired_cost = 0.0
        for item in acquired:
            modality = item.modality.value
            acquired_counts[modality] += 1.0
            acquired_cost += item.acquisition_cost
            item_tokens = set(normalize_text(item.text).split())
            for token in item_tokens:
                vector[self._hash_index(f"acquired:{modality}:{token}")] += 1.0
            for token in set(query_tokens) & item_tokens:
                vector[self._hash_index(f"acquired_shared:{modality}:{token}")] += 1.0

        remaining_counts = {"subtitle": 0.0, "frame": 0.0, "segment": 0.0}
        remaining_top_scores = {"subtitle": 0.0, "frame": 0.0, "segment": 0.0}
        for modality in ("subtitle", "frame", "segment"):
            items = remaining_by_modality.get(modality, ())
            remaining_counts[modality] = float(len(items))
            if not items:
                continue

            top_item = items[0]
            remaining_top_scores[modality] = float(top_item.retrieval_score)
            top_tokens = set(normalize_text(top_item.text).split())
            for token in top_tokens:
                vector[self._hash_index(f"remaining_top:{modality}:{token}")] += 1.0
            for token in set(query_tokens) & top_tokens:
                vector[self._hash_index(f"remaining_shared:{modality}:{token}")] += 1.0

        numeric_features = np.asarray(
            [
                float(step_index) / max(float(max_steps), 1.0),
                float(len(acquired)),
                acquired_cost,
                float(current_prediction.confidence),
                float(max(current_prediction.option_scores) if current_prediction.option_scores else 0.0),
                acquired_counts["subtitle"],
                acquired_counts["frame"],
                acquired_counts["segment"],
                remaining_counts["subtitle"],
                remaining_counts["frame"],
                remaining_counts["segment"],
                remaining_top_scores["subtitle"],
                remaining_top_scores["frame"],
                remaining_top_scores["segment"],
            ],
            dtype=np.float32,
        )
        vector[self.text_feature_dim :] = numeric_features

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector


class SequentialPolicy(Protocol):
    def run(
        self,
        example: QuestionExample,
        candidate_pool: dict[str, tuple[EvidenceItem, ...]],
        max_items: int = 6,
    ) -> AcquisitionTrace:
        ...


class KeywordSequentialPolicy:
    def __init__(
        self,
        answerer: Answerer,
        stop_confidence: float = 0.80,
        min_items_before_stop: int = 1,
    ) -> None:
        self.answerer = answerer
        self.stop_confidence = stop_confidence
        self.min_items_before_stop = min_items_before_stop

    def preferred_modalities(self, question: str) -> list[str]:
        lowered = question.lower()
        matches = []
        for modality, keywords in KEYWORD_MODALITY_MAP.items():
            score = sum(keyword in lowered for keyword in keywords)
            matches.append((score, modality))
        ranked = [modality for _, modality in sorted(matches, reverse=True)]
        return ranked

    def run(
        self,
        example: QuestionExample,
        candidate_pool: dict[str, tuple[EvidenceItem, ...]],
        max_items: int = 6,
    ) -> AcquisitionTrace:
        acquired: list[EvidenceItem] = []
        steps: list[AcquisitionStep] = []
        ordered_modalities = self.preferred_modalities(example.question)
        modality_offsets = {key: 0 for key in candidate_pool}

        for step_index in range(max_items):
            prediction = self.answerer.predict(example, tuple(acquired))
            if (
                prediction.confidence >= self.stop_confidence
                and len(acquired) >= self.min_items_before_stop
            ):
                steps.append(
                    AcquisitionStep(
                        step_index=step_index,
                        action="stop",
                        selected_item=None,
                        confidence_after_step=prediction.confidence,
                    )
                )
                return AcquisitionTrace(steps=tuple(steps), final_prediction=prediction)

            selected: EvidenceItem | None = None
            selected_modality = "stop"
            for modality in ordered_modalities:
                index = modality_offsets.get(modality, 0)
                items = candidate_pool.get(modality, ())
                if index < len(items):
                    selected = items[index]
                    selected_modality = modality
                    modality_offsets[modality] = index + 1
                    break

            if selected is None:
                final_prediction = self.answerer.predict(example, tuple(acquired))
                steps.append(
                    AcquisitionStep(
                        step_index=step_index,
                        action="stop",
                        selected_item=None,
                        confidence_after_step=final_prediction.confidence,
                    )
                )
                return AcquisitionTrace(steps=tuple(steps), final_prediction=final_prediction)

            acquired.append(selected)
            updated_prediction = self.answerer.predict(example, tuple(acquired))
            steps.append(
                AcquisitionStep(
                    step_index=step_index,
                    action=f"acquire_{selected_modality}",
                    selected_item=selected,
                    confidence_after_step=updated_prediction.confidence,
                )
            )

        final_prediction = self.answerer.predict(example, tuple(acquired))
        steps.append(
            AcquisitionStep(
                step_index=len(steps),
                action="stop",
                selected_item=None,
                confidence_after_step=final_prediction.confidence,
            )
        )
        return AcquisitionTrace(steps=tuple(steps), final_prediction=final_prediction)


class TrainableSequentialPolicy:
    model_name = "trainable_linear_policy"

    def __init__(
        self,
        weights: np.ndarray,
        feature_extractor: HashedPolicyFeatureExtractor,
        answerer: Answerer,
        config: SequentialPolicyConfig,
        history: list[dict[str, float]] | None = None,
    ) -> None:
        self.weights = weights.astype(np.float32, copy=False)
        self.feature_extractor = feature_extractor
        self.answerer = answerer
        self.config = config
        self.history = history or []

    def featurize_state(
        self,
        example: QuestionExample,
        acquired: tuple[EvidenceItem, ...],
        remaining_by_modality: dict[str, tuple[EvidenceItem, ...]],
        step_index: int,
        max_steps: int,
        current_prediction: ModelPrediction | None = None,
    ) -> np.ndarray:
        prediction = current_prediction or self.answerer.predict(example, acquired)
        return self.feature_extractor.transform(
            example=example,
            acquired=acquired,
            remaining_by_modality=remaining_by_modality,
            current_prediction=prediction,
            step_index=step_index,
            max_steps=max_steps,
        )

    def predict_action(
        self,
        example: QuestionExample,
        acquired: tuple[EvidenceItem, ...],
        remaining_by_modality: dict[str, tuple[EvidenceItem, ...]],
        step_index: int,
        max_steps: int,
    ) -> tuple[str, dict[str, float]]:
        current_prediction = self.answerer.predict(example, acquired)
        features = self.featurize_state(
            example=example,
            acquired=acquired,
            remaining_by_modality=remaining_by_modality,
            step_index=step_index,
            max_steps=max_steps,
            current_prediction=current_prediction,
        )
        logits = features @ self.weights
        mask = action_mask(
            remaining_by_modality,
            acquired_count=len(acquired),
            min_items_before_stop=self.config.min_items_before_stop,
        )
        probabilities = _masked_softmax(logits, mask)
        action_index = int(np.argmax(probabilities)) if probabilities.size else ACTION_TO_INDEX["stop"]
        return INDEX_TO_ACTION[action_index], {
            INDEX_TO_ACTION[index]: float(probability)
            for index, probability in enumerate(probabilities)
            if mask[index] > 0
        }

    def run(
        self,
        example: QuestionExample,
        candidate_pool: dict[str, tuple[EvidenceItem, ...]],
        max_items: int = 6,
    ) -> AcquisitionTrace:
        acquired: list[EvidenceItem] = []
        steps: list[AcquisitionStep] = []
        offsets = {"subtitle": 0, "frame": 0, "segment": 0}

        for step_index in range(max_items):
            remaining_by_modality = {
                modality: candidate_pool.get(modality, ())[offsets[modality] :]
                for modality in ("subtitle", "frame", "segment")
            }
            action, _ = self.predict_action(
                example=example,
                acquired=tuple(acquired),
                remaining_by_modality=remaining_by_modality,
                step_index=step_index,
                max_steps=max_items,
            )

            if action == "stop":
                final_prediction = self.answerer.predict(example, tuple(acquired))
                steps.append(
                    AcquisitionStep(
                        step_index=step_index,
                        action="stop",
                        selected_item=None,
                        confidence_after_step=final_prediction.confidence,
                    )
                )
                return AcquisitionTrace(steps=tuple(steps), final_prediction=final_prediction)

            modality = ACTION_TO_MODALITY[action]
            if not remaining_by_modality[modality]:
                final_prediction = self.answerer.predict(example, tuple(acquired))
                steps.append(
                    AcquisitionStep(
                        step_index=step_index,
                        action="stop",
                        selected_item=None,
                        confidence_after_step=final_prediction.confidence,
                    )
                )
                return AcquisitionTrace(steps=tuple(steps), final_prediction=final_prediction)

            selected = remaining_by_modality[modality][0]
            offsets[modality] += 1
            acquired.append(selected)
            updated_prediction = self.answerer.predict(example, tuple(acquired))
            steps.append(
                AcquisitionStep(
                    step_index=step_index,
                    action=action,
                    selected_item=selected,
                    confidence_after_step=updated_prediction.confidence,
                )
            )

        final_prediction = self.answerer.predict(example, tuple(acquired))
        steps.append(
            AcquisitionStep(
                step_index=len(steps),
                action="stop",
                selected_item=None,
                confidence_after_step=final_prediction.confidence,
            )
        )
        return AcquisitionTrace(steps=tuple(steps), final_prediction=final_prediction)

    def save(self, model_dir: str | Path) -> None:
        output_dir = Path(model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(output_dir / "weights.npz", weights=self.weights)
        metadata = {
            "model_name": self.model_name,
            "config": asdict(self.config),
            "feature_extractor": {
                "text_feature_dim": self.feature_extractor.text_feature_dim,
                "numeric_feature_names": list(self.feature_extractor.numeric_feature_names),
            },
            "history": self.history,
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        answerer: Answerer,
    ) -> "TrainableSequentialPolicy":
        input_dir = Path(model_dir)
        metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
        weights = np.load(input_dir / "weights.npz")["weights"]
        feature_extractor = HashedPolicyFeatureExtractor(
            text_feature_dim=int(metadata["feature_extractor"]["text_feature_dim"]),
            numeric_feature_names=tuple(metadata["feature_extractor"]["numeric_feature_names"]),
        )
        config = SequentialPolicyConfig(**metadata["config"])
        history = metadata.get("history", [])
        return cls(
            weights=weights,
            feature_extractor=feature_extractor,
            answerer=answerer,
            config=config,
            history=history,
        )

    @classmethod
    def fit(
        cls,
        train_states: list[PolicyTrainingState],
        answerer: Answerer,
        validation_states: list[PolicyTrainingState] | None = None,
        config: SequentialPolicyConfig | None = None,
    ) -> "TrainableSequentialPolicy":
        if not train_states:
            raise ValueError("TrainableSequentialPolicy.fit requires at least one training state.")

        resolved_config = config or SequentialPolicyConfig()
        feature_extractor = HashedPolicyFeatureExtractor(text_feature_dim=resolved_config.text_feature_dim)
        train_features, train_labels, train_masks = cls._prepare_dataset(
            train_states,
            feature_extractor=feature_extractor,
            answerer=answerer,
            config=resolved_config,
        )
        validation_features, validation_labels, validation_masks = cls._prepare_dataset(
            validation_states or [],
            feature_extractor=feature_extractor,
            answerer=answerer,
            config=resolved_config,
        )

        rng = np.random.default_rng(resolved_config.seed)
        weights = np.zeros((feature_extractor.total_dim, len(POLICY_ACTIONS)), dtype=np.float32)
        best_weights = weights.copy()
        best_score = float("-inf")
        best_epoch = 0
        epochs_without_improvement = 0
        history: list[dict[str, float]] = []

        for epoch in range(1, resolved_config.epochs + 1):
            order = rng.permutation(len(train_features))
            for batch_start in range(0, len(order), resolved_config.batch_size):
                batch_indices = order[batch_start : batch_start + resolved_config.batch_size]
                gradient = np.zeros_like(weights)
                for index in batch_indices:
                    features = train_features[index]
                    label = train_labels[index]
                    mask = train_masks[index]
                    logits = features @ weights
                    probabilities = _masked_softmax(logits, mask)
                    probabilities[label] -= 1.0
                    gradient += np.outer(features, probabilities)

                gradient /= max(len(batch_indices), 1)
                gradient += resolved_config.weight_decay * weights
                weights -= resolved_config.learning_rate * gradient

            train_metrics = cls._dataset_metrics(train_features, train_labels, train_masks, weights)
            validation_metrics = (
                cls._dataset_metrics(validation_features, validation_labels, validation_masks, weights)
                if validation_features
                else train_metrics
            )
            history_entry = {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
            }
            history.append(history_entry)

            score = validation_metrics["accuracy"] - validation_metrics["loss"]
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_weights = weights.copy()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= resolved_config.patience:
                break

        model = cls(
            weights=best_weights,
            feature_extractor=feature_extractor,
            answerer=answerer,
            config=resolved_config,
            history=history,
        )
        model.history.append({"best_epoch": float(best_epoch)})
        return model

    @staticmethod
    def _prepare_dataset(
        states: list[PolicyTrainingState],
        feature_extractor: HashedPolicyFeatureExtractor,
        answerer: Answerer,
        config: SequentialPolicyConfig,
    ) -> tuple[list[np.ndarray], list[int], list[np.ndarray]]:
        features: list[np.ndarray] = []
        labels: list[int] = []
        masks: list[np.ndarray] = []

        for state in states:
            if state.gold_action not in ACTION_TO_INDEX:
                raise ValueError(f"Unsupported policy action: {state.gold_action}")

            remaining_by_modality = state.remaining_by_modality()
            mask = action_mask(
                remaining_by_modality,
                acquired_count=len(state.acquired),
                min_items_before_stop=config.min_items_before_stop,
            )
            label = ACTION_TO_INDEX[state.gold_action]
            if mask[label] == 0:
                if state.gold_action == "stop":
                    continue
                raise ValueError(
                    f"Gold action `{state.gold_action}` is not valid for step {state.step_index} "
                    f"of example {state.example.example_id}."
                )

            current_prediction = answerer.predict(state.example, state.acquired)
            features.append(
                feature_extractor.transform(
                    example=state.example,
                    acquired=state.acquired,
                    remaining_by_modality=remaining_by_modality,
                    current_prediction=current_prediction,
                    step_index=state.step_index,
                    max_steps=state.max_steps,
                )
            )
            labels.append(label)
            masks.append(mask)

        return features, labels, masks

    @staticmethod
    def _dataset_metrics(
        features: list[np.ndarray],
        labels: list[int],
        masks: list[np.ndarray],
        weights: np.ndarray,
    ) -> dict[str, float]:
        if not features:
            return {"loss": 0.0, "accuracy": 0.0}

        losses = []
        correct = 0
        for feature, label, mask in zip(features, labels, masks, strict=True):
            logits = feature @ weights
            probabilities = _masked_softmax(logits, mask)
            losses.append(-float(np.log(max(probabilities[label], 1e-8))))
            if int(np.argmax(probabilities)) == label:
                correct += 1

        return {
            "loss": float(np.mean(losses)),
            "accuracy": correct / len(features),
        }


def build_policy(
    name: str,
    answerer: Answerer,
    model_dir: str | None = None,
    *,
    min_items_before_stop: int = 1,
) -> SequentialPolicy:
    if name == "keyword":
        return KeywordSequentialPolicy(answerer, min_items_before_stop=min_items_before_stop)
    if name == "linear":
        if not model_dir:
            raise ValueError("A model directory is required for the trainable sequential policy.")
        return TrainableSequentialPolicy.load(model_dir, answerer=answerer)
    raise ValueError(f"Unsupported policy: {name}")
