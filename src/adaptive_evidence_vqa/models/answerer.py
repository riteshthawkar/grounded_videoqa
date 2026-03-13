import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from adaptive_evidence_vqa.schemas import EvidenceItem, ModelPrediction, QuestionExample
from adaptive_evidence_vqa.utils import jaccard_overlap, normalize_text, softmax

ExampleWithEvidence = tuple[QuestionExample, tuple[EvidenceItem, ...]]


class Answerer(Protocol):
    def predict(
        self,
        example: QuestionExample,
        evidence: tuple[EvidenceItem, ...],
    ) -> ModelPrediction:
        ...


class LexicalAnswerer:
    """A lightweight answerer for early pipeline integration."""

    def score_option(
        self,
        question: str,
        option_text: str,
        evidence: tuple[EvidenceItem, ...],
    ) -> float:
        option_query = f"{question} {option_text}"
        evidence_score = sum(jaccard_overlap(option_query, item.text) for item in evidence)
        option_overlap = jaccard_overlap(question, option_text)
        return evidence_score + (0.25 * option_overlap)

    def predict(
        self,
        example: QuestionExample,
        evidence: tuple[EvidenceItem, ...],
    ) -> ModelPrediction:
        scores = [
            self.score_option(example.question, option.text, evidence)
            for option in example.options
        ]
        probabilities = softmax(scores)
        predicted_index = max(range(len(scores)), key=scores.__getitem__)
        return ModelPrediction(
            predicted_index=predicted_index,
            option_scores=tuple(scores),
            confidence=probabilities[predicted_index],
            supporting_evidence=evidence,
        )


@dataclass(slots=True)
class LinearAnswererConfig:
    text_feature_dim: int = 4096
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.2
    weight_decay: float = 1e-4
    patience: int = 4
    seed: int = 13


@dataclass(slots=True)
class HashedFeatureExtractor:
    text_feature_dim: int = 4096
    numeric_feature_names: tuple[str, ...] = field(
        default_factory=lambda: (
            "query_evidence_overlap_sum",
            "query_evidence_overlap_max",
            "subtitle_overlap_sum",
            "frame_overlap_sum",
            "segment_overlap_sum",
            "evidence_count",
            "evidence_cost",
            "max_retrieval_score",
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
        question: str,
        option_text: str,
        evidence: tuple[EvidenceItem, ...],
    ) -> np.ndarray:
        vector = np.zeros(self.total_dim, dtype=np.float32)
        query_text = f"{question} {option_text}"
        query_tokens = set(normalize_text(query_text).split())

        subtitle_overlap_sum = 0.0
        frame_overlap_sum = 0.0
        segment_overlap_sum = 0.0
        query_evidence_overlap_sum = 0.0
        query_evidence_overlap_max = 0.0
        max_retrieval_score = 0.0
        evidence_cost = 0.0

        for item in evidence:
            overlap = jaccard_overlap(query_text, item.text)
            query_evidence_overlap_sum += overlap
            query_evidence_overlap_max = max(query_evidence_overlap_max, overlap)
            evidence_cost += item.acquisition_cost
            max_retrieval_score = max(max_retrieval_score, item.retrieval_score)

            evidence_tokens = set(normalize_text(item.text).split())
            shared_tokens = query_tokens & evidence_tokens
            for token in shared_tokens:
                shared_weight = 1.0 + max(item.retrieval_score, 0.0)
                vector[self._hash_index(f"shared:{token}")] += shared_weight
                vector[self._hash_index(f"shared:{item.modality.value}:{token}")] += shared_weight

            if item.modality.value == "subtitle":
                subtitle_overlap_sum += overlap
            elif item.modality.value == "frame":
                frame_overlap_sum += overlap
            elif item.modality.value == "segment":
                segment_overlap_sum += overlap

        numeric_features = np.asarray(
            [
                query_evidence_overlap_sum,
                query_evidence_overlap_max,
                subtitle_overlap_sum,
                frame_overlap_sum,
                segment_overlap_sum,
                float(len(evidence)),
                evidence_cost,
                max_retrieval_score,
            ],
            dtype=np.float32,
        )
        vector[self.text_feature_dim :] = numeric_features

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector


def _softmax_numpy(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=np.float32)
    shifted = values - np.max(values)
    exps = np.exp(shifted)
    return exps / np.clip(np.sum(exps), 1e-8, None)


class TrainableLinearAnswerer:
    """Linear multiple-choice scorer over hashed text and overlap features."""

    model_name = "trainable_linear"

    def __init__(
        self,
        weights: np.ndarray,
        feature_extractor: HashedFeatureExtractor,
        config: LinearAnswererConfig,
        history: list[dict[str, float]] | None = None,
    ) -> None:
        self.weights = weights.astype(np.float32, copy=False)
        self.feature_extractor = feature_extractor
        self.config = config
        self.history = history or []

    def featurize_option(
        self,
        question: str,
        option_text: str,
        evidence: tuple[EvidenceItem, ...],
    ) -> np.ndarray:
        return self.feature_extractor.transform(question, option_text, evidence)

    def option_scores(
        self,
        example: QuestionExample,
        evidence: tuple[EvidenceItem, ...],
    ) -> np.ndarray:
        features = np.stack(
            [
                self.featurize_option(example.question, option.text, evidence)
                for option in example.options
            ]
        )
        return features @ self.weights

    def predict(
        self,
        example: QuestionExample,
        evidence: tuple[EvidenceItem, ...],
    ) -> ModelPrediction:
        scores = self.option_scores(example, evidence)
        probabilities = _softmax_numpy(scores)
        predicted_index = int(np.argmax(scores)) if scores.size else 0
        return ModelPrediction(
            predicted_index=predicted_index,
            option_scores=tuple(float(score) for score in scores),
            confidence=float(probabilities[predicted_index]) if probabilities.size else 0.0,
            supporting_evidence=evidence,
        )

    def save(self, model_dir: str | Path) -> None:
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        np.savez(model_path / "weights.npz", weights=self.weights)
        metadata = {
            "model_name": self.model_name,
            "config": asdict(self.config),
            "feature_extractor": {
                "text_feature_dim": self.feature_extractor.text_feature_dim,
                "numeric_feature_names": list(self.feature_extractor.numeric_feature_names),
            },
            "history": self.history,
        }
        (model_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_dir: str | Path) -> "TrainableLinearAnswerer":
        model_path = Path(model_dir)
        metadata = json.loads((model_path / "metadata.json").read_text(encoding="utf-8"))
        weights = np.load(model_path / "weights.npz")["weights"]
        feature_extractor = HashedFeatureExtractor(
            text_feature_dim=int(metadata["feature_extractor"]["text_feature_dim"]),
            numeric_feature_names=tuple(metadata["feature_extractor"]["numeric_feature_names"]),
        )
        config = LinearAnswererConfig(**metadata["config"])
        history = metadata.get("history", [])
        return cls(weights=weights, feature_extractor=feature_extractor, config=config, history=history)

    @classmethod
    def fit(
        cls,
        train_examples: list[ExampleWithEvidence],
        validation_examples: list[ExampleWithEvidence] | None = None,
        config: LinearAnswererConfig | None = None,
    ) -> "TrainableLinearAnswerer":
        if not train_examples:
            raise ValueError("TrainableLinearAnswerer.fit requires at least one training example.")

        resolved_config = config or LinearAnswererConfig()
        feature_extractor = HashedFeatureExtractor(text_feature_dim=resolved_config.text_feature_dim)
        train_features, train_labels = cls._prepare_dataset(train_examples, feature_extractor)
        validation_features, validation_labels = cls._prepare_dataset(
            validation_examples or [],
            feature_extractor,
        )

        rng = np.random.default_rng(resolved_config.seed)
        weights = np.zeros(feature_extractor.total_dim, dtype=np.float32)
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
                    scores = features @ weights
                    probabilities = _softmax_numpy(scores)
                    probabilities[label] -= 1.0
                    gradient += probabilities @ features

                gradient /= max(len(batch_indices), 1)
                gradient += resolved_config.weight_decay * weights
                weights -= resolved_config.learning_rate * gradient

            train_metrics = cls._dataset_metrics(train_features, train_labels, weights)
            validation_metrics = (
                cls._dataset_metrics(validation_features, validation_labels, weights)
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
            config=resolved_config,
            history=history,
        )
        model.history.append({"best_epoch": float(best_epoch)})
        return model

    @staticmethod
    def _prepare_dataset(
        examples: list[ExampleWithEvidence],
        feature_extractor: HashedFeatureExtractor,
    ) -> tuple[list[np.ndarray], list[int]]:
        features: list[np.ndarray] = []
        labels: list[int] = []
        for example, evidence in examples:
            if example.answer_index is None:
                continue
            features.append(
                np.stack(
                    [
                        feature_extractor.transform(example.question, option.text, evidence)
                        for option in example.options
                    ]
                )
            )
            labels.append(example.answer_index)
        return features, labels

    @staticmethod
    def _dataset_metrics(
        features: list[np.ndarray],
        labels: list[int],
        weights: np.ndarray,
    ) -> dict[str, float]:
        if not features:
            return {"loss": 0.0, "accuracy": 0.0}

        losses: list[float] = []
        correct = 0
        for option_features, label in zip(features, labels, strict=False):
            scores = option_features @ weights
            probabilities = _softmax_numpy(scores)
            losses.append(-float(np.log(np.clip(probabilities[label], 1e-8, 1.0))))
            correct += int(int(np.argmax(scores)) == label)

        return {
            "loss": float(sum(losses) / len(losses)),
            "accuracy": float(correct / len(features)),
        }


def build_answerer(
    name: str,
    model_dir: str | Path | None = None,
    *,
    model_name: str = "openai/clip-vit-base-patch32",
    device: str | None = None,
) -> Answerer:
    if name == "lexical":
        return LexicalAnswerer()
    if name == "linear":
        if model_dir is None:
            raise ValueError("A model directory is required when loading the linear answerer.")
        return TrainableLinearAnswerer.load(model_dir)
    if name == "frozen_multimodal":
        from adaptive_evidence_vqa.models.frozen_multimodal_answerer import (
            FrozenMultimodalAnswerer,
            FrozenMultimodalAnswererConfig,
        )

        return FrozenMultimodalAnswerer(
            config=FrozenMultimodalAnswererConfig(
                model_name=model_name,
                device=device,
            )
        )
    raise ValueError(f"Unsupported answerer: {name}")
