from dataclasses import dataclass

import numpy as np

from adaptive_evidence_vqa.retrieval.hybrid import ClipTextEncoder, TextEncoder
from adaptive_evidence_vqa.schemas import EvidenceItem, ModelPrediction, QuestionExample
from adaptive_evidence_vqa.utils import jaccard_overlap, softmax


@dataclass(slots=True)
class FrozenMultimodalAnswererConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    device: str | None = None
    subtitle_weight: float = 1.0
    frame_weight: float = 1.0
    segment_weight: float = 1.1
    lexical_weight: float = 0.20
    retrieval_score_weight: float = 0.10
    max_vs_mean_mix: float = 0.60
    option_prior_weight: float = 0.10


class FrozenMultimodalAnswerer:
    """A stronger frozen baseline using CLIP text embeddings and visual features.

    This is intentionally lightweight: it keeps the answerer frozen while
    using the repo's existing precomputed frame and segment features.
    The goal is to provide a more credible answerer backend than the
    lexical and hashed-linear baselines without introducing heavy training.
    """

    model_name = "frozen_multimodal"

    def __init__(
        self,
        config: FrozenMultimodalAnswererConfig | None = None,
        text_encoder: TextEncoder | None = None,
    ) -> None:
        self.config = config or FrozenMultimodalAnswererConfig()
        self.text_encoder = text_encoder or ClipTextEncoder(
            model_name=self.config.model_name,
            device=self.config.device,
        )
        self._text_cache: dict[str, np.ndarray] = {}
        self._feature_cache: dict[str, dict[str, np.ndarray]] = {}

    def _text_embedding(self, text: str) -> np.ndarray:
        cache_key = text.strip() or " "
        if cache_key not in self._text_cache:
            self._text_cache[cache_key] = self.text_encoder.encode([cache_key])[0]
        return self._text_cache[cache_key]

    def _load_feature_file(self, feature_path: str) -> dict[str, np.ndarray]:
        if feature_path not in self._feature_cache:
            with np.load(feature_path) as data:
                self._feature_cache[feature_path] = {
                    key: data[key].astype(np.float32, copy=False)
                    for key in data.files
                }
        return self._feature_cache[feature_path]

    def _visual_feature(self, evidence: EvidenceItem) -> np.ndarray | None:
        feature_path = evidence.metadata.get("visual_feature_path")
        feature_index = evidence.metadata.get("feature_index")
        if not isinstance(feature_path, str) or not feature_path:
            return None
        if not isinstance(feature_index, int):
            return None

        array_name = None
        if evidence.modality.value == "frame":
            array_name = "frame_embeddings"
        elif evidence.modality.value == "segment":
            array_name = "segment_embeddings"
        if array_name is None:
            return None

        feature_store = self._load_feature_file(feature_path)
        if array_name not in feature_store:
            return None
        matrix = feature_store[array_name]
        if feature_index < 0 or feature_index >= len(matrix):
            return None
        return matrix[feature_index]

    def _modality_score(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        items: tuple[EvidenceItem, ...],
        modality_weight: float,
    ) -> float:
        if not items:
            return 0.0

        item_scores = []
        for item in items:
            if item.modality.value == "subtitle":
                evidence_embedding = self._text_embedding(item.text)
                semantic_score = float(np.dot(query_embedding, evidence_embedding))
            else:
                feature_vector = self._visual_feature(item)
                if feature_vector is None:
                    evidence_embedding = self._text_embedding(item.text)
                    semantic_score = float(np.dot(query_embedding, evidence_embedding))
                else:
                    semantic_score = float(np.dot(query_embedding, feature_vector))

            lexical_score = jaccard_overlap(query_text, item.text) if item.text else 0.0
            retrieval_bonus = max(item.retrieval_score, 0.0)
            item_scores.append(
                semantic_score
                + (self.config.lexical_weight * lexical_score)
                + (self.config.retrieval_score_weight * retrieval_bonus)
            )

        max_score = max(item_scores)
        mean_score = sum(item_scores) / len(item_scores)
        blended_score = (
            self.config.max_vs_mean_mix * max_score
            + (1.0 - self.config.max_vs_mean_mix) * mean_score
        )
        return modality_weight * blended_score

    def score_option(
        self,
        question: str,
        option_text: str,
        evidence: tuple[EvidenceItem, ...],
    ) -> float:
        query_text = f"{question} {option_text}"
        query_embedding = self._text_embedding(query_text)

        subtitles = tuple(item for item in evidence if item.modality.value == "subtitle")
        frames = tuple(item for item in evidence if item.modality.value == "frame")
        segments = tuple(item for item in evidence if item.modality.value == "segment")

        option_prior = self.config.option_prior_weight * jaccard_overlap(question, option_text)
        return (
            option_prior
            + self._modality_score(query_text, query_embedding, subtitles, self.config.subtitle_weight)
            + self._modality_score(query_text, query_embedding, frames, self.config.frame_weight)
            + self._modality_score(query_text, query_embedding, segments, self.config.segment_weight)
        )

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
        predicted_index = max(range(len(scores)), key=scores.__getitem__) if scores else 0
        return ModelPrediction(
            predicted_index=predicted_index,
            option_scores=tuple(float(score) for score in scores),
            confidence=float(probabilities[predicted_index]) if probabilities else 0.0,
            supporting_evidence=evidence,
        )
