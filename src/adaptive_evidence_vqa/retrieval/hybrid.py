from dataclasses import replace
from pathlib import Path
from typing import Protocol

import numpy as np

from adaptive_evidence_vqa.retrieval.base import BM25Retriever, Retriever
from adaptive_evidence_vqa.retrieval.visual_features import clip_output_to_numpy
from adaptive_evidence_vqa.schemas import EvidenceItem


class TextEncoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        ...


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors.astype(np.float32, copy=False)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return (vectors / norms).astype(np.float32, copy=False)


class ClipTextEncoder:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "ClipTextEncoder requires the vision dependencies. "
                "Install the project with `.[vision]` inside the Conda environment."
            ) from exc

        self._torch = torch
        self.device = device or self._resolve_device(torch)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _resolve_device(self, torch_module) -> str:
        if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
            return "cuda"
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.config.projection_dim), dtype=np.float32)

        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        with self._torch.no_grad():
            text_features = self.model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        return _normalize_rows(clip_output_to_numpy(text_features))


class FeatureBackedVisualRetriever:
    def __init__(
        self,
        text_encoder: TextEncoder,
        array_name: str,
    ) -> None:
        self.text_encoder = text_encoder
        self.array_name = array_name
        self._feature_cache: dict[str, dict[str, np.ndarray]] = {}

    def _feature_file(self, evidence: EvidenceItem) -> str:
        feature_path = evidence.metadata.get("visual_feature_path")
        if not isinstance(feature_path, str) or not feature_path:
            raise ValueError(
                f"Evidence item {evidence.evidence_id} is missing `visual_feature_path` metadata."
            )
        return feature_path

    def _feature_vector(self, evidence: EvidenceItem) -> np.ndarray:
        feature_path = self._feature_file(evidence)
        if feature_path not in self._feature_cache:
            with np.load(feature_path) as data:
                self._feature_cache[feature_path] = {
                    key: data[key].astype(np.float32, copy=False)
                    for key in data.files
                }

        feature_index = evidence.metadata.get("feature_index")
        if not isinstance(feature_index, int):
            raise ValueError(
                f"Evidence item {evidence.evidence_id} is missing integer `feature_index` metadata."
            )

        matrix = self._feature_cache[feature_path][self.array_name]
        return matrix[feature_index]

    def score(
        self,
        query: str,
        evidence: EvidenceItem,
        pool: tuple[EvidenceItem, ...] | None = None,
    ) -> float:
        query_embedding = self.text_encoder.encode([query])[0]
        feature_vector = self._feature_vector(evidence)
        return float(np.dot(query_embedding, feature_vector))

    def retrieve(
        self,
        query: str,
        pool: tuple[EvidenceItem, ...],
        top_k: int,
    ) -> tuple[EvidenceItem, ...]:
        if top_k <= 0 or not pool:
            return ()

        query_embedding = self.text_encoder.encode([query])[0]
        scored = []
        for item in pool:
            feature_vector = self._feature_vector(item)
            score = float(np.dot(query_embedding, feature_vector))
            scored.append((item, score))

        ranked = sorted(scored, key=lambda pair: pair[1], reverse=True)
        return tuple(
            replace(item, retrieval_score=score)
            for item, score in ranked[:top_k]
        )


class HybridClipRetriever:
    """BM25 for subtitles and CLIP similarity for frames/segments."""

    def __init__(
        self,
        text_encoder: TextEncoder | None = None,
        subtitle_retriever: Retriever | None = None,
    ) -> None:
        encoder = text_encoder or ClipTextEncoder()
        self.subtitle_retriever = subtitle_retriever or BM25Retriever()
        self.frame_retriever = FeatureBackedVisualRetriever(
            text_encoder=encoder,
            array_name="frame_embeddings",
        )
        self.segment_retriever = FeatureBackedVisualRetriever(
            text_encoder=encoder,
            array_name="segment_embeddings",
        )

    def _delegate(self, pool: tuple[EvidenceItem, ...]) -> Retriever:
        if not pool:
            return self.subtitle_retriever
        modality = pool[0].modality.value
        if modality == "subtitle":
            return self.subtitle_retriever
        if modality == "frame":
            return self.frame_retriever
        if modality == "segment":
            return self.segment_retriever
        raise ValueError(f"Unsupported modality for retrieval: {modality}")

    def score(
        self,
        query: str,
        evidence: EvidenceItem,
        pool: tuple[EvidenceItem, ...] | None = None,
    ) -> float:
        if pool is None:
            pool = (evidence,)
        return self._delegate(pool).score(query, evidence, pool=pool)

    def retrieve(
        self,
        query: str,
        pool: tuple[EvidenceItem, ...],
        top_k: int,
    ) -> tuple[EvidenceItem, ...]:
        return self._delegate(pool).retrieve(query, pool, top_k)
