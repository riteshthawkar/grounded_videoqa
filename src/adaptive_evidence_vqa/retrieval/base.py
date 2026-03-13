import math
from collections import Counter
from dataclasses import dataclass, replace
from typing import Protocol

from adaptive_evidence_vqa.schemas import EvidenceItem, QuestionExample
from adaptive_evidence_vqa.utils import jaccard_overlap, normalize_text

MODALITY_KEYS = ("subtitle", "frame", "segment")


@dataclass(frozen=True, slots=True)
class RetrievalAllocation:
    subtitle: int = 0
    frame: int = 0
    segment: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "subtitle": self.subtitle,
            "frame": self.frame,
            "segment": self.segment,
        }


def build_query(example: QuestionExample, include_options: bool = True) -> str:
    fields = [example.question]
    if include_options:
        fields.extend(option.text for option in example.options)
    return " ".join(fields)


def coerce_modality_limits(
    top_k_per_modality: int | dict[str, int] | RetrievalAllocation,
) -> dict[str, int]:
    if isinstance(top_k_per_modality, int):
        return {key: top_k_per_modality for key in MODALITY_KEYS}
    if isinstance(top_k_per_modality, RetrievalAllocation):
        return top_k_per_modality.to_dict()

    return {key: int(top_k_per_modality.get(key, 0)) for key in MODALITY_KEYS}


def flatten_candidate_pool(
    candidate_pool: dict[str, tuple[EvidenceItem, ...]],
) -> tuple[EvidenceItem, ...]:
    flattened = []
    for key in MODALITY_KEYS:
        flattened.extend(candidate_pool.get(key, ()))
    return tuple(sorted(flattened, key=lambda item: item.retrieval_score, reverse=True))


class Retriever(Protocol):
    def score(self, query: str, evidence: EvidenceItem, pool: tuple[EvidenceItem, ...] | None = None) -> float:
        ...

    def retrieve(
        self,
        query: str,
        pool: tuple[EvidenceItem, ...],
        top_k: int,
    ) -> tuple[EvidenceItem, ...]:
        ...


class LexicalRetriever:
    """A trivial retriever used for early integration testing.

    This is intentionally weak. It gives us a runnable end-to-end baseline
    before we replace it with learned text, image, and video encoders.
    """

    def score(
        self,
        query: str,
        evidence: EvidenceItem,
        pool: tuple[EvidenceItem, ...] | None = None,
    ) -> float:
        return jaccard_overlap(query, evidence.text)

    def retrieve(
        self,
        query: str,
        pool: tuple[EvidenceItem, ...],
        top_k: int,
    ) -> tuple[EvidenceItem, ...]:
        ranked = sorted(
            pool,
            key=lambda item: self.score(query, item),
            reverse=True,
        )
        return tuple(
            replace(item, retrieval_score=self.score(query, item))
            for item in ranked[:top_k]
        )


class BM25Retriever:
    """A stronger sparse baseline for retrieval over small candidate pools."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def _tokenize(self, text: str) -> tuple[str, ...]:
        normalized = normalize_text(text)
        if not normalized:
            return ()
        return tuple(normalized.split())

    def _pool_stats(self, pool: tuple[EvidenceItem, ...]) -> tuple[dict[str, int], dict[str, Counter[str]], float]:
        document_frequency: dict[str, int] = {}
        term_frequencies: dict[str, Counter[str]] = {}
        document_lengths = []

        for item in pool:
            tokens = self._tokenize(item.text)
            counts = Counter(tokens)
            term_frequencies[item.evidence_id] = counts
            document_lengths.append(sum(counts.values()))
            for token in counts:
                document_frequency[token] = document_frequency.get(token, 0) + 1

        average_length = sum(document_lengths) / len(document_lengths) if document_lengths else 0.0
        return document_frequency, term_frequencies, average_length

    def _score_with_stats(
        self,
        query_tokens: tuple[str, ...],
        evidence: EvidenceItem,
        document_frequency: dict[str, int],
        term_frequencies: dict[str, Counter[str]],
        average_length: float,
        num_documents: int,
    ) -> float:
        term_counts = term_frequencies[evidence.evidence_id]
        document_length = sum(term_counts.values())
        score = 0.0

        for token in set(query_tokens):
            tf = term_counts.get(token, 0)
            if tf == 0:
                continue
            df = document_frequency.get(token, 0)
            idf = math.log(1.0 + ((num_documents - df + 0.5) / (df + 0.5)))
            denominator = tf + self.k1 * (
                1.0 - self.b + self.b * (document_length / max(average_length, 1e-8))
            )
            score += idf * ((tf * (self.k1 + 1.0)) / max(denominator, 1e-8))

        return score

    def score(
        self,
        query: str,
        evidence: EvidenceItem,
        pool: tuple[EvidenceItem, ...] | None = None,
    ) -> float:
        if pool is None:
            raise ValueError("BM25Retriever.score requires the candidate pool for corpus statistics.")

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0

        document_frequency, term_frequencies, average_length = self._pool_stats(pool)
        return self._score_with_stats(
            query_tokens=query_tokens,
            evidence=evidence,
            document_frequency=document_frequency,
            term_frequencies=term_frequencies,
            average_length=average_length,
            num_documents=len(pool),
        )

    def retrieve(
        self,
        query: str,
        pool: tuple[EvidenceItem, ...],
        top_k: int,
    ) -> tuple[EvidenceItem, ...]:
        query_tokens = self._tokenize(query)
        if not query_tokens or not pool:
            return ()

        document_frequency, term_frequencies, average_length = self._pool_stats(pool)
        num_documents = len(pool)
        scored = [
            (
                item,
                self._score_with_stats(
                    query_tokens=query_tokens,
                    evidence=item,
                    document_frequency=document_frequency,
                    term_frequencies=term_frequencies,
                    average_length=average_length,
                    num_documents=num_documents,
                ),
            )
            for item in pool
        ]
        ranked = sorted(scored, key=lambda pair: pair[1], reverse=True)
        return tuple(
            replace(item, retrieval_score=score)
            for item, score in ranked[:top_k]
        )


def build_named_retriever(
    name: str,
    *,
    visual_model_name: str = "openai/clip-vit-base-patch32",
    visual_device: str | None = None,
) -> Retriever:
    if name == "lexical":
        return LexicalRetriever()
    if name == "bm25":
        return BM25Retriever()
    if name == "hybrid_clip":
        from adaptive_evidence_vqa.retrieval.hybrid import ClipTextEncoder, HybridClipRetriever

        return HybridClipRetriever(
            text_encoder=ClipTextEncoder(
                model_name=visual_model_name,
                device=visual_device,
            ),
        )
    raise ValueError(f"Unsupported retriever: {name}")


class CandidatePoolBuilder:
    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    def build(
        self,
        example: QuestionExample,
        top_k_per_modality: int | dict[str, int] | RetrievalAllocation = 4,
    ) -> dict[str, tuple[EvidenceItem, ...]]:
        query = build_query(example)
        limits = coerce_modality_limits(top_k_per_modality)
        return {
            "subtitle": self.retriever.retrieve(query, example.subtitles, limits["subtitle"]),
            "frame": self.retriever.retrieve(query, example.frames, limits["frame"]),
            "segment": self.retriever.retrieve(query, example.segments, limits["segment"]),
        }


class FixedBudgetRetriever:
    def __init__(self, retriever: Retriever) -> None:
        self.pool_builder = CandidatePoolBuilder(retriever)

    def retrieve(
        self,
        example: QuestionExample,
        allocation: RetrievalAllocation,
    ) -> tuple[EvidenceItem, ...]:
        return flatten_candidate_pool(self.pool_builder.build(example, top_k_per_modality=allocation))
