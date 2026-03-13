import numpy as np

from adaptive_evidence_vqa.data.tvqa import parse_tvqa_like_record
from adaptive_evidence_vqa.retrieval.base import (
    BM25Retriever,
    CandidatePoolBuilder,
    FixedBudgetRetriever,
    LexicalRetriever,
    RetrievalAllocation,
)
from adaptive_evidence_vqa.retrieval.hybrid import HybridClipRetriever


def toy_example():
    return parse_tvqa_like_record(
        {
            "example_id": "toy-002",
            "video_id": "video-002",
            "question": "Who waves before leaving the room?",
            "options": [
                "The woman waves goodbye.",
                "The doctor sits down.",
                "The man opens the window.",
                "The child starts running.",
                "Nobody leaves the room.",
            ],
            "answer_index": 0,
            "subtitles": [
                {"text": "The woman waves goodbye before leaving.", "start": 1.0, "end": 2.0},
                {"text": "The doctor is still in the room.", "start": 2.0, "end": 3.0},
            ],
            "frames": [
                {"text": "A woman stands near the door.", "time": 1.5},
                {"text": "A doctor sits at a desk.", "time": 2.5},
            ],
            "segments": [
                {"text": "The woman waves and exits through the door.", "start": 1.0, "end": 3.0},
                {"text": "The doctor keeps reading files.", "start": 3.0, "end": 5.0},
            ],
        }
    )


def test_candidate_pool_builder_respects_per_modality_limits() -> None:
    example = toy_example()
    builder = CandidatePoolBuilder(LexicalRetriever())
    allocation = RetrievalAllocation(subtitle=1, frame=2, segment=1)

    pool = builder.build(example, top_k_per_modality=allocation)

    assert len(pool["subtitle"]) == 1
    assert len(pool["frame"]) == 2
    assert len(pool["segment"]) == 1
    assert pool["subtitle"][0].retrieval_score > 0.0


def test_fixed_budget_retriever_returns_combined_selection() -> None:
    example = toy_example()
    retriever = FixedBudgetRetriever(LexicalRetriever())
    allocation = RetrievalAllocation(subtitle=1, frame=1, segment=1)

    selected = retriever.retrieve(example, allocation)

    assert len(selected) == 3
    assert {item.modality.value for item in selected} == {"subtitle", "frame", "segment"}


def test_bm25_retriever_ranks_relevant_subtitle_first() -> None:
    example = toy_example()
    builder = CandidatePoolBuilder(BM25Retriever())

    pool = builder.build(example, top_k_per_modality=RetrievalAllocation(subtitle=2, frame=0, segment=0))

    assert pool["subtitle"][0].text == "The woman waves goodbye before leaving."


class StubTextEncoder:
    def encode(self, texts: list[str]) -> np.ndarray:
        query = texts[0].lower()
        if "woman" in query:
            return np.asarray([[1.0, 0.0]], dtype=np.float32)
        return np.asarray([[0.0, 1.0]], dtype=np.float32)


def test_hybrid_clip_retriever_uses_visual_feature_store(tmp_path) -> None:
    feature_path = tmp_path / "visual_features.npz"
    np.savez(
        feature_path,
        frame_embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        segment_embeddings=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        frame_times=np.asarray([1.5, 2.5], dtype=np.float32),
    )
    example = parse_tvqa_like_record(
        {
            "example_id": "toy-visual",
            "video_id": "video-visual",
            "question": "Who waves before leaving the room?",
            "options": [
                "The woman waves goodbye.",
                "The doctor sits down.",
            ],
            "answer_index": 0,
            "subtitles": [
                {"text": "The woman waves goodbye before leaving.", "start": 1.0, "end": 2.0},
                {"text": "The doctor is still in the room.", "start": 2.0, "end": 3.0},
            ],
            "frames": [
                {
                    "text": "",
                    "time": 1.5,
                    "source_path": "/tmp/frame0.jpg",
                    "metadata": {"feature_index": 0, "visual_feature_path": str(feature_path)},
                },
                {
                    "text": "",
                    "time": 2.5,
                    "source_path": "/tmp/frame1.jpg",
                    "metadata": {"feature_index": 1, "visual_feature_path": str(feature_path)},
                },
            ],
            "segments": [
                {
                    "text": "",
                    "start": 1.0,
                    "end": 2.0,
                    "source_path": "/tmp/seg0.mp4",
                    "metadata": {"feature_index": 0, "visual_feature_path": str(feature_path)},
                },
                {
                    "text": "",
                    "start": 2.0,
                    "end": 3.0,
                    "source_path": "/tmp/seg1.mp4",
                    "metadata": {"feature_index": 1, "visual_feature_path": str(feature_path)},
                },
            ],
            "metadata": {"visual_feature_path": str(feature_path)},
        }
    )
    builder = CandidatePoolBuilder(HybridClipRetriever(text_encoder=StubTextEncoder(), subtitle_retriever=BM25Retriever()))

    pool = builder.build(example, top_k_per_modality=RetrievalAllocation(subtitle=1, frame=1, segment=1))

    assert pool["subtitle"][0].text == "The woman waves goodbye before leaving."
    assert pool["frame"][0].start_time == 1.5
    assert pool["segment"][0].start_time == 2.0
