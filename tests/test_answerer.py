import numpy as np

from adaptive_evidence_vqa.data.tvqa import parse_tvqa_like_record
from adaptive_evidence_vqa.models.answerer import (
    LinearAnswererConfig,
    TrainableLinearAnswerer,
)
from adaptive_evidence_vqa.models.frozen_multimodal_answerer import FrozenMultimodalAnswerer


def build_training_examples():
    example_a = parse_tvqa_like_record(
        {
            "example_id": "toy-a",
            "video_id": "video-a",
            "question": "What fruit is on the table?",
            "options": [
                "An apple is on the table.",
                "A banana is on the table.",
            ],
            "answer_index": 0,
            "subtitles": [
                {"text": "There is a shiny apple on the table.", "start": 0.0, "end": 1.0},
            ],
            "frames": [
                {"text": "A person points at an apple.", "time": 0.5},
            ],
            "segments": [
                {"text": "The scene focuses on the apple on the table.", "start": 0.0, "end": 2.0},
            ],
        }
    )
    example_b = parse_tvqa_like_record(
        {
            "example_id": "toy-b",
            "video_id": "video-b",
            "question": "Which animal is running outside?",
            "options": [
                "A cat is running outside.",
                "A dog is running outside.",
            ],
            "answer_index": 1,
            "subtitles": [
                {"text": "A happy dog runs across the yard.", "start": 0.0, "end": 1.0},
            ],
            "frames": [
                {"text": "The dog is visible in the yard.", "time": 0.5},
            ],
            "segments": [
                {"text": "The dog keeps running outside.", "start": 0.0, "end": 2.0},
            ],
        }
    )
    return [
        (example_a, example_a.evidence_pool),
        (example_b, example_b.evidence_pool),
    ]


def test_trainable_linear_answerer_fits_toy_examples() -> None:
    dataset = build_training_examples()
    model = TrainableLinearAnswerer.fit(
        train_examples=dataset,
        config=LinearAnswererConfig(
            text_feature_dim=256,
            epochs=40,
            batch_size=2,
            learning_rate=0.5,
            weight_decay=0.0,
            patience=10,
            seed=7,
        ),
    )

    predictions = [
        model.predict(example, evidence).predicted_index
        for example, evidence in dataset
    ]

    assert predictions == [0, 1]


def test_trainable_linear_answerer_save_and_load(tmp_path) -> None:
    dataset = build_training_examples()
    model = TrainableLinearAnswerer.fit(
        train_examples=dataset,
        config=LinearAnswererConfig(
            text_feature_dim=256,
            epochs=30,
            batch_size=2,
            learning_rate=0.5,
            weight_decay=0.0,
            patience=8,
            seed=11,
        ),
    )
    model_dir = tmp_path / "linear-answerer"
    model.save(model_dir)

    loaded = TrainableLinearAnswerer.load(model_dir)
    example, evidence = dataset[0]

    assert loaded.predict(example, evidence).predicted_index == model.predict(example, evidence).predicted_index


def test_trainable_linear_answerer_needs_evidence_to_choose_nondefault_option() -> None:
    dataset = build_training_examples()
    model = TrainableLinearAnswerer.fit(
        train_examples=dataset,
        config=LinearAnswererConfig(
            text_feature_dim=256,
            epochs=40,
            batch_size=2,
            learning_rate=0.5,
            weight_decay=0.0,
            patience=10,
            seed=5,
        ),
    )
    example, evidence = dataset[1]

    with_evidence = model.predict(example, evidence)
    without_evidence = model.predict(example, ())

    assert with_evidence.predicted_index == 1
    assert without_evidence.predicted_index == 0


class StubTextEncoder:
    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            lowered = text.lower()
            if "blue" in lowered:
                embeddings.append([1.0, 0.0])
            elif "red" in lowered:
                embeddings.append([0.0, 1.0])
            else:
                embeddings.append([0.5, 0.5])
        return np.asarray(embeddings, dtype=np.float32)


def test_frozen_multimodal_answerer_uses_visual_feature_store(tmp_path) -> None:
    feature_path = tmp_path / "visual_features.npz"
    np.savez(
        feature_path,
        frame_embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        segment_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        frame_times=np.asarray([1.0, 2.0], dtype=np.float32),
    )
    example = parse_tvqa_like_record(
        {
            "example_id": "toy-visual-answerer",
            "video_id": "video-visual-answerer",
            "question": "What color is the shirt?",
            "options": [
                "The shirt is blue.",
                "The shirt is red.",
            ],
            "answer_index": 0,
            "subtitles": [
                {"text": "They keep talking in the room.", "start": 0.0, "end": 1.0},
            ],
            "frames": [
                {
                    "text": "",
                    "time": 1.0,
                    "metadata": {"feature_index": 0, "visual_feature_path": str(feature_path)},
                },
                {
                    "text": "",
                    "time": 2.0,
                    "metadata": {"feature_index": 1, "visual_feature_path": str(feature_path)},
                },
            ],
            "segments": [
                {
                    "text": "",
                    "start": 0.5,
                    "end": 1.5,
                    "metadata": {"feature_index": 0, "visual_feature_path": str(feature_path)},
                },
            ],
        }
    )
    answerer = FrozenMultimodalAnswerer(text_encoder=StubTextEncoder())

    prediction = answerer.predict(example, example.evidence_pool)

    assert prediction.predicted_index == 0
