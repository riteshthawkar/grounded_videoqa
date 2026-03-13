from adaptive_evidence_vqa.data.tvqa import parse_tvqa_like_record
from adaptive_evidence_vqa.models.answerer import LexicalAnswerer
from adaptive_evidence_vqa.models.oracle import MinimalEvidenceOracle, OracleConfig
from adaptive_evidence_vqa.schemas import ModelPrediction


class KeywordAnswerer:
    def predict(self, example, evidence):
        joined = " ".join(item.text.lower() for item in evidence)
        if "goodbye" in joined:
            predicted_index = 0
            scores = (3.0, 1.0)
        else:
            predicted_index = 1
            scores = (1.0, 3.0)
        return ModelPrediction(
            predicted_index=predicted_index,
            option_scores=scores,
            confidence=0.9,
            supporting_evidence=evidence,
        )


def test_oracle_returns_subset_not_superset() -> None:
    example = parse_tvqa_like_record(
        {
            "example_id": "toy-001",
            "video_id": "video-001",
            "question": "What does the woman say before she leaves the room?",
            "options": [
                "She says goodbye to John.",
                "She opens the window.",
            ],
            "answer_index": 0,
            "subtitles": [
                {"text": "I have to go now, goodbye John.", "start": 12.0, "end": 14.0},
                {"text": "The room is quiet after she leaves.", "start": 14.0, "end": 16.0},
            ],
            "frames": [],
            "segments": [
                {"text": "The woman walks to the door and exits.", "start": 12.0, "end": 15.0},
            ],
        }
    )
    answerer = LexicalAnswerer()
    oracle = MinimalEvidenceOracle(answerer)
    seed = example.evidence_pool

    subset = oracle.minimal_subset(example, seed)

    assert len(subset) <= len(seed)


def test_oracle_acquisition_trace_ends_with_stop() -> None:
    example = parse_tvqa_like_record(
        {
            "example_id": "toy-002",
            "video_id": "video-002",
            "question": "What does the woman say before she leaves the room?",
            "options": [
                "She says goodbye to John.",
                "She opens the window.",
            ],
            "answer_index": 0,
            "subtitles": [
                {"text": "I have to go now, goodbye John.", "start": 12.0, "end": 14.0},
                {"text": "The room is quiet after she leaves.", "start": 14.0, "end": 16.0},
            ],
            "frames": [],
            "segments": [
                {"text": "The woman walks to the door and exits.", "start": 12.0, "end": 15.0},
            ],
        }
    )
    answerer = LexicalAnswerer()
    oracle = MinimalEvidenceOracle(answerer)

    trace = oracle.acquisition_trace(example, example.evidence_pool)

    assert trace.steps
    assert trace.steps[-1].action == "stop"


def test_oracle_respects_temporal_grounding_constraint() -> None:
    example = parse_tvqa_like_record(
        {
            "example_id": "toy-003",
            "video_id": "video-003",
            "question": "What does the woman say before she leaves the room?",
            "options": [
                "She says goodbye to John.",
                "She opens the window.",
            ],
            "answer_index": 0,
            "temporal_grounding": [12.0, 14.0],
            "subtitles": [
                {"text": "Goodbye John.", "start": 12.0, "end": 14.0},
                {"text": "Goodbye John.", "start": 40.0, "end": 42.0},
            ],
            "frames": [],
            "segments": [],
        }
    )
    oracle = MinimalEvidenceOracle(
        KeywordAnswerer(),
        config=OracleConfig(min_temporal_iou=0.5),
    )

    subset = oracle.minimal_subset(example, example.subtitles)

    assert len(subset) == 1
    assert subset[0].start_time == 12.0
    assert subset[0].end_time == 14.0


def test_oracle_keeps_seed_when_correctness_constraint_is_unmet() -> None:
    example = parse_tvqa_like_record(
        {
            "example_id": "toy-004",
            "video_id": "video-004",
            "question": "What does the woman say before she leaves the room?",
            "options": [
                "She says goodbye to John.",
                "She opens the window.",
            ],
            "answer_index": 0,
            "subtitles": [
                {"text": "She opens the window.", "start": 12.0, "end": 14.0},
                {"text": "The room is quiet after she leaves.", "start": 14.0, "end": 16.0},
            ],
            "frames": [],
            "segments": [],
        }
    )
    oracle = MinimalEvidenceOracle(KeywordAnswerer())

    subset = oracle.minimal_subset(example, example.evidence_pool)

    assert subset == example.evidence_pool
    assert not oracle.seed_satisfies_constraints(example, example.evidence_pool)


def test_oracle_config_from_mode_enables_expected_constraints() -> None:
    prediction_preserving = OracleConfig.from_mode("prediction_preserving")
    correctness_only = OracleConfig.from_mode("correctness_only")
    sufficiency_mode = OracleConfig.from_mode(
        "correctness_plus_sufficiency",
        min_sufficiency=0.75,
    )
    grounded_mode = OracleConfig.from_mode(
        "correctness_plus_sufficiency_plus_grounding",
        min_sufficiency=0.75,
        min_temporal_iou=0.2,
    )

    assert not prediction_preserving.require_gold_answer
    assert prediction_preserving.min_sufficiency == 0.0
    assert correctness_only.require_gold_answer
    assert correctness_only.min_temporal_iou == 0.0
    assert sufficiency_mode.min_sufficiency == 0.75
    assert sufficiency_mode.min_temporal_iou == 0.0
    assert grounded_mode.min_sufficiency == 0.75
    assert grounded_mode.min_temporal_iou == 0.2
