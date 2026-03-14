from scripts.train_policy import parse_trace_record

from adaptive_evidence_vqa.models.policy import (
    PolicyTrainingState,
    SequentialPolicyConfig,
    TrainableSequentialPolicy,
    action_mask,
)
from adaptive_evidence_vqa.schemas import (
    AcquisitionTrace,
    AnswerOption,
    EvidenceItem,
    Modality,
    ModelPrediction,
    QuestionExample,
)


class StubAnswerer:
    def predict(self, example, evidence):
        joined = " ".join(item.text.lower() for item in evidence)
        if "goodbye" in joined:
            predicted_index = 0
            scores = (3.0, 1.0)
            confidence = 0.9
        elif "blue" in joined:
            predicted_index = 0
            scores = (3.0, 1.0)
            confidence = 0.9
        else:
            predicted_index = 1
            scores = (1.0, 3.0)
            confidence = 0.6
        return ModelPrediction(
            predicted_index=predicted_index,
            option_scores=scores,
            confidence=confidence,
            supporting_evidence=evidence,
        )


def make_example(example_id: str, question: str, options: tuple[str, str], subtitles, frames):
    return QuestionExample(
        example_id=example_id,
        video_id=example_id,
        question=question,
        options=tuple(AnswerOption(index=index, text=text) for index, text in enumerate(options)),
        answer_index=0,
        subtitles=tuple(subtitles),
        frames=tuple(frames),
        segments=(),
    )


def test_trainable_policy_learns_modality_actions() -> None:
    subtitle_goodbye = EvidenceItem(
        evidence_id="sub-1",
        modality=Modality.SUBTITLE,
        text="Goodbye John.",
        start_time=1.0,
        end_time=2.0,
        retrieval_score=2.0,
    )
    subtitle_irrelevant = EvidenceItem(
        evidence_id="sub-2",
        modality=Modality.SUBTITLE,
        text="The room is quiet.",
        start_time=2.0,
        end_time=3.0,
        retrieval_score=1.0,
    )
    frame_blue = EvidenceItem(
        evidence_id="frame-1",
        modality=Modality.FRAME,
        text="A blue shirt is visible.",
        start_time=1.5,
        end_time=1.5,
        retrieval_score=2.0,
    )

    example_subtitle = make_example(
        "ex-1",
        "Who says goodbye to John?",
        ("The woman says goodbye to John.", "She opens the window."),
        subtitles=(subtitle_goodbye, subtitle_irrelevant),
        frames=(frame_blue,),
    )
    example_frame = make_example(
        "ex-2",
        "What color is the shirt?",
        ("The shirt is blue.", "The shirt is red."),
        subtitles=(subtitle_irrelevant,),
        frames=(frame_blue,),
    )

    train_states = [
        PolicyTrainingState(
            example=example_subtitle,
            acquired=(),
            remaining_subtitles=example_subtitle.subtitles,
            remaining_frames=example_subtitle.frames,
            remaining_segments=(),
            gold_action="acquire_subtitle",
            step_index=0,
            max_steps=2,
        ),
        PolicyTrainingState(
            example=example_subtitle,
            acquired=(subtitle_goodbye,),
            remaining_subtitles=(subtitle_irrelevant,),
            remaining_frames=example_subtitle.frames,
            remaining_segments=(),
            gold_action="stop",
            step_index=1,
            max_steps=2,
        ),
        PolicyTrainingState(
            example=example_frame,
            acquired=(),
            remaining_subtitles=example_frame.subtitles,
            remaining_frames=example_frame.frames,
            remaining_segments=(),
            gold_action="acquire_frame",
            step_index=0,
            max_steps=2,
        ),
        PolicyTrainingState(
            example=example_frame,
            acquired=(frame_blue,),
            remaining_subtitles=example_frame.subtitles,
            remaining_frames=(),
            remaining_segments=(),
            gold_action="stop",
            step_index=1,
            max_steps=2,
        ),
    ]

    model = TrainableSequentialPolicy.fit(
        train_states=train_states,
        validation_states=None,
        answerer=StubAnswerer(),
        config=SequentialPolicyConfig(
            text_feature_dim=512,
            epochs=50,
            batch_size=2,
            learning_rate=0.5,
            weight_decay=0.0,
            patience=10,
            seed=7,
        ),
    )

    subtitle_trace = model.run(
        example_subtitle,
        candidate_pool={
            "subtitle": example_subtitle.subtitles,
            "frame": example_subtitle.frames,
            "segment": (),
        },
        max_items=2,
    )
    frame_trace = model.run(
        example_frame,
        candidate_pool={
            "subtitle": example_frame.subtitles,
            "frame": example_frame.frames,
            "segment": (),
        },
        max_items=2,
    )

    assert isinstance(subtitle_trace, AcquisitionTrace)
    assert subtitle_trace.steps[0].action == "acquire_subtitle"
    assert subtitle_trace.steps[-1].action == "stop"
    assert frame_trace.steps[0].action == "acquire_frame"
    assert frame_trace.steps[-1].action == "stop"


def test_parse_trace_record_preserves_visual_metadata() -> None:
    example, seed_evidence, trace_steps = parse_trace_record(
        {
            "example_id": "ex-visual",
            "video_id": "video-visual",
            "question": "What color is the shirt?",
            "options": ["Blue", "Red"],
            "answer_index": 0,
            "temporal_grounding": [1.0, 2.0],
            "seed_evidence": [
                {
                    "evidence_id": "frame-1",
                    "modality": "frame",
                    "text": "A blue shirt is visible.",
                    "start_time": 1.5,
                    "end_time": 1.5,
                    "source_path": "/tmp/frame-1.jpg",
                    "retrieval_score": 1.7,
                    "acquisition_cost": 1.0,
                    "metadata": {
                        "visual_feature_path": "/tmp/features.npz",
                        "feature_index": 0,
                    },
                }
            ],
            "trace": [
                {
                    "step_index": 0,
                    "action": "acquire_frame",
                    "selected_evidence_id": "frame-1",
                    "confidence_after_step": 0.9,
                },
                {
                    "step_index": 1,
                    "action": "stop",
                    "selected_evidence_id": None,
                    "confidence_after_step": 0.95,
                },
            ],
        }
    )

    assert example.temporal_grounding == (1.0, 2.0)
    assert seed_evidence[0].source_path == "/tmp/frame-1.jpg"
    assert seed_evidence[0].metadata["visual_feature_path"] == "/tmp/features.npz"
    assert trace_steps[0]["action"] == "acquire_frame"


def test_action_mask_blocks_stop_until_minimum_items_are_acquired() -> None:
    frame_item = EvidenceItem(
        evidence_id="frame-1",
        modality=Modality.FRAME,
        text="A blue shirt is visible.",
        start_time=1.5,
        end_time=1.5,
        retrieval_score=2.0,
    )

    blocked_mask = action_mask(
        {"subtitle": (), "frame": (frame_item,), "segment": ()},
        acquired_count=0,
        min_items_before_stop=1,
    )
    allowed_mask = action_mask(
        {"subtitle": (), "frame": (frame_item,), "segment": ()},
        acquired_count=1,
        min_items_before_stop=1,
    )

    assert blocked_mask.sum() == 1.0
    assert blocked_mask[-1] == 0.0
    assert allowed_mask[-1] == 1.0


def test_trainable_policy_filters_invalid_stop_only_states() -> None:
    frame_item = EvidenceItem(
        evidence_id="frame-1",
        modality=Modality.FRAME,
        text="A blue shirt is visible.",
        start_time=1.5,
        end_time=1.5,
        retrieval_score=2.0,
    )
    example = make_example(
        "ex-stop-filter",
        "What color is the shirt?",
        ("The shirt is blue.", "The shirt is red."),
        subtitles=(),
        frames=(frame_item,),
    )

    train_states = [
        PolicyTrainingState(
            example=example,
            acquired=(),
            remaining_subtitles=(),
            remaining_frames=example.frames,
            remaining_segments=(),
            gold_action="stop",
            step_index=0,
            max_steps=2,
        ),
        PolicyTrainingState(
            example=example,
            acquired=(),
            remaining_subtitles=(),
            remaining_frames=example.frames,
            remaining_segments=(),
            gold_action="acquire_frame",
            step_index=0,
            max_steps=2,
        ),
        PolicyTrainingState(
            example=example,
            acquired=(frame_item,),
            remaining_subtitles=(),
            remaining_frames=(),
            remaining_segments=(),
            gold_action="stop",
            step_index=1,
            max_steps=2,
        ),
    ]

    model = TrainableSequentialPolicy.fit(
        train_states=train_states,
        validation_states=None,
        answerer=StubAnswerer(),
        config=SequentialPolicyConfig(
            text_feature_dim=256,
            epochs=10,
            batch_size=2,
            learning_rate=0.5,
            weight_decay=0.0,
            patience=5,
            seed=3,
            min_items_before_stop=1,
        ),
    )

    trace = model.run(
        example,
        candidate_pool={"subtitle": (), "frame": example.frames, "segment": ()},
        max_items=2,
    )

    assert trace.steps[0].action == "acquire_frame"
