from dataclasses import dataclass, field
from enum import Enum


class Modality(str, Enum):
    SUBTITLE = "subtitle"
    FRAME = "frame"
    SEGMENT = "segment"


@dataclass(frozen=True, slots=True)
class AnswerOption:
    index: int
    text: str


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    evidence_id: str
    modality: Modality
    text: str
    start_time: float | None = None
    end_time: float | None = None
    source_path: str | None = None
    retrieval_score: float = 0.0
    acquisition_cost: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QuestionExample:
    example_id: str
    video_id: str
    question: str
    options: tuple[AnswerOption, ...]
    answer_index: int | None = None
    temporal_grounding: tuple[float, float] | None = None
    subtitles: tuple[EvidenceItem, ...] = ()
    frames: tuple[EvidenceItem, ...] = ()
    segments: tuple[EvidenceItem, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def evidence_pool(self) -> tuple[EvidenceItem, ...]:
        return self.subtitles + self.frames + self.segments


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    predicted_index: int
    option_scores: tuple[float, ...]
    confidence: float
    supporting_evidence: tuple[EvidenceItem, ...] = ()


@dataclass(frozen=True, slots=True)
class AcquisitionStep:
    step_index: int
    action: str
    selected_item: EvidenceItem | None
    confidence_after_step: float


@dataclass(frozen=True, slots=True)
class AcquisitionTrace:
    steps: tuple[AcquisitionStep, ...]
    final_prediction: ModelPrediction
