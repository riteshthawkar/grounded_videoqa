from dataclasses import dataclass

from adaptive_evidence_vqa.eval.metrics import (
    max_temporal_iou_for_target_spans,
    sufficiency,
    temporal_target_spans,
)
from adaptive_evidence_vqa.models.answerer import Answerer
from adaptive_evidence_vqa.schemas import (
    AcquisitionStep,
    AcquisitionTrace,
    EvidenceItem,
    ModelPrediction,
    QuestionExample,
)

ORACLE_MODES = (
    "prediction_preserving",
    "correctness_only",
    "correctness_plus_sufficiency",
    "correctness_plus_sufficiency_plus_grounding",
)


@dataclass(frozen=True, slots=True)
class OracleConfig:
    mode: str = "custom"
    min_sufficiency: float = 0.0
    min_temporal_iou: float = 0.0
    require_gold_answer: bool = True

    @classmethod
    def from_mode(
        cls,
        mode: str,
        *,
        min_sufficiency: float = 0.8,
        min_temporal_iou: float = 0.1,
    ) -> "OracleConfig":
        if mode not in ORACLE_MODES:
            raise ValueError(f"Unsupported oracle mode: {mode}")

        require_gold_answer = mode != "prediction_preserving"
        resolved_min_sufficiency = 0.0
        resolved_min_temporal_iou = 0.0

        if mode in {"correctness_plus_sufficiency", "correctness_plus_sufficiency_plus_grounding"}:
            resolved_min_sufficiency = min_sufficiency
        if mode == "correctness_plus_sufficiency_plus_grounding":
            resolved_min_temporal_iou = min_temporal_iou

        return cls(
            mode=mode,
            min_sufficiency=resolved_min_sufficiency,
            min_temporal_iou=resolved_min_temporal_iou,
            require_gold_answer=require_gold_answer,
        )

    def to_dict(self) -> dict[str, float | bool | str]:
        return {
            "mode": self.mode,
            "min_sufficiency": self.min_sufficiency,
            "min_temporal_iou": self.min_temporal_iou,
            "require_gold_answer": self.require_gold_answer,
        }


class MinimalEvidenceOracle:
    """Greedy backward elimination over a candidate set.

    This approximates a minimal sufficient evidence subset by repeatedly
    removing items whose deletion still preserves correctness and any
    configured faithfulness constraints.

    Seed evidence is sorted by ``evidence_id`` before elimination so that
    the resulting minimal subset is deterministic regardless of the order
    in which the retriever returned items.
    """

    # Relative weight for the correctness term in the forward-selection
    # score used to order acquisition traces.  The value must be large
    # enough to dominate sufficiency (≤1), temporal IoU (≤1), and
    # confidence (≤1) so that preserving the target answer is always
    # preferred.  Exposed as a class attribute so callers can override it
    # if needed.
    correctness_weight: float = 4.0

    def __init__(
        self,
        answerer: Answerer,
        config: OracleConfig | None = None,
    ) -> None:
        self.answerer = answerer
        self.config = config or OracleConfig()

    def predict(
        self,
        example: QuestionExample,
        evidence: tuple[EvidenceItem, ...],
    ) -> ModelPrediction:
        return self.answerer.predict(example, evidence)

    def _target_index(
        self,
        example: QuestionExample,
        reference_prediction: ModelPrediction,
    ) -> int:
        if self.config.require_gold_answer and example.answer_index is not None:
            return example.answer_index
        return reference_prediction.predicted_index

    def _meets_constraints(
        self,
        example: QuestionExample,
        evidence: tuple[EvidenceItem, ...],
        reference_prediction: ModelPrediction,
        candidate_prediction: ModelPrediction,
    ) -> bool:
        if candidate_prediction.predicted_index != self._target_index(example, reference_prediction):
            return False

        if self.config.min_sufficiency > 0.0 and example.answer_index is not None:
            candidate_sufficiency = sufficiency(reference_prediction, candidate_prediction, example.answer_index)
            if candidate_sufficiency < self.config.min_sufficiency:
                return False

        if self.config.min_temporal_iou > 0.0 and example.temporal_grounding is not None:
            temporal_iou = max_temporal_iou_for_target_spans(
                evidence,
                temporal_target_spans(example.temporal_grounding, example.metadata),
            )
            if temporal_iou < self.config.min_temporal_iou:
                return False

        return True

    def seed_satisfies_constraints(
        self,
        example: QuestionExample,
        evidence: tuple[EvidenceItem, ...],
    ) -> bool:
        prediction = self.predict(example, evidence)
        return self._meets_constraints(
            example=example,
            evidence=evidence,
            reference_prediction=prediction,
            candidate_prediction=prediction,
        )

    def _forward_score(
        self,
        example: QuestionExample,
        evidence: tuple[EvidenceItem, ...],
        reference_prediction: ModelPrediction,
    ) -> tuple[float, ModelPrediction]:
        prediction = self.predict(example, evidence)
        score = 0.0

        if prediction.predicted_index == self._target_index(example, reference_prediction):
            score += self.correctness_weight

        if example.answer_index is not None:
            score += sufficiency(reference_prediction, prediction, example.answer_index)

        if example.temporal_grounding is not None:
            score += max_temporal_iou_for_target_spans(
                evidence,
                temporal_target_spans(example.temporal_grounding, example.metadata),
            )

        score += prediction.confidence
        return score, prediction

    def minimal_subset(
        self,
        example: QuestionExample,
        seed_evidence: tuple[EvidenceItem, ...],
    ) -> tuple[EvidenceItem, ...]:
        current = sorted(seed_evidence, key=lambda item: item.evidence_id)
        if not current:
            return ()

        reference = self.predict(example, tuple(current))
        if not self._meets_constraints(example, tuple(current), reference, reference):
            return tuple(current)

        changed = True
        while changed and current:
            changed = False
            for index in range(len(current)):
                candidate = current[:index] + current[index + 1 :]
                prediction = self.predict(example, tuple(candidate))
                if self._meets_constraints(example, tuple(candidate), reference, prediction):
                    current = candidate
                    changed = True
                    break
        return tuple(current)

    def acquisition_trace(
        self,
        example: QuestionExample,
        seed_evidence: tuple[EvidenceItem, ...],
    ) -> AcquisitionTrace:
        subset = self.minimal_subset(example, seed_evidence)
        reference = self.predict(example, seed_evidence)
        acquired: list[EvidenceItem] = []
        steps: list[AcquisitionStep] = []
        remaining = list(subset)
        while remaining:
            best_index = 0
            best_prediction: ModelPrediction | None = None
            best_score = float("-inf")

            for index, item in enumerate(remaining):
                candidate_evidence = tuple(acquired + [item])
                score, prediction = self._forward_score(example, candidate_evidence, reference)
                if score > best_score:
                    best_score = score
                    best_index = index
                    best_prediction = prediction

            item = remaining.pop(best_index)
            acquired.append(item)
            prediction = best_prediction if best_prediction is not None else self.predict(example, tuple(acquired))
            steps.append(
                AcquisitionStep(
                    step_index=len(steps),
                    action=f"acquire_{item.modality.value}",
                    selected_item=item,
                    confidence_after_step=prediction.confidence,
                )
            )

        final_prediction = self.predict(example, tuple(acquired))
        steps.append(
            AcquisitionStep(
                step_index=len(steps),
                action="stop",
                selected_item=None,
                confidence_after_step=final_prediction.confidence,
            )
        )
        return AcquisitionTrace(steps=tuple(steps), final_prediction=final_prediction)
