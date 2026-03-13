from pathlib import Path

from adaptive_evidence_vqa.data.base import load_jsonl
from adaptive_evidence_vqa.data.tvqa import parse_tvqa_like_record
from adaptive_evidence_vqa.schemas import QuestionExample


def load_normalized_examples(path: str | Path) -> list[QuestionExample]:
    return [parse_tvqa_like_record(record) for record in load_jsonl(path)]
