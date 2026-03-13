import argparse
import json

from adaptive_evidence_vqa.config import ProjectConfig
from adaptive_evidence_vqa.data.tvqa import parse_tvqa_like_record
from adaptive_evidence_vqa.models.answerer import LexicalAnswerer
from adaptive_evidence_vqa.models.oracle import MinimalEvidenceOracle
from adaptive_evidence_vqa.models.policy import KeywordSequentialPolicy
from adaptive_evidence_vqa.retrieval.base import CandidatePoolBuilder, LexicalRetriever


def toy_record() -> dict:
    return {
        "example_id": "toy-001",
        "video_id": "video-001",
        "question": "What does the woman say before she leaves the room?",
        "options": [
            "She says goodbye to John.",
            "She opens the window.",
            "She sits on the sofa.",
            "She turns off the television.",
            "She starts cooking dinner.",
        ],
        "answer_index": 0,
        "subtitles": [
            {"text": "I have to go now, goodbye John.", "start": 12.0, "end": 14.0},
            {"text": "The room is quiet after she leaves.", "start": 14.0, "end": 16.0},
        ],
        "frames": [
            {"text": "A woman stands near the door.", "time": 13.0},
            {"text": "A sofa is visible in the room.", "time": 10.0},
        ],
        "segments": [
            {"text": "The woman walks to the door and exits.", "start": 12.0, "end": 15.0},
            {"text": "A man remains seated after she leaves.", "start": 15.0, "end": 18.0},
        ],
    }


def cmd_print_config() -> None:
    config = ProjectConfig()
    print(json.dumps(config.to_dict(), indent=2, sort_keys=True))


def cmd_toy_run() -> None:
    example = parse_tvqa_like_record(toy_record())
    retriever = LexicalRetriever()
    pool_builder = CandidatePoolBuilder(retriever)
    answerer = LexicalAnswerer()
    oracle = MinimalEvidenceOracle(answerer)
    policy = KeywordSequentialPolicy(answerer)

    candidate_pool = pool_builder.build(example, top_k_per_modality=2)
    flat_seed = candidate_pool["subtitle"] + candidate_pool["frame"] + candidate_pool["segment"]
    oracle_subset = oracle.minimal_subset(example, flat_seed)
    trace = policy.run(example, candidate_pool, max_items=4)

    print("Predicted option:", trace.final_prediction.predicted_index)
    print("Confidence:", round(trace.final_prediction.confidence, 4))
    print("Oracle subset size:", len(oracle_subset))
    print("Acquisition trace:")
    for step in trace.steps:
        evidence_id = step.selected_item.evidence_id if step.selected_item else "-"
        print(f"  step={step.step_index} action={step.action} evidence={evidence_id} conf={step.confidence_after_step:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive evidence acquisition for grounded VideoQA.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("print-config", help="Print the default project configuration.")
    subparsers.add_parser("toy-run", help="Run the toy end-to-end retrieval, oracle, and policy pipeline.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "print-config":
        cmd_print_config()
        return
    if args.command == "toy-run":
        cmd_toy_run()
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
