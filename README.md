# Adaptive Minimal Evidence Acquisition for Grounded VideoQA

This repository is the working codebase for the project on adaptive evidence acquisition and minimal sufficient evidence for grounded VideoQA.

The repository is intentionally built as a fresh research codebase instead of forking one older paper repo wholesale. That is the right tradeoff for this project:

- the official TVQA repository is useful as a data and evaluation reference, but it is tightly coupled to its original modeling stack;
- the TVQA+ / STAGE code is useful for grounding supervision and annotations, but it uses an older environment and is not a good base for new retrieval-policy work;
- FrozenBiLM is the cleanest modern baseline reference for VideoQA training structure, but its modeling assumptions are still different from sequential evidence acquisition.

The plan is to use those projects as references and, where appropriate, port only the parts we need:

- dataset adapters and preprocessing conventions from TVQA / TVQA+;
- evaluation logic for localized or grounded VideoQA where available;
- training and experiment organization inspired by more modern baseline code.

## Research Objective

The method we are building is:

1. construct a multimodal evidence pool from subtitles, frames, and temporal segments;
2. score answer options with evidence-conditioned features;
3. approximate minimal sufficient evidence subsets offline;
4. learn a sequential acquisition policy that chooses which evidence to acquire next and when to stop.

The main scientific target is the tradeoff between:

- answer accuracy,
- evidence faithfulness,
- evidence compactness,
- acquisition cost.

## Objective And Novelty

The current project direction is:

- formulate grounded VideoQA as a sequential evidence-acquisition problem rather than a fixed-context prediction problem;
- study minimal sufficient evidence as a model-relative property rather than an absolute one;
- retrieve from three evidence sources: subtitle chunks, keyframes, and short temporal segments;
- approximate constrained minimal evidence sets offline under explicit oracle modes;
- compare weak and stronger answerers to measure how evidence size, faithfulness, and transfer change.

The main novelty claim is not ``use three modalities.''
That is already well explored in the literature.
The actual contribution we are targeting is:

- model-relative minimal evidence rather than generic top-$K$ retrieval;
- budget-aware evidence acquisition under a fixed evidence cap;
- grounded and faithful evidence selection, not only answer accuracy;
- a clear accuracy-cost-faithfulness evaluation protocol;
- transfer analysis across answerer strength, so the paper studies whether a "minimal" evidence set is stable across models.

This repository is therefore organized as a research codebase for:

1. strong fixed-budget baselines;
2. constrained oracle construction;
3. trainable sequential acquisition policies;
4. reproducible ablations and evaluation.

## Current Implementation Status

What is implemented today:

- TVQA and TVQA+ normalization into a shared JSONL schema.
- NExT-GQA normalization from official CSV annotations, grounding JSON, frame-time JSON, and video-id maps.
- Candidate-pool construction from normalized records.
- Real visual artifact materialization from source videos with `ffmpeg`.
- CLIP-based frame and segment feature extraction.
- Three retrieval modes: lexical, BM25, and hybrid CLIP retrieval.
- A fixed-budget evaluation runner with accuracy, sufficiency, comprehensiveness, temporal IoU, and oracle-validity reporting.
- A trainable linear evidence-conditioned answerer baseline.
- A stronger frozen multimodal answerer backend using CLIP text embeddings plus the existing visual feature store.
- A constrained minimal-evidence oracle with explicit modes for prediction-preserving, correctness-only, correctness-plus-sufficiency, and correctness-plus-sufficiency-plus-grounding comparisons.
- Oracle trace export for imitation learning.
- A trainable sequential acquisition policy over `acquire_subtitle`, `acquire_frame`, `acquire_segment`, and `stop`.
- A model-relative study runner that compares minimal-evidence subsets across two answerers and records subset overlap, modality agreement, temporal agreement, and transfer gaps.
- A multi-GPU HPC-oriented orchestration script for the current end-to-end pipeline.
- A focused NExT-GQA HPC runner that packages the validated single-GPU experiment flow into one script.

What has already been validated:

- unit and integration coverage over preprocessing, retrieval, visual materialization, answerer, oracle, and policy code;
- a workspace-local end-to-end dry run through normalization, candidate building, visual extraction, CLIP features, hybrid retrieval, answerer training, oracle export, policy training, and sequential evaluation;
- the current test suite passes in the Conda environment.

Dry-run outputs from the latest local validation are under:

- `outputs/dry_run/normalized`
- `outputs/dry_run/candidates`
- `outputs/dry_run/features`
- `outputs/dry_run/models`
- `outputs/dry_run/outputs`

Important limitations of the current codebase:

- the new frozen multimodal answerer is stronger than the linear baseline, but it is still a frozen baseline rather than a competitive fine-tuned VLM;
- the policy currently chooses modality-level acquisition actions, not individual item reranking within a modality;
- the local dry run was on synthetic TVQA-shaped data and is only a plumbing validation, not a scientific result;
- the final paper experiments still need real TVQA / TVQA+ runs, multi-seed evaluation, and ablations;
- the repository now has NExT-GQA preprocessing support, but that benchmark has not yet been run end to end on real data.

## Repository Layout

```text
configs/                    Experiment configuration templates
docs/                       Proposal, report, and course material
scripts/                    Entry-point scripts for preprocessing and training
src/adaptive_evidence_vqa/  Python package
tests/                      Unit tests for core logic
```

Within the package:

```text
data/       Dataset schemas and adapters
retrieval/  Evidence pool construction and candidate retrieval interfaces
models/     Answerer, oracle, and acquisition policy modules
eval/       Accuracy, grounding, and sufficiency metrics
```

The first concrete dataset support is now wired for:

- raw `TVQA` question JSONL + subtitle JSONL;
- raw `TVQA+` question JSON + subtitle JSON.
- raw `NExT-GQA` CSV plus optional `gsub_*.json`, `frame2time_*.json` / `upbd_*.json`, and `map_vid_vidorID.json`.

Both are normalized into a common JSONL schema before training.
After normalization, a separate candidate-pool stage generates subtitle chunks, segment windows,
and frame timestamps for retrieval and oracle construction.

## Quick Start

Use the project Conda environment instead of the global interpreter:

```bash
conda env create -f environment.yml
conda activate adaptive-evidence-vqa
```

If the environment already exists, update it in place:

```bash
conda env update -f environment.yml --prune
conda activate adaptive-evidence-vqa
```

Run the toy pipeline:

```bash
python -m adaptive_evidence_vqa toy-run
```

Print the default configuration:

```bash
python -m adaptive_evidence_vqa print-config
```

Normalize raw TVQA annotations:

```bash
python scripts/prepare_tvqa.py \
  --qa-path /path/to/tvqa_train.jsonl \
  --subtitles-path /path/to/tvqa_preprocessed_subtitles.jsonl \
  --output-path data/normalized/tvqa_train.jsonl
```

Normalize raw TVQA+ annotations:

```bash
python scripts/prepare_tvqa_plus.py \
  --qa-path /path/to/tvqa_plus_train_preprocessed.json \
  --subtitles-path /path/to/tvqa_plus_subtitles.json \
  --output-path data/normalized/tvqa_plus_train.jsonl
```

Normalize raw NExT-GQA annotations:

```bash
python scripts/prepare_nextgqa.py \
  --qa-path /path/to/nextgqa/val.csv \
  --gsub-path /path/to/nextgqa/gsub_val.json \
  --frame-times-path /path/to/nextgqa/frame2time_val.json \
  --video-map-path /path/to/nextgqa/map_vid_vidorID.json \
  --output-path data/normalized/nextgqa_val.jsonl
```

For training splits, add `--video-root /path/to/videos` so the script can probe full-video duration when no
grounded span or frame-time file is available:

```bash
python scripts/prepare_nextgqa.py \
  --qa-path /path/to/nextgqa/train.csv \
  --video-map-path /path/to/nextgqa/map_vid_vidorID.json \
  --video-root /path/to/videos \
  --output-path data/normalized/nextgqa_train.jsonl
```

Build a candidate pool from normalized examples:

```bash
python scripts/build_candidate_pool.py \
  --input-path data/normalized/tvqa_train.jsonl \
  --output-path data/candidates/tvqa_train_candidates.jsonl
```

Materialize real frame artifacts and optional segment clips from source videos:

```bash
python scripts/materialize_visual_evidence.py \
  --input-path data/candidates/tvqa_train_candidates.jsonl \
  --video-root /path/to/tvqa_videos \
  --output-path data/candidates/tvqa_train_visual.jsonl \
  --frames-dir data/artifacts/frames \
  --segments-dir data/artifacts/segments \
  --extract-segments
```

Extract CLIP features for materialized frame and segment evidence:

```bash
python scripts/extract_clip_features.py \
  --input-path data/candidates/tvqa_train_visual.jsonl \
  --output-path data/candidates/tvqa_train_visual_features.jsonl \
  --feature-dir data/features/clip
```

Run the validated end-to-end NExT-GQA experiment flow on an HPC machine:

```bash
bash scripts/run_nextgqa_experiment.sh \
  --data-root /path/to/nextgqa \
  --video-root /path/to/nextgqa/videos \
  --run-root /path/to/output/run \
  --conda-prefix /path/to/conda-env \
  --train-limit 500 \
  --val-limit 200
```

Aggregate multiple seeded runs into a paper-ready mean/std table:

```bash
python scripts/aggregate_run_summaries.py \
  --run-roots runs/nextgqa_500_200_seed13 runs/nextgqa_500_200_seed21 runs/nextgqa_500_200_seed34 \
  --output-json runs/nextgqa_500_200_aggregate.json \
  --output-markdown runs/nextgqa_500_200_aggregate.md
```

Run a fixed-allocation baseline on candidate pools:

```bash
python scripts/run_fixed_budget_baseline.py \
  --input-path data/candidates/tvqa_train_candidates.jsonl \
  --summary-output outputs/fixed_budget_summary.json \
  --predictions-output outputs/fixed_budget_predictions.jsonl \
  --answerer lexical \
  --retriever bm25 \
  --subtitle-k 2 \
  --frame-k 2 \
  --segment-k 2
```

Train the fixed-budget linear answerer baseline:

```bash
python scripts/train_answerer.py \
  --train-path data/candidates/tvqa_train_candidates.jsonl \
  --validation-path data/candidates/tvqa_val_candidates.jsonl \
  --model-dir outputs/answerer_linear_bm25 \
  --retriever bm25 \
  --subtitle-k 2 \
  --frame-k 2 \
  --segment-k 2
```

Evaluate the trained answerer with the same baseline runner:

```bash
python scripts/run_fixed_budget_baseline.py \
  --input-path data/candidates/tvqa_val_candidates.jsonl \
  --summary-output outputs/fixed_budget_linear_summary.json \
  --answerer linear \
  --answerer-model-dir outputs/answerer_linear_bm25 \
  --retriever bm25 \
  --subtitle-k 2 \
  --frame-k 2 \
  --segment-k 2
```

Export oracle acquisition traces from a retrieved seed set:

```bash
python scripts/export_oracle_traces.py \
  --input-path data/candidates/tvqa_train_candidates.jsonl \
  --output-path outputs/oracle_traces.jsonl \
  --answerer linear \
  --answerer-model-dir outputs/answerer_linear_bm25 \
  --retriever bm25 \
  --oracle-mode correctness_plus_sufficiency_plus_grounding \
  --oracle-min-sufficiency 0.8 \
  --oracle-min-temporal-iou 0.1 \
  --subtitle-k 2 \
  --frame-k 2 \
  --segment-k 2
```

By default, invalid oracle traces are skipped. Use `--include-invalid-traces` only for debugging, not
for the main imitation-learning experiments.

Compare model-relative minimal evidence across two answerers:

```bash
python scripts/run_model_relative_study.py \
  --input-path data/candidates/tvqa_val_visual_features.jsonl \
  --output-dir outputs/model_relative_linear_vs_frozen \
  --answerer-a linear \
  --answerer-a-model-dir outputs/answerer_linear_bm25 \
  --answerer-a-label linear \
  --answerer-b frozen_multimodal \
  --answerer-b-label frozen_multimodal \
  --retriever hybrid_clip \
  --oracle-mode correctness_plus_sufficiency_plus_grounding \
  --oracle-min-sufficiency 0.8 \
  --oracle-min-temporal-iou 0.1 \
  --subtitle-k 2 \
  --frame-k 2 \
  --segment-k 2
```

Train the sequential imitation policy from oracle traces:

```bash
python scripts/train_policy.py \
  --train-traces-path outputs/oracle_traces.jsonl \
  --validation-traces-path outputs/oracle_val_traces.jsonl \
  --model-dir outputs/policy_linear \
  --answerer linear \
  --answerer-model-dir outputs/answerer_linear_bm25
```

Evaluate the learned sequential policy on candidate pools:

```bash
python scripts/run_sequential_policy.py \
  --input-path data/candidates/tvqa_val_candidates.jsonl \
  --summary-output outputs/sequential_policy_summary.json \
  --answerer linear \
  --answerer-model-dir outputs/answerer_linear_bm25 \
  --policy linear \
  --policy-model-dir outputs/policy_linear \
  --retriever bm25 \
  --subtitle-k 2 \
  --frame-k 2 \
  --segment-k 2 \
  --max-items 6
```

For an HPC-style end-to-end run across multiple GPUs, use:

```bash
bash scripts/run_multigpu_pipeline.sh \
  --dataset tvqa \
  --train-qa /path/to/tvqa_train.jsonl \
  --train-subtitles /path/to/tvqa_preprocessed_subtitles.jsonl \
  --val-qa /path/to/tvqa_val.jsonl \
  --val-subtitles /path/to/tvqa_preprocessed_subtitles.jsonl \
  --video-root /path/to/tvqa_videos \
  --run-root /path/to/experiment_run \
  --gpus 0,1
```

This runner parallelizes the CLIP-heavy stages across GPUs and keeps the current linear answerer/policy stages on CPU.

Run the test suite from the Conda environment:

```bash
pytest
```

## Recommended Reading Order For New Contributors

For someone joining the project midstream, the fastest way to get oriented is:

1. read this README for the research objective, novelty, and current implementation status;
2. read [docs/report/main.tex](/Users/ritesh.thawkar/Desktop/Masters_study/SEM_2/Deep_Learning/Project/docs/report/main.tex) for the literature review and research positioning;
3. inspect [configs/experiment.template.yaml](/Users/ritesh.thawkar/Desktop/Masters_study/SEM_2/Deep_Learning/Project/configs/experiment.template.yaml) for the current experiment assumptions;
4. inspect `scripts/` in pipeline order: preprocessing, candidate building, visual extraction, answerer training, oracle export, policy training, and evaluation;
5. inspect `outputs/dry_run/` for a fully executed miniature run.

## External References

These are references, not vendored dependencies:

- TVQA official repository: <https://github.com/jayleicn/TVQA>
- TVQA+ / STAGE official repository: <https://github.com/jayleicn/TVQAplus>
- FrozenBiLM VideoQA repository: <https://github.com/antoyang/FrozenBiLM>

## Immediate Next Implementation Steps

1. run the end-to-end pipeline on real TVQA / TVQA+ train and validation splits instead of smoke examples;
2. benchmark the CLIP-backed hybrid retriever on real materialized visual evidence and compare it fairly against sparse baselines;
3. add a second, clearly stronger answerer tier beyond the current frozen multimodal baseline;
4. run the new NExT-GQA preprocessing path on real data and validate the benchmark end to end;
5. add ablations for oracle modes, stop behavior, and action-level imitation accuracy;
6. benchmark robustness under noisy subtitles, missing modalities, and tighter acquisition budgets;
7. prepare paper-ready tables and figures for accuracy, cost, faithfulness, grounding, and transfer tradeoffs.
