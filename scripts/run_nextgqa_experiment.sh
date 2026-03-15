#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_nextgqa_experiment.sh \
    --data-root /path/to/nextgqa \
    --video-root /path/to/nextgqa/videos \
    --run-root /path/to/output/run \
    --conda-prefix /path/to/conda-env

Required:
  --data-root           Directory containing NExT-GQA annotations.
  --video-root          Directory containing NExT-GQA videos.
  --run-root            Output directory for this experiment run.

Environment activation:
  --conda-prefix        Conda environment path to activate.
  --conda-env           Conda environment name to activate.
  If neither is set, the script assumes the correct environment is already active.

Optional:
  --train-limit         Number of train examples to normalize (default: 500)
  --val-limit           Number of val examples to normalize (default: 200)
  --seed                Random seed for policy training (default: 13)
  --device              Device for CLIP-backed stages (default: cuda)
  --feature-batch-size  CLIP feature batch size (default: 32)
  --model-name          CLIP model name (default: openai/clip-vit-base-patch32)
  --subtitle-k          Retrieved subtitle items (default: 0)
  --frame-k             Retrieved frame items (default: 3)
  --segment-k           Retrieved segment items (default: 3)
  --oracle-mode         Oracle mode (default: correctness_plus_sufficiency)
  --oracle-min-sufficiency  Minimum sufficiency threshold (default: 0.8)
  --max-items           Policy acquisition cap (default: 6)
  --min-items-before-stop  Minimum items before stop is allowed (default: 1)

Expected files under --data-root:
  train.csv
  val.csv
  map_vid_vidorID.json
  gsub_val.json
  frame2time_val.json
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Required file not found: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Required directory not found: $path" >&2
    exit 1
  fi
}

activate_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not available on PATH." >&2
    exit 1
  fi
  eval "$(conda shell.bash hook)"
  if [[ -n "$CONDA_PREFIX_PATH" ]]; then
    conda activate "$CONDA_PREFIX_PATH"
  elif [[ -n "$CONDA_ENV_NAME" ]]; then
    conda activate "$CONDA_ENV_NAME"
  fi
}

run_logged() {
  local log_path="$1"
  shift
  log "Running: $*"
  "$@" >"$log_path" 2>&1
}

append_limit_arg() {
  local limit="$1"
  if [[ -n "$limit" ]]; then
    printf -- "--limit=%s" "$limit"
  fi
}

DATA_ROOT=""
VIDEO_ROOT=""
RUN_ROOT=""
CONDA_PREFIX_PATH=""
CONDA_ENV_NAME=""
TRAIN_LIMIT="500"
VAL_LIMIT="200"
SEED="13"
DEVICE="cuda"
FEATURE_BATCH_SIZE="32"
MODEL_NAME="openai/clip-vit-base-patch32"
SUBTITLE_K="0"
FRAME_K="3"
SEGMENT_K="3"
ORACLE_MODE="correctness_plus_sufficiency"
ORACLE_MIN_SUFFICIENCY="0.8"
MAX_ITEMS="6"
MIN_ITEMS_BEFORE_STOP="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --video-root)
      VIDEO_ROOT="$2"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --conda-prefix)
      CONDA_PREFIX_PATH="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV_NAME="$2"
      shift 2
      ;;
    --train-limit)
      TRAIN_LIMIT="$2"
      shift 2
      ;;
    --val-limit)
      VAL_LIMIT="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --feature-batch-size)
      FEATURE_BATCH_SIZE="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --subtitle-k)
      SUBTITLE_K="$2"
      shift 2
      ;;
    --frame-k)
      FRAME_K="$2"
      shift 2
      ;;
    --segment-k)
      SEGMENT_K="$2"
      shift 2
      ;;
    --oracle-mode)
      ORACLE_MODE="$2"
      shift 2
      ;;
    --oracle-min-sufficiency)
      ORACLE_MIN_SUFFICIENCY="$2"
      shift 2
      ;;
    --max-items)
      MAX_ITEMS="$2"
      shift 2
      ;;
    --min-items-before-stop)
      MIN_ITEMS_BEFORE_STOP="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$DATA_ROOT" || -z "$VIDEO_ROOT" || -z "$RUN_ROOT" ]]; then
  usage
  exit 1
fi

require_dir "$DATA_ROOT"
require_dir "$VIDEO_ROOT"
require_file "$DATA_ROOT/train.csv"
require_file "$DATA_ROOT/val.csv"
require_file "$DATA_ROOT/map_vid_vidorID.json"
require_file "$DATA_ROOT/gsub_val.json"
require_file "$DATA_ROOT/frame2time_val.json"

if [[ -n "$CONDA_PREFIX_PATH" ]]; then
  require_dir "$CONDA_PREFIX_PATH"
fi

activate_conda

mkdir -p "$RUN_ROOT"/{normalized,candidates,artifacts/frames,artifacts/segments,features/clip,models,outputs,logs}

TRAIN_LIMIT_ARG="$(append_limit_arg "$TRAIN_LIMIT")"
VAL_LIMIT_ARG="$(append_limit_arg "$VAL_LIMIT")"

log "Preparing NExT-GQA train split"
run_logged \
  "$RUN_ROOT/logs/train.prepare.log" \
  python scripts/prepare_nextgqa.py \
  --qa-path "$DATA_ROOT/train.csv" \
  --video-map-path "$DATA_ROOT/map_vid_vidorID.json" \
  --video-root "$VIDEO_ROOT" \
  --output-path "$RUN_ROOT/normalized/train.jsonl" \
  ${TRAIN_LIMIT_ARG:+$TRAIN_LIMIT_ARG}

log "Preparing NExT-GQA val split"
run_logged \
  "$RUN_ROOT/logs/val.prepare.log" \
  python scripts/prepare_nextgqa.py \
  --qa-path "$DATA_ROOT/val.csv" \
  --gsub-path "$DATA_ROOT/gsub_val.json" \
  --frame-times-path "$DATA_ROOT/frame2time_val.json" \
  --video-map-path "$DATA_ROOT/map_vid_vidorID.json" \
  --output-path "$RUN_ROOT/normalized/val.jsonl" \
  ${VAL_LIMIT_ARG:+$VAL_LIMIT_ARG}

for split in train val; do
  log "Building candidate pool for $split"
  run_logged \
    "$RUN_ROOT/logs/${split}.candidate_pool.log" \
    python scripts/build_candidate_pool.py \
    --input-path "$RUN_ROOT/normalized/${split}.jsonl" \
    --output-path "$RUN_ROOT/candidates/${split}.jsonl"

  log "Materializing visual evidence for $split"
  run_logged \
    "$RUN_ROOT/logs/${split}.materialize.log" \
    python scripts/materialize_visual_evidence.py \
    --input-path "$RUN_ROOT/candidates/${split}.jsonl" \
    --video-root "$VIDEO_ROOT" \
    --output-path "$RUN_ROOT/candidates/${split}.visual.jsonl" \
    --frames-dir "$RUN_ROOT/artifacts/frames" \
    --segments-dir "$RUN_ROOT/artifacts/segments" \
    --overwrite

  log "Extracting CLIP features for $split"
  run_logged \
    "$RUN_ROOT/logs/${split}.features.log" \
    python scripts/extract_clip_features.py \
    --input-path "$RUN_ROOT/candidates/${split}.visual.jsonl" \
    --output-path "$RUN_ROOT/candidates/${split}.visual_features.jsonl" \
    --feature-dir "$RUN_ROOT/features/clip" \
    --device "$DEVICE" \
    --batch-size "$FEATURE_BATCH_SIZE" \
    --model-name "$MODEL_NAME"
done

log "Running fixed-budget baseline"
run_logged \
  "$RUN_ROOT/logs/fixed_budget.log" \
  python scripts/run_fixed_budget_baseline.py \
  --input-path "$RUN_ROOT/candidates/val.visual_features.jsonl" \
  --summary-output "$RUN_ROOT/outputs/fixed_budget_frozen.summary.json" \
  --predictions-output "$RUN_ROOT/outputs/fixed_budget_frozen.predictions.jsonl" \
  --answerer frozen_multimodal \
  --answerer-device "$DEVICE" \
  --answerer-model-name "$MODEL_NAME" \
  --retriever hybrid_clip \
  --visual-device "$DEVICE" \
  --visual-model-name "$MODEL_NAME" \
  --subtitle-k "$SUBTITLE_K" \
  --frame-k "$FRAME_K" \
  --segment-k "$SEGMENT_K" \
  --oracle-mode "$ORACLE_MODE" \
  --oracle-min-sufficiency "$ORACLE_MIN_SUFFICIENCY"

log "Running keyword policy baseline"
run_logged \
  "$RUN_ROOT/logs/sequential_keyword.log" \
  python scripts/run_sequential_policy.py \
  --input-path "$RUN_ROOT/candidates/val.visual_features.jsonl" \
  --summary-output "$RUN_ROOT/outputs/sequential_policy_keyword_frozen.summary.json" \
  --predictions-output "$RUN_ROOT/outputs/sequential_policy_keyword_frozen.predictions.jsonl" \
  --answerer frozen_multimodal \
  --answerer-device "$DEVICE" \
  --answerer-model-name "$MODEL_NAME" \
  --policy keyword \
  --retriever hybrid_clip \
  --visual-device "$DEVICE" \
  --visual-model-name "$MODEL_NAME" \
  --subtitle-k "$SUBTITLE_K" \
  --frame-k "$FRAME_K" \
  --segment-k "$SEGMENT_K" \
  --max-items "$MAX_ITEMS" \
  --min-items-before-stop "$MIN_ITEMS_BEFORE_STOP"

for split in train val; do
  log "Exporting oracle traces for $split"
  run_logged \
    "$RUN_ROOT/logs/${split}.oracle.log" \
    python scripts/export_oracle_traces.py \
    --input-path "$RUN_ROOT/candidates/${split}.visual_features.jsonl" \
    --output-path "$RUN_ROOT/outputs/${split}_oracle_traces_frozen.jsonl" \
    --answerer frozen_multimodal \
    --answerer-device "$DEVICE" \
    --answerer-model-name "$MODEL_NAME" \
    --retriever hybrid_clip \
    --visual-device "$DEVICE" \
    --visual-model-name "$MODEL_NAME" \
    --subtitle-k "$SUBTITLE_K" \
    --frame-k "$FRAME_K" \
    --segment-k "$SEGMENT_K" \
    --oracle-mode "$ORACLE_MODE" \
    --oracle-min-sufficiency "$ORACLE_MIN_SUFFICIENCY"
done

log "Training learned policy"
run_logged \
  "$RUN_ROOT/logs/policy.train.log" \
  python scripts/train_policy.py \
  --train-traces-path "$RUN_ROOT/outputs/train_oracle_traces_frozen.jsonl" \
  --validation-traces-path "$RUN_ROOT/outputs/val_oracle_traces_frozen.jsonl" \
  --model-dir "$RUN_ROOT/models/policy_hybrid_frozen" \
  --answerer frozen_multimodal \
  --answerer-device "$DEVICE" \
  --answerer-model-name "$MODEL_NAME" \
  --epochs 10 \
  --seed "$SEED" \
  --min-items-before-stop "$MIN_ITEMS_BEFORE_STOP"

log "Evaluating learned policy"
run_logged \
  "$RUN_ROOT/logs/sequential_learned.log" \
  python scripts/run_sequential_policy.py \
  --input-path "$RUN_ROOT/candidates/val.visual_features.jsonl" \
  --summary-output "$RUN_ROOT/outputs/sequential_policy_frozen.summary.json" \
  --predictions-output "$RUN_ROOT/outputs/sequential_policy_frozen.predictions.jsonl" \
  --answerer frozen_multimodal \
  --answerer-device "$DEVICE" \
  --answerer-model-name "$MODEL_NAME" \
  --policy linear \
  --policy-model-dir "$RUN_ROOT/models/policy_hybrid_frozen" \
  --retriever hybrid_clip \
  --visual-device "$DEVICE" \
  --visual-model-name "$MODEL_NAME" \
  --subtitle-k "$SUBTITLE_K" \
  --frame-k "$FRAME_K" \
  --segment-k "$SEGMENT_K" \
  --max-items "$MAX_ITEMS" \
  --min-items-before-stop "$MIN_ITEMS_BEFORE_STOP"

log "Experiment finished. Summary files:"
printf '  %s\n' \
  "$RUN_ROOT/outputs/fixed_budget_frozen.summary.json" \
  "$RUN_ROOT/outputs/sequential_policy_keyword_frozen.summary.json" \
  "$RUN_ROOT/outputs/sequential_policy_frozen.summary.json"

python - <<PY
import json
from pathlib import Path

paths = [
    Path("$RUN_ROOT/outputs/fixed_budget_frozen.summary.json"),
    Path("$RUN_ROOT/outputs/sequential_policy_keyword_frozen.summary.json"),
    Path("$RUN_ROOT/outputs/sequential_policy_frozen.summary.json"),
]
for path in paths:
    payload = json.loads(path.read_text())
    metrics = payload.get("metrics", {})
    print(path.name)
    print(json.dumps({
        "accuracy": metrics.get("accuracy"),
        "selected_evidence_cost": metrics.get("selected_evidence_cost"),
        "selected_evidence_count": metrics.get("selected_evidence_count"),
        "selected_temporal_iou": metrics.get("selected_temporal_iou"),
    }, indent=2))
PY
