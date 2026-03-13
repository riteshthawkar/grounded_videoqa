#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_multigpu_pipeline.sh \
    --dataset tvqa \
    --train-qa /path/to/tvqa_train.jsonl \
    --train-subtitles /path/to/tvqa_subtitles.jsonl \
    --val-qa /path/to/tvqa_val.jsonl \
    --val-subtitles /path/to/tvqa_subtitles.jsonl \
    --video-root /path/to/tvqa_videos \
    --run-root /path/to/output/run \
    --gpus 0,1

  For NExT-GQA (--train-subtitles / --val-subtitles are not required):

  bash scripts/run_multigpu_pipeline.sh \
    --dataset nextgqa \
    --train-qa /path/to/nextgqa/train.csv \
    --val-qa /path/to/nextgqa/val.csv \
    --video-root /path/to/videos \
    --video-map /path/to/map_vid_vidorID.json \
    --val-gsub /path/to/gsub_val.json \
    --val-frame-times /path/to/frame2time_val.json \
    --run-root /path/to/output/run \
    --gpus 0,1

Optional:
  --test-qa /path/to/test.jsonl
  --test-subtitles /path/to/test_subtitles.jsonl
  --conda-env adaptive-evidence-vqa
  --feature-model openai/clip-vit-base-patch32
  --feature-batch-size 32
  --subtitle-k 2
  --frame-k 2
  --segment-k 2
  --text-feature-dim 4096
  --answerer-epochs 20
  --policy-epochs 20
  --oracle-mode correctness_plus_sufficiency_plus_grounding
  --oracle-min-sufficiency 0.8
  --oracle-min-temporal-iou 0.1
  --max-items 6
  --extract-segments
  --hf-offline
  --skip-model-relative    skip the cross-answerer model-relative study

  NExT-GQA optional metadata:
  --video-map /path/to/map_vid_vidorID.json
  --train-gsub /path/to/gsub_train.json
  --val-gsub /path/to/gsub_val.json
  --test-gsub /path/to/gsub_test.json
  --train-frame-times /path/to/frame2time_train.json
  --val-frame-times /path/to/frame2time_val.json
  --test-frame-times /path/to/frame2time_test.json

Notes:
  - GPU acceleration is used for CLIP feature extraction and hybrid retrieval stages.
  - The linear answerer and linear policy remain CPU-bound today.
  - After evaluation the pipeline runs a model-relative comparison (linear vs
    frozen_multimodal) unless --skip-model-relative is set.
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
  conda activate "$CONDA_ENV"
}

run_logged() {
  local log_path="$1"
  shift
  log "Running: $*"
  "$@" >"$log_path" 2>&1
}

run_logged_gpu() {
  local gpu_id="$1"
  local log_path="$2"
  shift 2
  log "Running on GPU ${gpu_id}: $*"
  CUDA_VISIBLE_DEVICES="$gpu_id" "$@" >"$log_path" 2>&1
}

gpu_for_index() {
  local index="$1"
  local count="${#GPU_LIST[@]}"
  printf '%s' "${GPU_LIST[$((index % count))]}"
}

# --- TVQA / TVQA+ split preparation (requires subtitles) ----------------
prepare_split_tvqa() {
  local split="$1"
  local qa_path="$2"
  local subtitles_path="$3"
  local normalized_path="$RUN_ROOT/normalized/${split}.jsonl"
  local candidates_path="$RUN_ROOT/candidates/${split}.jsonl"
  local visual_path="$RUN_ROOT/candidates/${split}.visual.jsonl"

  require_file "$qa_path"
  require_file "$subtitles_path"

  run_logged \
    "$RUN_ROOT/logs/${split}.prepare.log" \
    python "scripts/${PREPARE_SCRIPT}" \
    --qa-path "$qa_path" \
    --subtitles-path "$subtitles_path" \
    --output-path "$normalized_path"

  build_candidates_and_materialize "$split" "$normalized_path" "$candidates_path" "$visual_path"
}

# --- NExT-GQA split preparation (subtitles not required) -----------------
prepare_split_nextgqa() {
  local split="$1"
  local qa_path="$2"
  local gsub_path="$3"
  local frame_times_path="$4"
  local normalized_path="$RUN_ROOT/normalized/${split}.jsonl"
  local candidates_path="$RUN_ROOT/candidates/${split}.jsonl"
  local visual_path="$RUN_ROOT/candidates/${split}.visual.jsonl"

  require_file "$qa_path"

  local cmd=(python scripts/prepare_nextgqa.py
    --qa-path "$qa_path"
    --output-path "$normalized_path"
    --video-root "$VIDEO_ROOT")

  if [[ -n "$VIDEO_MAP" ]]; then
    cmd+=(--video-map-path "$VIDEO_MAP")
  fi
  if [[ -n "$gsub_path" && -f "$gsub_path" ]]; then
    cmd+=(--gsub-path "$gsub_path")
  fi
  if [[ -n "$frame_times_path" && -f "$frame_times_path" ]]; then
    cmd+=(--frame-times-path "$frame_times_path")
  fi

  run_logged "$RUN_ROOT/logs/${split}.prepare.log" "${cmd[@]}"

  build_candidates_and_materialize "$split" "$normalized_path" "$candidates_path" "$visual_path"
}

# --- Shared candidate-pool + visual materialization ----------------------
build_candidates_and_materialize() {
  local split="$1"
  local normalized_path="$2"
  local candidates_path="$3"
  local visual_path="$4"

  run_logged \
    "$RUN_ROOT/logs/${split}.candidate_pool.log" \
    python scripts/build_candidate_pool.py \
    --input-path "$normalized_path" \
    --output-path "$candidates_path"

  local materialize_cmd=(python scripts/materialize_visual_evidence.py
    --input-path "$candidates_path"
    --video-root "$VIDEO_ROOT"
    --output-path "$visual_path"
    --frames-dir "$RUN_ROOT/artifacts/frames"
    --segments-dir "$RUN_ROOT/artifacts/segments"
    --overwrite)

  if [[ "$EXTRACT_SEGMENTS" == "1" ]]; then
    materialize_cmd+=(--extract-segments)
  fi

  run_logged "$RUN_ROOT/logs/${split}.materialize.log" "${materialize_cmd[@]}"
}

launch_feature_job() {
  local split="$1"
  local gpu_id="$2"
  local visual_path="$RUN_ROOT/candidates/${split}.visual.jsonl"
  local feature_output_path="$RUN_ROOT/candidates/${split}.visual_features.jsonl"

  run_logged_gpu \
    "$gpu_id" \
    "$RUN_ROOT/logs/${split}.features.log" \
    python scripts/extract_clip_features.py \
    --input-path "$visual_path" \
    --output-path "$feature_output_path" \
    --feature-dir "$RUN_ROOT/features/clip" \
    --model-name "$FEATURE_MODEL" \
    --batch-size "$FEATURE_BATCH_SIZE" \
    --device cuda &
  FEATURE_PIDS+=("$!")
}

wait_for_jobs() {
  local pid
  for pid in "$@"; do
    wait "$pid"
  done
}

# ======================== argument defaults ================================
DATASET=""
TRAIN_QA=""
TRAIN_SUBTITLES=""
VAL_QA=""
VAL_SUBTITLES=""
TEST_QA=""
TEST_SUBTITLES=""
VIDEO_ROOT=""
RUN_ROOT=""
CONDA_ENV="adaptive-evidence-vqa"
GPU_IDS="0"
FEATURE_MODEL="openai/clip-vit-base-patch32"
FEATURE_BATCH_SIZE="32"
SUBTITLE_K="2"
FRAME_K="2"
SEGMENT_K="2"
TEXT_FEATURE_DIM="4096"
ANSWERER_EPOCHS="20"
POLICY_EPOCHS="20"
ORACLE_MODE="correctness_plus_sufficiency_plus_grounding"
ORACLE_MIN_SUFFICIENCY="0.8"
ORACLE_MIN_TEMPORAL_IOU="0.1"
MAX_ITEMS="6"
EXTRACT_SEGMENTS="0"
HF_OFFLINE="0"
SKIP_MODEL_RELATIVE="0"

# NExT-GQA optional metadata
VIDEO_MAP=""
TRAIN_GSUB=""
VAL_GSUB=""
TEST_GSUB=""
TRAIN_FRAME_TIMES=""
VAL_FRAME_TIMES=""
TEST_FRAME_TIMES=""

# ======================== argument parsing =================================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --train-qa) TRAIN_QA="$2"; shift 2 ;;
    --train-subtitles) TRAIN_SUBTITLES="$2"; shift 2 ;;
    --val-qa) VAL_QA="$2"; shift 2 ;;
    --val-subtitles) VAL_SUBTITLES="$2"; shift 2 ;;
    --test-qa) TEST_QA="$2"; shift 2 ;;
    --test-subtitles) TEST_SUBTITLES="$2"; shift 2 ;;
    --video-root) VIDEO_ROOT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --gpus) GPU_IDS="$2"; shift 2 ;;
    --feature-model) FEATURE_MODEL="$2"; shift 2 ;;
    --feature-batch-size) FEATURE_BATCH_SIZE="$2"; shift 2 ;;
    --subtitle-k) SUBTITLE_K="$2"; shift 2 ;;
    --frame-k) FRAME_K="$2"; shift 2 ;;
    --segment-k) SEGMENT_K="$2"; shift 2 ;;
    --text-feature-dim) TEXT_FEATURE_DIM="$2"; shift 2 ;;
    --answerer-epochs) ANSWERER_EPOCHS="$2"; shift 2 ;;
    --policy-epochs) POLICY_EPOCHS="$2"; shift 2 ;;
    --oracle-mode) ORACLE_MODE="$2"; shift 2 ;;
    --oracle-min-sufficiency) ORACLE_MIN_SUFFICIENCY="$2"; shift 2 ;;
    --oracle-min-temporal-iou) ORACLE_MIN_TEMPORAL_IOU="$2"; shift 2 ;;
    --max-items) MAX_ITEMS="$2"; shift 2 ;;
    --extract-segments) EXTRACT_SEGMENTS="1"; shift 1 ;;
    --hf-offline) HF_OFFLINE="1"; shift 1 ;;
    --skip-model-relative) SKIP_MODEL_RELATIVE="1"; shift 1 ;;
    --video-map) VIDEO_MAP="$2"; shift 2 ;;
    --train-gsub) TRAIN_GSUB="$2"; shift 2 ;;
    --val-gsub) VAL_GSUB="$2"; shift 2 ;;
    --test-gsub) TEST_GSUB="$2"; shift 2 ;;
    --train-frame-times) TRAIN_FRAME_TIMES="$2"; shift 2 ;;
    --val-frame-times) VAL_FRAME_TIMES="$2"; shift 2 ;;
    --test-frame-times) TEST_FRAME_TIMES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

# ======================== validation =======================================
if [[ -z "$DATASET" || -z "$TRAIN_QA" || -z "$VAL_QA" || -z "$VIDEO_ROOT" || -z "$RUN_ROOT" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

case "$DATASET" in
  tvqa)
    PREPARE_SCRIPT="prepare_tvqa.py"
    if [[ -z "$TRAIN_SUBTITLES" || -z "$VAL_SUBTITLES" ]]; then
      echo "--train-subtitles and --val-subtitles are required for dataset=tvqa" >&2
      exit 1
    fi
    ;;
  tvqa_plus)
    PREPARE_SCRIPT="prepare_tvqa_plus.py"
    if [[ -z "$TRAIN_SUBTITLES" || -z "$VAL_SUBTITLES" ]]; then
      echo "--train-subtitles and --val-subtitles are required for dataset=tvqa_plus" >&2
      exit 1
    fi
    ;;
  nextgqa)
    PREPARE_SCRIPT="prepare_nextgqa.py"
    ;;
  *)
    echo "Unsupported dataset: $DATASET (expected tvqa, tvqa_plus, or nextgqa)" >&2
    exit 1
    ;;
esac

require_dir "$VIDEO_ROOT"
mkdir -p \
  "$RUN_ROOT/logs" \
  "$RUN_ROOT/normalized" \
  "$RUN_ROOT/candidates" \
  "$RUN_ROOT/artifacts/frames" \
  "$RUN_ROOT/artifacts/segments" \
  "$RUN_ROOT/features/clip" \
  "$RUN_ROOT/models" \
  "$RUN_ROOT/outputs"

activate_conda

export PYTHONUNBUFFERED=1
if [[ "$HF_OFFLINE" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "At least one GPU id must be provided via --gpus." >&2
  exit 1
fi

# ======================== Stage 1: prepare splits ==========================
log "Preparing splits"
if [[ "$DATASET" == "nextgqa" ]]; then
  prepare_split_nextgqa "train" "$TRAIN_QA" "$TRAIN_GSUB" "$TRAIN_FRAME_TIMES"
  prepare_split_nextgqa "val" "$VAL_QA" "$VAL_GSUB" "$VAL_FRAME_TIMES"
  if [[ -n "$TEST_QA" ]]; then
    prepare_split_nextgqa "test" "$TEST_QA" "$TEST_GSUB" "$TEST_FRAME_TIMES"
  fi
else
  prepare_split_tvqa "train" "$TRAIN_QA" "$TRAIN_SUBTITLES"
  prepare_split_tvqa "val" "$VAL_QA" "$VAL_SUBTITLES"
  if [[ -n "$TEST_QA" && -n "$TEST_SUBTITLES" ]]; then
    prepare_split_tvqa "test" "$TEST_QA" "$TEST_SUBTITLES"
  fi
fi

# ======================== Stage 2: CLIP features ===========================
log "Extracting CLIP features"
FEATURE_PIDS=()
SPLITS=("train" "val")
HAS_TEST="0"
if [[ "$DATASET" == "nextgqa" && -n "$TEST_QA" ]]; then
  SPLITS+=("test"); HAS_TEST="1"
elif [[ -n "$TEST_QA" && -n "$TEST_SUBTITLES" ]]; then
  SPLITS+=("test"); HAS_TEST="1"
fi
for index in "${!SPLITS[@]}"; do
  split="${SPLITS[$index]}"
  launch_feature_job "$split" "$(gpu_for_index "$index")"
done
wait_for_jobs "${FEATURE_PIDS[@]}"

PRIMARY_GPU="$(gpu_for_index 0)"
SECONDARY_GPU="$(gpu_for_index 1)"

# ======================== Stage 3: train linear answerer ===================
log "Training fixed-budget linear answerer"
run_logged_gpu \
  "$PRIMARY_GPU" \
  "$RUN_ROOT/logs/train_answerer.log" \
  python scripts/train_answerer.py \
  --train-path "$RUN_ROOT/candidates/train.visual_features.jsonl" \
  --validation-path "$RUN_ROOT/candidates/val.visual_features.jsonl" \
  --model-dir "$RUN_ROOT/models/answerer_hybrid_linear" \
  --retriever hybrid_clip \
  --visual-device cuda \
  --subtitle-k "$SUBTITLE_K" \
  --frame-k "$FRAME_K" \
  --segment-k "$SEGMENT_K" \
  --text-feature-dim "$TEXT_FEATURE_DIM" \
  --epochs "$ANSWERER_EPOCHS"

# ======================== Stage 4: oracle traces (linear) ==================
log "Exporting oracle traces (linear answerer)"
ORACLE_PIDS=()
run_logged_gpu \
  "$PRIMARY_GPU" \
  "$RUN_ROOT/logs/train_oracle_linear.log" \
  python scripts/export_oracle_traces.py \
  --input-path "$RUN_ROOT/candidates/train.visual_features.jsonl" \
  --output-path "$RUN_ROOT/outputs/train_oracle_traces_linear.jsonl" \
  --answerer linear \
  --answerer-model-dir "$RUN_ROOT/models/answerer_hybrid_linear" \
  --retriever hybrid_clip \
  --visual-device cuda \
  --subtitle-k "$SUBTITLE_K" \
  --frame-k "$FRAME_K" \
  --segment-k "$SEGMENT_K" \
  --oracle-mode "$ORACLE_MODE" \
  --oracle-min-sufficiency "$ORACLE_MIN_SUFFICIENCY" \
  --oracle-min-temporal-iou "$ORACLE_MIN_TEMPORAL_IOU" &
ORACLE_PIDS+=("$!")

run_logged_gpu \
  "$SECONDARY_GPU" \
  "$RUN_ROOT/logs/val_oracle_linear.log" \
  python scripts/export_oracle_traces.py \
  --input-path "$RUN_ROOT/candidates/val.visual_features.jsonl" \
  --output-path "$RUN_ROOT/outputs/val_oracle_traces_linear.jsonl" \
  --answerer linear \
  --answerer-model-dir "$RUN_ROOT/models/answerer_hybrid_linear" \
  --retriever hybrid_clip \
  --visual-device cuda \
  --subtitle-k "$SUBTITLE_K" \
  --frame-k "$FRAME_K" \
  --segment-k "$SEGMENT_K" \
  --oracle-mode "$ORACLE_MODE" \
  --oracle-min-sufficiency "$ORACLE_MIN_SUFFICIENCY" \
  --oracle-min-temporal-iou "$ORACLE_MIN_TEMPORAL_IOU" &
ORACLE_PIDS+=("$!")

wait_for_jobs "${ORACLE_PIDS[@]}"

# ======================== Stage 5: oracle traces (frozen_multimodal) =======
log "Exporting oracle traces (frozen_multimodal answerer)"
ORACLE_FM_PIDS=()
run_logged_gpu \
  "$PRIMARY_GPU" \
  "$RUN_ROOT/logs/train_oracle_frozen.log" \
  python scripts/export_oracle_traces.py \
  --input-path "$RUN_ROOT/candidates/train.visual_features.jsonl" \
  --output-path "$RUN_ROOT/outputs/train_oracle_traces_frozen.jsonl" \
  --answerer frozen_multimodal \
  --answerer-model-name "$FEATURE_MODEL" \
  --answerer-device cuda \
  --retriever hybrid_clip \
  --visual-device cuda \
  --subtitle-k "$SUBTITLE_K" \
  --frame-k "$FRAME_K" \
  --segment-k "$SEGMENT_K" \
  --oracle-mode "$ORACLE_MODE" \
  --oracle-min-sufficiency "$ORACLE_MIN_SUFFICIENCY" \
  --oracle-min-temporal-iou "$ORACLE_MIN_TEMPORAL_IOU" &
ORACLE_FM_PIDS+=("$!")

run_logged_gpu \
  "$SECONDARY_GPU" \
  "$RUN_ROOT/logs/val_oracle_frozen.log" \
  python scripts/export_oracle_traces.py \
  --input-path "$RUN_ROOT/candidates/val.visual_features.jsonl" \
  --output-path "$RUN_ROOT/outputs/val_oracle_traces_frozen.jsonl" \
  --answerer frozen_multimodal \
  --answerer-model-name "$FEATURE_MODEL" \
  --answerer-device cuda \
  --retriever hybrid_clip \
  --visual-device cuda \
  --subtitle-k "$SUBTITLE_K" \
  --frame-k "$FRAME_K" \
  --segment-k "$SEGMENT_K" \
  --oracle-mode "$ORACLE_MODE" \
  --oracle-min-sufficiency "$ORACLE_MIN_SUFFICIENCY" \
  --oracle-min-temporal-iou "$ORACLE_MIN_TEMPORAL_IOU" &
ORACLE_FM_PIDS+=("$!")

wait_for_jobs "${ORACLE_FM_PIDS[@]}"

# ======================== Stage 6: train policies ==========================
log "Training sequential policies from both oracle sources"

run_logged \
  "$RUN_ROOT/logs/train_policy_from_linear.log" \
  python scripts/train_policy.py \
  --train-traces-path "$RUN_ROOT/outputs/train_oracle_traces_linear.jsonl" \
  --validation-traces-path "$RUN_ROOT/outputs/val_oracle_traces_linear.jsonl" \
  --model-dir "$RUN_ROOT/models/policy_from_linear" \
  --answerer linear \
  --answerer-model-dir "$RUN_ROOT/models/answerer_hybrid_linear" \
  --text-feature-dim "$TEXT_FEATURE_DIM" \
  --epochs "$POLICY_EPOCHS"

run_logged \
  "$RUN_ROOT/logs/train_policy_from_frozen.log" \
  python scripts/train_policy.py \
  --train-traces-path "$RUN_ROOT/outputs/train_oracle_traces_frozen.jsonl" \
  --validation-traces-path "$RUN_ROOT/outputs/val_oracle_traces_frozen.jsonl" \
  --model-dir "$RUN_ROOT/models/policy_from_frozen" \
  --answerer frozen_multimodal \
  --answerer-model-name "$FEATURE_MODEL" \
  --text-feature-dim "$TEXT_FEATURE_DIM" \
  --epochs "$POLICY_EPOCHS"

# ======================== Stage 7: evaluation ==============================
log "Evaluating fixed-budget and sequential policies"
EVAL_PIDS=()

# -- val: fixed-budget with linear answerer
run_logged_gpu \
  "$PRIMARY_GPU" \
  "$RUN_ROOT/logs/val_fixed_budget_linear.log" \
  python scripts/run_fixed_budget_baseline.py \
  --input-path "$RUN_ROOT/candidates/val.visual_features.jsonl" \
  --summary-output "$RUN_ROOT/outputs/val_fixed_budget_linear.summary.json" \
  --predictions-output "$RUN_ROOT/outputs/val_fixed_budget_linear.predictions.jsonl" \
  --answerer linear \
  --answerer-model-dir "$RUN_ROOT/models/answerer_hybrid_linear" \
  --retriever hybrid_clip \
  --visual-device cuda \
  --subtitle-k "$SUBTITLE_K" \
  --frame-k "$FRAME_K" \
  --segment-k "$SEGMENT_K" \
  --oracle-mode "$ORACLE_MODE" \
  --oracle-min-sufficiency "$ORACLE_MIN_SUFFICIENCY" \
  --oracle-min-temporal-iou "$ORACLE_MIN_TEMPORAL_IOU" &
EVAL_PIDS+=("$!")

# -- val: fixed-budget with frozen_multimodal answerer
run_logged_gpu \
  "$SECONDARY_GPU" \
  "$RUN_ROOT/logs/val_fixed_budget_frozen.log" \
  python scripts/run_fixed_budget_baseline.py \
  --input-path "$RUN_ROOT/candidates/val.visual_features.jsonl" \
  --summary-output "$RUN_ROOT/outputs/val_fixed_budget_frozen.summary.json" \
  --predictions-output "$RUN_ROOT/outputs/val_fixed_budget_frozen.predictions.jsonl" \
  --answerer frozen_multimodal \
  --answerer-model-name "$FEATURE_MODEL" \
  --answerer-device cuda \
  --retriever hybrid_clip \
  --visual-device cuda \
  --subtitle-k "$SUBTITLE_K" \
  --frame-k "$FRAME_K" \
  --segment-k "$SEGMENT_K" \
  --oracle-mode "$ORACLE_MODE" \
  --oracle-min-sufficiency "$ORACLE_MIN_SUFFICIENCY" \
  --oracle-min-temporal-iou "$ORACLE_MIN_TEMPORAL_IOU" &
EVAL_PIDS+=("$!")

wait_for_jobs "${EVAL_PIDS[@]}"

# -- val: sequential policy trained from linear oracle
EVAL_PIDS=()
run_logged_gpu \
  "$PRIMARY_GPU" \
  "$RUN_ROOT/logs/val_sequential_policy_from_linear.log" \
  python scripts/run_sequential_policy.py \
  --input-path "$RUN_ROOT/candidates/val.visual_features.jsonl" \
  --summary-output "$RUN_ROOT/outputs/val_sequential_policy_from_linear.summary.json" \
  --predictions-output "$RUN_ROOT/outputs/val_sequential_policy_from_linear.predictions.jsonl" \
  --answerer linear \
  --answerer-model-dir "$RUN_ROOT/models/answerer_hybrid_linear" \
  --policy linear \
  --policy-model-dir "$RUN_ROOT/models/policy_from_linear" \
  --retriever hybrid_clip \
  --visual-device cuda \
  --subtitle-k "$SUBTITLE_K" \
  --frame-k "$FRAME_K" \
  --segment-k "$SEGMENT_K" \
  --max-items "$MAX_ITEMS" &
EVAL_PIDS+=("$!")

# -- val: sequential policy trained from frozen oracle
run_logged_gpu \
  "$SECONDARY_GPU" \
  "$RUN_ROOT/logs/val_sequential_policy_from_frozen.log" \
  python scripts/run_sequential_policy.py \
  --input-path "$RUN_ROOT/candidates/val.visual_features.jsonl" \
  --summary-output "$RUN_ROOT/outputs/val_sequential_policy_from_frozen.summary.json" \
  --predictions-output "$RUN_ROOT/outputs/val_sequential_policy_from_frozen.predictions.jsonl" \
  --answerer frozen_multimodal \
  --answerer-model-name "$FEATURE_MODEL" \
  --answerer-device cuda \
  --policy linear \
  --policy-model-dir "$RUN_ROOT/models/policy_from_frozen" \
  --retriever hybrid_clip \
  --visual-device cuda \
  --subtitle-k "$SUBTITLE_K" \
  --frame-k "$FRAME_K" \
  --segment-k "$SEGMENT_K" \
  --max-items "$MAX_ITEMS" &
EVAL_PIDS+=("$!")

wait_for_jobs "${EVAL_PIDS[@]}"

# -- optional: test split
if [[ "$HAS_TEST" == "1" ]]; then
  log "Evaluating on test split"
  EVAL_PIDS=()
  run_logged_gpu \
    "$PRIMARY_GPU" \
    "$RUN_ROOT/logs/test_fixed_budget_linear.log" \
    python scripts/run_fixed_budget_baseline.py \
    --input-path "$RUN_ROOT/candidates/test.visual_features.jsonl" \
    --summary-output "$RUN_ROOT/outputs/test_fixed_budget_linear.summary.json" \
    --predictions-output "$RUN_ROOT/outputs/test_fixed_budget_linear.predictions.jsonl" \
    --answerer linear \
    --answerer-model-dir "$RUN_ROOT/models/answerer_hybrid_linear" \
    --retriever hybrid_clip \
    --visual-device cuda \
    --subtitle-k "$SUBTITLE_K" \
    --frame-k "$FRAME_K" \
    --segment-k "$SEGMENT_K" \
    --oracle-mode "$ORACLE_MODE" \
    --oracle-min-sufficiency "$ORACLE_MIN_SUFFICIENCY" \
    --oracle-min-temporal-iou "$ORACLE_MIN_TEMPORAL_IOU" &
  EVAL_PIDS+=("$!")

  run_logged_gpu \
    "$SECONDARY_GPU" \
    "$RUN_ROOT/logs/test_fixed_budget_frozen.log" \
    python scripts/run_fixed_budget_baseline.py \
    --input-path "$RUN_ROOT/candidates/test.visual_features.jsonl" \
    --summary-output "$RUN_ROOT/outputs/test_fixed_budget_frozen.summary.json" \
    --predictions-output "$RUN_ROOT/outputs/test_fixed_budget_frozen.predictions.jsonl" \
    --answerer frozen_multimodal \
    --answerer-model-name "$FEATURE_MODEL" \
    --answerer-device cuda \
    --retriever hybrid_clip \
    --visual-device cuda \
    --subtitle-k "$SUBTITLE_K" \
    --frame-k "$FRAME_K" \
    --segment-k "$SEGMENT_K" \
    --oracle-mode "$ORACLE_MODE" \
    --oracle-min-sufficiency "$ORACLE_MIN_SUFFICIENCY" \
    --oracle-min-temporal-iou "$ORACLE_MIN_TEMPORAL_IOU" &
  EVAL_PIDS+=("$!")

  wait_for_jobs "${EVAL_PIDS[@]}"
fi

# ======================== Stage 8: model-relative study ====================
if [[ "$SKIP_MODEL_RELATIVE" == "0" ]]; then
  log "Running model-relative study (linear vs frozen_multimodal)"
  run_logged_gpu \
    "$PRIMARY_GPU" \
    "$RUN_ROOT/logs/val_model_relative.log" \
    python scripts/run_model_relative_study.py \
    --input-path "$RUN_ROOT/candidates/val.visual_features.jsonl" \
    --output-dir "$RUN_ROOT/outputs/model_relative_linear_vs_frozen" \
    --answerer-a linear \
    --answerer-a-model-dir "$RUN_ROOT/models/answerer_hybrid_linear" \
    --answerer-a-label linear \
    --answerer-b frozen_multimodal \
    --answerer-b-model-name "$FEATURE_MODEL" \
    --answerer-b-device cuda \
    --answerer-b-label frozen_multimodal \
    --retriever hybrid_clip \
    --visual-model-name "$FEATURE_MODEL" \
    --visual-device cuda \
    --subtitle-k "$SUBTITLE_K" \
    --frame-k "$FRAME_K" \
    --segment-k "$SEGMENT_K" \
    --oracle-mode "$ORACLE_MODE" \
    --oracle-min-sufficiency "$ORACLE_MIN_SUFFICIENCY" \
    --oracle-min-temporal-iou "$ORACLE_MIN_TEMPORAL_IOU"
fi

log "Pipeline completed. Outputs are in $RUN_ROOT"
