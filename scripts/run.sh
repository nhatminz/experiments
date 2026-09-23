#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAIN_REPO="${MAIN_REPO:-$(cd "${EXPERIMENT_DIR}/../BellmanOPD_analysis" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ----------------------- USER CONFIG -----------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MAX_STEPS="${MAX_STEPS:-100}"
export NUM_PROBE_ROLLOUTS="${NUM_PROBE_ROLLOUTS:-4}"
export FUTURE_HORIZON="${FUTURE_HORIZON:-256}"
export TOP_K="${TOP_K:-16}"
export LEARNING_RATE="${LEARNING_RATE:-5e-6}"
export SEED="${SEED:-42}"
export ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION="${ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION:-0.60}"
export ROLLOUT_VLLM_MAX_MODEL_LEN="${ROLLOUT_VLLM_MAX_MODEL_LEN:-5500}"
# Optional path overrides use the same semantics as BellmanOPD_analysis:
# STUDENT_MODEL, TEACHER_MODEL, TRAIN_DATA, PROMPT_KEY, TRAIN_DATA_SPLIT,
# STORAGE_ROOT, NORMAL_ROLLOUT_HORIZON, MIN_ORIGINAL_SUFFIX_TOKENS.
# -----------------------------------------------------------

if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "Experiment is single-GPU; set CUDA_VISIBLE_DEVICES to exactly one GPU" >&2
  exit 2
fi

RUN_NAME="${RUN_NAME:-cmt_state_intervention_topk${TOP_K}_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXPERIMENT_DIR}/outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"

if [[ ! -f "${MAIN_REPO}/b200_experiment/trainer.py" ]]; then
  echo "Invalid MAIN_REPO: ${MAIN_REPO}" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite non-empty output: ${OUTPUT_DIR}" >&2
  exit 1
fi

export PYTHONPATH="${MAIN_REPO}:${EXPERIMENT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
echo "Main repo: ${MAIN_REPO}"
echo "Output: ${OUTPUT_DIR}"
echo "Interventions: ${MAX_STEPS}"
echo "Paired suffix rollouts/state: ${NUM_PROBE_ROLLOUTS}"
echo "Future horizon: ${FUTURE_HORIZON}"
echo "Student Top-K: ${TOP_K}"
echo "Learning rate: ${LEARNING_RATE}"

cd "${EXPERIMENT_DIR}"
exec "${PYTHON_BIN}" run_experiment.py \
  --main-repo "${MAIN_REPO}" \
  --config "${EXPERIMENT_DIR}/config.yaml" \
  --output-dir "${OUTPUT_DIR}"
