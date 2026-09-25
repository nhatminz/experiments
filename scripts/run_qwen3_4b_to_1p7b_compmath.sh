#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${EXPERIMENT_DIR}/.." && pwd)"
MAIN_REPO="${MAIN_REPO:-${WORKSPACE_DIR}/BellmanOPD_analysis}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ====================== USER CONFIG ======================
export STORAGE_ROOT="${STORAGE_ROOT:-/workspace/storage-shared}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TEACHER_MODEL="${TEACHER_MODEL:-${STORAGE_ROOT}/models/Qwen3-4B}"
export STUDENT_MODEL="${STUDENT_MODEL:-${STORAGE_ROOT}/nlp/tungdd11/stable-on-policy-distillation/OPD/model/Qwen3-1.7B-Base}"
export TRAIN_DATA="${TRAIN_DATA:-${STORAGE_ROOT}/nlp/minhpn19/data/competition_math/data/train-00000-of-00001.parquet}"
export PROMPT_KEY="${PROMPT_KEY:-problem}"
export TRAIN_DATA_SPLIT="${TRAIN_DATA_SPLIT:-null}"

export NUM_STATES="${NUM_STATES:-300}"
export K_ROLLOUTS="${K_ROLLOUTS:-${NUM_CONTINUATIONS:-8}}"
export NUM_CONTINUATIONS="${K_ROLLOUTS}"
export FUTURE_HORIZON="${FUTURE_HORIZON:-128}"
export TOP_K="${TOP_K:-16}"
export LEARNING_RATE="${LEARNING_RATE:-5e-6}"
export SEED="${SEED:-42}"
export CANDIDATE_ROLLOUT_HORIZON="${CANDIDATE_ROLLOUT_HORIZON:-4096}"
export CANDIDATE_COLLECTION_BATCH_SIZE="${CANDIDATE_COLLECTION_BATCH_SIZE:-8}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
export ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION="${ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION:-0.60}"
export ROLLOUT_VLLM_MAX_MODEL_LEN="${ROLLOUT_VLLM_MAX_MODEL_LEN:-5500}"
# =========================================================

if [[ -z "${CUDA_VISIBLE_DEVICES//[[:space:]]/}" || "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "This mechanistic protocol uses exactly one B200; set CUDA_VISIBLE_DEVICES to one ID." >&2
  exit 2
fi
if [[ ! -f "${MAIN_REPO}/b200_experiment/models.py" ]]; then
  echo "Invalid MAIN_REPO: ${MAIN_REPO}" >&2
  exit 2
fi

RUN_NAME="${RUN_NAME:-figure1_qwen3_4b_to_1p7b_compmath_n${NUM_STATES}_k${TOP_K}_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXPERIMENT_DIR}/outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${WORKSPACE_DIR}:${MAIN_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
echo "Protocol: fixed-base single-state reverse-KL (no CMT/LIFT quantities)"
echo "Teacher: ${TEACHER_MODEL}"
echo "Student: ${STUDENT_MODEL}"
echo "Data: ${TRAIN_DATA} [${PROMPT_KEY}, enable_thinking=false]"
echo "States=${NUM_STATES} continuations/state=${NUM_CONTINUATIONS} future_horizon=${FUTURE_HORIZON}"
echo "Local update: Student Top-${TOP_K}, AdamW lr=${LEARNING_RATE}, one step"
echo "Sampling: temperature=${ROLLOUT_TEMPERATURE} top_p=${ROLLOUT_TOP_P}"
echo "Output: ${OUTPUT_DIR}"

cd "${WORKSPACE_DIR}"
exec "${PYTHON_BIN}" -m Experiment.run_experiment \
  --main-repo "${MAIN_REPO}" \
  --config "${EXPERIMENT_DIR}/config.yaml" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
