#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# One intervention run is sequential (before -> update -> after).  Multiple
# GPUs therefore execute independent seeds, with exactly one process/vLLM
# replica assigned to each physical GPU.
GPU_LIST="${GPU_LIST:-0,1,2,3}"
SEEDS="${SEEDS:-42 43 44 45}"
PARALLEL_TAG="${PARALLEL_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${EXPERIMENT_DIR}/outputs/_parallel_logs/${PARALLEL_TAG}}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
read -r -a SEED_VALUES <<< "${SEEDS}"
if (( ${#GPUS[@]} == 0 || ${#GPUS[@]} != ${#SEED_VALUES[@]} )); then
  echo "GPU_LIST and SEEDS must contain the same non-zero number of entries" >&2
  echo "Example: GPU_LIST=0,1,2,3 SEEDS='42 43 44 45' $0" >&2
  exit 2
fi

mkdir -p "${LOG_ROOT}"
declare -A SEEN_GPUS=()
for index in "${!GPUS[@]}"; do
  gpu="${GPUS[$index]//[[:space:]]/}"
  if [[ -z "${gpu}" || "${gpu}" == *,* ]]; then
    echo "Invalid single-GPU entry at index ${index}: '${GPUS[$index]}'" >&2
    exit 2
  fi
  if [[ -n "${SEEN_GPUS[$gpu]:-}" ]]; then
    echo "GPU_LIST assigns physical GPU ${gpu} more than once; refusing to launch" >&2
    exit 2
  fi
  SEEN_GPUS["${gpu}"]=1
done

declare -a PIDS=()
declare -a NAMES=()
for index in "${!GPUS[@]}"; do
  gpu="${GPUS[$index]//[[:space:]]/}"
  seed="${SEED_VALUES[$index]}"
  name="${RUN_NAME_PREFIX:-cmt_state_intervention}_seed${seed}_${PARALLEL_TAG}"
  log_path="${LOG_ROOT}/${name}.log"
  echo "Starting ${name}: physical GPU ${gpu}, seed ${seed}, log ${log_path}"
  env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    SEED="${seed}" \
    RUN_NAME="${name}" \
    bash "${SCRIPT_DIR}/run.sh" "$@" >"${log_path}" 2>&1 &
  PIDS+=("$!")
  NAMES+=("${name}")
done

status=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    echo "Completed ${NAMES[$index]}"
  else
    echo "FAILED ${NAMES[$index]}; see ${LOG_ROOT}/${NAMES[$index]}.log" >&2
    status=1
  fi
done
echo "Parallel logs: ${LOG_ROOT}"
exit "${status}"
