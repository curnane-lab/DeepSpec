#!/usr/bin/env bash

# NPU launch script for DeepSpec DFlash/DSpark evaluation (Qwen3.5-4B).
# Mirrors scripts/eval/eval.sh but uses Ascend NPUs instead of CUDA GPUs.
#
# Usage:
#   export TARGET_MODEL_PATH=/path/to/Qwen3.5-4B
#   bash scripts/eval/eval_npu.sh [dflash|dspark]
#
# The optional positional argument selects the draft checkpoint type:
#   dflash  - ${HOME}/checkpoints/deepspec/dflash_block8_qwen3_5_4b/step_latest
#   dspark  - ${HOME}/checkpoints/deepspec/dspark_block5_qwen3_5_4b/step_latest
#
# You can also set it via the DEEPSPEC_CONFIG_TYPE environment variable:
#   export DEEPSPEC_CONFIG_TYPE=dspark
#   bash scripts/eval/eval_npu.sh

# Determine config type: positional arg overrides env var, default to dflash.
config_type_arg="${1:-${DEEPSPEC_CONFIG_TYPE:-dflash}}"
config_type=$(echo "${config_type_arg}" | tr '[:upper:]' '[:lower:]')

if [[ "${config_type}" != "dflash" && "${config_type}" != "dspark" ]]; then
    echo "ERROR: Unsupported config type '${config_type_arg}'."
    echo "Usage: bash scripts/eval/eval_npu.sh [dflash|dspark]"
    echo "Or set DEEPSPEC_CONFIG_TYPE=dflash|dspark"
    exit 1
fi

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

export DEEPSPEC_DEVICE=${DEEPSPEC_DEVICE:-npu}

# Infer the number of local processes from the visible NPU devices.
# torch.npu.device_count() may report all physical devices on some Ascend
# setups, which causes HCCL initialization errors when only a subset is
# allocated to the job. Count ASCEND_RT_VISIBLE_DEVICES explicitly.
_npu_visible=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}
_nproc=$(echo "${_npu_visible}" | tr ',' '\n' | wc -l)

# Target model must be a local path in offline/air-gapped NPU environments.
# Set it before running, e.g.:
#   export TARGET_MODEL_PATH=/data1/f00538480/Qwen3.5-4B
if [ -z "${TARGET_MODEL_PATH}" ]; then
    echo "ERROR: TARGET_MODEL_PATH is not set." >&2
    echo "Set it to the local directory containing the target model weights." >&2
    exit 1
fi
if [ ! -d "${TARGET_MODEL_PATH}" ]; then
    echo "ERROR: TARGET_MODEL_PATH=${TARGET_MODEL_PATH} does not exist." >&2
    exit 1
fi

if [ "${config_type}" = "dspark" ]; then
    draft_name_or_path=${DRAFT_MODEL_PATH:-${HOME}/checkpoints/deepspec/dspark_block5_qwen3_5_4b/step_latest}
else
    draft_name_or_path=${DRAFT_MODEL_PATH:-${HOME}/checkpoints/deepspec/dflash_block8_qwen3_5_4b/step_latest}
fi

if [ ! -d "${draft_name_or_path}" ]; then
    echo "ERROR: Draft checkpoint not found at ${draft_name_or_path}" >&2
    echo "Train the draft model first, or override with DRAFT_MODEL_PATH=<path>." >&2
    exit 1
fi

echo "Evaluating ${config_type} draft model: ${draft_name_or_path}"
echo "Spawning ${_nproc} local NPU process(es) from ASCEND_RT_VISIBLE_DEVICES=${_npu_visible}"
python eval.py \
    --target_name_or_path "${TARGET_MODEL_PATH}" \
    --draft_name_or_path "${draft_name_or_path}" \
    --nproc "${_nproc}"
