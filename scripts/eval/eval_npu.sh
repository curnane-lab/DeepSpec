#!/usr/bin/env bash

# NPU launch script for DeepSpec DFlash evaluation (Qwen3.5-4B).
# Mirrors scripts/eval/eval.sh but uses Ascend NPUs instead of CUDA GPUs.

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

export DEEPSPEC_DEVICE=${DEEPSPEC_DEVICE:-npu}

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

python eval.py \
    --target_name_or_path ${TARGET_MODEL_PATH} \
    --draft_name_or_path ${HOME}/checkpoints/deepspec/dflash_block8_qwen3_5_4b/step_latest
