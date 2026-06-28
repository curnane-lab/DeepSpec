#!/usr/bin/env bash

# NPU launch script for DeepSpec DFlash evaluation (Qwen3.5-4B).
# Mirrors scripts/eval/eval.sh but uses Ascend NPUs instead of CUDA GPUs.

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

export DEEPSPEC_DEVICE=${DEEPSPEC_DEVICE:-npu}

# Default to a local weights directory for offline/air-gapped environments.
export TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-Qwen3.5-4B}

python eval.py \
    --target_name_or_path ${TARGET_MODEL_PATH} \
    --draft_name_or_path ${HOME}/checkpoints/deepspec/dflash_block8_qwen3_5_4b/step_latest
