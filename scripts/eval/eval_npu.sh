#!/usr/bin/env bash

# NPU launch script for DeepSpec DFlash evaluation.
# Mirrors scripts/eval/eval.sh but uses Ascend NPUs instead of CUDA GPUs.

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

export DEEPSPEC_DEVICE=${DEEPSPEC_DEVICE:-npu}

target_name_or_path=Qwen/Qwen3-4B
draft_name_or_path=${HOME}/checkpoints/deepspec/dflash_block8_qwen3_4b/step_latest
python eval.py \
    --target_name_or_path ${target_name_or_path} \
    --draft_name_or_path ${draft_name_or_path}
