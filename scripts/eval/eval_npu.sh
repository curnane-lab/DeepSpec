#!/usr/bin/env bash

# NPU launch script for DeepSpec DFlash evaluation (Qwen3.5-4B).
# Mirrors scripts/eval/eval.sh but uses Ascend NPUs instead of CUDA GPUs.

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

export DEEPSPEC_DEVICE=${DEEPSPEC_DEVICE:-npu}

# Match this to the target model used by the draft checkpoint.
target_name_or_path=Qwen/Qwen3.5-4B

# Training writes checkpoints under ~/checkpoints/<project_name>/<exp_name>/step_*.
# Use step_latest for the most recent checkpoint, or replace it with step_<N>.
draft_name_or_path=${HOME}/checkpoints/deepspec/dflash_block8_qwen3_5_4b/step_latest
python eval.py \
    --target_name_or_path ${target_name_or_path} \
    --draft_name_or_path ${draft_name_or_path}
