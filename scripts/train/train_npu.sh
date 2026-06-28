#!/usr/bin/env bash

# NPU launch script for DeepSpec DFlash training (Qwen3.5-4B).
# Mirrors scripts/train/train.sh but uses Ascend NPUs instead of CUDA GPUs.
#
# Usage:
#   export TARGET_MODEL_PATH=/path/to/Qwen3.5-4B
#   bash scripts/train/train_npu.sh
#
# Environment:
#   TARGET_MODEL_PATH        - local path to Qwen3.5-4B weights (default: Qwen3.5-4B)
#   ASCEND_RT_VISIBLE_DEVICES  - comma-separated list of NPU IDs (default: 0,1,2,3,4,5,6,7)
#   MASTER_ADDR / MASTER_PORT  - distributed coordinator (default: 127.0.0.1:29500)
#   RANK / WORLD_SIZE          - node rank / total nodes (default: 0 / 1)

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

# Optional: reduce NPU memory fragmentation observed on Ascend 910B.
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:32}

# Force the codebase to use the NPU device abstraction.
export DEEPSPEC_DEVICE=${DEEPSPEC_DEVICE:-npu}

# Default to a local weights directory for offline/air-gapped environments.
export TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-Qwen3.5-4B}

target_cache_dir=${target_cache_dir:-${HOME}/.cache/deepspec/qwen3_5_4b_target_cache}

# The training step requires the target cache to be prepared in advance.
if [[ ! -f "${target_cache_dir}/manifest.json" ]]; then
    echo "ERROR: Target cache not found at ${target_cache_dir}/manifest.json"
    echo ""
    echo "Please prepare the target cache first, for example:"
    echo "  export TARGET_MODEL_PATH=${TARGET_MODEL_PATH}"
    echo "  export DEEPSPEC_DEVICE=npu"
    echo "  python scripts/data/prepare_target_cache.py \\"
    echo "      --config config/dflash/dflash_qwen3_5_4b.py \\"
    echo "      --train_data_path <path_to_train.jsonl> \\"
    echo "      --output_dir ${target_cache_dir}"
    echo ""
    echo "Or set a different cache directory via the target_cache_dir variable."
    exit 1
fi

python train.py \
    --config config/dflash/dflash_qwen3_5_4b.py \
    --opts "data.target_cache_path=${target_cache_dir}"
