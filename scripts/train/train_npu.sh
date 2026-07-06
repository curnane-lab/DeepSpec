#!/usr/bin/env bash

# NPU launch script for DeepSpec DFlash/DSpark training (Qwen3.5-4B).
# Mirrors scripts/train/train.sh but uses Ascend NPUs instead of CUDA GPUs.
#
# Usage:
#   export TARGET_MODEL_PATH=/path/to/Qwen3.5-4B
#   bash scripts/train/train_npu.sh [dflash|dspark]
#
# The optional positional argument selects the training config type:
#   dflash  - config/dflash/dflash_qwen3_5_4b.py (default)
#   dspark  - config/dspark/dspark_qwen3_5_4b.py
#
# You can also set it via the DEEPSPEC_CONFIG_TYPE environment variable:
#   export DEEPSPEC_CONFIG_TYPE=dspark
#   bash scripts/train/train_npu.sh
#
# Environment:
#   TARGET_MODEL_PATH          - local path to Qwen3.5-4B weights (default: Qwen3.5-4B)
#   DEEPSPEC_CONFIG_TYPE       - dflash or dspark (default: dflash)
#   ASCEND_RT_VISIBLE_DEVICES  - comma-separated list of NPU IDs (default: 0,1,2,3,4,5,6,7)
#   MASTER_ADDR / MASTER_PORT  - distributed coordinator (default: 127.0.0.1:29500)
#   RANK / WORLD_SIZE          - node rank / total nodes (default: 0 / 1)

# Determine config type: positional arg overrides env var, default to dflash.
config_type_arg="${1:-${DEEPSPEC_CONFIG_TYPE:-dflash}}"
config_type=$(echo "${config_type_arg}" | tr '[:upper:]' '[:lower:]')

if [[ "${config_type}" != "dflash" && "${config_type}" != "dspark" ]]; then
    echo "ERROR: Unsupported config type '${config_type_arg}'."
    echo "Usage: bash scripts/train/train_npu.sh [dflash|dspark]"
    echo "Or set DEEPSPEC_CONFIG_TYPE=dflash|dspark"
    exit 1
fi

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

config_path="config/${config_type}/${config_type}_qwen3_5_4b.py"
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

echo "Launching ${config_type} training with config: ${config_path}"
python train.py \
    --config "${config_path}" \
    --opts "data.target_cache_path=${target_cache_dir}"
