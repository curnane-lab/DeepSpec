# DeepSpec NPU Adaptation Notes

This branch (`add_npu_support`) adds Ascend NPU support for the **DFlash**
draft model path, with the default example targeting **Qwen3.5-4B**.

## What works on NPU

- DFlash training for Qwen3.5-4B via `scripts/train/train_npu.sh`
- DFlash evaluation for Qwen3.5-4B via `scripts/eval/eval_npu.sh`
- Target cache preparation for Qwen3.5-4B via `scripts/data/prepare_target_cache.py`
- FSDP distributed training with the `hccl` backend

The Qwen3 architecture files are reused for Qwen3.5 because the two model
families share the same transformer implementation (`Qwen3PreTrainedModel`).

## Key changes

1. **Device abstraction layer** (`deepspec/utils/device.py`)
   - Auto-detects `cuda` / `npu` / `cpu`.
   - Can be forced via `DEEPSPEC_DEVICE=npu`.
   - Provides unified `device_count`, `set_device`, `get_backend`, `empty_cache`,
     RNG state helpers, etc.

2. **Distributed backend** (`deepspec/utils/distributed.py`)
   - Uses `hccl` on NPU, `nccl` on CUDA, `gloo` on CPU.
   - Override via `DEEPSPEC_DIST_BACKEND` if needed.

3. **Attention fallback** (`deepspec/modeling/dspark/common.py`)
   - `torch.nn.attention.flex_attention` is disabled on NPU.
   - A 4D additive attention mask fallback is used for SDPA/eager instead.

4. **Default attention backend** (`deepspec/modeling/dspark/{qwen3,gemma4}/config.py`)
   - Defaults to `"sdpa"` when `DEEPSPEC_DEVICE=npu`, otherwise `"flex_attention"`.

5. **Prefetcher** (`deepspec/data/device_prefetcher.py`)
   - CUDA path still uses a dedicated stream for overlap.
   - NPU/CPU path falls back to synchronous H2D transfer.

6. **torch.compile**
   - Automatically skipped on NPU in `BaseTrainer`.

7. **Checkpoint RNG state**
   - Device-agnostic save/load via `deepspec/utils/device.py` helpers.

## Environment setup

```bash
# Install torch-npu for your CANN version, e.g.:
# pip install torch-npu==2.9.1.post4

# Optional memory tuning
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:32

# Force NPU mode (auto-detected if torch.npu.is_available())
export DEEPSPEC_DEVICE=npu

# Path to your local Qwen3.5-4B weights (default: Qwen3.5-4B)
export TARGET_MODEL_PATH=/path/to/Qwen3.5-4B
```

## Data preparation (required before training)

Training reads pre-computed target hidden states from a **target cache**.
You must build this cache once before launching training.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export DEEPSPEC_DEVICE=npu
export TARGET_MODEL_PATH=/data1/f00538480/models/Qwen3.5-4B

target_cache_dir=${HOME}/.cache/deepspec/qwen3_5_4b_target_cache

python scripts/data/prepare_target_cache.py \
    --config config/dflash/dflash_qwen3_5_4b.py \
    --train_data_path <path_to_train.jsonl> \
    --output_dir ${target_cache_dir} \
    --local_batch_size 1 \
    --num_workers 4
```

This writes `manifest.json` and binary shards under `${target_cache_dir}`.
`scripts/train/train_npu.sh` checks for this manifest and fails with a
helpful message if it is missing.

Note: the `scripts/data/launch_sglang_server.sh` helper uses SGLang, which
still requires a CUDA-compatible inference engine for answer regeneration.
For NPU-only environments, replace it with an OpenAI-compatible server backed
by an NPU inference engine (e.g. vLLM-ascend / MindIE) when generating the
initial training answers.

## Running DFlash training on NPU (Qwen3.5-4B)

Place or symlink the Qwen3.5-4B weights at `Qwen3.5-4B` under the repo root,
or set `TARGET_MODEL_PATH` to the directory containing the weights, then run:

```bash
export TARGET_MODEL_PATH=/data1/f00538480/models/Qwen3.5-4B
bash scripts/train/train_npu.sh
```

This uses `config/dflash/dflash_qwen3_5_4b.py`, which selects
`target_layer_ids=[1, 8, 15, 22, 29]` for Qwen3.5-4B.

## Running DFlash evaluation on NPU (Qwen3.5-4B)

```bash
export TARGET_MODEL_PATH=/data1/f00538480/models/Qwen3.5-4B
bash scripts/eval/eval_npu.sh
```

## Adapting to other Qwen3.5 sizes

Copy `config/dflash/dflash_qwen3_5_4b.py`, update:

- `model.target_model_name_or_path`
- `model.target_layer_ids` (must be strictly increasing and inside the target
  model's layer range; `-1` is allowed for the embedding output)
- `model.mask_token_id` (the `[MASK]` token id of the Qwen3.5 tokenizer)
- `exp_name` and `data.target_cache_path`

Then launch with `--config <your_config.py>`.

## Known limitations

- **Eagle3** is not yet adapted; its `FusedLogSoftmaxLoss` uses Triton and
  asserts `logits.is_cuda`.
- **Gemma4** SDPA path has been updated but may need additional testing on
  Ascend kernels that do not support all SDPA variants.
- Some Gemma4-specific fused operations may require eager fallback.
- The original Qwen3-4B configs are left untouched; this branch's NPU scripts
  default to Qwen3.5-4B with a local-weights default.

## References

NPU adaptation patterns are informed by `SpecForge_npu_v2` (see
`E:\work\ref\SpecForge_npu_v2`), especially the device abstraction and the
strategy of disabling `flex_attention` on Ascend NPUs.
