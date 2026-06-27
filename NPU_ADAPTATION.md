# DeepSpec NPU Adaptation Notes

This branch (`add_npu_support`) adds Ascend NPU support for the **DFlash**
draft model path (Qwen3/Gemma4 target families). Eagle3 support is not yet
included because its Triton-based fused loss requires additional work.

## What works on NPU

- DFlash training via `scripts/train/train_npu.sh`
- DFlash evaluation via `scripts/eval/eval_npu.sh`
- Target cache preparation via `scripts/data/prepare_target_cache.py`
- FSDP distributed training with the `hccl` backend

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
```

## Running DFlash training on NPU

```bash
bash scripts/train/train_npu.sh
```

## Running DFlash evaluation on NPU

```bash
bash scripts/eval/eval_npu.sh
```

## Data preparation on NPU

The `prepare_target_cache.py` script is device-agnostic except for the target
model forward. Run it with the NPU environment set:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export DEEPSPEC_DEVICE=npu
python scripts/data/prepare_target_cache.py \
    --config config/dflash/dflash_qwen3_4b.py \
    --train_data_path <path_to_train.jsonl> \
    --output_dir <target_cache_dir>
```

Note: the `scripts/data/launch_sglang_server.sh` helper uses SGLang, which
still requires a CUDA-compatible inference engine for answer regeneration.
For NPU-only environments, replace it with an OpenAI-compatible server backed
by an NPU inference engine (e.g. vLLM-ascend / MindIE).

## Known limitations

- **Eagle3** is not yet adapted; its `FusedLogSoftmaxLoss` uses Triton and
  asserts `logits.is_cuda`.
- **Gemma4** SDPA path has been updated but may need additional testing on
  Ascend kernels that do not support all SDPA variants.
- Some Gemma4-specific fused operations may require eager fallback.

## References

NPU adaptation patterns are informed by `SpecForge_npu_v2` (see
`E:\work\ref\SpecForge_npu_v2`), especially the device abstraction and the
strategy of disabling `flex_attention` on Ascend NPUs.
