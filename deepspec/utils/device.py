"""Device abstraction utilities for DeepSpec.

This module centralizes device/backend detection so the same codebase can run
on NVIDIA GPUs (CUDA), Ascend NPUs (NPU), and CPU fallback without sprinkling
`torch.cuda` calls throughout the training/evaluation logic.

Adapted from patterns used in SpecForge NPU adaptations.
"""

import os

import torch


__all__ = [
    "get_device_type",
    "get_local_device",
    "device_count",
    "set_device",
    "get_current_device",
    "get_backend",
    "empty_cache",
    "manual_seed_all",
    "get_rng_state",
    "set_rng_state",
]


def get_device_type() -> str:
    """Auto-detect the available accelerator type.

    Priority:
      1. ``DEEPSPEC_DEVICE`` environment variable.
      2. NVIDIA CUDA (``torch.cuda``).
      3. Ascend NPU (``torch.npu``).
      4. CPU fallback.
    """
    dt = os.environ.get("DEEPSPEC_DEVICE", None)
    if dt:
        return dt
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    return "cpu"


def get_local_device() -> torch.device:
    """Return the local torch.device for the current process rank."""
    device_type = get_device_type()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return torch.device(device_type, local_rank)


def device_count() -> int:
    """Return the number of visible devices on this node."""
    device_type = get_device_type()
    if device_type == "cuda":
        return torch.cuda.device_count()
    if device_type == "npu":
        return torch.npu.device_count()
    return 1


def set_device(local_rank: int) -> None:
    """Bind the current process to the given local rank's device."""
    device_type = get_device_type()
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
    elif device_type == "npu":
        torch.npu.set_device(local_rank)


def get_current_device() -> int:
    """Return the current device id for the active accelerator."""
    device_type = get_device_type()
    if device_type == "cuda":
        return torch.cuda.current_device()
    if device_type == "npu":
        return torch.npu.current_device()
    return 0


def get_backend() -> str:
    """Return the distributed backend appropriate for the active accelerator.

    The value can be overridden via the ``DEEPSPEC_DIST_BACKEND`` environment
    variable (useful for testing or exotic setups).
    """
    backend = os.environ.get("DEEPSPEC_DIST_BACKEND", None)
    if backend:
        return backend
    return {
        "cuda": "nccl",
        "npu": "hccl",
        "cpu": "gloo",
    }[get_device_type()]


def empty_cache() -> None:
    """Clear the device memory cache when supported."""
    device_type = get_device_type()
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "npu":
        torch.npu.empty_cache()


def manual_seed_all(seed: int) -> None:
    """Set the seed on all devices on this node."""
    torch.manual_seed(seed)
    device_type = get_device_type()
    if device_type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device_type == "npu":
        torch.npu.manual_seed_all(seed)


def get_rng_state():
    """Return the RNG state for the active accelerator."""
    device_type = get_device_type()
    if device_type == "cuda":
        return torch.cuda.get_rng_state()
    if device_type == "npu":
        return torch.npu.get_rng_state()
    return torch.get_rng_state()


def set_rng_state(state) -> None:
    """Restore the RNG state for the active accelerator."""
    device_type = get_device_type()
    if device_type == "cuda":
        torch.cuda.set_rng_state(state)
    elif device_type == "npu":
        torch.npu.set_rng_state(state)
    else:
        torch.set_rng_state(state)
