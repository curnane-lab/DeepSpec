"""Device-agnostic asynchronous prefetcher.

On CUDA this overlaps DataLoader iteration and H2D transfer with compute using
a dedicated CUDA stream.  On NPU and other accelerators without a public stream
API it falls back to synchronous device transfer so training can still run.
"""

from threading import Thread

import torch

from deepspec.utils.device import get_device_type


def move_batch_to_device(batch, device):
    moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    # Embedding lookup requires int64; cast on device to avoid bloating host-to-device transfer.
    if moved["input_ids"].dtype != torch.long:
        moved["input_ids"] = moved["input_ids"].to(torch.long)
    return moved


class _SyncPrefetcher:
    """Fallback prefetcher that simply moves batches to the device synchronously."""

    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        self._iter = iter(self.dataloader)
        return self

    def __next__(self):
        try:
            batch = next(self._iter)
        except StopIteration:
            raise StopIteration
        return move_batch_to_device(batch, self.device)

    def __len__(self):
        return len(self.dataloader)


class _CUDAPrefetcher:
    """CUDA-specific stream prefetcher."""

    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)

    def __iter__(self):
        self._iter = iter(self.dataloader)
        self._done = False
        self._gpu_batch = None
        self._thread = None
        self._fetch_and_transfer()
        return self

    def _fetch_and_transfer(self):
        try:
            cpu_batch = next(self._iter)
        except StopIteration:
            self._done = True
            return
        with torch.cuda.stream(self.stream):
            self._gpu_batch = move_batch_to_device(cpu_batch, self.device)

    def __next__(self):
        if self._thread is not None:
            self._thread.join()
            self._thread = None

        if self._done:
            raise StopIteration

        current = torch.cuda.current_stream(self.device)
        current.wait_stream(self.stream)
        batch = self._gpu_batch

        for value in batch.values():
            value.record_stream(current)

        self._thread = Thread(target=self._fetch_and_transfer, daemon=True)
        self._thread.start()

        return batch

    def __len__(self):
        return len(self.dataloader)


def DevicePrefetcher(dataloader, device):
    """Return the appropriate prefetcher for the active device type."""
    if get_device_type() == "cuda":
        return _CUDAPrefetcher(dataloader, device)
    return _SyncPrefetcher(dataloader, device)


# Backward-compatible alias.
CUDAPrefetcher = _CUDAPrefetcher
