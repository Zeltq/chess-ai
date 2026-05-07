"""Centralized GPU inference for MCTS.

Two evaluators with the same callable contract::

    policies_np, values_np = evaluator(state_tensors)

`state_tensors` is a CPU torch.Tensor of shape (B, C, H, W); the result is
two numpy arrays — softmaxed policy logits (B, A) and value scalars (B,).

`LocalEvaluator` runs the model directly in the calling thread. Use it for
sequential self-play (one self-play game at a time).

`InferenceServer` is callable too, but every call is enqueued and executed
by a single background thread that batches across producers. Use it when
several MCTS threads run concurrently — the batched forward keeps GPU
utilisation high even though each MCTS only asks for a few leaves per
chunk.
"""
import threading
import time
from concurrent.futures import Future
from queue import Empty, Queue

import torch


def _amp_dtype_for(device):
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class LocalEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self._amp_dtype = _amp_dtype_for(device)
        self._channels_last = device.type == "cuda"

    def __call__(self, state_tensors):
        x = state_tensors.to(self.device, non_blocking=True)
        if self._channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self._amp_dtype,
            enabled=self.device.type == "cuda",
        ):
            policy_logits, values = self.model(x)
        policies = torch.softmax(policy_logits.float(), dim=-1).cpu().numpy()
        values_np = values.float().squeeze(-1).cpu().numpy()
        return policies, values_np


class InferenceServer:
    """Background thread that batches forward calls from many producers."""

    def __init__(self, model, device, max_batch_size=256, max_wait_seconds=0.002):
        self.model = model
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max_wait_seconds
        self._amp_dtype = _amp_dtype_for(device)
        self._channels_last = device.type == "cuda"
        self._queue: Queue = Queue()
        self._stop = threading.Event()
        self._stats_lock = threading.Lock()
        self._batches = 0
        self._positions = 0
        self._max_observed = 0
        self._thread = threading.Thread(
            target=self._loop, name="inference-server", daemon=True
        )
        self._thread.start()

    def __call__(self, state_tensors):
        future: Future = Future()
        self._queue.put((state_tensors, future))
        return future.result()

    def stop(self):
        self._stop.set()
        self._queue.put(None)
        self._thread.join(timeout=10)

    def stats_and_reset(self):
        with self._stats_lock:
            stats = {
                "batches": self._batches,
                "positions": self._positions,
                "avg_batch": self._positions / max(1, self._batches),
                "max_batch": self._max_observed,
            }
            self._batches = 0
            self._positions = 0
            self._max_observed = 0
        return stats

    def _loop(self):
        while not self._stop.is_set():
            first = self._queue.get()
            if first is None:
                return
            tensors = [first[0]]
            futures = [first[1]]
            current_n = first[0].shape[0]
            deadline = time.perf_counter() + self.max_wait_seconds
            while current_n < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                except Empty:
                    break
                if item is None:
                    self._stop.set()
                    break
                tensors.append(item[0])
                futures.append(item[1])
                current_n += item[0].shape[0]
            self._run_batch(tensors, futures)

    def _run_batch(self, tensors, futures):
        sizes = [t.shape[0] for t in tensors]
        big = torch.cat(tensors, dim=0).to(self.device, non_blocking=True)
        if self._channels_last:
            big = big.contiguous(memory_format=torch.channels_last)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self._amp_dtype,
            enabled=self.device.type == "cuda",
        ):
            policy_logits, values = self.model(big)
        policies = torch.softmax(policy_logits.float(), dim=-1).cpu().numpy()
        values_np = values.float().squeeze(-1).cpu().numpy()
        offset = 0
        for sz, fut in zip(sizes, futures):
            fut.set_result(
                (policies[offset:offset + sz], values_np[offset:offset + sz])
            )
            offset += sz
        with self._stats_lock:
            total = sum(sizes)
            self._batches += 1
            self._positions += total
            if total > self._max_observed:
                self._max_observed = total
