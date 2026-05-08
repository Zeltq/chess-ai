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

import numpy as np
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


class MPInferenceServer:
    """Inference server fed by worker *processes* over multiprocessing queues.

    Each worker has a dedicated response queue (response_queues[worker_id]).
    Workers send (worker_id, request_id, np_array) on the shared
    request_queue; the server thread drains, batches, runs forward on the
    GPU, and returns (request_id, policies_np, values_np) on the worker's
    response queue.

    Tensors are exchanged as numpy arrays — pickled by the queue's
    underlying pipe — to keep IPC simple. For typical chess workloads
    (~20KB per request) the pickle cost is tiny relative to MCTS work.
    """

    def __init__(
        self, model, device, request_queue, response_queues,
        max_batch_size=256, max_wait_seconds=0.002,
    ):
        self.model = model
        self.device = device
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max_wait_seconds
        self._amp_dtype = _amp_dtype_for(device)
        self._channels_last = device.type == "cuda"
        self._stop = threading.Event()
        self._stats_lock = threading.Lock()
        self._batches = 0
        self._positions = 0
        self._max_observed = 0
        self._thread = threading.Thread(
            target=self._loop, name="mp-inference-server", daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            self.request_queue.put_nowait(None)
        except Exception:
            pass
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
            try:
                first = self.request_queue.get(timeout=0.1)
            except Empty:
                continue
            if first is None:
                return
            requests = [first]
            current_n = first[2].shape[0]
            deadline = time.perf_counter() + self.max_wait_seconds
            while current_n < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    item = self.request_queue.get(timeout=remaining)
                except Empty:
                    break
                if item is None:
                    self._stop.set()
                    break
                requests.append(item)
                current_n += item[2].shape[0]
            self._run_batch(requests)

    def _run_batch(self, requests):
        sizes = [r[2].shape[0] for r in requests]
        big = np.concatenate([r[2] for r in requests], axis=0)
        x = torch.from_numpy(big).to(self.device, non_blocking=True)
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
        offset = 0
        for sz, (worker_id, request_id, _arr) in zip(sizes, requests):
            self.response_queues[worker_id].put(
                (
                    request_id,
                    policies[offset:offset + sz].copy(),
                    values_np[offset:offset + sz].copy(),
                )
            )
            offset += sz
        with self._stats_lock:
            total = sum(sizes)
            self._batches += 1
            self._positions += total
            if total > self._max_observed:
                self._max_observed = total


class MPInferenceClient:
    """Worker-side proxy. Same callable contract as LocalEvaluator.

    Constructed in each worker process; sends requests on the shared
    request_queue, blocks on its dedicated response_queue.
    """

    def __init__(self, worker_id, request_queue, response_queue):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._next_id = 0

    def __call__(self, state_tensors):
        if hasattr(state_tensors, "numpy"):
            arr = state_tensors.detach().contiguous().numpy()
        else:
            arr = np.ascontiguousarray(state_tensors)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        request_id = self._next_id
        self._next_id += 1
        self.request_queue.put((self.worker_id, request_id, arr))
        resp_id, policies, values = self.response_queue.get()
        if resp_id != request_id:
            raise RuntimeError(
                f"out-of-order inference response: expected {request_id},"
                f" got {resp_id}"
            )
        return policies, values
