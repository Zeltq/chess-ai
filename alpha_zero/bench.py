"""Microbenchmarks for the chess-AZ hot paths.

Run from alpha_zero/:
    python bench.py
    python bench.py --section forward    # only forward-pass scaling
    python bench.py --section selfplay   # end-to-end pos/s

Prints a small report. Not part of the test suite — meant as a baseline
tool for tracking the effect of changes.
"""
import argparse
import time
from contextlib import contextmanager

import chess
import numpy as np
import torch

from env.chess_env import Chess
from mcts.inference import InferenceServer, LocalEvaluator
from model.net import AlphaZeroNet
from training.self_play import self_play_game


@contextmanager
def section(name):
    print(f"\n=== {name} ===")
    t = time.perf_counter()
    yield
    print(f"  total: {time.perf_counter() - t:.2f}s")


def random_board(rng, plies):
    b = chess.Board()
    for _ in range(plies):
        moves = list(b.legal_moves)
        if not moves:
            break
        b.push(moves[int(rng.integers(0, len(moves)))])
        if b.is_game_over():
            break
    return b


def bench_encode(repeat=5000):
    g = Chess()
    rng = np.random.default_rng(0)
    b = random_board(rng, 30)
    t = time.perf_counter()
    for _ in range(repeat):
        g.encode_state(b)
    elapsed = time.perf_counter() - t
    print(f"  encode_state: {elapsed * 1e6 / repeat:.1f} us/call ({repeat} reps)")


def bench_legal_actions(repeat=5000):
    g = Chess()
    rng = np.random.default_rng(0)
    b = random_board(rng, 30)
    t = time.perf_counter()
    for _ in range(repeat):
        g.get_valid_actions(b)
    elapsed = time.perf_counter() - t
    print(f"  get_valid_actions: {elapsed * 1e6 / repeat:.1f} us/call ({repeat} reps)")


def bench_forward(device, channels=224, blocks=8):
    m = AlphaZeroNet(channels=channels, num_res_blocks=blocks).to(device)
    if device.type == "cuda":
        m = m.to(memory_format=torch.channels_last)
    m.eval()
    amp_dtype = (
        torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16 if device.type == "cuda"
        else torch.float32
    )
    print(f"  model: {channels}ch x {blocks} blocks, dtype={amp_dtype}")
    for batch in (1, 8, 32, 64, 128, 256):
        x = torch.zeros(batch, 19, 8, 8, device=device)
        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        # warmup
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"
        ):
            for _ in range(5):
                m(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        n = 50
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"
        ):
            for _ in range(n):
                m(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t
        per_call = elapsed * 1000 / n
        per_pos = elapsed * 1e6 / (n * batch)
        print(
            f"  batch={batch:4d}: {per_call:6.2f} ms/call, "
            f"{per_pos:6.1f} us/pos, {batch * n / elapsed:7.0f} pos/s"
        )


def bench_selfplay(device, parallel=4, sims=32, max_moves=40,
                   channels=64, blocks=2):
    m = AlphaZeroNet(channels=channels, num_res_blocks=blocks).to(device)
    if device.type == "cuda":
        m = m.to(memory_format=torch.channels_last)
    m.eval()

    g = Chess()
    print(f"  model: {channels}ch x {blocks} blocks, sims={sims}, "
          f"max_moves={max_moves}")

    print("  -- sequential (LocalEvaluator)")
    ev = LocalEvaluator(m, device)
    t = time.perf_counter()
    samples, _, _, _, _ = self_play_game(
        g, ev, num_simulations=sims, max_moves=max_moves, mcts_batch_size=8
    )
    elapsed = time.perf_counter() - t
    print(f"     1 game: {len(samples)} pos in {elapsed:.2f}s "
          f"= {len(samples) / elapsed:.1f} pos/s")

    print(f"  -- threaded (InferenceServer, {parallel} threads)")
    import concurrent.futures as cf
    server = InferenceServer(m, device, max_batch_size=128, max_wait_seconds=0.002)

    def run_one(_):
        return self_play_game(
            Chess(), server, num_simulations=sims, max_moves=max_moves,
            mcts_batch_size=8,
        )

    t = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=parallel) as pool:
        results = list(pool.map(run_one, range(parallel)))
    elapsed = time.perf_counter() - t
    total = sum(len(r[0]) for r in results)
    stats = server.stats_and_reset()
    server.stop()
    print(f"     {parallel} games: {total} pos in {elapsed:.2f}s "
          f"= {total / elapsed:.1f} pos/s")
    print(f"     server: {stats['batches']} batches, "
          f"avg {stats['avg_batch']:.0f}, max {stats['max_batch']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--section", choices=["all", "encode", "actions", "forward", "selfplay"],
        default="all",
    )
    parser.add_argument("--channels", type=int, default=224)
    parser.add_argument("--res-blocks", type=int, default=8)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--sims", type=int, default=32)
    parser.add_argument("--max-moves", type=int, default=40)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        print(f"device: {torch.cuda.get_device_name(0)} "
              f"(bf16={torch.cuda.is_bf16_supported()})")
    else:
        print("device: cpu")

    if args.section in ("all", "encode"):
        with section("encode_state / get_valid_actions"):
            bench_encode()
            bench_legal_actions()

    if args.section in ("all", "forward"):
        with section("forward pass scaling"):
            bench_forward(device, channels=args.channels, blocks=args.res_blocks)

    if args.section in ("all", "selfplay"):
        with section("self-play throughput"):
            # Use the same (channels, res_blocks) the user trains with so
            # the numbers match real workloads.
            bench_selfplay(
                device, parallel=args.parallel, sims=args.sims,
                max_moves=args.max_moves,
                channels=args.channels, blocks=args.res_blocks,
            )


if __name__ == "__main__":
    main()
