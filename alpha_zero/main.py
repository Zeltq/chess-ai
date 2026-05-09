import argparse
import csv
from datetime import UTC, datetime
import math
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
import chess
import chess.engine
import chess.pgn
import torch
from torch.utils.tensorboard import SummaryWriter

from env.chess_env import Chess
from model.net import AlphaZeroNet
from training.buffer import ReplayBuffer
from training.self_play import find_immediate_checkmate_action, self_play_game
from training.train import build_batch, train_on_batch
from mcts.mcts import MCTS
from mcts.inference import (
    LocalEvaluator,
    MPInferenceClient,
    MPInferenceServer,
)


def _selfplay_worker(
    worker_id, request_queue, response_queue,
    job_queue, result_queue, base_game_kwargs, seed, cwd,
):
    """Worker process for parallel self-play.

    Each worker runs an independent MCTS over CPU; inference goes through
    a process-shared GPU server in the main process. Workers stay alive
    across iterations and pull jobs from `job_queue`. A None job is the
    shutdown sentinel.
    """
    import sys
    os.chdir(cwd)
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    import random as _random
    if seed is not None:
        _random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
    torch.set_num_threads(1)

    from env.chess_env import Chess as _Chess
    from training.self_play import self_play_game as _spg

    game = _Chess()
    evaluator = MPInferenceClient(worker_id, request_queue, response_queue)

    while True:
        job = job_queue.get()
        if job is None:
            break
        game_id, capture_reward_scale = job
        kwargs = dict(base_game_kwargs)
        kwargs["capture_reward_scale"] = capture_reward_scale
        t0 = time.perf_counter()
        try:
            samples, final_state, moves, result, stats = _spg(
                game=game, evaluator=evaluator, **kwargs,
            )
        except Exception as exc:  # surface failures rather than hang the queue
            result_queue.put((game_id, None, None, None, None, None, 0.0, repr(exc)))
            continue
        elapsed = time.perf_counter() - t0
        result_queue.put(
            (game_id, samples, final_state, moves, result, stats, elapsed, None)
        )


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero chess training and play")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a chess model")
    train_parser.add_argument("--iterations", type=int, default=20)
    train_parser.add_argument("--games-per-iteration", type=int, default=8)
    train_parser.add_argument("--simulations", type=int, default=64)
    train_parser.add_argument(
        "--max-moves",
        type=int,
        default=400,
        help="Self-play game cap in half-moves. Default 400 is about 200 full moves.",
    )
    train_parser.add_argument("--epochs", type=int, default=4)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument(
        "--lr-milestones", type=int, nargs="+", default=None,
        help="Iterations at which to multiply lr by --lr-decay-factor. "
             "Default: 50%% and 75%% of --iterations.",
    )
    train_parser.add_argument("--lr-decay-factor", type=float, default=0.1)
    train_parser.add_argument("--draw-score", type=float, default=0.45)
    train_parser.add_argument("--capture-reward-scale", type=float, default=0.01)
    train_parser.add_argument("--capture-reward-cap", type=float, default=0.2)
    train_parser.add_argument(
        "--capture-reward-ramp-iterations", type=int, default=None,
        help="Linearly decay capture_reward_scale to zero over this many "
             "iterations (default: max(1, iterations // 3)). Capture-reward "
             "shaping helps bootstrap early but biases value targets toward "
             "materialism in the long run, so it ramps off.",
    )
    train_parser.add_argument("--buffer-size", type=int, default=50000)
    train_parser.add_argument(
        "--mirror-augment", action="store_true",
        help="Augment training samples with horizontal-mirror copies (50%% "
             "chance per sample). Effectively doubles data diversity but "
             "introduces small distribution shift on positions with castling "
             "rights, since chess geometry is not perfectly file-symmetric.",
    )
    train_parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    train_parser.add_argument("--records-dir", type=Path, default=Path("games"))
    train_parser.add_argument("--metrics-dir", type=Path, default=Path("metrics"))
    train_parser.add_argument("--metrics-file", type=Path, default=None)
    train_parser.add_argument("--tensorboard-dir", type=Path, default=Path("runs"))
    train_parser.add_argument("--no-tensorboard", action="store_true")
    train_parser.add_argument("--no-save-buffer", action="store_true")
    train_parser.add_argument("--channels", type=int, default=224)
    train_parser.add_argument("--res-blocks", type=int, default=8)
    train_parser.add_argument(
        "--record-games", type=int, default=None,
        help="Max games per iteration to save as PGN. Default: all games.",
    )
    train_parser.add_argument("--time-limit-minutes", type=float, default=None)
    train_parser.add_argument(
        "--mcts-batch-size", type=int, default=8,
        help="Leaves evaluated per NN forward pass in MCTS. Higher = better GPU utilization.",
    )
    train_parser.add_argument(
        "--fpu-reduction", type=float, default=0.25,
        help="First-Play Urgency reduction. Estimated Q for unvisited children "
             "is parent.value - fpu_reduction. Set 0 for the legacy Q=0 behavior.",
    )
    train_parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature applied to MCTS visit counts during the "
             "opening (until --temperature-drop-move). 1.0 = match AZ paper.",
    )
    train_parser.add_argument(
        "--temperature-drop-move", type=int, default=30,
        help="Half-move index after which temperature drops to 0 (greedy).",
    )
    train_parser.add_argument(
        "--adjudicate-material", type=float, default=None,
        help="Material lead (in pawn units) that triggers a decisive result "
             "if held for --adjudicate-consecutive plies after move "
             "--adjudicate-min-move. Bootstraps win/loss signal in early "
             "training when MCTS would otherwise draw everything by "
             "repetition. Recommended: 5.0. None disables.",
    )
    train_parser.add_argument("--adjudicate-min-move", type=int, default=40)
    train_parser.add_argument("--adjudicate-consecutive", type=int, default=3)
    train_parser.add_argument(
        "--c-puct-base", type=float, default=None,
        help="Enable AlphaZero log-c_puct: c(N) = log((1+N+base)/base) + init. "
             "Recommended only with high simulation counts; default disables.",
    )
    train_parser.add_argument(
        "--c-puct-init", type=float, default=1.25,
        help="Init term in the log-c_puct formula (only used with --c-puct-base).",
    )
    train_parser.add_argument(
        "--parallel-games", type=int, default=1,
        help="Number of self-play games to run in parallel as Python threads "
             "sharing one GPU model via the InferenceServer. Default 1 = "
             "sequential with a LocalEvaluator.",
    )
    train_parser.add_argument(
        "--server-batch-size", type=int, default=256,
        help="Max batch the inference server will assemble before flushing. "
             "Larger = better GPU utilisation, slightly higher latency.",
    )
    train_parser.add_argument(
        "--server-max-wait-ms", type=float, default=2.0,
        help="How long the inference server waits for more requests before "
             "flushing a partial batch.",
    )
    train_parser.add_argument("--eval-engine", type=Path, default=None)
    train_parser.add_argument("--eval-time", type=float, default=0.05)
    train_parser.add_argument("--eval-games", type=int, default=1)
    train_parser.add_argument("--resume", type=Path, default=None)
    train_parser.add_argument(
        "--reset-lr", action="store_true",
        help="Ignore saved scheduler state and reset LR to --learning-rate with new milestones.",
    )
    train_parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed numpy / random / torch RNGs for reproducibility. CUDA is "
             "still non-deterministic at the kernel level even with a seed.",
    )

    play_parser = subparsers.add_parser("play", help="Play against a saved checkpoint")
    play_parser.add_argument("--checkpoint", type=Path, required=True)
    play_parser.add_argument("--color", choices=["white", "black"], default="white")
    play_parser.add_argument("--simulations", type=int, default=128)

    gui_parser = subparsers.add_parser("gui", help="Play against a checkpoint in GUI")
    gui_parser.add_argument("--checkpoint", type=Path, required=True)
    gui_parser.add_argument("--color", choices=["white", "black"], default="white")
    gui_parser.add_argument("--simulations", type=int, default=128)

    viewer_parser = subparsers.add_parser("pgn-viewer", help="Preview recorded PGN games")
    viewer_parser.add_argument("--records-dir", type=Path, default=Path("games"))
    viewer_parser.add_argument("--pgn", type=Path, default=None)
    viewer_parser.add_argument("--latest", action="store_true")
    viewer_parser.add_argument("--poll-seconds", type=float, default=2.0)

    return parser.parse_args()


def create_model(device, model_config=None):
    config = model_config or {}
    model = AlphaZeroNet(**config).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


def autocast_dtype(device):
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def maybe_compile(model, device):
    """Compile only the forward method so state_dict / save / load stay simple."""
    if device.type != "cuda":
        return model
    try:
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
    except Exception as exc:
        print_kv("compile", [("skipped", exc)])
    return model


def describe_device(device):
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        cap = torch.cuda.get_device_capability(index)
        return f"cuda:{index} ({name}, sm_{cap[0]}{cap[1]})"
    return "cpu"


def select_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print_kv("device", [("cuda", "yes"), ("name", describe_device(device))])
    else:
        device = torch.device("cpu")
        print_kv("device", [("cuda", "no — falling back to CPU")])
    return device


def timestamp_slug():
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def print_title(title):
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")


def print_kv(prefix, items):
    print(f"{prefix:<10} | " + " | ".join(f"{key}: {value}" for key, value in items))


def save_checkpoint(path, model, optimizer, scheduler, iteration, replay_buffer_size, metadata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": model.get_config(),
        "iteration": iteration,
        "replay_buffer_size": replay_buffer_size,
        "saved_at": datetime.now(UTC).isoformat(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def save_buffer(replay_buffer, path):
    data = list(replay_buffer.buffer)
    if not data:
        return 0
    states = np.stack([x[0] for x in data])
    policies = np.stack([x[1] for x in data])
    values = np.array([x[2] for x in data], dtype=np.float32)
    path = Path(path)
    np.savez_compressed(path, states=states, policies=policies, values=values)
    saved = path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")
    return saved.stat().st_size


def load_buffer(replay_buffer, path):
    data = np.load(path)
    for state, policy, value in zip(data["states"], data["policies"], data["values"]):
        replay_buffer.buffer.append((state, policy, float(value)))


def append_metrics_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_self_play_pgn(path, moves, result, iteration, game_index):
    path.parent.mkdir(parents=True, exist_ok=True)
    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "AlphaZero self-play"
    pgn_game.headers["Site"] = "local"
    pgn_game.headers["Date"] = datetime.now(UTC).strftime("%Y.%m.%d")
    pgn_game.headers["Round"] = f"{iteration}.{game_index}"
    pgn_game.headers["White"] = f"AlphaZero iter {iteration}"
    pgn_game.headers["Black"] = f"AlphaZero iter {iteration}"
    pgn_game.headers["Result"] = result

    node = pgn_game
    board = chess.Board()
    for move in moves:
        node = node.add_variation(move)
        board.push(move)

    with path.open("w", encoding="utf-8") as handle:
        print(pgn_game, file=handle, end="\n\n")


def score_to_cp(score, pov_color):
    return score.pov(pov_color).score(mate_score=100000)


def first_engine_info(info):
    return info[0] if isinstance(info, list) else info


def move_accuracy_from_cp_loss(cp_loss):
    capped_loss = min(max(cp_loss, 0), 1000)
    return 100.0 * math.exp(-capped_loss / 250.0)


def evaluate_game_with_engine(engine, moves, time_limit):
    board = chess.Board()
    cp_losses = []
    accuracies = []

    for move in moves:
        if move not in board.legal_moves:
            break

        side_to_move = board.turn
        before_info = engine.analyse(
            board,
            chess.engine.Limit(time=time_limit),
            multipv=1,
        )
        before_info = first_engine_info(before_info)
        best_score = score_to_cp(before_info["score"], side_to_move)

        board.push(move)
        after_info = engine.analyse(
            board,
            chess.engine.Limit(time=time_limit),
            multipv=1,
        )
        after_info = first_engine_info(after_info)
        played_score = score_to_cp(after_info["score"], side_to_move)

        cp_loss = max(0, best_score - played_score)
        cp_losses.append(cp_loss)
        accuracies.append(move_accuracy_from_cp_loss(cp_loss))

    if not cp_losses:
        return None

    return {
        "moves": len(cp_losses),
        "avg_cp_loss": sum(cp_losses) / len(cp_losses),
        "avg_accuracy": sum(accuracies) / len(accuracies),
    }


def train_command(args):
    if not 0.0 <= args.draw_score <= 1.0:
        raise ValueError("--draw-score must be between 0 and 1")
    if args.max_moves is not None and args.max_moves <= 0:
        raise ValueError("--max-moves must be positive")
    if args.capture_reward_scale < 0.0:
        raise ValueError("--capture-reward-scale must be non-negative")
    if not 0.0 <= args.capture_reward_cap <= 1.0:
        raise ValueError("--capture-reward-cap must be between 0 and 1")

    if args.seed is not None:
        import random as _random
        _random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = select_device()
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    game = Chess()

    model_config = {
        "channels": args.channels,
        "num_res_blocks": args.res_blocks,
    }
    resumed_checkpoint = None
    if args.resume is not None:
        resumed_checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        if "model_config" in resumed_checkpoint:
            model_config = resumed_checkpoint["model_config"]

    model = create_model(device, model_config)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    milestones = args.lr_milestones or [
        int(0.5 * args.iterations),
        int(0.75 * args.iterations),
    ]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=args.lr_decay_factor
    )

    amp_dtype = autocast_dtype(device)
    # bf16 has fp32 exponent range; no GradScaler needed. fp16 still needs it.
    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda" and amp_dtype == torch.float16
        else None
    )
    # Self-play parallelism: each worker game runs in its own *process*,
    # which sidesteps Python's GIL. Inference is centralized in the main
    # process via MPInferenceServer; workers ship encoded states over a
    # multiprocessing.Queue and block on per-worker response queues.
    inference_server = None
    mp_workers = []
    mp_request_queue = None
    mp_response_queues = None
    mp_job_queue = None
    mp_result_queue = None
    if args.parallel_games > 1:
        ctx = mp.get_context("spawn")
        mp_request_queue = ctx.Queue()
        mp_response_queues = [ctx.Queue() for _ in range(args.parallel_games)]
        mp_job_queue = ctx.Queue()
        mp_result_queue = ctx.Queue()
        # Server starts now so the spawned workers can immediately push
        # requests as soon as they finish initialising.
        inference_server = MPInferenceServer(
            model,
            device,
            mp_request_queue,
            mp_response_queues,
            max_batch_size=args.server_batch_size,
            max_wait_seconds=args.server_max_wait_ms / 1000.0,
        )
    replay_buffer = ReplayBuffer(args.buffer_size, mirror_augment=args.mirror_augment)
    run_id = timestamp_slug()
    start_iteration = 1
    resumed_metrics_file = None
    if resumed_checkpoint is not None:
        model.load_state_dict(resumed_checkpoint["model_state_dict"])
        if "optimizer_state_dict" in resumed_checkpoint:
            optimizer.load_state_dict(resumed_checkpoint["optimizer_state_dict"])
        if not args.reset_lr and "scheduler_state_dict" in resumed_checkpoint:
            scheduler.load_state_dict(resumed_checkpoint["scheduler_state_dict"])
        if args.reset_lr:
            for pg in optimizer.param_groups:
                pg["lr"] = args.learning_rate
        start_iteration = int(resumed_checkpoint.get("iteration", 0)) + 1
        metadata = resumed_checkpoint.get("metadata", {})
        resumed_metrics_file = metadata.get("metrics_file")
        resumed_milestones = metadata.get("lr_milestones")
        if not args.reset_lr and args.lr_milestones is None and resumed_milestones:
            milestones = resumed_milestones

        buffer_path = args.resume.parent / "buffer.npz"
        if buffer_path.exists():
            load_buffer(replay_buffer, buffer_path)
            print_kv("resume", [("buffer", f"{len(replay_buffer)} positions loaded from {buffer_path.name}")])

    # Compile after weights are loaded so the first forward pass triggers
    # compile on real shapes. Skipped automatically when an InferenceServer
    # is in play (variable batch sizes would re-trigger compilation).
    if inference_server is None:
        model = maybe_compile(model, device)

    local_evaluator = LocalEvaluator(model, device)

    eval_engine = None
    if args.eval_engine is not None:
        eval_engine = chess.engine.SimpleEngine.popen_uci(str(args.eval_engine))

    draw_value = 2.0 * args.draw_score - 1.0
    game_kwargs = {
        "num_simulations": args.simulations,
        "max_moves": args.max_moves,
        "draw_value": draw_value,
        "capture_reward_scale": args.capture_reward_scale,
        "capture_reward_cap": args.capture_reward_cap,
        "mcts_batch_size": args.mcts_batch_size,
        "fpu_reduction": args.fpu_reduction,
        "temperature": args.temperature,
        "temperature_drop_move": args.temperature_drop_move,
        "c_puct_base": args.c_puct_base,
        "c_puct_init": args.c_puct_init,
        "adjudicate_material": args.adjudicate_material,
        "adjudicate_min_move": args.adjudicate_min_move,
        "adjudicate_consecutive": args.adjudicate_consecutive,
    }
    if args.metrics_file is not None:
        metrics_file = args.metrics_file
    elif resumed_metrics_file:
        metrics_file = Path(resumed_metrics_file)
    else:
        metrics_file = args.metrics_dir / f"{run_id}_training_metrics.csv"

    print_title("AlphaZero chess training")
    print_kv(
        "run",
        [
            ("id", run_id),
            ("device", device),
            ("start_iter", start_iteration),
            ("target_iter", args.iterations),
        ],
    )
    print_kv(
        "selfplay",
        [
            ("games/iter", args.games_per_iteration),
            ("sims", args.simulations),
            ("max_moves", args.max_moves),
            ("draw", f"{args.draw_score:g} -> {draw_value:+.2f}"),
            (
                "capture",
                f"{args.capture_reward_scale:g} cap {args.capture_reward_cap:g}",
            ),
        ],
    )
    print_kv(
        "train",
        [
            ("epochs", args.epochs),
            ("batch", args.batch_size),
            ("buffer", args.buffer_size),
            ("lr", args.learning_rate),
            ("wd", args.weight_decay),
            ("lr_milestones", milestones),
            ("lr_factor", args.lr_decay_factor),
        ],
    )
    if args.resume is not None:
        print_kv("resume", [("checkpoint", args.resume)])
    if args.time_limit_minutes is not None:
        print_kv("limit", [("time", f"{args.time_limit_minutes:g} min")])
    if args.eval_engine is not None:
        print_kv(
            "audit",
            [
                ("engine", args.eval_engine),
                ("time/move", f"{args.eval_time:g}s"),
                ("games/iter", args.eval_games),
            ],
        )
    print_kv("metrics", [("file", metrics_file)])
    num_params = sum(p.numel() for p in model.parameters())
    print_kv(
        "model",
        [
            ("channels", model.channels),
            ("res_blocks", model.num_res_blocks),
            ("params", f"{num_params:,}"),
        ],
    )

    writer = None
    if not args.no_tensorboard:
        tb_log_dir = args.tensorboard_dir / run_id
        writer = SummaryWriter(log_dir=str(tb_log_dir))
        print_kv("tensorboard", [("dir", tb_log_dir)])
        try:
            dummy = torch.zeros(
                1, model.input_channels, model.board_size, model.board_size,
                device=device,
            )
            model.eval()
            writer.add_graph(model, dummy)
        except Exception as exc:
            print_kv("tensorboard", [("graph", f"skipped: {exc}")])

    global_step = 0
    global_game = 0
    completed_iteration_times = []
    training_started_at = time.perf_counter()
    time_limit_seconds = (
        args.time_limit_minutes * 60
        if args.time_limit_minutes is not None
        else None
    )
    total_iterations = args.iterations - start_iteration + 1

    capture_ramp_iters = max(
        1,
        args.capture_reward_ramp_iterations
        if args.capture_reward_ramp_iterations is not None
        else args.iterations // 3,
    )

    for iteration in range(start_iteration, args.iterations + 1):
        iteration_started_at = time.perf_counter()
        model.eval()
        iteration_samples = 0
        results = {"white_wins": 0, "black_wins": 0, "draws": 0}
        engine_metrics = []

        # Linear ramp-down: full scale at iteration 1, zero from
        # capture_ramp_iters onward.
        ramp_progress = max(0.0, 1.0 - (iteration - 1) / capture_ramp_iters)
        game_kwargs["capture_reward_scale"] = (
            args.capture_reward_scale * ramp_progress
        )

        if completed_iteration_times:
            avg_iter_time = sum(completed_iteration_times) / len(completed_iteration_times)
            remaining = args.iterations - iteration + 1
            print_title(
                f"Iteration {iteration}/{args.iterations}"
                f"  |  ETA {format_duration(remaining * avg_iter_time)}"
            )
        else:
            print_title(f"Iteration {iteration}/{args.iterations}")

        game_times = []

        # Lazily start workers on the first iteration that uses them so the
        # initial spawn cost (numba JIT, torch import) overlaps with iter 1
        # only and not with arg parsing.
        if mp_request_queue is not None and not mp_workers:
            for wid in range(args.parallel_games):
                p = mp.get_context("spawn").Process(
                    target=_selfplay_worker,
                    args=(
                        wid,
                        mp_request_queue,
                        mp_response_queues[wid],
                        mp_job_queue,
                        mp_result_queue,
                        game_kwargs,
                        args.seed,
                        os.getcwd(),
                    ),
                    daemon=False,
                )
                p.start()
                mp_workers.append(p)

        if mp_request_queue is not None:
            # Dispatch this iteration's games. capture_reward_scale rides on
            # the job tuple so the per-iter ramp value reaches workers without
            # any state broadcast.
            scale = game_kwargs["capture_reward_scale"]
            for gi in range(1, args.games_per_iteration + 1):
                mp_job_queue.put((gi, scale))

            def _drain():
                completed = 0
                while completed < args.games_per_iteration:
                    item = mp_result_queue.get()
                    completed += 1
                    game_id, samples, final_state, moves, result, st, elapsed, exc = item
                    if exc is not None:
                        raise RuntimeError(f"worker game {game_id} failed: {exc}")
                    yield game_id, samples, final_state, moves, result, st, elapsed
            games_source = _drain()
        else:
            evaluator = local_evaluator
            def _seq():
                for gi in range(1, args.games_per_iteration + 1):
                    t0 = time.perf_counter()
                    s, fs, mv, r, st = self_play_game(game=game, evaluator=evaluator, **game_kwargs)
                    yield gi, s, fs, mv, r, st, time.perf_counter() - t0
            games_source = _seq()

        iteration_reused_total = 0.0
        iteration_reused_moves = 0

        for (
            game_index, samples, final_state, moves, result, game_stats, game_elapsed
        ) in games_source:
            iteration_reused_total += game_stats["reused_visits_avg"] * game_stats["reused_visits_moves"]
            iteration_reused_moves += game_stats["reused_visits_moves"]
            game_times.append(game_elapsed)
            replay_buffer.add_game(samples)
            iteration_samples += len(samples)
            global_game += 1

            outcome = final_state.outcome(claim_draw=True)
            if outcome is not None and outcome.winner is not None:
                if outcome.winner == chess.WHITE:
                    results["white_wins"] += 1
                else:
                    results["black_wins"] += 1
            else:
                results["draws"] += 1

            pgn_path = None
            if args.record_games is None or game_index <= args.record_games:
                pgn_path = (
                    args.records_dir
                    / f"{run_id}_iter_{iteration:04d}_game_{game_index:02d}.pgn"
                )
                save_self_play_pgn(pgn_path, moves, result, iteration, game_index)

            metrics = None
            if eval_engine is not None and game_index <= args.eval_games:
                eval_started_at = time.perf_counter()
                try:
                    metrics = evaluate_game_with_engine(eval_engine, moves, args.eval_time)
                except chess.engine.EngineError as exc:
                    print_kv("audit", [("game", game_index), ("skipped", exc)])
                eval_elapsed = time.perf_counter() - eval_started_at
                if metrics is not None:
                    engine_metrics.append(metrics)

            completed_count = len(game_times)
            if mp_request_queue is None:
                avg_game_time = sum(game_times) / completed_count
                eta_sp_str = format_duration(
                    (args.games_per_iteration - completed_count) * avg_game_time
                )
            else:
                eta_sp_str = "parallel"

            game_items = [
                ("game", f"{game_index}/{args.games_per_iteration}"),
                ("result", result),
                ("pos", len(samples)),
                ("speed", f"{len(samples) / max(game_elapsed, 1e-9):.1f} pos/s"),
                ("eta_sp", eta_sp_str),
            ]
            if pgn_path is not None:
                game_items.append(("pgn", pgn_path.name))
            if metrics is not None:
                game_items.extend(
                    [
                        ("acc", f"{metrics['avg_accuracy']:.1f}%"),
                        ("cpl", f"{metrics['avg_cp_loss']:.0f}"),
                        ("eval", format_duration(eval_elapsed)),
                    ]
                )
            print_kv("selfplay", game_items)

            if writer is not None:
                game_result_val = (
                    1.0 if result == "1-0"
                    else -1.0 if result == "0-1"
                    else 0.0
                )
                writer.add_scalar("selfplay/game_positions", len(samples), global_game)
                writer.add_scalar(
                    "selfplay/game_speed_pos_per_s",
                    len(samples) / max(game_elapsed, 1e-9),
                    global_game,
                )
                writer.add_scalar("selfplay/game_duration_s", game_elapsed, global_game)
                writer.add_scalar("selfplay/game_result", game_result_val, global_game)
                writer.add_scalar("selfplay/game_moves", len(moves), global_game)
                writer.flush()

        summary_items = [
            ("positions", iteration_samples),
            ("buffer", len(replay_buffer)),
            (
                "W/B/D",
                f"{results['white_wins']}/{results['black_wins']}/{results['draws']}",
            ),
        ]
        if iteration_reused_moves > 0:
            summary_items.append(
                ("reused_v",
                 f"{iteration_reused_total / iteration_reused_moves:.1f}")
            )
        if inference_server is not None:
            ifs = inference_server.stats_and_reset()
            if ifs["batches"] > 0:
                summary_items.append(("ifs_batch", f"{ifs['avg_batch']:.0f}"))
                summary_items.append(("ifs_max", str(ifs["max_batch"])))
            # Side effect: by reading we reset, so the per-iter window is clean.
            # Stash for the writer block below.
            self_play_inference_stats = ifs
        else:
            self_play_inference_stats = None
        print_kv("summary", summary_items)
        if engine_metrics:
            avg_accuracy = sum(
                metric["avg_accuracy"] for metric in engine_metrics
            ) / len(engine_metrics)
            avg_cp_loss = sum(
                metric["avg_cp_loss"] for metric in engine_metrics
            ) / len(engine_metrics)
            print_kv(
                "audit avg",
                [
                    ("acc", f"{avg_accuracy:.1f}%"),
                    ("cpl", f"{avg_cp_loss:.1f}"),
                    ("games", len(engine_metrics)),
                ],
            )
        else:
            avg_accuracy = None
            avg_cp_loss = None

        model.train()
        if len(replay_buffer) == 0:
            raise RuntimeError("Replay buffer is empty after self-play")

        loss_history = []
        policy_loss_history = []
        value_loss_history = []
        for epoch in range(1, args.epochs + 1):
            epoch_started_at = time.perf_counter()
            steps = max(1, len(replay_buffer) // args.batch_size)
            epoch_losses = []
            trained_samples = 0
            for _ in range(steps):
                batch_samples = replay_buffer.sample(args.batch_size)
                batch = build_batch(batch_samples)
                metrics = train_on_batch(model, optimizer, batch, device, scaler)
                epoch_losses.append(metrics["loss"])
                policy_loss_history.append(metrics["policy_loss"])
                value_loss_history.append(metrics["value_loss"])
                trained_samples += len(batch_samples)
                global_step += 1
                if writer is not None:
                    writer.add_scalar("loss/total", metrics["loss"], global_step)
                    writer.add_scalar("loss/policy", metrics["policy_loss"], global_step)
                    writer.add_scalar("loss/value", metrics["value_loss"], global_step)
                    writer.add_scalar(
                        "lr",
                        optimizer.param_groups[0]["lr"],
                        global_step,
                    )

            average_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            epoch_elapsed = time.perf_counter() - epoch_started_at
            loss_history.extend(epoch_losses)
            print_kv(
                "train",
                [
                    ("epoch", f"{epoch}/{args.epochs}"),
                    ("loss", f"{average_epoch_loss:.4f}"),
                    (
                        "speed",
                        f"{trained_samples / max(epoch_elapsed, 1e-9):.1f} samples/s",
                    ),
                ],
            )

        checkpoint_name = f"{run_id}_alpha_zero_chess_iter_{iteration:04d}.pt"
        checkpoint_path = args.checkpoint_dir / checkpoint_name
        metadata = {
            "run_id": run_id,
            "metrics_file": str(metrics_file),
            "games_per_iteration": args.games_per_iteration,
            "simulations": args.simulations,
            "max_moves": args.max_moves,
            "draw_score": args.draw_score,
            "draw_value": draw_value,
            "capture_reward_scale": args.capture_reward_scale,
            "capture_reward_cap": args.capture_reward_cap,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr_milestones": milestones,
            "lr_decay_factor": args.lr_decay_factor,
            "average_loss": sum(loss_history) / len(loss_history),
            "engine_eval": {
                "engine": str(args.eval_engine) if args.eval_engine else None,
                "time": args.eval_time,
                "games": len(engine_metrics),
                "average_accuracy": avg_accuracy,
                "average_cp_loss": avg_cp_loss,
            },
        }
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            iteration,
            len(replay_buffer),
            metadata=metadata,
        )
        save_checkpoint(
            args.checkpoint_dir / "latest.pt",
            model,
            optimizer,
            scheduler,
            iteration,
            len(replay_buffer),
            metadata=metadata,
        )
        if not args.no_save_buffer:
            buffer_path = args.checkpoint_dir / "buffer.npz"
            buffer_bytes = save_buffer(replay_buffer, buffer_path)
            print_kv("buffer", [
                ("saved", buffer_path.name),
                ("positions", len(replay_buffer)),
                ("size", f"{buffer_bytes / 1024 / 1024:.1f} MB"),
            ])
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr != args.learning_rate:
            print_kv("lr", [("decayed_to", f"{current_lr:.2e}")])
        if writer is not None:
            writer.add_scalar("train/lr", current_lr, iteration)
        iteration_elapsed = time.perf_counter() - iteration_started_at
        completed_iteration_times.append(iteration_elapsed)
        completed_iterations = iteration - start_iteration + 1
        remaining_iterations = total_iterations - completed_iterations
        average_iteration_time = sum(completed_iteration_times) / len(completed_iteration_times)
        eta = remaining_iterations * average_iteration_time
        print_kv(
            "checkpoint",
            [
                ("file", checkpoint_path.name),
                ("iter_time", format_duration(iteration_elapsed)),
                ("eta", format_duration(eta)),
            ],
        )
        if writer is not None:
            avg_loss = sum(loss_history) / len(loss_history)
            avg_policy_loss = sum(policy_loss_history) / len(policy_loss_history)
            avg_value_loss = sum(value_loss_history) / len(value_loss_history)
            writer.add_scalar("selfplay/white_wins", results["white_wins"], iteration)
            writer.add_scalar("selfplay/black_wins", results["black_wins"], iteration)
            writer.add_scalar("selfplay/draws", results["draws"], iteration)
            writer.add_scalar("selfplay/positions", iteration_samples, iteration)
            writer.add_scalar("selfplay/buffer_size", len(replay_buffer), iteration)
            writer.add_scalar(
                "selfplay/speed_pos_per_s",
                iteration_samples / max(iteration_elapsed, 1e-9),
                iteration,
            )
            writer.add_scalar("train/avg_loss", avg_loss, iteration)
            writer.add_scalar("train/avg_policy_loss", avg_policy_loss, iteration)
            writer.add_scalar("train/avg_value_loss", avg_value_loss, iteration)
            writer.add_scalar("time/iteration_seconds", iteration_elapsed, iteration)
            if avg_accuracy is not None:
                writer.add_scalar("audit/accuracy", avg_accuracy, iteration)
            if avg_cp_loss is not None:
                writer.add_scalar("audit/cp_loss", avg_cp_loss, iteration)
            if iteration_reused_moves > 0:
                writer.add_scalar(
                    "mcts/avg_reused_visits",
                    iteration_reused_total / iteration_reused_moves,
                    iteration,
                )
            if self_play_inference_stats is not None and self_play_inference_stats["batches"] > 0:
                writer.add_scalar(
                    "inference/avg_batch_size",
                    self_play_inference_stats["avg_batch"],
                    iteration,
                )
                writer.add_scalar(
                    "inference/max_batch_size",
                    self_play_inference_stats["max_batch"],
                    iteration,
                )
                writer.add_scalar(
                    "inference/positions_per_iter",
                    self_play_inference_stats["positions"],
                    iteration,
                )
            writer.flush()

        append_metrics_row(
            metrics_file,
            {
                "saved_at": datetime.now(UTC).isoformat(),
                "run_id": run_id,
                "iteration": iteration,
                "games_per_iteration": args.games_per_iteration,
                "simulations": args.simulations,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "draw_score": args.draw_score,
                "capture_reward_scale": args.capture_reward_scale,
                "capture_reward_cap": args.capture_reward_cap,
                "positions": iteration_samples,
                "buffer_size": len(replay_buffer),
                "white_wins": results["white_wins"],
                "black_wins": results["black_wins"],
                "draws": results["draws"],
                "average_loss": sum(loss_history) / len(loss_history),
                "engine_accuracy": avg_accuracy,
                "engine_avg_cp_loss": avg_cp_loss,
                "engine_games": len(engine_metrics),
                "iteration_seconds": round(iteration_elapsed, 3),
                "eta_seconds": round(eta, 3),
                "checkpoint_file": str(checkpoint_path),
            },
        )
        if (
            time_limit_seconds is not None
            and time.perf_counter() - training_started_at >= time_limit_seconds
        ):
            print_kv("stop", [("reason", "time limit reached after checkpoint")])
            break

    if mp_workers:
        # Send shutdown sentinels and join workers cleanly.
        for _ in mp_workers:
            mp_job_queue.put(None)
        for p in mp_workers:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
    if inference_server is not None:
        inference_server.stop()
    if eval_engine is not None:
        eval_engine.quit()
    if writer is not None:
        writer.close()


def choose_model_action(game, state, model, num_simulations):
    mate_action = find_immediate_checkmate_action(game, state)
    if mate_action is not None:
        return mate_action

    model.eval()
    device = next(model.parameters()).device
    evaluator = LocalEvaluator(model, device)
    mcts = MCTS(c_puct=1.5)
    root = mcts.run(
        game,
        state,
        evaluator,
        num_simulations=num_simulations,
        add_exploration_noise=False,
    )
    best_idx = int(np.argmax(root.child_visits))
    return int(root.actions[best_idx])


def prompt_human_move(board):
    while True:
        user_input = input("Your move (UCI, e.g. e2e4): ").strip()
        try:
            move = chess.Move.from_uci(user_input)
        except ValueError:
            print("Invalid move format. Try again.")
            continue
        if move in board.legal_moves:
            return move
        print("Illegal move in this position. Try again.")


def play_command(args):
    device = select_device()
    game = Chess()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = create_model(device, checkpoint.get("model_config"))
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"Loaded checkpoint {args.checkpoint} "
        f"(iteration {checkpoint.get('iteration', 'unknown')})"
    )

    board = game.get_initial_state()
    human_color = chess.WHITE if args.color == "white" else chess.BLACK

    while not game.is_terminal(board):
        print("\n" + str(board))
        print(f"FEN: {board.fen()}")
        side_label = "White" if board.turn == chess.WHITE else "Black"
        print(f"Turn: {side_label}")

        if board.turn == human_color:
            move = prompt_human_move(board)
            board.push(move)
            continue

        action = choose_model_action(game, board, model, args.simulations)
        move = game.action_to_move(action, board)
        print(f"Model plays: {move.uci()}")
        board.push(move)

    print("\n" + str(board))
    print(f"Game over: {game.get_result(board)}")


class PGNViewer:
    light_square = "#f0d9b5"
    dark_square = "#b58863"
    last_move_square = "#f2e085"
    pieces = {
        "P": "♙",
        "N": "♘",
        "B": "♗",
        "R": "♖",
        "Q": "♕",
        "K": "♔",
        "p": "♟",
        "n": "♞",
        "b": "♝",
        "r": "♜",
        "q": "♛",
        "k": "♚",
    }

    def __init__(self, records_dir, pgn_path=None, latest=False, poll_ms=2000):
        import tkinter as tk

        self.tk = tk
        self.records_dir = Path(records_dir)
        self.latest = latest
        self.poll_ms = poll_ms
        self.current_path = None
        self.moves = []
        self.move_index = 0
        self.last_move = None

        self.root = self.tk.Tk()
        self.root.title("AlphaZero PGN viewer")
        self.info = self.tk.StringVar()
        self.status = self.tk.StringVar()
        self.buttons = {}

        board_frame = self.tk.Frame(self.root)
        board_frame.pack(padx=12, pady=12)
        for row in range(8):
            for col in range(8):
                button = self.tk.Button(
                    board_frame,
                    width=4,
                    height=2,
                    font=("Arial", 28),
                    state="disabled",
                    disabledforeground="#111111",
                )
                button.grid(row=row, column=col)
                self.buttons[(row, col)] = button

        controls = self.tk.Frame(self.root)
        controls.pack(padx=12, pady=(0, 8))
        self.tk.Button(controls, text="Back", command=self.back).grid(
            row=0,
            column=0,
            padx=4,
        )
        self.tk.Button(controls, text="Forward", command=self.forward).grid(
            row=0,
            column=1,
            padx=4,
        )
        self.tk.Button(controls, text="Reload latest", command=self.load_latest).grid(
            row=0,
            column=2,
            padx=4,
        )
        self.tk.Button(controls, text="Reload file", command=self.reload_current).grid(
            row=0,
            column=3,
            padx=4,
        )

        self.tk.Label(self.root, textvariable=self.info, font=("Arial", 11)).pack(
            padx=12,
            pady=(0, 4),
        )
        self.tk.Label(self.root, textvariable=self.status, font=("Arial", 10)).pack(
            padx=12,
            pady=(0, 12),
        )

        if pgn_path is not None:
            self.load_pgn(Path(pgn_path))
        elif latest:
            self.load_latest()
        else:
            self.status.set("Pass --pgn <file> or --latest.")

        self.render()
        if self.latest:
            self.root.after(500, self.poll_latest)

    def run(self):
        self.root.mainloop()

    def pgn_files(self):
        if not self.records_dir.exists():
            return []
        return sorted(
            self.records_dir.glob("*.pgn"),
            key=lambda path: (path.stat().st_mtime, path.name),
        )

    def load_pgn(self, path):
        try:
            with path.open("r", encoding="utf-8") as handle:
                pgn_game = chess.pgn.read_game(handle)
        except OSError:
            return False

        if pgn_game is None:
            return False

        self.current_path = path
        self.moves = list(pgn_game.mainline_moves())
        self.move_index = 0
        self.last_move = None
        self.render()
        return True

    def poll_latest(self):
        self.load_latest(silent=True)
        self.update_status()
        self.root.after(self.poll_ms, self.poll_latest)

    def load_latest(self, silent=False):
        files = self.pgn_files()
        if not files:
            if not silent:
                self.status.set("No PGN files yet. Waiting for recorded games.")
            return

        latest_path = files[-1]
        if self.current_path == latest_path and silent:
            return
        self.load_pgn(latest_path)

    def reload_current(self):
        if self.current_path is None:
            self.status.set("No PGN file is loaded.")
            return
        self.load_pgn(self.current_path)

    def back(self):
        if self.move_index > 0:
            self.move_index -= 1
            self.render()

    def forward(self):
        if self.move_index < len(self.moves):
            self.move_index += 1
            self.render()

    def board_at_current_move(self):
        board = chess.Board()
        self.last_move = None
        for move in self.moves[: self.move_index]:
            board.push(move)
            self.last_move = move
        return board

    def render(self):
        board = self.board_at_current_move()
        highlighted = set()
        if self.last_move is not None:
            highlighted = {self.last_move.from_square, self.last_move.to_square}

        for row in range(8):
            for col in range(8):
                square = chess.square(col, 7 - row)
                piece = board.piece_at(square)
                color = self.light_square if (row + col) % 2 == 0 else self.dark_square
                if square in highlighted:
                    color = self.last_move_square
                self.buttons[(row, col)].configure(
                    text=self.pieces.get(piece.symbol(), "") if piece else "",
                    bg=color,
                    disabledforeground="#111111",
                )

        file_label = self.current_path.name if self.current_path else "waiting for PGN"
        outcome = board.outcome(claim_draw=True)
        result = outcome.result() if outcome is not None else "*"
        self.info.set(
            f"{file_label} | move {self.move_index}/{len(self.moves)} | "
            f"result {result}"
        )
        self.update_status()

    def update_status(self):
        self.status.set(
            "Use Forward/Back to replay. "
            "Reload latest loads the newest saved PGN."
        )


def pgn_viewer_command(args):
    if args.pgn is None and not args.latest:
        raise ValueError("Use --pgn <file> or --latest")
    PGNViewer(
        records_dir=args.records_dir,
        pgn_path=args.pgn,
        latest=args.latest,
        poll_ms=max(250, int(args.poll_seconds * 1000)),
    ).run()


class ChessGUI:
    light_square = "#f0d9b5"
    dark_square = "#b58863"
    selected_square = "#f6f669"
    legal_square = "#a9d18e"
    pieces = {
        "P": "♙",
        "N": "♘",
        "B": "♗",
        "R": "♖",
        "Q": "♕",
        "K": "♔",
        "p": "♟",
        "n": "♞",
        "b": "♝",
        "r": "♜",
        "q": "♛",
        "k": "♚",
    }

    def __init__(self, args):
        import tkinter as tk
        from tkinter import messagebox

        self.tk = tk
        self.messagebox = messagebox
        self.device = select_device()
        self.game = Chess()
        self.checkpoint = torch.load(
            args.checkpoint, map_location=self.device, weights_only=False
        )
        self.model = create_model(self.device, self.checkpoint.get("model_config"))
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.board = self.game.get_initial_state()
        self.human_color = chess.WHITE if args.color == "white" else chess.BLACK
        self.simulations = args.simulations
        self.selected = None

        self.root = self.tk.Tk()
        self.root.title(
            f"AlphaZero Chess - iteration {self.checkpoint.get('iteration', '?')}"
        )
        self.status = tk.StringVar()
        self.buttons = {}

        board_frame = self.tk.Frame(self.root)
        board_frame.pack(padx=12, pady=12)
        for row in range(8):
            for col in range(8):
                button = self.tk.Button(
                    board_frame,
                    width=4,
                    height=2,
                    font=("Arial", 28),
                    command=lambda r=row, c=col: self.on_square_click(r, c),
                )
                button.grid(row=row, column=col)
                self.buttons[(row, col)] = button

        self.tk.Label(self.root, textvariable=self.status, font=("Arial", 12)).pack(
            padx=12, pady=(0, 12)
        )
        self.render()
        self.root.after(250, self.maybe_model_move)

    def run(self):
        self.root.mainloop()

    def row_col_to_square(self, row, col):
        if self.human_color == chess.WHITE:
            return chess.square(col, 7 - row)
        return chess.square(7 - col, row)

    def square_to_row_col(self, square):
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        if self.human_color == chess.WHITE:
            return 7 - rank_idx, file_idx
        return rank_idx, 7 - file_idx

    def render(self):
        legal_targets = set()
        if self.selected is not None:
            legal_targets = {
                move.to_square
                for move in self.board.legal_moves
                if move.from_square == self.selected
            }

        for row in range(8):
            for col in range(8):
                square = self.row_col_to_square(row, col)
                piece = self.board.piece_at(square)
                button = self.buttons[(row, col)]
                color = self.light_square if (row + col) % 2 == 0 else self.dark_square
                if square == self.selected:
                    color = self.selected_square
                elif square in legal_targets:
                    color = self.legal_square
                button.configure(
                    text=self.pieces.get(piece.symbol(), "") if piece else "",
                    bg=color,
                    activebackground=color,
                )

        if self.game.is_terminal(self.board):
            self.status.set(f"Game over: {self.game.get_result(self.board)}")
        elif self.board.turn == self.human_color:
            self.status.set("Your move")
        else:
            self.status.set("Model is thinking...")

    def on_square_click(self, row, col):
        if self.game.is_terminal(self.board) or self.board.turn != self.human_color:
            return

        square = self.row_col_to_square(row, col)
        piece = self.board.piece_at(square)
        if self.selected is None:
            if piece is not None and piece.color == self.human_color:
                self.selected = square
                self.render()
            return

        move = chess.Move(self.selected, square)
        promotion_move = chess.Move(self.selected, square, promotion=chess.QUEEN)
        if promotion_move in self.board.legal_moves:
            move = promotion_move

        if move in self.board.legal_moves:
            self.board.push(move)
            self.selected = None
            self.render()
            self.root.after(100, self.maybe_model_move)
            return

        if piece is not None and piece.color == self.human_color:
            self.selected = square
        else:
            self.selected = None
        self.render()

    def maybe_model_move(self):
        if self.game.is_terminal(self.board) or self.board.turn == self.human_color:
            self.render()
            return

        self.render()
        self.root.update_idletasks()
        try:
            action = choose_model_action(
                self.game,
                self.board,
                self.model,
                self.simulations,
            )
            move = self.game.action_to_move(action, self.board)
            self.board.push(move)
        except Exception as exc:
            self.messagebox.showerror("Model move failed", str(exc))
        self.render()


def gui_command(args):
    ChessGUI(args).run()


def main():
    args = parse_args()
    if args.command == "train":
        train_command(args)
    elif args.command == "play":
        play_command(args)
    elif args.command == "gui":
        gui_command(args)
    elif args.command == "pgn-viewer":
        pgn_viewer_command(args)
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
