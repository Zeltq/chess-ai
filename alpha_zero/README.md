# AlphaZero Chess

A from-scratch AlphaZero-style chess engine in PyTorch. No openings, no endgame
tables, no Stockfish labels, no human games — the model learns purely from
self-play.

This is the package-level README. The detailed user guide (commands, flags,
architecture diagrams, monitoring) lives in the repository's root
[`README.md`](../README.md).

## Quick reference

Install (from the repository root):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r alpha_zero/requirements.txt
```

Run all commands from `alpha_zero/`.

```bash
# minimal smoke test
python main.py train --iterations 2 --games-per-iteration 4 \
                     --simulations 32 --parallel-games 4 \
                     --channels 64 --res-blocks 2

# real training (see root README for the full recommended config)
python main.py train --iterations 200 --games-per-iteration 24 \
                     --simulations 200 --parallel-games 8 \
                     --channels 128 --res-blocks 8 --buffer-size 50000

# resume
python main.py train --resume checkpoints/latest.pt

# play (GUI)
python main.py gui --checkpoint checkpoints/latest.pt --color white

# play (terminal, UCI moves)
python main.py play --checkpoint checkpoints/latest.pt --color white

# preview self-play games
python main.py pgn-viewer --latest --poll-seconds 5
```

## How it works

Each self-play move is selected by MCTS over the policy-value network. The
network outputs:

- `policy`: prior over 4672 chess action slots
- `value`: expected result from the side-to-move perspective

After a self-play game finishes, every position becomes a training sample
`(encoded_position, MCTS_visit_distribution, final_game_result)`. The loss is
policy cross-entropy plus value MSE.

Several improvements layered on top of the base AlphaZero recipe:

- **Tree reuse.** The chosen child becomes the next move's root; visits and
  cached state propagate forward.
- **Virtual loss + leaf batching.** Each MCTS chunk gathers `mcts-batch-size`
  leaves with virtual loss before doing a single forward pass.
- **First-Play Urgency.** Unvisited children's Q is initialized to
  `parent.value − fpu_reduction` instead of 0.
- **Centralized GPU inference + worker processes.** With `--parallel-games > 1`
  each game runs in its own process (independent GIL); a single GPU model in
  the main process batches forward calls from all of them. See `mcts/inference.py`.
- **bfloat16 + channels_last + TF32** on Ampere/newer GPUs, automatic.
- **Bitboard-vectorized state encoding** (single `np.unpackbits` over all 12
  piece bitboards).

## Why early games are mostly draws

An untrained model with low simulation counts shuffles pieces until a draw
rule fires. Move-50 / 3-fold repetition / insufficient material all kick in
quickly. Loss going down means the network is starting to imitate its own
MCTS — that's progress, but it does not immediately translate to strong play.

To accelerate progress, raise `--iterations`, `--games-per-iteration`,
`--simulations`. The default `--max-moves 400` caps half-move count to keep
early low-signal games from dominating the replay buffer; if hit, the game is
treated as a draw.

## Files

```text
alpha_zero/
  bench.py             # microbenchmarks (encode / forward / selfplay)
  env/
    chess_env.py       # python-chess wrapper, board encoding (19×8×8)
    game.py            # Game interface
  mcts/
    mcts.py            # MCTS with virtual-loss, tree-reuse, FPU
    node.py            # Node with parallel-array children for vectorized PUCT
    inference.py       # LocalEvaluator + MPInferenceServer/Client
  model/
    net.py             # policy-value ResNet
  training/
    self_play.py       # self-play game generation
    train.py           # one optimizer step
    buffer.py          # replay buffer (+ optional file-mirror augmentation)
    augment.py         # mirror state / policy maps
  utils/
    fast_chess.py      # numba: action <-> components, normalization
  tests/               # pytest: encoding invariants + MCTS regressions
  main.py              # CLI: train / play / gui / pgn-viewer
```

For all flags, monitoring instructions, and tuning advice see the root
[`README.md`](../README.md).
