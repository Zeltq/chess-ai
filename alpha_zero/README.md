# AlphaZero Chess

This is a small from-scratch AlphaZero-style chess project in PyTorch. It does
not use openings, endgame tables, Stockfish labels, or human games. The model
learns only from self-play.

## Install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r alpha_zero/requirements.txt
```

For CUDA, install the PyTorch build that matches your system from the official
PyTorch instructions, then install the remaining requirements if needed.

`numba` is used for hot numeric chess action encoding/decoding paths. The first
run may spend a little time compiling cached JIT functions; later runs reuse the
cache.

## Train

Run a small training session:

```bash
python alpha_zero/main.py train --iterations 5 --games-per-iteration 4 --simulations 32
```

Useful options:

```bash
python alpha_zero/main.py train \
  --iterations 50 \
  --games-per-iteration 16 \
  --simulations 128 \
  --epochs 4 \
  --batch-size 64
```

Approximate one-hour run:

```bash
python alpha_zero/main.py train \
  --iterations 999 \
  --time-limit-minutes 60 \
  --games-per-iteration 6 \
  --simulations 64 \
  --epochs 3 \
  --batch-size 128
```

The trainer stops only after saving a checkpoint, so it can run a little over
the requested time limit.

Training logs are printed as compact rows:

```text
selfplay  | game: 1/6 | result: 1/2-1/2 | pos: 83 | speed: 5.4 pos/s
train     | epoch: 1/3 | loss: 2.3140 | speed: 910.2 samples/s
checkpoint | file: ...pt | iter_time: 8m 12s | eta: 51m 04s
```

By default a draw is trained as `0.45` game points, which maps to value target
`-0.10` on the neural network's `[-1, 1]` scale for both sides. That makes a
draw mildly unattractive without treating it like a loss. You can tune it:

```bash
python alpha_zero/main.py train --draw-score 0.4
```

That maps drawn games to value target `-0.2` for both sides.

Self-play adds a tiny material shaping signal to the value target for captures:
with the default `--capture-reward-scale 0.01`, taking a pawn is worth `0.01`,
a minor piece `0.03`, a rook `0.05`, and a queen `0.09`. The total capture
shaping is capped by `--capture-reward-cap 0.2`, so winning or losing the game
remains much more important than material.

```bash
python alpha_zero/main.py train --capture-reward-scale 0.01 --capture-reward-cap 0.2
```

Self-play also avoids moves that immediately produce a claimable draw when at
least one non-drawing legal move exists. Disable that only for rule-pure
experiments:

```bash
python alpha_zero/main.py train --allow-immediate-draws
```

Optional engine-only audit metrics can be logged with any UCI engine, such as
Stockfish. These values are never added to MCTS, replay buffer, loss, or model
inputs; they are only printed for you:

```bash
python alpha_zero/main.py train \
  --eval-engine /usr/games/stockfish \
  --eval-time 0.05 \
  --eval-games 1
```

The audit logs average move accuracy and average centipawn loss for the first
`--eval-games` self-play games in each iteration. Engine analysis can slow
training down, so keep `--eval-time` small at first.

Iteration metrics are also written to CSV so you can build graphs later:

```bash
python alpha_zero/main.py train \
  --eval-engine /usr/games/stockfish \
  --metrics-dir metrics
```

By default the trainer writes something like:

```text
metrics/20260429_120501_training_metrics.csv
```

You can force a specific file with:

```bash
python alpha_zero/main.py train --metrics-file metrics/my_run.csv
```

When you resume from a checkpoint, the trainer reuses the previous metrics file
from checkpoint metadata unless you override it with `--metrics-file`.

The trainer saves a checkpoint after every iteration:

```text
checkpoints/20260428_194512_alpha_zero_chess_iter_0001.pt
checkpoints/20260428_194512_alpha_zero_chess_iter_0002.pt
checkpoints/latest.pt
```

It also saves one or two self-play PGN files per iteration by default:

```text
games/20260428_194512_iter_0001_game_01.pgn
games/20260428_194512_iter_0001_game_02.pgn
```

The timestamp is generated when a training run starts, so new runs do not
overwrite older checkpoints or PGN records. `latest.pt` is intentionally
overwritten as a convenience pointer to the newest checkpoint.

Change the amount of recorded games with:

```bash
python alpha_zero/main.py train --record-games 1
```

## Preview PGN Games

You can open a PGN viewer at any time, in another terminal, while training is
running or after it has finished.

Open the latest recorded game and keep polling for newer PGN files:

```bash
python alpha_zero/main.py pgn-viewer --latest
```

Open a specific game:

```bash
python alpha_zero/main.py pgn-viewer \
  --pgn games/20260428_194512_iter_0001_game_01.pgn
```

The viewer has `Back`, `Forward`, `Reload latest`, and `Reload file` buttons.
`--latest` keeps checking `games/` for newly saved self-play games.

## Resume Training

You can continue from a checkpoint:

```bash
python alpha_zero/main.py train \
  --resume checkpoints/20260428_194512_alpha_zero_chess_iter_0005.pt \
  --iterations 20 \
  --games-per-iteration 8 \
  --simulations 64
```

`--iterations` is the final target iteration number. If the checkpoint is
iteration 5 and you pass `--iterations 20`, training continues from iteration 6
through 20.

The checkpoint stores model weights and optimizer state. The replay buffer is
not persisted, so resumed training starts with the learned model but collects a
fresh self-play buffer.

## Play In GUI

Launch a simple click-to-move board:

```bash
python alpha_zero/main.py gui \
  --checkpoint checkpoints/latest.pt \
  --color white \
  --simulations 128
```

Controls:

- Left click one of your pieces.
- Left click the destination square.
- Pawn promotion is promoted to queen automatically.

If you run this from WSL, your environment needs working GUI support for
`tkinter`.

## Play In Terminal

The text mode is still available:

```bash
python alpha_zero/main.py play \
  --checkpoint checkpoints/20260428_194512_alpha_zero_chess_iter_0005.pt \
  --color black
```

Moves are entered in UCI format, for example `e2e4`, `g1f3`, or `e7e8q`.

## How It Works

Each self-play move is selected by MCTS, except that an immediate checkmate is
played directly when one is legally available. The neural network gives two
outputs:

- `policy`: prior probabilities for 4672 chess action slots.
- `value`: expected result from the side-to-move perspective.

After a self-play game finishes, every position becomes a training sample:

```text
encoded position, MCTS visit distribution, final game result
```

Training minimizes policy loss plus value loss. Then the updated network is used
for the next round of self-play.

## Why Early Games Are Draws

All-draw logs at the beginning are normal for this toy setup. An untrained model
combined with low MCTS simulations tends to shuffle pieces until chess draw
rules end the game. The loss going down means the network is learning to imitate
its own MCTS search, but that does not immediately mean it is strong.

To make progress more visible, increase:

- `--iterations`
- `--games-per-iteration`
- `--simulations`

`--max-moves` caps self-play game length in half-moves and defaults to `400`,
which is about 200 full moves. If the limit is reached, the game is treated as a
draw. This keeps early random models from spending too much time in very long
low-signal games and prevents repetitive positions from dominating the replay
buffer.

By default the chess environment uses normal `python-chess` outcomes with draw
claims enabled for automated play, but the self-play move picker avoids
immediate claimable draws if another legal move is available. Games can still
end by checkmate, stalemate, insufficient material, unavoidable repetition,
fivefold repetition, the 50-move rule, the 75-move rule, or the self-play move
cap. Move history is preserved so repetition rules work correctly.

This implementation is intentionally pure self-play. It can learn interesting
behavior, but reaching something like stable chess.com 1000 Elo will likely
require much more compute, more games, and stronger evaluation tooling.

## Project Layout

```text
alpha_zero/
  env/
    chess_env.py     # chess rules, state encoding, action encoding
    game.py          # environment interface
  mcts/
    mcts.py          # Monte Carlo Tree Search
    node.py          # search tree node
  model/
    net.py           # policy-value neural network
  training/
    buffer.py        # replay buffer
    self_play.py     # self-play game generation
    train.py         # batch training
  main.py            # CLI: train, play, gui, pgn-viewer
```
