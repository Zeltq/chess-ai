"""Quick value-head sanity check: does the model see material?

Loads checkpoints/latest.pt and prints value-head outputs for a few
hand-crafted positions. If material-changing positions all return
similar values, value head has lost material discrimination.

Run from alpha_zero/:
    python diagnose_value.py
"""
import sys
import torch
import chess
import numpy as np

from env.chess_env import Chess
from model.net import AlphaZeroNet


def evaluate(model, device, fen, label):
    g = Chess()
    board = chess.Board(fen)
    state = g.encode_state(board).unsqueeze(0).to(device).contiguous(
        memory_format=torch.channels_last
    )
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        policy_logits, value = model(state)
    v = float(value.float().cpu().item())
    policy = torch.softmax(policy_logits.float(), dim=-1)[0].cpu().numpy()
    # Top-3 predicted moves
    valid = g.get_valid_actions(board)
    masked = policy[valid]
    top3_local = np.argsort(masked)[-3:][::-1]
    top3 = []
    for li in top3_local:
        a = int(valid[li])
        try:
            mv = g.action_to_move(a, board)
            top3.append(f"{mv.uci()}({masked[li]:.2f})")
        except Exception:
            top3.append(f"?({masked[li]:.2f})")
    side = "W" if board.turn == chess.WHITE else "B"
    print(f"  {side}-to-move | value={v:+.3f} | top3 policy: {' '.join(top3)}  | {label}")


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/latest.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = AlphaZeroNet(**ckpt["model_config"]).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {ckpt_path} (iter {ckpt.get('iteration', '?')})")
    print()

    # 1. Starting position — should be ~0
    print("=== Baseline ===")
    evaluate(model, device,
             "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
             "starting position (expected value ~ 0.0)")

    # 2. White up a queen — same setup but no black queen
    print()
    print("=== Material asymmetry ===")
    evaluate(model, device,
             "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
             "white +Q (black queen removed) — expected: STRONGLY positive (+0.5..+0.9)")
    evaluate(model, device,
             "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
             "black +Q (white queen removed) — expected: STRONGLY negative (-0.5..-0.9)")

    # 3. White up a rook
    evaluate(model, device,
             "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
             "white +R (black queen-rook removed) — expected: positive (+0.3..+0.7)")

    # 4. Down material
    evaluate(model, device,
             "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w Kkq - 0 1",
             "white -R (own queen-rook removed) — expected: negative (-0.3..-0.7)")

    # 5. Endgame KQ vs K — clearly winning
    print()
    print("=== Tactical positions ===")
    evaluate(model, device,
             "4k3/8/8/8/8/8/8/4K2Q w - - 0 1",
             "white K+Q vs lone K — expected: very positive (+0.7..+0.95)")

    # 6. Position where white can take a free queen on b4 with pawn a3
    evaluate(model, device,
             "4k3/8/8/8/1q6/P7/8/4K3 w - - 0 1",
             "white pawn a3 can take free queen on b4 (axb4) — expected: positive (+0.3..+0.7)")

    print()
    print("Interpretation:")
    print("  - If 'white +Q' value is similar to baseline (both near 0) -> value head LOST material understanding.")
    print("  - If 'white -Q' value is also near 0 -> definitely value collapse (predicts constant).")
    print("  - If material positions correctly rank by value -> value head works, problem is elsewhere.")


if __name__ == "__main__":
    main()
