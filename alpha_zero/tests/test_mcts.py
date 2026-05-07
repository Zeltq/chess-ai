"""MCTS behavioural regression tests.

Run from the alpha_zero/ directory:
    python -m pytest tests/test_mcts.py -q
"""
import numpy as np
import torch
import chess

from env.chess_env import Chess
from mcts.mcts import MCTS
from mcts.inference import LocalEvaluator
from model.net import AlphaZeroNet


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tiny_net(device):
    m = AlphaZeroNet(channels=32, num_res_blocks=1).to(device)
    if device.type == "cuda":
        m = m.to(memory_format=torch.channels_last)
    m.eval()
    return m


def test_run_returns_root_with_expected_visits():
    device = _device()
    g = Chess()
    ev = LocalEvaluator(_tiny_net(device), device)
    mcts = MCTS(c_puct=1.5, batch_size=4)

    root = mcts.run(g, g.get_initial_state(), ev, num_simulations=32)
    # root.visit_count == sum(child_visits) for fully expanded root.
    assert root.visit_count == int(root.child_visits.sum())
    assert root.expanded()
    assert root.actions.shape[0] == len(list(chess.Board().legal_moves))


def test_virtual_loss_balanced():
    """After MCTS run, no virtual losses should remain (counts match real
    visits; value sums look reasonable bounded)."""
    device = _device()
    g = Chess()
    ev = LocalEvaluator(_tiny_net(device), device)
    mcts = MCTS(c_puct=1.5, batch_size=8)
    root = mcts.run(g, g.get_initial_state(), ev, num_simulations=64)

    # Total visits == sum over children + visits accumulated at root from
    # simulations that touched it (each simulation increments root once).
    assert root.visit_count == 64
    # Children visits should sum to 64 (every sim picks exactly one child).
    assert int(root.child_visits.sum()) == 64
    # No child should have a runaway value sum.
    assert np.all(np.abs(root.child_value_sums) <= root.child_visits + 1)


def test_tree_reuse_carries_visits():
    device = _device()
    g = Chess()
    ev = LocalEvaluator(_tiny_net(device), device)
    mcts = MCTS(c_puct=1.5, batch_size=4)

    state = g.get_initial_state()
    root = mcts.run(g, state, ev, num_simulations=64, add_exploration_noise=True)
    assert root.visit_count == 64

    best_idx = int(np.argmax(root.child_visits))
    carried = int(root.child_visits[best_idx])
    action = int(root.actions[best_idx])

    next_state, _, _ = g.step(state, action)
    new_root = root.child_nodes[best_idx]
    new_root.detach_as_root()

    assert new_root.visit_count == carried, (
        f"expected {carried} carried visits, got {new_root.visit_count}"
    )

    root2 = mcts.run(
        g, next_state, ev, num_simulations=64, add_exploration_noise=True,
        root=new_root,
    )
    # root2 visits = carried + new sims (modulo root visit accounting).
    assert root2.visit_count >= 64 + carried - 1


def test_mate_in_one_found_at_low_sims():
    """Construct a position with mate-in-1 and check MCTS surfaces it."""
    device = _device()
    ev = LocalEvaluator(_tiny_net(device), device)
    g = Chess()
    # Famous Fool's-mate-style: black to move, Qh4# would mate.
    # Use a clean construction: 1.f3 e5 2.g4 -> black plays Qh4#.
    board = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4"):
        board.push(chess.Move.from_uci(uci))
    assert board.turn == chess.BLACK

    mate_move = chess.Move.from_uci("d8h4")
    mate_action = g.move_to_action(mate_move, board.turn)

    mcts = MCTS(c_puct=1.5, batch_size=4)
    root = mcts.run(g, board, ev, num_simulations=128)
    best_idx = int(np.argmax(root.child_visits))
    assert int(root.actions[best_idx]) == mate_action, (
        "MCTS did not converge on mate-in-1"
    )
