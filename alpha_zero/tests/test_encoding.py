"""Encoding invariants for the chess action space and board encoder.

Run from the alpha_zero/ directory:
    python -m pytest tests/test_encoding.py -q
"""
import numpy as np
import chess

from env.chess_env import Chess
from training.augment import MIRROR_ACTION_MAP, mirror_state, mirror_policy
from utils.fast_chess import action_to_components, move_to_action_index


def _random_board(rng, max_plies=120):
    b = chess.Board()
    plies = int(rng.integers(0, max_plies + 1))
    for _ in range(plies):
        moves = list(b.legal_moves)
        if not moves:
            break
        b.push(moves[int(rng.integers(0, len(moves)))])
        if b.is_game_over():
            break
    return b


def test_move_to_action_to_move_roundtrip():
    """Every legal move encodes and decodes back to itself."""
    rng = np.random.default_rng(0)
    g = Chess()
    checked = 0
    for _ in range(100):
        board = _random_board(rng)
        for move in board.legal_moves:
            action = g.move_to_action(move, board.turn)
            decoded = g.action_to_move(action, board)
            assert decoded == move, (
                f"move {move.uci()} -> action {action} -> {decoded.uci()}"
                f" on FEN {board.fen()}"
            )
            checked += 1
    assert checked > 100, f"only checked {checked} moves"


def test_mirror_action_map_is_involution():
    arr = MIRROR_ACTION_MAP
    assert arr.shape == (4672,)
    assert np.array_equal(arr[arr], np.arange(4672))


def test_mirror_action_decodes_to_mirrored_move():
    """Mirroring an action should yield the action of the file-mirrored move."""
    rng = np.random.default_rng(1)
    g = Chess()
    for _ in range(40):
        board = _random_board(rng)
        for move in board.legal_moves:
            a = g.move_to_action(move, board.turn)
            a_mirror = MIRROR_ACTION_MAP[a]
            from_sq, to_sq, promo = action_to_components(
                int(a_mirror), board.turn == chess.BLACK,
                board.piece_at(move.from_square ^ 7).piece_type
                if board.piece_at(move.from_square ^ 7) is not None else 0,
            )
            # Mirror's from-square == file-flipped original from-square
            assert (from_sq ^ 7) == move.from_square, (
                f"original from {move.from_square}, mirrored decode {from_sq}"
            )
            assert (to_sq ^ 7) == move.to_square, (
                f"original to {move.to_square}, mirrored decode {to_sq}"
            )


def test_state_mirror_involution():
    """state[:,:,::-1][:,:,::-1] == state."""
    rng = np.random.default_rng(2)
    g = Chess()
    for _ in range(20):
        board = _random_board(rng)
        s = g.encode_state(board).numpy()
        m = mirror_state(s)
        m2 = mirror_state(m)
        assert np.array_equal(m2, s)


def test_mirror_policy_preserves_total_mass():
    rng = np.random.default_rng(3)
    g = Chess()
    for _ in range(20):
        board = _random_board(rng)
        if board.is_game_over():
            continue
        policy = np.zeros(4672, dtype=np.float32)
        for move in board.legal_moves:
            policy[g.move_to_action(move, board.turn)] = np.float32(rng.random())
        s = policy.sum()
        if s == 0:
            continue
        policy /= s
        mp = mirror_policy(policy)
        assert np.isclose(mp.sum(), 1.0, atol=1e-6)
        # Double mirror = identity
        assert np.allclose(mirror_policy(mp), policy)


def test_encode_state_shape_and_dtype():
    g = Chess()
    s = g.encode_state(chess.Board())
    assert tuple(s.shape) == (19, 8, 8)
    assert s.dtype.is_floating_point
