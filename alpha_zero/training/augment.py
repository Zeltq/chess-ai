"""Horizontal-mirror augmentation for chess training samples.

A file-flipped copy of a position is geometrically equivalent for piece
moves and pawn structure, so the network can learn from both. Castling
geometry is asymmetric (king/queen side), so the augmentation introduces
a small distribution shift on positions that retain castling rights —
acceptable in practice, opt-in via --mirror-augment.
"""
import numpy as np


def _build_action_mirror_map(num_actions=4672, planes_per_square=73):
    """Index map a -> a' such that action a on the mirrored board equals
    action a' on the original board.

    The encoding (utils/fast_chess.py) splits the 4672 action space as
    `oriented_from_square * 73 + plane`. We mirror the file of the from-
    square (sq XOR 7) and the per-plane direction:
      - sliding planes 0..55: direction 0..7 -> direction with df sign flipped
      - knight planes 56..63: each (df, dr) maps to (-df, dr)
      - promotion planes 64..72: 3 directions (forward / cap-left / cap-right)
        with cap-left <-> cap-right swap; 3 piece choices unchanged.
    """
    plane_mirror = np.empty(planes_per_square, dtype=np.int32)

    sliding_dir_mirror = [0, 1, 3, 2, 5, 4, 7, 6]
    for d in range(8):
        for dist in range(7):
            plane_mirror[d * 7 + dist] = sliding_dir_mirror[d] * 7 + dist

    # Knight order (matches fast_chess._knight_index):
    #   0:(1,2) 1:(2,1) 2:(2,-1) 3:(1,-2) 4:(-1,-2) 5:(-2,-1) 6:(-2,1) 7:(-1,2)
    # Flipping df gives:
    #   (1,2)  -> (-1,2)  = idx 7
    #   (2,1)  -> (-2,1)  = idx 6
    #   (2,-1) -> (-2,-1) = idx 5
    #   (1,-2) -> (-1,-2) = idx 4
    knight_mirror = [7, 6, 5, 4, 3, 2, 1, 0]
    for k in range(8):
        plane_mirror[56 + k] = 56 + knight_mirror[k]

    # Promotion: direction encoding is {0:forward, 1:cap-left(df=-1), 2:cap-right(df=+1)}.
    promo_dir_mirror = [0, 2, 1]
    for d in range(3):
        for p in range(3):
            plane_mirror[64 + d * 3 + p] = 64 + promo_dir_mirror[d] * 3 + p

    mapping = np.empty(num_actions, dtype=np.int32)
    for sq in range(64):
        mirrored_sq = sq ^ 7  # XOR with 7 flips file bits, leaves rank
        base = sq * planes_per_square
        m_base = mirrored_sq * planes_per_square
        for plane in range(planes_per_square):
            mapping[base + plane] = m_base + plane_mirror[plane]
    return mapping


# Built once at import. The mirror is an involution: mapping[mapping[a]] == a.
MIRROR_ACTION_MAP = _build_action_mirror_map()


def mirror_state(state):
    """Flip board file axis. Caller may need to .copy() if downstream
    expects contiguous memory.
    """
    return state[:, :, ::-1]


def mirror_policy(policy):
    """Apply MIRROR_ACTION_MAP. Mirror is an involution so a single fancy
    index suffices.
    """
    return policy[MIRROR_ACTION_MAP]


def mirror_sample(state, policy):
    return mirror_state(state).copy(), mirror_policy(policy).copy()
