import chess
import numpy as np
import torch

from .game import Game
from utils.fast_chess import action_to_components, move_to_action_index, moves_to_actions


_PIECE_TYPE_ORDER = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)


def _piece_planes(bbs_uint64, mirror):
    """Convert N piece bitboards to (N, 8, 8) plane block in one shot.

    Single unpackbits call beats per-bitboard unpacking by ~10x on tiny
    buffers because numpy's per-call overhead dominates otherwise.
    Caller assumes little-endian host (x86/ARM-LE).
    """
    bits = np.unpackbits(bbs_uint64.view(np.uint8), bitorder="little")
    planes = bits.reshape(-1, 8, 8)  # planes[i, rank, file]
    if not mirror:
        planes = planes[:, ::-1, :]  # white-perspective: flip ranks top-down
    return planes


class Chess(Game):
    """Chess environment using the AlphaZero 8x8x73 policy encoding."""

    board_size = 8
    planes_per_square = 73
    action_space = 8 * 8 * planes_per_square

    _DIRECTIONS = [
        (0, 1),   # N
        (0, -1),  # S
        (1, 0),   # E
        (-1, 0),  # W
        (1, 1),   # NE
        (-1, 1),  # NW
        (1, -1),  # SE
        (-1, -1), # SW
    ]
    _KNIGHT_DELTAS = [
        (1, 2),
        (2, 1),
        (2, -1),
        (1, -2),
        (-1, -2),
        (-2, -1),
        (-2, 1),
        (-1, 2),
    ]
    _PROMOTION_DELTAS = [
        (0, 1),   # forward
        (-1, 1),  # capture left
        (1, 1),   # capture right
    ]
    _PROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

    def get_initial_state(self):
        return chess.Board()

    def copy_state(self, state, stack=True):
        return state.copy(stack=stack)

    def get_valid_actions(self, state):
        moves = list(state.legal_moves)
        n = len(moves)
        if n == 0:
            return np.empty(0, dtype=np.int32)
        from_sq = np.empty(n, dtype=np.int32)
        to_sq = np.empty(n, dtype=np.int32)
        promo = np.empty(n, dtype=np.int32)
        for i, m in enumerate(moves):
            from_sq[i] = m.from_square
            to_sq[i] = m.to_square
            promo[i] = m.promotion or 0
        return moves_to_actions(from_sq, to_sq, promo, state.turn == chess.BLACK)

    def step(self, state, action, stack=True):
        move = self.action_to_move(action, state)
        if move not in state.legal_moves:
            raise ValueError(f"Illegal move for current position: {move.uci()}")

        next_state = state.copy(stack=stack)
        next_state.push(move)
        if stack:
            done = self.is_terminal(next_state)
            reward = self.get_reward(next_state) if done else 0.0
        else:
            done = self.is_terminal_fast(next_state)
            reward = self.get_reward_fast(next_state) if done else 0.0
        return next_state, reward, done

    def is_terminal(self, state):
        return state.outcome(claim_draw=True) is not None

    def get_reward(self, state):
        outcome = state.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == state.turn else -1.0

    def is_terminal_fast(self, state):
        """Cheap terminal check for MCTS rollouts: skips claim_draw (50-move/3-fold).

        Misses repetition/50-move draws inside the search tree, but those are
        rare at MCTS depth and missing them only delays exploration cutoff —
        the real game loop still uses the strict is_terminal.
        """
        if state.is_checkmate():
            return True
        if state.is_stalemate():
            return True
        if state.is_insufficient_material():
            return True
        return False

    def get_reward_fast(self, state):
        """Reward matching is_terminal_fast: only checkmate gives ±1."""
        if state.is_checkmate():
            # state.turn is the side that just got mated and has no legal reply.
            return -1.0
        return 0.0

    def get_result(self, state):
        outcome = state.outcome(claim_draw=True)
        return outcome.result() if outcome is not None else "*"

    def encode_state(self, state):
        # Bitboard-driven encoding. Avoids state.mirror() (allocates a full
        # board copy and rebuilds piece tables) and the per-piece Python
        # loop. Single unpackbits over all 12 piece bitboards beats per-piece
        # loops by ~3-5x on tiny buffers.
        is_black = state.turn == chess.BLACK
        own = chess.BLACK if is_black else chess.WHITE
        opp = chess.WHITE if is_black else chess.BLACK

        bbs = np.array(
            [state.pieces_mask(pt, own) for pt in _PIECE_TYPE_ORDER]
            + [state.pieces_mask(pt, opp) for pt in _PIECE_TYPE_ORDER],
            dtype=np.uint64,
        )

        encoded = np.empty((19, 8, 8), dtype=np.float32)
        encoded[:12] = _piece_planes(bbs, is_black)  # uint8 -> float32 cast on assign

        encoded[12].fill(1.0 if state.has_kingside_castling_rights(own) else 0.0)
        encoded[13].fill(1.0 if state.has_queenside_castling_rights(own) else 0.0)
        encoded[14].fill(1.0 if state.has_kingside_castling_rights(opp) else 0.0)
        encoded[15].fill(1.0 if state.has_queenside_castling_rights(opp) else 0.0)

        encoded[16].fill(0.0)
        if state.ep_square is not None:
            ep_rank = state.ep_square >> 3
            ep_file = state.ep_square & 7
            row = ep_rank if is_black else (7 - ep_rank)
            encoded[16, row, ep_file] = 1.0

        encoded[17].fill(min(state.halfmove_clock, 100) / 100.0)
        encoded[18].fill(1.0)

        return torch.from_numpy(encoded)

    def move_to_action(self, move, turn):
        promotion = move.promotion or 0
        action = move_to_action_index(
            move.from_square,
            move.to_square,
            promotion,
            turn == chess.BLACK,
        )
        if action < 0:
            raise ValueError(f"Unsupported move encoding: {move.uci()}")
        return int(action)

    def action_to_move(self, action, state):
        oriented_from = action // self.planes_per_square
        real_from = self._deorient_square(oriented_from, state.turn)
        piece = state.piece_at(real_from)
        piece_type = piece.piece_type if piece is not None else 0
        real_from, real_to, promotion = action_to_components(
            int(action),
            state.turn == chess.BLACK,
            piece_type,
        )
        if real_from < 0:
            raise ValueError(f"Decoded move is off-board for action {action}")
        return chess.Move(
            real_from,
            real_to,
            promotion=promotion if promotion else None,
        )

    def _canonical_board(self, state):
        if state.turn == chess.WHITE:
            return state
        mirrored = state.mirror()
        mirrored.turn = chess.WHITE
        return mirrored

    def _orient_square(self, square, turn):
        return square if turn == chess.WHITE else chess.square_mirror(square)

    def _deorient_square(self, square, turn):
        return square if turn == chess.WHITE else chess.square_mirror(square)

    def _ray_plane(self, df, dr, promotion):
        step_file = 0 if df == 0 else int(df / abs(df))
        step_rank = 0 if dr == 0 else int(dr / abs(dr))
        distance = max(abs(df), abs(dr))
        if distance < 1 or distance > 7:
            raise ValueError(f"Unsupported ray move delta ({df}, {dr})")
        if (step_file, step_rank) not in self._DIRECTIONS:
            raise ValueError(f"Unsupported ray direction ({df}, {dr})")
        direction_idx = self._DIRECTIONS.index((step_file, step_rank))
        return direction_idx * 7 + (distance - 1)

    def _promotion_plane(self, df, promotion):
        if df not in (-1, 0, 1):
            raise ValueError(f"Unsupported promotion file delta {df}")
        direction_idx = {-1: 1, 0: 0, 1: 2}[df]
        piece_idx = self._PROMOTION_PIECES.index(promotion)
        return 64 + direction_idx * 3 + piece_idx
