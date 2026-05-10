import chess
import numpy as np

from mcts.mcts import MCTS
from utils.fast_chess import apply_temperature, normalize_nonnegative


PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}
CAPTURE_REWARD_DISCOUNT = 0.98


# Endgame curriculum: a pool of starting positions that force the model to
# learn mating technique. From scratch, self-play almost never reaches these
# positions (games end earlier by adjudication / 50-move / max_moves), so
# the model never learns to checkmate K+R vs K, K+Q vs K, etc.
# When --endgame-curriculum > 0, each self-play game starts from a sampled
# position from this list with the given probability. Adjudication is
# disabled for these games — the model must actually deliver checkmate
# (or accept a draw by 50-move / stalemate / max_moves).
ENDGAME_STARTING_FENS = [
    # --- K+Q vs K (easiest — ~10 moves to mate) ---
    "8/8/8/4k3/8/8/8/3QK3 w - - 0 1",
    "4k3/8/8/8/8/8/8/Q3K3 w - - 0 1",
    "8/4k3/8/8/8/8/4Q3/4K3 w - - 0 1",
    # Black side variants
    "3qk3/8/8/8/4K3/8/8/8 b - - 0 1",
    "q3k3/8/8/8/8/8/8/4K3 b - - 0 1",
    # --- K+R vs K (~15-20 moves; the famous test) ---
    "8/8/8/4k3/8/8/8/3RK3 w - - 0 1",
    "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
    "8/4k3/8/8/8/8/4R3/4K3 w - - 0 1",
    "3rk3/8/8/8/4K3/8/8/8 b - - 0 1",
    "r3k3/8/8/8/8/8/8/4K3 b - - 0 1",
    # --- K + 2R vs K (gateway, easier than K+R) ---
    "8/8/8/4k3/8/8/R7/3RK3 w - - 0 1",
    # --- K + P vs K endgames (square rule, opposition) ---
    "4k3/8/8/8/8/4P3/8/4K3 w - - 0 1",
    "4k3/8/8/4P3/8/8/8/4K3 w - - 0 1",
    "8/4k3/8/4P3/4K3/8/8/8 w - - 0 1",
    # K + Pawn race
    "8/4k3/8/8/8/8/P7/4K3 w - - 0 1",
    # Two pawns vs K
    "4k3/8/8/8/4P3/4P3/8/4K3 w - - 0 1",
    # --- K + 2Q vs K + Q (queen technique with counterplay) ---
    "8/8/8/4k3/4q3/8/8/Q2QK3 w - - 0 1",
]


def _sample_action(actions, probabilities, temperature):
    if temperature <= 1e-6:
        return int(actions[np.argmax(probabilities)])

    scaled = apply_temperature(probabilities, temperature)
    return int(np.random.choice(actions, p=scaled))


def _material_balance(state):
    """Material differential from White's perspective, in pawn units."""
    balance = 0.0
    for _, piece in state.piece_map().items():
        v = PIECE_VALUES.get(piece.piece_type, 0.0)
        balance += v if piece.color == chess.WHITE else -v
    return balance


def find_immediate_checkmate_action(game, state):
    for move in state.legal_moves:
        next_state = game.copy_state(state)
        next_state.push(move)
        if next_state.is_checkmate():
            return game.move_to_action(move, state.turn)
    return None


def _captured_piece_value(state, move):
    if state.is_en_passant(move):
        capture_square = chess.square(
            chess.square_file(move.to_square),
            chess.square_rank(move.from_square),
        )
    else:
        capture_square = move.to_square

    captured_piece = state.piece_at(capture_square)
    if captured_piece is None:
        return 0.0
    return PIECE_VALUES.get(captured_piece.piece_type, 0.0)


def _future_capture_bonus(history, capture_rewards, scale, cap):
    if scale <= 0.0 or cap <= 0.0:
        return [0.0] * len(history)

    bonuses = []
    for index, item in enumerate(history):
        player = item["player"]
        bonus = 0.0
        discount = 1.0
        for future in range(index, len(capture_rewards)):
            mover, reward = capture_rewards[future]
            if reward:
                bonus += discount * reward if mover == player else -discount * reward
            discount *= CAPTURE_REWARD_DISCOUNT
        bonus *= scale
        bonuses.append(float(np.clip(bonus, -cap, cap)))
    return bonuses


def self_play_game(
    game,
    evaluator,
    num_simulations,
    temperature=1.0,
    temperature_drop_move=30,
    c_puct=1.5,
    max_moves=None,
    draw_value=0,
    capture_reward_scale=0.01,
    capture_reward_cap=0.2,
    mcts_batch_size=8,
    fpu_reduction=0.25,
    c_puct_base=None,
    c_puct_init=1.25,
    adjudicate_material=None,
    adjudicate_min_move=40,
    adjudicate_consecutive=3,
    endgame_curriculum=0.0,
):
    mcts = MCTS(
        c_puct=c_puct,
        batch_size=mcts_batch_size,
        fpu_reduction=fpu_reduction,
        c_puct_base=c_puct_base,
        c_puct_init=c_puct_init,
    )

    # Endgame curriculum: with probability `endgame_curriculum` start from a
    # sampled endgame position. Adjudication is disabled for these games so
    # the model is forced to actually deliver checkmate (otherwise the game
    # ends in a draw via 50-move / max_moves and the value target reflects
    # that — useful negative signal for "you had advantage but didn't win").
    started_from_endgame = (
        endgame_curriculum > 0.0
        and np.random.random() < endgame_curriculum
    )
    if started_from_endgame:
        fen = ENDGAME_STARTING_FENS[
            np.random.randint(0, len(ENDGAME_STARTING_FENS))
        ]
        state = chess.Board(fen)
        starting_fen = fen
        # Disable adjudication for this game — force the model to actually
        # deliver mate; otherwise it never learns the technique.
        effective_adjudicate_material = None
    else:
        state = game.get_initial_state()
        starting_fen = None
        effective_adjudicate_material = adjudicate_material

    history = []
    moves = []
    capture_rewards = []
    move_index = 0
    current_root = None  # MCTS subtree carried over from the previous move.
    reused_visits_total = 0
    reused_visits_moves = 0
    # Material adjudication: ends the game with a decisive result if one side
    # has held a material lead of at least `adjudicate_material` pawn units
    # for `adjudicate_consecutive` plies after move `adjudicate_min_move`.
    # Bootstraps win/loss signal in early training when MCTS would otherwise
    # draw everything by repetition.
    adjudicated_result = None  # "1-0" / "0-1" / None
    material_streak = 0
    last_streak_sign = 0

    while not game.is_terminal(state) and (max_moves is None or move_index < max_moves):
        if current_root is not None:
            reused_visits_total += current_root.visit_count
            reused_visits_moves += 1
        root = mcts.run(
            game,
            state,
            evaluator,
            num_simulations=num_simulations,
            add_exploration_noise=True,
            root=current_root,
        )

        actions = root.actions
        visit_counts = root.child_visits.astype(np.float32, copy=False)
        visit_probs = normalize_nonnegative(visit_counts)

        policy_target = np.zeros(game.action_space, dtype=np.float32)
        policy_target[actions] = visit_probs

        # Reuse the encoded tensor MCTS already produced for the root —
        # avoids encoding the same position twice per move. Stored as
        # numpy so downstream consumers (replay buffer, mirror augment,
        # IPC) all see the same type.
        encoded_state = root.encoded_state
        if encoded_state is None:
            encoded_state = game.encode_state(state)
        if hasattr(encoded_state, "numpy"):
            encoded_state = encoded_state.numpy()
        history.append(
            {
                "state": encoded_state,
                "policy": policy_target,
                "player": state.turn,
            }
        )

        current_temperature = temperature if move_index < temperature_drop_move else 0.0
        action = _sample_action(actions, visit_probs, current_temperature)
        move = game.action_to_move(action, state)
        mover = state.turn
        captured_value = _captured_piece_value(state, move)
        state, _, _ = game.step(state, action)
        moves.append(move)
        capture_rewards.append((mover, captured_value))
        move_index += 1

        # Tree reuse: the chosen child becomes the new root. It is guaranteed
        # to be expanded (it had the highest sampled visit count, so MCTS
        # selected it at least once and ran expand_and_evaluate on it).
        idx = root.action_to_idx(int(action))
        next_root = root.child_nodes[idx] if idx >= 0 else None
        if next_root is not None and next_root.expanded():
            next_root.detach_as_root()
            current_root = next_root
        else:
            current_root = None

        # Material-balance adjudication. Tracks consecutive plies where the
        # signed material lead exceeds the threshold; firing breaks the loop
        # with a decisive result.
        if (
            effective_adjudicate_material is not None
            and move_index >= adjudicate_min_move
        ):
            balance = _material_balance(state)
            if balance >= effective_adjudicate_material:
                sign = 1
            elif balance <= -effective_adjudicate_material:
                sign = -1
            else:
                sign = 0
            if sign != 0 and sign == last_streak_sign:
                material_streak += 1
            elif sign != 0:
                material_streak = 1
                last_streak_sign = sign
            else:
                material_streak = 0
                last_streak_sign = 0
            if material_streak >= adjudicate_consecutive:
                adjudicated_result = "1-0" if last_streak_sign > 0 else "0-1"
                break
        else:
            current_root = None

    # Adjudication overrides natural outcome detection: we end the game
    # decisively even though python-chess sees an in-progress (or drawable)
    # position. value targets follow the adjudicated winner.
    if adjudicated_result is not None:
        result = adjudicated_result
        white_outcome = 1.0 if adjudicated_result == "1-0" else -1.0
        decisive = True
    else:
        result = game.get_result(state)
        if result == "*":
            result = "1/2-1/2"
        outcome = state.outcome(claim_draw=True)
        if outcome is not None and outcome.winner is not None:
            white_outcome = 1.0 if outcome.winner else -1.0
            decisive = True
        else:
            white_outcome = draw_value
            decisive = False

    samples = []
    capture_bonuses = _future_capture_bonus(
        history,
        capture_rewards,
        capture_reward_scale,
        capture_reward_cap,
    )
    for item, capture_bonus in zip(history, capture_bonuses):
        if decisive:
            value = white_outcome if item["player"] == chess.WHITE else -white_outcome
        else:
            value = draw_value
        value = float(np.clip(value + capture_bonus, -1.0, 1.0))
        samples.append((item["state"], item["policy"], value))

    stats = {
        "moves": move_index,
        "reused_visits_avg": (
            reused_visits_total / reused_visits_moves
            if reused_visits_moves > 0 else 0.0
        ),
        "reused_visits_moves": reused_visits_moves,
        "adjudicated": adjudicated_result is not None,
        "endgame_curriculum": started_from_endgame,
        "starting_fen": starting_fen,
    }
    return samples, state, moves, result, stats
