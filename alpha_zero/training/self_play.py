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


def _sample_action(actions, probabilities, temperature):
    if temperature <= 1e-6:
        return int(actions[np.argmax(probabilities)])

    scaled = apply_temperature(probabilities, temperature)
    return int(np.random.choice(actions, p=scaled))


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
    temperature=0.8,
    temperature_drop_move=20,
    c_puct=1.5,
    max_moves=None,
    draw_value=0,
    capture_reward_scale=0.01,
    capture_reward_cap=0.2,
    mcts_batch_size=8,
    fpu_reduction=0.25,
):
    mcts = MCTS(c_puct=c_puct, batch_size=mcts_batch_size, fpu_reduction=fpu_reduction)
    state = game.get_initial_state()
    history = []
    moves = []
    capture_rewards = []
    move_index = 0
    current_root = None  # MCTS subtree carried over from the previous move.

    while not game.is_terminal(state) and (max_moves is None or move_index < max_moves):
        root = mcts.run(
            game,
            state,
            evaluator,
            num_simulations=num_simulations,
            add_exploration_noise=True,
            root=current_root,
        )

        actions = np.array(list(root.children.keys()), dtype=np.int32)
        visit_counts = np.array(
            [root.children[action].visit_count for action in actions],
            dtype=np.float32,
        )
        visit_probs = normalize_nonnegative(visit_counts)

        policy_target = np.zeros(game.action_space, dtype=np.float32)
        policy_target[actions] = visit_probs

        # Reuse the encoded tensor MCTS already produced for the root —
        # avoids encoding the same position twice per move.
        encoded_state = root.encoded_state
        if encoded_state is None:
            encoded_state = game.encode_state(state)
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
        next_root = root.children.get(action)
        if next_root is not None and next_root.expanded():
            next_root.parent = None
            current_root = next_root
        else:
            current_root = None

    result = game.get_result(state)
    if result == "*":
        result = "1/2-1/2"
    outcome = state.outcome(claim_draw=True)
    if outcome is not None and outcome.winner is not None:
        white_outcome = 1.0 if outcome.winner else -1.0
    else:
        white_outcome = draw_value

    samples = []
    capture_bonuses = _future_capture_bonus(
        history,
        capture_rewards,
        capture_reward_scale,
        capture_reward_cap,
    )
    for item, capture_bonus in zip(history, capture_bonuses):
        if outcome is not None and outcome.winner is not None:
            value = white_outcome if item["player"] == chess.WHITE else -white_outcome
        else:
            value = draw_value
        value = float(np.clip(value + capture_bonus, -1.0, 1.0))
        samples.append((item["state"], item["policy"], value))

    return samples, state, moves, result
