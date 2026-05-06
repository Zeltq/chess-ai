try:
    from numba import njit
except ImportError:  # pragma: no cover - optional speed dependency
    njit = None

import numpy as np


PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
PLANES_PER_SQUARE = 73


def _maybe_jit(fn):
    if njit is None:
        return fn
    return njit(cache=True)(fn)


@_maybe_jit
def _orient_square(square, is_black):
    return square ^ 56 if is_black else square


@_maybe_jit
def _direction_index(df, dr):
    if df == 0 and dr == 1:
        return 0
    if df == 0 and dr == -1:
        return 1
    if df == 1 and dr == 0:
        return 2
    if df == -1 and dr == 0:
        return 3
    if df == 1 and dr == 1:
        return 4
    if df == -1 and dr == 1:
        return 5
    if df == 1 and dr == -1:
        return 6
    if df == -1 and dr == -1:
        return 7
    return -1


@_maybe_jit
def _knight_index(df, dr):
    if df == 1 and dr == 2:
        return 0
    if df == 2 and dr == 1:
        return 1
    if df == 2 and dr == -1:
        return 2
    if df == 1 and dr == -2:
        return 3
    if df == -1 and dr == -2:
        return 4
    if df == -2 and dr == -1:
        return 5
    if df == -2 and dr == 1:
        return 6
    if df == -1 and dr == 2:
        return 7
    return -1


@_maybe_jit
def _promotion_piece_index(promotion):
    if promotion == KNIGHT:
        return 0
    if promotion == BISHOP:
        return 1
    if promotion == ROOK:
        return 2
    return -1


@_maybe_jit
def move_to_action_index(from_square, to_square, promotion, is_black):
    oriented_from = _orient_square(from_square, is_black)
    oriented_to = _orient_square(to_square, is_black)

    from_file = oriented_from & 7
    from_rank = oriented_from >> 3
    to_file = oriented_to & 7
    to_rank = oriented_to >> 3

    df = to_file - from_file
    dr = to_rank - from_rank

    promotion_index = _promotion_piece_index(promotion)
    if promotion_index >= 0:
        if df == 0:
            direction_index = 0
        elif df == -1:
            direction_index = 1
        elif df == 1:
            direction_index = 2
        else:
            return -1
        plane = 64 + direction_index * 3 + promotion_index
        return oriented_from * PLANES_PER_SQUARE + plane

    knight_index = _knight_index(df, dr)
    if knight_index >= 0:
        return oriented_from * PLANES_PER_SQUARE + 56 + knight_index

    step_file = 0
    if df > 0:
        step_file = 1
    elif df < 0:
        step_file = -1

    step_rank = 0
    if dr > 0:
        step_rank = 1
    elif dr < 0:
        step_rank = -1

    direction_index = _direction_index(step_file, step_rank)
    if direction_index < 0:
        return -1

    distance = abs(df)
    if abs(dr) > distance:
        distance = abs(dr)
    if distance < 1 or distance > 7:
        return -1

    plane = direction_index * 7 + (distance - 1)
    return oriented_from * PLANES_PER_SQUARE + plane


@_maybe_jit
def action_to_components(action, is_black, piece_type):
    oriented_from = action // PLANES_PER_SQUARE
    plane = action % PLANES_PER_SQUARE
    from_file = oriented_from & 7
    from_rank = oriented_from >> 3
    promotion = 0

    if plane < 56:
        direction_index = plane // 7
        distance = (plane % 7) + 1
        if direction_index == 0:
            df, dr = 0, 1
        elif direction_index == 1:
            df, dr = 0, -1
        elif direction_index == 2:
            df, dr = 1, 0
        elif direction_index == 3:
            df, dr = -1, 0
        elif direction_index == 4:
            df, dr = 1, 1
        elif direction_index == 5:
            df, dr = -1, 1
        elif direction_index == 6:
            df, dr = 1, -1
        else:
            df, dr = -1, -1
        to_file = from_file + df * distance
        to_rank = from_rank + dr * distance
    elif plane < 64:
        knight_index = plane - 56
        if knight_index == 0:
            df, dr = 1, 2
        elif knight_index == 1:
            df, dr = 2, 1
        elif knight_index == 2:
            df, dr = 2, -1
        elif knight_index == 3:
            df, dr = 1, -2
        elif knight_index == 4:
            df, dr = -1, -2
        elif knight_index == 5:
            df, dr = -2, -1
        elif knight_index == 6:
            df, dr = -2, 1
        else:
            df, dr = -1, 2
        to_file = from_file + df
        to_rank = from_rank + dr
    else:
        promo_index = plane - 64
        direction = promo_index // 3
        piece_index = promo_index % 3
        if direction == 0:
            df, dr = 0, 1
        elif direction == 1:
            df, dr = -1, 1
        else:
            df, dr = 1, 1
        to_file = from_file + df
        to_rank = from_rank + dr
        if piece_index == 0:
            promotion = KNIGHT
        elif piece_index == 1:
            promotion = BISHOP
        else:
            promotion = ROOK

    if to_file < 0 or to_file >= 8 or to_rank < 0 or to_rank >= 8:
        return -1, -1, 0

    oriented_to = to_rank * 8 + to_file
    if promotion == 0 and piece_type == PAWN and to_rank == 7:
        promotion = QUEEN

    real_from = _orient_square(oriented_from, is_black)
    real_to = _orient_square(oriented_to, is_black)
    return real_from, real_to, promotion


@_maybe_jit
def normalize_nonnegative(values):
    total = 0.0
    for value in values:
        if value > 0.0:
            total += value

    normalized = np.empty_like(values)
    if total <= 0.0:
        uniform = 1.0 / len(values)
        for index in range(len(values)):
            normalized[index] = uniform
        return normalized

    for index in range(len(values)):
        value = values[index]
        normalized[index] = value / total if value > 0.0 else 0.0
    return normalized


@_maybe_jit
def apply_temperature(probabilities, temperature):
    scaled = np.empty_like(probabilities)
    if temperature <= 1e-6:
        for index in range(len(probabilities)):
            scaled[index] = probabilities[index]
        return scaled

    exponent = 1.0 / temperature
    for index in range(len(probabilities)):
        scaled[index] = probabilities[index] ** exponent
    return normalize_nonnegative(scaled)
