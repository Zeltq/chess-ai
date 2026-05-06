import numpy as np


class TicTacToe:
    def __init__(self):
        self.board_size = 3

    def get_initial_state(self):
        """Возвращает начальное состояние (пустая доска 3x3)."""
        return np.zeros((self.board_size, self.board_size), dtype=np.int8)

    def encode_state(self, state):
        """
        Кодирует состояние в тензор для нейросети.
        Возвращает numpy array (3, 3, 3) - 3 канала:
        канал 0: пустые клетки
        канал 1: X (1)
        канал 2: O (-1)
        """
        encoded = np.zeros((3, 3, 3), dtype=np.float32)
        encoded[0] = state == 0  # Пустые клетки
        encoded[1] = state == 1  # X
        encoded[2] = state == -1  # O
        return encoded.transpose(2, 0, 1)  # (C, H, W) для PyTorch

    def get_valid_actions(self, state):
        """Возвращает список легальных ходов (индексы 0-8)."""
        return np.where(state.flatten() == 0)[0].tolist()

    def is_terminal(self, state):
        """Проверяет, окончена ли игра."""
        return (
            self._check_win(state, 1)
            or self._check_win(state, -1)
            or len(self.get_valid_actions(state)) == 0
        )

    def get_reward(self, state):
        """
        Возвращает награду: 1.0 (X выиграл), -1.0 (O выиграл), 0.0 (ничья/игра продолжается).
        """
        if self._check_win(state, 1):
            return 1.0
        elif self._check_win(state, -1):
            return -1.0
        return 0.0

    def make_move(self, state, action):
        """Возвращает новое состояние после применения хода."""
        row, col = divmod(action, 3)
        new_state = state.copy()
        new_state[row, col] = self.get_current_player(state)
        return new_state

    def get_current_player(self, state):
        """
        Определяет, чей сейчас ход.
        X начинает, ходы чередуются.
        Возвращает 1 для X, -1 для O.
        """
        moves_made = np.sum(state != 0)
        return 1 if moves_made % 2 == 0 else -1

    def action_space_size(self):
        """Возвращает размер пространства действий (9 для Tic-Tac-Toe)."""
        return 9

    def _check_win(self, state, player):
        """Проверяет, выиграл ли указанный игрок."""
        # Строки
        for i in range(3):
            if np.all(state[i, :] == player):
                return True
        # Столбцы
        for i in range(3):
            if np.all(state[:, i] == player):
                return True
        # Диагонали
        if np.all(np.diag(state) == player):
            return True
        if np.all(np.diag(np.fliplr(state)) == player):
            return True
        return False

    def display(self, state):
        """Красивый вывод доски (для отладки)."""
        symbols = {0: ".", 1: "X", -1: "O"}
        print()
        for i in range(3):
            row = " ".join(symbols[state[i, j]] for j in range(3))
            print(f"  {row}")
        print()
