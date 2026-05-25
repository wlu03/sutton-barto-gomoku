"""Gomoku environment.

Two-player zero-sum, perfect information. The env tracks whose turn it is and
returns rewards from the acting player's point of view, so a single-agent RL
loop (or self-play) can drive both sides through the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BLACK = 1
WHITE = -1
EMPTY = 0

WIN_LEN = 5
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


@dataclass
class StepResult:
    board: np.ndarray         # (N, N) int8, view of internal state — copy if you need to keep it
    to_play: int              # player about to act next (BLACK or WHITE)
    reward: float             # from the POV of the player who just moved
    terminal: bool
    winner: int               # BLACK, WHITE, or EMPTY (draw / non-terminal)


class GomokuEnv:
    """Minimal Gomoku env. Black moves first."""

    def __init__(self, size: int = 15) -> None:
        self.size = size
        self._board = np.zeros((size, size), dtype=np.int8)
        self._to_play = BLACK
        self._winner = EMPTY
        self._done = False
        self._move_count = 0

    # ----- core API -----

    def reset(self) -> StepResult:
        self._board.fill(EMPTY)
        self._to_play = BLACK
        self._winner = EMPTY
        self._done = False
        self._move_count = 0
        return StepResult(self._board, self._to_play, 0.0, False, EMPTY)

    def step(self, action: int) -> StepResult:
        if self._done:
            raise RuntimeError("step() called on a terminated episode; call reset() first")
        r, c = divmod(action, self.size)
        if not (0 <= r < self.size and 0 <= c < self.size):
            raise ValueError(f"action {action} out of bounds for size {self.size}")
        if self._board[r, c] != EMPTY:
            raise ValueError(f"square ({r},{c}) already occupied")

        mover = self._to_play
        self._board[r, c] = mover
        self._move_count += 1

        if _is_winning_move(self._board, r, c, mover):
            self._winner = mover
            self._done = True
            return StepResult(self._board, -mover, 1.0, True, mover)
        if self._move_count == self.size * self.size:
            self._done = True
            return StepResult(self._board, -mover, 0.0, True, EMPTY)

        self._to_play = -mover
        return StepResult(self._board, self._to_play, 0.0, False, EMPTY)

    # ----- helpers -----

    def legal_actions(self) -> np.ndarray:
        """Flat indices of empty squares."""
        return np.flatnonzero(self._board.ravel() == EMPTY)

    def legal_mask(self) -> np.ndarray:
        """Boolean mask of shape (size*size,) where True means legal."""
        return (self._board.ravel() == EMPTY)

    @property
    def board(self) -> np.ndarray:
        return self._board

    @property
    def to_play(self) -> int:
        return self._to_play

    @property
    def done(self) -> bool:
        return self._done

    @property
    def winner(self) -> int:
        return self._winner

    def render(self) -> str:
        glyph = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        rows = ["  " + " ".join(f"{c:2d}" for c in range(self.size))]
        for r in range(self.size):
            cells = " ".join(f" {glyph[int(v)]}" for v in self._board[r])
            rows.append(f"{r:2d} {cells}")
        return "\n".join(rows)


def _is_winning_move(board: np.ndarray, r: int, c: int, player: int) -> bool:
    n = board.shape[0]
    for dr, dc in DIRS:
        count = 1
        rr, cc = r + dr, c + dc
        while 0 <= rr < n and 0 <= cc < n and board[rr, cc] == player:
            count += 1
            rr += dr
            cc += dc
        rr, cc = r - dr, c - dc
        while 0 <= rr < n and 0 <= cc < n and board[rr, cc] == player:
            count += 1
            rr -= dr
            cc -= dc
        if count >= WIN_LEN:
            return True
    return False
