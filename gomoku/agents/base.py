"""Abstract agent interface.

All Gomoku agents — bandit, tabular, linear, deep — implement this. The env
hands the agent the current board and whose turn it is; the agent returns a
flat action index. Learning hooks (`observe`, `end_episode`) are optional and
no-op by default so non-learning agents can stay minimal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):
    @abstractmethod
    def act(self, board: np.ndarray, to_play: int, legal_mask: np.ndarray) -> int:
        """Return a flat action index in [0, board.size).

        `board` is a (N, N) int8 array (BLACK=1, WHITE=-1, EMPTY=0).
        `to_play` is BLACK or WHITE — the player the agent is choosing for.
        `legal_mask` is a (N*N,) bool array; True means the square is empty.
        """

    def observe(
        self,
        board: np.ndarray,
        to_play: int,
        action: int,
        reward: float,
        next_board: np.ndarray,
        next_to_play: int,
        terminal: bool,
    ) -> None:
        """Hook for learning agents. No-op by default."""

    def end_episode(self) -> None:
        """Hook called after a terminal step. No-op by default."""

    def reset(self) -> None:
        """Reset per-episode state (e.g. eligibility traces). No-op by default."""
