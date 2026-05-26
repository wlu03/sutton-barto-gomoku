from __future__ import annotations
import numpy as np
from gomoku.agents.bandits.base import BanditAgent

class UCBAgent(BanditAgent):
    def __init__(
            self,
            n_arms: int,
            c: float = 2.0,
            rng: np.random.Genearator | None = None,
    ) -> None: 
        self.c = c
        super().__init__(n_arms, rng)
    def reset(self) -> None: 
        self.Q = np.zeros(self.n_arms, dtype=np.float64) # estimated state
        self.N = np.zeros(self.n_arms, dtype=np.int64) # per arm counter
        self.t = 0 # total selections 
    
    def select_action(self) -> int:
        self.t += 1
        bonus = np.full(self.n_arms, np.inf)
        tried = self.N > 0 # boolean mask of the N array, must have tried once
        bonus[tried] = self.c * np.sqrt(np.log(self.t) / self.N[tried])
        ucb = self.Q + bonus
        candidates = np.flatnonzero(ucb == ucb.max())
        return int(self.rng.choice(candidates))
    
    def update(self, action: int, reward: float) -> None:
        self.N[action] += 1 
        self.Q[action] += (reward - self.Q[action]) / self.N[action]
