"""
    Covers three variants in this class: 
        - Sample-average (alpha=None)
        - Constant step size 
        - Optimistic initial values
    Greedy is just when epsilon is equal to 0
"""
from __future__ import annotations
import numpy as np
from gomoku.agents.bandits.base import BanditAgent

class EpsilonGreedyAgent(BanditAgent): 
    def __init__(
            self,
            n_arms: int,
            epsilon: float = 0.1,
            alpha: float | None = None,
            q_init: float = 0.0,
            rng: np.random.Generator | None = None
    ) -> None:
        self.epsilon = epsilon
        self.alpha = alpha
        self.q_init = q_init
        super().__init__(n_arms, rng)
    
    def reset(self) -> None:
        # Action-value estimates
        self.Q = np.full(self.n_arms, self.q_init, dtype=np.float64)
        # Number of time ach arm has been selected so far. 
        self.N = np.zeros(self.n_arms, dtype=np.int64)

    def select_action(self) -> int:
        # select from 0 to n_arm-1 of what action to choose 
        if self.rng.random() < self.epsilon: # explore with probability less than epsilon
            # thus explore random
            return int(self.rng.integers(0,self.n_arms))
        # else select the one with the best estimate Q
            # self.Q = [0.2, 0.2, 0.7, ..., 0.3]
            # pick the best arm which is 0.7
        best_val = self.Q.max() # 0.7 
        is_best = (self.Q == best_val) # boolean mask, turns array into bool array
        candidates = np.flatnonzero(is_best) # indices where mask is True
        chosen = self.rng.choice(candidates) # randomly choose 
        return int(chosen) # return the value
    
    def update(self, action: int, reward: float) -> None: 
        self.N[action] += 1 
        k = self.N[action]
        q_k = self.Q[action]

        if self.alpha is None: 
            step = 1/k
        else:
            step = self.alpha

        q_k_plus_1 = q_k + step * (reward - q_k)
        self.Q[action] = q_k_plus_1