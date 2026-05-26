from __future__ import annotations
import numpy as np
from gomoku.agents.bandits.base import BanditAgent

class GradientAgent(BanditAgent):
    """
    Unlike Q-value methods (epsilon greedy and UCB) this agent does not
    estimate expected rewards. It learns a preference H(a) per arm and 
    samples actions from a softmax over those preferences. The update is a 
    stochasitc gradient ascent step on E[R_t]
    """
    def __init__(
            self, 
            n_arms: int,
            alpha: float = 0.1,
            use_baseline: bool = True,
            rng: np.random.Generator | None = None,
    ) -> None: 
        self.alpha = alpha 
        self.use_baseline = use_baseline
        super().__init__(n_arms, rng)
    
    def reset(self) -> None:
        # Preference vector
        self.H = np.zeros(self.n_arms, dtype=np.float64)
        self.R_bar = 0.0 # average of the results
        self.n_updates = 0
        self._pi = None # cached softmax from the most recent select actions
    
    # pi(a) = e^H(a) / \sum_b e^H(b)
    def _softmax(self) -> np.ndarray:
        shifted = self.H - self.H.max()
        exp_h = np.exp(shifted)
        return exp_h / exp_h.sum()
    
        # example H = [2, 0, 1]
            # 1) shift so max is 0:   h = [0,-2,-1]
            # 2) exponetiate          e = [1, 0.135, 0.368]
            # 3) normalize to sum=1   π = [0.665, 0.090, 0.245]
    def select_action(self) -> int:
        self._pi = self._softmax()
        return int(self.rng.choice(self.n_arms, p=self._pi))
    
        # rng.choice(..., p=self._pi) samples an arm using the softmax probabilities. 
        # With the example above, it'd give arm 0 ~66.5% of the time, etc.

    def update(self, action: int, reward: float) -> None:
        pi = self._pi

        # Advantage: reward vs. the running average (or 0 if baseline off).
        baseline = self.R_bar if self.use_baseline else 0.0
        advantage = reward - baseline

        # Gradient update in two pieces:
        #   every arm:   H[a]   -= alpha * advantage * pi[a]
        #   chosen arm:  H[A_t] += alpha * advantage
        # Together, the chosen arm ends up with +alpha * advantage * (1 - pi[A_t]).
        self.H -= self.alpha * advantage * pi
        self.H[action] += self.alpha * advantage

        # Update the baseline AFTER the gradient step.
        # at step t as the mean of R_1..R_{t-1}, so the current reward folds
        # in only after the preference update has used the pre-step baseline.
        if self.use_baseline:
            self.n_updates += 1
            self.R_bar += (reward - self.R_bar) / self.n_updates