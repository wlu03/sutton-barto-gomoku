from __future__ import annotations
import numpy as np
class BanditTestbed: 
    def __init__(
            self, 
            n_arms: int = 10,
            rng: np.random.Generator | None = None,
            nonstationary: bool = False,
            walk_std: float = 0.01,
            reward_std: float = 1.0,
    ) -> None:
        self.n_arms = n_arms
        self.rng = rng if rng is not None else np.random.default_rng()
        self.nonstationary = nonstationary
        self.walk_std = walk_std
        self.reward_std = reward_std
        self.reset()
    def reset(self, seed: int | None = None) -> None:
        """Draw a fresh problem: new q* per arm. Call between independent runs."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.q_star = self.rng.standard_normal(self.n_arms)
    
    def step(self, action: int) -> float:
        """Pull arm 'action', return a sampled reward. Drifts q* if nonstationary"""
        reward = self.rng.normal(self.q_star[action], self.reward_std)
        if self.nonstationary:
            self.q_star += self.rng.normal(0.0, self.walk_std, size=self.n_arms)
        return float(reward)
    
    @property
    def optimal_action(self) -> int:
        return int(np.argmax(self.q_star))