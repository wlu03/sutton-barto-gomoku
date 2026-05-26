from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod

class BanditAgent(ABC):
    """Interface for n-armed bandit algorithms"""

    def __init__(self, n_arms: int, rng: np.random.Generator | None = None):
        self.n_arms = n_arms
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()
    @abstractmethod
    def reset(self) -> None: 
        """Clear all learned state. Called one per independent run"""
    @abstractmethod
    def select_action(self) -> int: 
        """Return an arm index in [0, n_arms]"""
    @abstractmethod
    def update(self, action: int, reward: float) -> None: 
        """Update internal estimates after observing (action, reward)"""