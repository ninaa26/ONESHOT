"""Base physics model."""

from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    """Base class for physics models."""

    def __init__(self, params: dict) -> None:
        """Store a dict of parameters (usually from YAML)."""
        self.p = params

    @abstractmethod
    def compute(self, state: np.ndarray) -> np.ndarray:
        """Compute the outputted force vector produced by the component.

        Args:
            state (np.ndarray): Current body state of the sailboat -> [x, y, psi, u, v, r]

        Returns:
            np.ndarray: Returns X and Y forces in newtons (within component frame)

        """
        ...
