"""Base physics model."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class State:
    """State class."""

    x:float
    y:float
    psi:tuple[float,float] # angle is cosine (tuple[0]) + i*sine (tuple[1])
    u: float
    v: float
    r: float

    @property
    def get_heading(self) -> float:
        """Returns angle in degress."""
        return float(np.degrees(np.arctan(self.psi[1]/self.psi[0])))

class Model(ABC):
    """Base class for physics models."""

    def __init__(self, params: dict) -> None:
        """Store a dict of parameters (usually from YAML)."""
        self.p = params

    @abstractmethod
    def compute(self, state: State) -> np.ndarray:
        """Compute the outputted force vector produced by the component.

        Args:
            state (np.ndarray): Current body state of the sailboat -> [x, y, psi, u, v, r]

        Returns:
            np.ndarray: Returns X and Y forces in newtons (within component frame)

        """
        ...
