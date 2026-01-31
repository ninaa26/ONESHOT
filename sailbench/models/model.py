"""Base physics model."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Pose:
    x: float  # global x position
    y: float  # global y position
    c: float  # cos of global heading angle
    s: float  # sin of global heading angle
    u: float  # body-frame x velocity
    v: float  # body-frame y velocity
    r: float  # body-frame angular velocity

    @property
    def heading(self) -> float:
        """Return the global heading angle in degrees."""
        return float(np.degrees(np.arctan2(self.s, self.c)))

    @property
    def velocity_magnitude(self) -> float:
        """Return the body-frame velocity magnitude."""
        return float(np.sqrt(self.u**2 + self.v**2))


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
