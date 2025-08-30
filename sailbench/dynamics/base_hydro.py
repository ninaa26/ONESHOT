"""
Abstract base for 3-DOF hull hydrodynamics.

Interface:
    compute(state, statedot=None) -> np.ndarray([X, Y, N])

state    : [u, v, r, x, y, psi]
statedot : optional [du, dv, dr, dx, dy, dpsi] (some models need ν̇)

All forces/moment are in BOAT BODY axes (X forward, Y starboard).
Units: N, N, N·m (SI).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class HydroModel3DOF(ABC):
    def __init__(self, params: dict):
        """Store a dict of parameters (usually from YAML)."""
        self.p = params

    @abstractmethod
    def compute(self, state: np.ndarray,
                      statedot: Optional[np.ndarray] = None) -> np.ndarray:
        """Return np.array([X, Y, N]) in body axes."""
        ...
