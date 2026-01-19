"""Base foil model."""

from abc import ABC, abstractmethod

import numpy as np

from sailbench.models.model import Model


class Foil(Model):
    """Base class for all components utilizing foil physics."""

    def __init__(self, params: dict) -> None:
        """Store a dict of parameters (usually from YAML)."""
        self.p = params
        self.Re = float(Re)
        self.alphas_deg = np.asarray(alphas, dtype=float)
        self.airfoil_name = p.get("airfoil_name", "NACA0012")

        # ensure cache dir exists
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_dir / f"{self.airfoil_name}_{cache_tag}_polar.json"

        # make the Airfoil
        self.foil = asb.Airfoil(name=self.airfoil_name)

        # build or load polar
        if not self.cache_file.exists():
            # First time: run XFoil and write cache
            self.foil.generate_polars(
                alphas=self.alphas_deg,
                Res=np.array([self.Re]),
                cache_filename=str(self.cache_file),
            )
        else:
            # Load existing cache (attaches CL_function/CD_function)
            self.foil.generate_polars(cache_filename=str(self.cache_file))
            # Try to restore alpha grid from cache if present
            try:
                with open(self.cache_file) as f:
                    data = json.load(f)
                alpha_list = data.get("alphas") or data.get("alpha")
                if alpha_list is not None:
                    self.alphas_deg = np.asarray(alpha_list, dtype=float)
            except Exception:
                pass

        # bounds for clipping
        self.alpha_min = float(self.alphas_deg.min())
        self.alpha_max = float(self.alphas_deg.max())

    @abstractmethod
    def compute(self, state: np.ndarray) -> np.ndarray:
        """Compute the outputted force vector produced by the component.

        Args:
            state (np.ndarray): Current body state of the sailboat -> [x, y, psi, u, v, r]

        Returns:
            np.ndarray: Returns X and Y forces in newtons (within component frame)

        """
        ...
