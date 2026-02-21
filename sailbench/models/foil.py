"""Base foil model."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from aerosandbox import Airfoil

from sailbench.models.constants import FOIL_CACHE
from sailbench.models.model import Model


class Foil(Model):
    """Base class for all components utilizing foil physics."""

    def __init__(self, params: dict[str, Any]) -> None:
        """Store a dict of parameters (usually from YAML) and generate an airfoil.

        Note that XFoil must be installed for airfoil generation, otherwise you can use
        the cached polar.

        Args:
            params (dict): Dictionary of parameters for the foil model.

        """
        self.p = params
        self.airfoil_name = self.p.get("airfoil_name", "NACA0012")

        # ensure cache dir exists
        cache_dir = Path(FOIL_CACHE)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_dir / f"{self.airfoil_name}_{self.p.get('model_type', 'foil')}.json"

        # generate the Airfoil
        self.foil = Airfoil(name=self.airfoil_name)
        self.alphas_deg = np.arange(self.p.get("alpha_min"), self.p.get("alpha_max"), step=1)  # type: np.ndarray
        # build or load polar
        if not self.cache_file.exists():
            # First time: run XFoil and write cache
            res_raw = self.p.get("res", 1e5)
            Res = np.atleast_1d(np.asarray(res_raw, dtype=float))
            self.foil.generate_polars(
                alphas=self.alphas_deg,
                Res=Res,
                cache_filename=str(self.cache_file),
                xfoil_kwargs={
                    "timeout": 120,
                    "max_iter": 1000,
                }
            )
        else:
            # Load existing cache (attaches CL_function/CD_function)
            self.foil.generate_polars(cache_filename=str(self.cache_file))
            # Try to restore alpha grid from cache if present
            with self.cache_file.open() as f:
                data = json.load(f)
            alpha_list = data.get("alphas") or data.get("alpha")
            if alpha_list is not None:
                self.alphas_deg = np.asarray(alpha_list, dtype=float)

        # bounds for clipping
        self.alpha_min = float(self.alphas_deg.min())
        self.alpha_max = float(self.alphas_deg.max())

    def cl_cd(self, alpha_rad: float, re: float) -> tuple[float, float]:
        """Return (CL, CD) for an input angle in radians. Alpha is clipped to the cached range.

        Args:
            alpha_rad (float): Angle of attack in radians.
            re (float): Reynolds number.

        Returns:
            tuple[float, float]: (CL, CD)

        """
        a_deg = float(np.clip(np.degrees(alpha_rad), self.alpha_min, self.alpha_max))
        cl = float(self.foil.CL_function(a_deg, Re=re, mach=0.0))
        cd = float(self.foil.CD_function(a_deg, Re=re, mach=0.0))
        return cl, cd
