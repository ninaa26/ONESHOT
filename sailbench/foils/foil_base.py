"""
Foil3DOF (XFoil-based superclass)
- Builds/loads XFoil polars via AeroSandbox and caches them to JSON.
- Child classes call cl_cd(alpha_rad) -> (CL, CD).
"""

from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import json
import numpy as np
import aerosandbox as asb


class Foil3DOF:
    def __init__(
        self,
        p: dict,
        *,
        cache_tag: str,                   # e.g. "rudder", "keel", "sail"
        alphas=np.arange(-25, 26, 1),     # degrees
        Re: float = 5e5,
        cache_dir: str = "cached_foils",
    ):
        self.p = p
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
                cache_filename=str(self.cache_file)
            )
        else:
            # Load existing cache (attaches CL_function/CD_function)
            self.foil.generate_polars(cache_filename=str(self.cache_file))
            # Try to restore alpha grid from cache if present
            try:
                with open(self.cache_file, "r") as f:
                    data = json.load(f)
                alpha_list = data.get("alphas") or data.get("alpha")
                if alpha_list is not None:
                    self.alphas_deg = np.asarray(alpha_list, dtype=float)
            except Exception:
                pass

        # bounds for clipping
        self.alpha_min = float(self.alphas_deg.min())
        self.alpha_max = float(self.alphas_deg.max())

    # ------------------------------------------------------------------
    @lru_cache(maxsize=None)
    def cl_cd(self, alpha_rad: float) -> tuple[float, float]:
        """Return (CL, CD) for an input angle in radians. Alpha is clipped to the cached range."""
        a_deg = float(np.clip(np.degrees(alpha_rad), self.alpha_min, self.alpha_max))
        CL = float(self.foil.CL_function(a_deg, Re=self.Re, mach=0.0))
        CD = float(self.foil.CD_function(a_deg, Re=self.Re, mach=0.0))
        return CL, CD
