from typing import Any

import numpy as np
from aerosandbox import Airfoil

from sailbench.models.model import Model


class Foil(Model):
    """Foil model with selectable backend (xfoil | neuralfoil)."""

    def __init__(self, params: dict[str, Any]) -> None:
        """Init neuralfoil parameters."""
        self.p = params
        self.airfoil_name = self.p.get("airfoil_name", "NACA0012")

        self.foil = Airfoil(name=self.airfoil_name)

        # Alpha limits (no cache dependency anymore)
        self.alpha_min = self.p.get("alpha_min", -20)
        self.alpha_max = self.p.get("alpha_max", 20)

    # ------------------------
    # CL/CD Interface
    # ------------------------
    def cl_cd(self, alpha_rad: float, re: float) -> tuple[float, float]:
        """Get CL and CD for a given angle of attack (in radians) and Reynolds number."""
        a_deg = float(np.clip(np.degrees(alpha_rad), self.alpha_min, self.alpha_max))

        result = self.foil.get_aero_from_neuralfoil(
            alpha=a_deg,
            Re=re,
            mach=0.0,
        )
        return float(result["CL"]), float(result["CD"])
