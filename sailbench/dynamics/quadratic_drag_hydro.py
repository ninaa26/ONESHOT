"""Linear hydrodynamic hull model."""

import numpy as np

from sailbench.tf.tf_tree import TFTree2D
from sailbench.models.model import Model, State
from sailbench.sim.registry import register


@register(
    "hull",
    "quadratic",
    name="Quadratic",
    blurb="Quadratic viscous damping from xu2, yv2 and nr2",
)
class QuadraticHydroModel(Model):
    """Quadratic viscous drag hydrodynamic hull model."""

    def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
        """Compute forces on hull model."""

        u, v, r = state.u, state.v, state.r


        # Quadratic damping coefficients
        xu2 = float(self.p.get("xu2", 0.0))  # [N·s²/m²]
        yv2 = float(self.p.get("yv2", 0.0))  # [N·s²/m²]
        nr2 = float(self.p.get("nr2", 0.0))  # [N·m·s²/rad²]

        # Quadratic drag
        x = -xu2 * u * abs(u)
        y = -yv2 * v * abs(v)
        n = -nr2 * r * abs(r)

        return np.array([x, y, n], dtype=float)
    