"""Linear hydrodynamic hull model."""

import numpy as np
from numpy.typing import NDArray

from sailbench.models.model import Model, State
from sailbench.sim.registry import register
from sailbench.tf.tf_tree import TFTree2D


@register(
    "hull",
    "linear",
    name="Linear",
    blurb="Linear viscous damping from xu1, yv1 and nr1",
)
class LinearHydroModel(Model):
    """Linear viscous drag hydrodynamic hull model."""

    def compute(self, state: State, tf_tree: TFTree2D) -> NDArray[np.float64]:
        """Compute forces on hull model.

        Args:
            state (State): State object
            tf_tree (TFTree2D): Unused; the hull works in the boat frame already.

        Returns:
            np.ndarray: Returns X and Y forces in newtons (within component frame)

        """
        u, v, r = state.u, state.v, state.r

        xu1 = float(self.p.get("xu1", 0.0))  # [N·s/m] surge
        yv1 = float(self.p.get("yv1", 0.0))  # [N·s/m] sway
        nr1 = float(self.p.get("nr1", 0.0))  # [N·m·s/rad] yaw damping

        x = -xu1 * u
        y = -yv1 * v
        # Yaw moment opposing rotation (positive r = starboard turn => negative moment)
        mz = -nr1 * r
        return np.array([x, y, mz], dtype=float)
