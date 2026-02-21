"""Basic keel foil model."""

from typing import Any

import numpy as np

from sailbench.tf.tf_tree import TFTree2D
import sailbench.utils.coordinate_helper as utils
from sailbench.models.foil import Foil
from sailbench.models.model import State


class BasicKeel(Foil):
    """Basic keel foil model."""

    def __init__(self, params: dict[str, Any]) -> None:
        """Initialize the BasicRudder model.

        Args:
            params (dict): Dictionary of parameters for the rudder model.

        """
        super().__init__(params)

    def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
        """Compute the lift and drag coefficients for the keel.

        Assume the keel is parallel to the axis of the boat.

        Args:
            state (State): Current boat state.
            tf_tree (TFTree2D): Current transform tree of the boat, used to get component positions.

        Returns:
            np.ndarray: Returns X and Y forces in newtons (within keel frame)

        """
        local_track_vector = tf_tree.vector_to_frame(np.array([state.u, state.v]), "world", "boat")
        local_track = np.degrees(np.arctan2(local_track_vector[1], local_track_vector[0]))
        aoa = -local_track  # Keel angle of attack is negative of local track

        # get lift and drag coefficients
        cl, cd = self.cl_cd(np.radians(aoa), re=self.p.get("re", 1e5))
        # compute dynamic pressure
        rho = self.p.get("water_density", 1000.0)  # kg/m^3
        v = utils.get_velocity_magnitude(state)
        q = 0.5 * rho * v**2
        # compute forces
        s = self.p.get("area", 1.0)  # m^2
        lift = cl * q * s
        drag = cd * q * s
        # Fluid-frame force
        f_fluid = np.array([-drag, lift])
        return tf_tree.vector_to_frame(f_fluid, "fluid", "boat")
