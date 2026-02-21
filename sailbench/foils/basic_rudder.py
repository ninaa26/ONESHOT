"""Basic rudder foil model."""

from typing import Any

import numpy as np

import sailbench.utils.coordinate_helper as utils
from sailbench.models.foil import Foil
from sailbench.models.model import State
from sailbench.tf.tf_tree import TFTree2D


class BasicRudder(Foil):
    """Basic rudder foil model."""

    def __init__(self, params: dict[str, Any]) -> None:
        """Initialize the BasicRudder model.

        Args:
            params (dict): Dictionary of parameters for the rudder model.

        """
        super().__init__(params)

    def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
        """Compute the lift and drag coefficients for the rudder.

        Args:
            state (np.ndarray): Current boat state.
            tf_tree (TFTree2D): Current transform tree of the boat, used to get component positions.

        Returns:
            np.ndarray: Returns X and Y forces in newtons (within rudder frame)


        """
        # get local track vector
        global_track_vector = np.array([[state.u], [state.v]])
        local_track_vector = utils.global_to_local(global_track_vector, state.psi)

        # angle of attack is angle of vector of negative track in rudder frame
        rudder_frame_track = tf_tree.vector_to_frame(local_track_vector, "boat", "rudder")
        angle_of_attack = -1 * np.arctan2(rudder_frame_track[1, 0], rudder_frame_track[0, 0])

        # get lift and drag coefficients
        cl, cd = self.cl_cd(angle_of_attack, re=self.p.get("re", 1e5))

        # compute dynamic pressure
        rho = self.p.get("water_density", 1000.0)  # kg/m^3
        u = state.u
        v = state.v
        v = np.hypot(u, v)
        q = 0.5 * rho * v**2

        # compute forces
        s = self.p.get("area", 1.0)  # m^2

        drag = cd * q * s
        lift = cl * q * s

        fx = -drag
        fy = lift

        rotate_into_rudder = np.array(
            [
                [np.cos(-1 * angle_of_attack), np.sin(-1 * angle_of_attack)],
                [-1 * np.sin(-1 * angle_of_attack), np.cos(-1 * angle_of_attack)],
            ],
        )

        # rotate lift and drag fluid --> rudder --> local frame
        f_prime = rotate_into_rudder @ np.array([[fx], [fy]])
        f_prime2 = tf_tree.vector_to_frame(f_prime, "rudder", "boat")

        return np.asarray([f_prime2[0, 0], f_prime2[1, 0]])
