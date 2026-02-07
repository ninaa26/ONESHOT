"""Basic rudder foil model."""

import numpy as np

import sailbench.utils.coordinate_helper as utils
from sailbench.models.foil import Foil
from sailbench.models.model import State


class BasicRudder(Foil):
    """Basic rudder foil model."""

    def __init__(self, params: dict) -> None:
        """Initialize the BasicRudder model.

        Args:
            params (dict): Dictionary of parameters for the rudder model.

        """
        super().__init__(params)

    def compute(self, state: State, angle_input: float = 0.0) -> np.ndarray:
        """Compute the lift and drag coefficients for the rudder.

        Args:
            state (np.ndarray): Current boat state.
            angle_input (float): Angle of rudder in degrees.

        Returns:
            np.ndarray: Returns X and Y forces in newtons (within rudder frame)

        """
        # get angle of attack in radians
        # alpha_rad = np.radians(angle_input)
        global_track_vector = np.ndarray(state.u, state.v)
        local_track_vector = utils.global_to_local(global_track_vector)


        #Rotation matrix: local to rudder frame
        rotate_into_rudder = np.array([
        [np.cosine(np.radians(angle_input)), np.sine(np.radians(angle_input))],
        [-1* np.sine(np.radians(angle_input)), np.cosine(np.radians(angle_input))],
        ])

        rudder_frame_track = local_track_vector @ rotate_into_rudder


        # get lift and drag coefficients
        cl, cd = self.cl_cd(alpha_rad, re=self.p.get("re", 1e5))

        # compute dynamic pressure
        rho = self.p.get("water_density", 1000.0)  # kg/m^3
        u = state.u
        v = state.v
        v = np.hypot(u, v)
        q = 0.5 * rho * v**2

        # compute forces
        s = self.p.get("area", 1.0)  # m^2
        cl, cd = self.cl_cd(np.radians(np.arctan2(rudder_frame_track[1], rudder_frame_track[0])), re=self.p.get("re", 1e5)) 
        lift = cl * q * s
        drag = cd * q * s

        # return forces in rudder frame (X forward, Y starboard)
        #TODO : convert back to local frame
        fx = -drag
        fy = -lift

        return np.array([fx, fy])
