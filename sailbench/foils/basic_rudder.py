"""Basic rudder foil model."""

import numpy as np
from sailbench.models.foil import Foil


class BasicRudder(Foil):
    """Basic rudder foil model."""

    def __init__(self, params: dict) -> None:
        """Initialize the BasicRudder model.

        Args:
            params (dict): Dictionary of parameters for the rudder model.

        """
        super().__init__(params)

    def compute(self, state: np.ndarray, angle_input: float = 0.0) -> np.ndarray:
        """Compute the lift and drag coefficients for the rudder.

        Args:
            state (np.ndarray): Current boat state.
            angle_input (float): Angle of rudder in degrees.

        Returns:
            np.ndarray: Returns X and Y forces in newtons (within rudder frame)

        """
        # get angle of attack in radians
        alpha_rad = np.radians(angle_input)

        # get lift and drag coefficients
        cl, cd = self.cl_cd(alpha_rad, re=self.p.get("re", 1e5))

        # compute dynamic pressure
        rho = self.p.get("water_density", 1000.0)  # kg/m^3
        u = state[3]
        v = state[4]
        V = np.hypot(u, v)
        q = 0.5 * rho * V**2

        # compute forces
        S = self.p.get("area", 1.0)  # m^2
        L = CL * q * S
        D = CD * q * S

        # return forces in rudder frame (X forward, Y starboard)
        Fx = -D
        Fy = -L
        return np.array([Fx, Fy])
