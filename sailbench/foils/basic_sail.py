"""Basic sail foil model."""

from typing import Any

import numpy as np

from sailbench.models.foil import Foil
from sailbench.models.model import State
import sailbench.utils.coordinate_helper as utils


class BasicSail(Foil):
    """Basic sail foil model."""

    def __init__(self, params: dict[str, Any]) -> None:
        """Initialize the BasicSail model.

        Args:
            params (dict): Dictionary of parameters for the sail model.

        """
        super().__init__(params)

    def compute(self, state: State, sail_angle: float = 0.0) -> np.ndarray:
        """Compute the lift and drag coefficients for the sail.

        Assume the sail is parallel to the axis of the boat.

        Args:
            state (np.ndarray): Current boat state.
            sail_angle (float): Angle of the sail relative to local frame. 

        Returns:
            np.ndarray: Returns X and Y forces in newtons (within the boat frame)

        """
        WIND_SPEED, WIND_ANGLE_DEG = self.p.get("wind_speed"), self.p.get("wind_dir_deg")

        sail_angle_radians = np.radians(sail_angle)
        R_local_to_sail  = np.array([
            [np.cos(sail_angle_radians), -np.sin(sail_angle_radians)],
            [np.sin(sail_angle_radians), np.cos(sail_angle_radians)],
        ])

        # Calculate boat velocity in sail frame (global -> local -> sail)
        v_global = np.array([state.u, state.v])
        v_local = utils.global_to_local(v_global, state.psi)
        v_sail = R_local_to_sail @ v_local

        # Calculate wind angle in sail frame (global -> local -> sail)
        wind_global = utils.wind_to_vector(WIND_SPEED, WIND_ANGLE_DEG)
        wind_local = utils.global_to_local(wind_global, state.psi)
        wind_sail = R_local_to_sail  @ wind_local

        # Calculate apparent wind vector in sail frame, this is the AoA
        apparent_wind = wind_sail - v_sail
        aoa = np.degrees(np.arctan2(apparent_wind[1], apparent_wind[0]))

        # Get lift and drag coefficients
        cl, cd = self.cl_cd(np.radians(-aoa), re=self.p.get("re", 1e5))

        # Compute forces
        rho = self.p.get("air_density", 1.225)  # kg/m^3

        V = np.linalg.norm(apparent_wind)
        q = 0.5 * rho * V**2
        s = self.p.get("area", 1.0)  # m^2
        lift = cl * q * s
        drag = cd * q * s

        # Forces in fluid frame; convert to boat frame using apparent wind angle in boat frame
        f_fluid = np.array([-drag, lift])
        apparent_wind_boat = wind_local - v_local
        flow_angle_boat_deg = np.degrees(np.arctan2(apparent_wind_boat[1], apparent_wind_boat[0]))
        return utils.fluid_frame_to_body_frame(f_fluid, flow_angle_boat_deg)