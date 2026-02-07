"""Basic sail foil model."""

from typing import Any

import numpy as np

from sailbench.models.foil import Foil
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

        # Calculate track in sail frame (global -> local -> sail)
        v_global = np.ndarray(state.u, state.v)
        # Convert to local frame
        v_local = utils.global_to_local(v_global, state.psi)
        # Convert to sail frame
        v_sail = R_local_to_sail  @ v_local

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
        q = 0.5 * rho * WIND_SPEED**2
        s = self.p.get("area", 1.0)  # m^2
        lift = cl * q * s
        drag = cd * q * s

        # Forces in the fluid frame
        f_fluid = np.array([-drag, lift])
        
        # Need to convert from fluid frame to sail frame first before converting to local frame
        f_fluid_sail = np.ndarray(R_local_to_sail @ f_fluid)
        return utils.fluid_frame_to_body_frame(f_fluid_sail, v_local)