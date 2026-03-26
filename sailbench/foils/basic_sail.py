"""Basic sail foil model."""

from typing import Any

import numpy as np

from sailbench.models.foil import Foil
from sailbench.models.model import State
from sailbench.tf.tf_tree import TFTree2D


class BasicSail(Foil):
    """Basic sail foil model."""

    def __init__(self, params: dict[str, Any]) -> None:
        """Initialize the BasicSail model.

        Args:
            params (dict): Dictionary of parameters for the sail model.

        """
        super().__init__(params)

    def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
        """Compute lift and drag forces for the sail.

        Uses tf_tree for all coordinate transforms. Sail angle comes from the
        "sail" frame. Rotates fluid frame → sail frame, then sail → boat via tf_tree.

        Args:
            state (State): Current boat state.
            tf_tree (TFTree2D): Transform tree with boat and sail frames.

        Returns:
            np.ndarray: X and Y forces in newtons (boat frame).

        """
        wind_speed = self.p.get("wind_speed", 0.0)
        wind_angle_deg = self.p.get("wind_dir_deg", 0.0)

        # Wind vector in world frame
        wind_rad = np.radians(wind_angle_deg)
        wind_global = np.array([
            wind_speed * np.cos(wind_rad),
            wind_speed * np.sin(wind_rad),
        ])
        # Boat velocity: state.u, state.v are body-frame; convert to world
        v_world = tf_tree.vector_to_frame(np.array([state.u, state.v]), "boat", "world")

        # Compute apparent wind in sail frame
        apparent_wind_global = wind_global - v_world
        apparent_wind_sail = tf_tree.vector_to_frame(apparent_wind_global, "world", "sail")

        aoa = np.degrees(np.arctan2(apparent_wind_sail[1], apparent_wind_sail[0]))

        cl, cd = self.cl_cd(np.radians(-aoa), re=self.p.get("re", 1e5))

        rho = self.p.get("air_density", 1.225)  # kg/m³
        V = np.linalg.norm(apparent_wind_sail)
        q = 0.5 * rho * V**2
        s = self.p.get("area", 1.0)  # m²
        lift = cl * q * s
        drag = cd * q * s

        # Force in fluid frame (x = wind direction; drag opposes motion => +drag along flow)
        f_fluid = np.array([drag, lift])

        # Rotate fluid → sail
        aoa_rad = np.radians(aoa)
        R = np.array([[np.cos(aoa_rad), -np.sin(aoa_rad)], [np.sin(aoa_rad), np.cos(aoa_rad)]])

        f_sail = R @ f_fluid
        # Rotate sail → boat using tf_tree
        sail_vec = tf_tree.vector_to_frame(f_sail, "sail", "boat")
        sail_vec[1] = 0
        return sail_vec
