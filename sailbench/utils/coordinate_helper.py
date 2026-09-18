"""Helper functions for coordinate transforms."""

import numpy as np

from sailbench.models.model import State
from sailbench.tf.tf_tree import TFTree2D, Transform2D


def fluid_transform_from_velocity(u: float, v: float) -> Transform2D:
    """Build the boat->fluid transform for a local flow velocity.

    The fluid frame's +x axis points along the direction the water travels
    relative to the boat, which is opposite its velocity. Foils resolve their
    drag along +x of this frame.

    Takes the velocity directly rather than a `State`, so a component sitting
    away from the centre of rotation can pass the flow it actually sees,
    including its yaw-rate contribution.

    Args:
        u (float): Local surge velocity [m/s].
        v (float): Local sway velocity [m/s].

    Returns:
        Transform2D: Rotation-only transform from the boat frame to the fluid frame.

    """
    speed = float(np.hypot(u, v))
    if speed <= 1e-6:
        return Transform2D(x=0.0, y=0.0, c=1.0, s=0.0)
    return Transform2D(x=0.0, y=0.0, c=-float(u) / speed, s=-float(v) / speed)


def apparent_wind_boat(
    state: State,
    tf_tree: TFTree2D,
    wind_speed: float,
    wind_dir_deg: float,
) -> np.ndarray:
    """Apparent wind in the boat frame, as the vector the air travels along.

    Shared by every above-water model so they cannot disagree about the wind.
    `wind_dir_deg` is the direction the true wind blows *to*, in the world frame.

    Args:
        state (State): Current body state of the sailboat.
        tf_tree (TFTree2D): Transform tree, used for the boat's heading.
        wind_speed (float): True wind speed [m/s].
        wind_dir_deg (float): Direction the true wind blows toward [deg].

    Returns:
        np.ndarray: Apparent wind vector in the boat frame [m/s].

    """
    wind_rad = np.radians(float(wind_dir_deg))
    wind_world = float(wind_speed) * np.array([np.cos(wind_rad), np.sin(wind_rad)], dtype=float)
    v_boat_world = tf_tree.vector_to_frame(np.array([state.u, state.v], dtype=float), "boat", "world")
    return np.asarray(tf_tree.vector_to_frame(wind_world - v_boat_world, "world", "boat"), dtype=float)
