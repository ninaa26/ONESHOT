"""Helper functions for coordinate transforms."""

import numpy as np
from sailbench.models.model import State


def get_global_track(state: State) -> float:
    """Get the boat's track angle in degrees.

    Args:
        state (np.ndarray): Current body state of the sailboat -> [x, y, psi, u, v, r]

    Returns:
        float: Boat track angle in degrees.

    """
    u = state.u
    v = state.v
    track_rad = np.arctan2(v, u)
    return float(np.degrees(track_rad))


def get_local_track(state: State) -> float:
    """Get the boat's local track angle in degrees.

    Args:
        state (State): Current body state of the sailboat -> [x, y, psi, u, v, r]

    Returns:
        float: Boat local track angle in degrees from 0 to 360.

    """
    psi = state.get_heading
    track_rad = get_global_track(state) - psi
    return float(np.degrees(track_rad) + 360) % 360


def get_velocity_magnitude(state: State) -> float:
    """Get the boat's velocity magnitude.

    Args:
        state (State): Current body state of the sailboat -> [x, y, psi, u, v, r]

    Returns:
        float: Boat velocity magnitude in m/s.

    """
    u = state.u
    v = state.v
    return float(np.hypot(u, v))


def fluid_frame_to_body_frame(forces_fluid: np.ndarray, local_track_deg: float) -> np.ndarray:
    """Convert forces from fluid frame to body frame.

    Args:
        forces_fluid (np.ndarray): Forces in fluid frame (x: lift, y: drag).
        local_track_deg (float): Local track angle in degrees.

    Returns:
        np.ndarray: Forces in body frame (+X, +Y).

    """
    local_track_rad = np.radians(local_track_deg)
    rotation_matrix = np.array(
        [[np.cos(local_track_rad), -np.sin(local_track_rad)], [np.sin(local_track_rad), np.cos(local_track_rad)]]
    )
    return np.asarray(rotation_matrix @ forces_fluid, dtype=np.float64)
