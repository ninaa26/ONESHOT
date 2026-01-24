"""Helper functions for coordinate transforms."""

import numpy as np


def get_global_track(state: np.ndarray) -> float:
    """Get the boat's track angle in degrees.

    Args:
        state (np.ndarray): Current body state of the sailboat -> [x, y, psi, u, v, r]

    Returns:
        float: Boat track angle in degrees.

    """
    u = state[3]
    v = state[4]
    track_rad = np.arctan2(v, u)
    return float(np.degrees(track_rad))


def get_local_track(state: np.ndarray) -> float:
    """Get the boat's local track angle in degrees.

    Args:
        state (np.ndarray): Current body state of the sailboat -> [x, y, psi, u, v, r]

    Returns:
        float: Boat local track angle in degrees from 0 to 360.

    """
    psi = state[2]
    track_rad = get_global_track(state) - psi
    return float(np.degrees(track_rad) + 360) % 360


def get_velocity_magnitude(state: np.ndarray) -> float:
    """Get the boat's velocity magnitude.

    Args:
        state (np.ndarray): Current body state of the sailboat -> [x, y, psi, u, v, r]

    Returns:
        float: Boat velocity magnitude in m/s.

    """
    u = state[3]
    v = state[4]
    return float(np.hypot(u, v))
