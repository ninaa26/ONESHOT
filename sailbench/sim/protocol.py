from __future__ import annotations

"""Message formats for browser <-> simulation communication.

This module defines the JSON payload shapes used between the Python
simulation core and a browser frontend (e.g. three.js).
"""

from dataclasses import dataclass
from typing import Any, Mapping, TypedDict, cast

from sailbench.models.model import State


class HeadingPayload(TypedDict):
    """Heading representation."""

    cos: float
    sin: float
    deg: float


class PositionPayload(TypedDict):
    """2D position in the simulation world."""

    x: float
    y: float


class BodyVelocityPayload(TypedDict):
    """Body-frame velocities."""

    u: float
    v: float
    r: float


class WorldVelocityPayload(TypedDict):
    """World-frame translational velocity."""

    vx: float
    vy: float


class BoatKinematicsPayload(TypedDict):
    """Kinematic state for a single boat."""

    position: PositionPayload
    heading: HeadingPayload
    velocity_body: BodyVelocityPayload
    velocity_world: WorldVelocityPayload


class WindPayload(TypedDict, total=False):
    """Wind information at the boat."""

    speed: float  # [m/s]
    dir_deg: float  # [deg] from +x axis, in world frame


class SailForcePayload(TypedDict, total=False):
    """Sail force resolved in the boat frame."""

    fx: float  # [N] surge (along +x boat axis)
    fy: float  # [N] sway  (along +y boat axis)


class ComponentForcePayload(TypedDict):
    """Per-component force in boat frame."""

    fx: float
    fy: float


class ForcesPayload(TypedDict, total=False):
    """Per-component and total forces for visualization."""

    hull: ComponentForcePayload
    keel: ComponentForcePayload
    rudder: ComponentForcePayload
    sail: ComponentForcePayload
    total: ComponentForcePayload


class StateMessagePayload(TypedDict, total=False):
    """Top-level state message sent from Python to the browser."""

    type: str
    t: float
    boat: BoatKinematicsPayload
    wind: WindPayload
    sail_force: SailForcePayload
    forces: ForcesPayload
    sail: dict[str, float]
    waypoint: PositionPayload


@dataclass(slots=True)
class ControlInputs:
    """Parsed control inputs coming from the browser.

    All fields are optional; ``None`` means \"no change\" for that control.

    ``rudder_deg`` and ``sail_deg`` are angles in degrees, consistent with
    the higher-level config and browser UI.
    """

    rudder_deg: float | None = None
    sail_deg: float | None = None
    wind_speed: float | None = None
    wind_dir_deg: float | None = None
    paused: bool | None = None
    reset: bool = False


def make_state_message(
    state: State,
    t: float,
    wind_speed: float | None = None,
    wind_dir_deg: float | None = None,
    sail_force: tuple[float, float] | None = None,
    forces: dict[str, tuple[float, float]] | None = None,
    sail_angle_deg: float | None = None,
    waypoint: tuple[float, float] | None = None,
) -> StateMessagePayload:
    """Convert an internal State into a JSON-ready state message.

    Args:
        state: Current simulation state in body/world coordinates.
        t: Simulation time in seconds.
    """
    c, s = state.psi
    u, v, r = state.u, state.v, state.r

    # World-frame velocity components
    vx = c * u - s * v
    vy = s * u + c * v

    heading_deg = state.get_heading

    msg: StateMessagePayload = {
        "type": "state",
        "t": float(t),
        "boat": {
            "position": {
                "x": state.x,
                "y": state.y,
            },
            "heading": {
                "cos": c,
                "sin": s,
                "deg": heading_deg,
            },
            "velocity_body": {
                "u": u,
                "v": v,
                "r": r,
            },
            "velocity_world": {
                "vx": vx,
                "vy": vy,
            },
        },
    }

    if wind_speed is not None or wind_dir_deg is not None:
        msg["wind"] = {
            "speed": float(wind_speed or 0.0),
            "dir_deg": float(wind_dir_deg or 0.0),
        }

    if sail_force is not None:
        fx, fy = sail_force
        msg["sail_force"] = {
            "fx": float(fx),
            "fy": float(fy),
        }

    if forces is not None:
        force_payload = cast(
            ForcesPayload,
            {
                name: {"fx": float(fx), "fy": float(fy)}
                for name, (fx, fy) in forces.items()
            },
        )
        msg["forces"] = force_payload

    if sail_angle_deg is not None:
        msg["sail"] = {
            "angle_deg": float(sail_angle_deg),
        }

    if waypoint is not None:
        msg["waypoint"] = {
            "x": float(waypoint[0]),
            "y": float(waypoint[1]),
        }

    return msg


def _get_float(data: Mapping[str, Any], name: str) -> float | None:
    value = data.get(name)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    msg = f"{name} must be a number if present"
    raise TypeError(msg)


def _get_bool(data: Mapping[str, Any], name: str) -> bool | None:
    value = data.get(name)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    msg = f"{name} must be a boolean if present"
    raise TypeError(msg)


def parse_control_message(data: Mapping[str, Any]) -> ControlInputs:
    """Parse a raw JSON mapping into structured ControlInputs.

    The expected JSON shape is::

        {
            "type": "control",
            "rudder_deg": float,     # optional, degrees
            "sail_deg": float,       # optional, degrees
            "wind_speed": float,     # optional, m/s
            "wind_dir_deg": float,   # optional, from +x, degrees
            "paused": bool,          # optional
            "reset": bool            # optional
        }
    """
    msg_type = data.get("type")
    if msg_type is not None and msg_type != "control":
        msg = f"Unsupported control message type: {msg_type!r}"
        raise ValueError(msg)

    return ControlInputs(
        rudder_deg=_get_float(data, "rudder_deg"),
        sail_deg=_get_float(data, "sail_deg"),
        wind_speed=_get_float(data, "wind_speed"),
        wind_dir_deg=_get_float(data, "wind_dir_deg"),
        paused=_get_bool(data, "paused"),
        reset=bool(data.get("reset", False)),
    )

