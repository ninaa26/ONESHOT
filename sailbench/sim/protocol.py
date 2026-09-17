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
    rudder: dict[str, float]
    waypoint: PositionPayload
    control: dict[str, str]


@dataclass(slots=True)
class ControlInputs:
    """Parsed control inputs coming from the browser.

    All fields are optional; ``None`` means \"no change\" for that control.

    ``rudder_deg`` is rudder angle in degrees.
    ``sail_deg`` is sheet limit angle in degrees (max |sail angle| from centerline).
    """

    rudder_deg: float | None = None
    sail_deg: float | None = None
    wind_speed: float | None = None
    wind_dir_deg: float | None = None
    paused: bool | None = None
    reset: bool = False


@dataclass(slots=True)
class SetupInputs:
    """What the browser's shipyard screen asked to sail.

    ``boat`` is a YAML filename under ``configs/``. ``parts`` maps a component
    (``sail``, ``keel``, ``rudder``, ``hull``) to the model option chosen for
    it; a component left out keeps the model its config names. ``helm`` is
    ``"manual"`` or the id of a trained policy from the catalog.
    """

    boat: str
    parts: dict[str, str]
    helm: str = "manual"


def make_state_message(
    state: State,
    t: float,
    wind_speed: float | None = None,
    wind_dir_deg: float | None = None,
    sail_force: tuple[float, float] | None = None,
    forces: dict[str, tuple[float, float]] | None = None,
    sail_angle_deg: float | None = None,
    rudder_angle_deg: float | None = None,
    waypoint: tuple[float, float] | None = None,
    control_mode: str | None = None,
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

    if rudder_angle_deg is not None:
        msg["rudder"] = {
            "angle_deg": float(rudder_angle_deg),
        }

    if waypoint is not None:
        msg["waypoint"] = {
            "x": float(waypoint[0]),
            "y": float(waypoint[1]),
        }

    if control_mode is not None:
        msg["control"] = {
            "mode": str(control_mode),
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
            "sail_deg": float,       # optional, sheet-limit degrees
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



def parse_setup_message(data: Mapping[str, Any]) -> SetupInputs:
    """Parse a raw JSON mapping into structured SetupInputs.

    The expected JSON shape is::

        {
            "type": "setup",
            "boat": "flingo_floty.yaml",
            "parts": {"sail": "orc_w_jib", "keel": "finite_span"},  # optional
            "helm": "manual"                                          # optional
        }

    Only the shape is checked here; whether the boat, parts and helm exist is
    for the runner, which holds the catalog.
    """
    msg_type = data.get("type")
    if msg_type != "setup":
        msg = f"Unsupported setup message type: {msg_type!r}"
        raise ValueError(msg)

    boat = data.get("boat")
    if not isinstance(boat, str) or not boat:
        msg = "setup needs a boat (config filename)"
        raise TypeError(msg)

    raw_parts = data.get("parts") or {}
    if not isinstance(raw_parts, Mapping):
        msg = "parts must be a mapping of component -> model if present"
        raise TypeError(msg)
    parts: dict[str, str] = {}
    for component, model in raw_parts.items():
        if not isinstance(component, str) or not isinstance(model, str):
            msg = "parts must map component names to model names"
            raise TypeError(msg)
        parts[component] = model

    helm = data.get("helm", "manual")
    if not isinstance(helm, str) or not helm:
        msg = "helm must be a non-empty string if present"
        raise TypeError(msg)

    return SetupInputs(boat=boat, parts=parts, helm=helm)
