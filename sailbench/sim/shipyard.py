"""What the browser's shipyard screen can offer, and how to build what it picks.

The catalog is derived, not declared: boats are whatever YAML files under
``configs/`` describe a boat, and an option is available for a boat only if the
hub can actually construct it on that config. So adding a boat is dropping in a
YAML, and a model that needs keys the config lacks (``jib_area``, ``span``) is
greyed out with the hub's own reason rather than blowing up after launch.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from sailbench.sim.part_config import SELECTOR_KEYS, compose, model_name
from sailbench.sim.registry import canonical as registry_canonical
from sailbench.sim.registry import options, unavailable
from sailbench.sim.sailboat_hub import CONFIG_PATH

if TYPE_CHECKING:
    from sailbench.sim.protocol import SetupInputs

RUNS_PATH = "runs/"

# The rows the shipyard shows, each mapping to what it actually selects:
# (the part models register under, the config key an override sets).
#
# The hull row is the odd one: it picks a skin-friction law, which is a choice
# *within* the hull model rather than a choice of hull model. The three hull
# models themselves are selectable by config now but are not offered here yet.
#
# The foils are overridden through `model` rather than `model_type` on purpose.
# Both spellings are accepted, `model` wins, and a config may use either -- so an
# override that wrote `model_type` would be quietly ignored on any boat that had
# already migrated to `model`.
PARTS: dict[str, tuple[str, str]] = {
    "sail": ("sail", "model"),
    "keel": ("keel", "model"),
    "rudder": ("rudder", "model"),
    "hull": ("friction", "friction_model"),
}

# Kept as the key each part's override writes, which is all callers need.
PART_KEY: dict[str, str] = {row: key for row, (_, key) in PARTS.items()}

# The stats the boat cards show, as (section, key, label, unit).
BOAT_STATS: list[tuple[str, str, str, str]] = [
    ("boat", "mass", "Mass", "kg"),
    ("boat", "inertia_z", "Yaw inertia", "kg·m²"),
    ("hull", "L", "Length", "m"),
    ("sail", "area", "Sail area", "m²"),
    ("keel", "area", "Keel area", "m²"),
    ("rudder", "area", "Rudder area", "m²"),
]


@dataclass(slots=True)
class Policy:
    """A trained checkpoint the helm can hand the boat to."""

    id: str
    name: str
    model_path: str
    config_path: str | None


@dataclass(slots=True)
class Catalog:
    """Everything the shipyard screen needs to render its choices."""

    boats: list[dict[str, Any]]
    parts: dict[str, list[dict[str, str]]]
    policies: list[Policy]
    default_boat: str
    default_helm: str = "manual"
    boat_ids: set[str] = field(init=False)

    def __post_init__(self) -> None:
        """Index the boats for validation."""
        self.boat_ids = {boat["id"] for boat in self.boats}

    def to_payload(self) -> dict[str, Any]:
        """Return the JSON the browser receives on connect."""
        return {
            "type": "catalog",
            "boats": self.boats,
            "parts": self.parts,
            "policies": [{"id": p.id, "name": p.name} for p in self.policies],
            "default_boat": self.default_boat,
            "default_helm": self.default_helm,
        }

    def policy(self, helm: str) -> Policy | None:
        """Return the policy a helm id names, or None for manual."""
        for policy in self.policies:
            if policy.id == helm:
                return policy
        return None


def _selected(part: str, section: Mapping[str, Any]) -> str:
    """Return the model a boat's section currently names, before canonicalisation.

    The foils accept either spelling of the selector key, so the reading of it
    lives in one place; the hull is not in a registry and names its friction law
    under a key of its own.
    """
    if part == "hull":
        return str(section.get(PART_KEY["hull"], "flat")).lower()
    return model_name(section)


def _canonical(part: str, value: object) -> str | None:
    """Map a config's selector value to the catalog option id for the same model."""
    if value is None:
        return None
    return registry_canonical(part, str(value))


def overrides_for(parts: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Turn a parts selection into hub overrides."""
    return {part: {PART_KEY[part]: option} for part, option in parts.items() if part in PART_KEY}


def _reason_unavailable(part: str, key: str, option_id: str, section: Mapping[str, Any]) -> str | None:
    """Return why this boat cannot use this model, or None if it can.

    The parameters are composed the way the hub would compose them for this
    option, so a boat carrying a `models` block is judged on the block that would
    actually be used rather than on the section as written. A row that selects
    something *inside* a model instead of the model itself -- the hull's friction
    law -- has nothing to compose, and its section is the parameters.
    """
    if key in SELECTOR_KEYS:
        try:
            params = compose(part, {**section, key: option_id})
        except (TypeError, ValueError) as exc:
            # A boat that offers `models` but has no block for this one says so
            # in its own words; that is the answer to whether it can be used.
            return str(exc)
    else:
        params = {**section, key: option_id}
    return unavailable(part, option_id, params)


def _pretty_name(stem: str) -> str:
    """Return a display name derived from a config's filename."""
    return stem.replace("_", " ").replace("-", " ").title()


def _boat_name(cfg: dict[str, Any], stem: str) -> str:
    """Return what to call this boat: its own `boat.name`, else its filename.

    Title-casing a filename cannot know that WPI is an acronym, so a boat that
    cares says what it is called.
    """
    declared = cfg.get("boat", {}).get("name")
    return str(declared) if declared else _pretty_name(stem)



# A foil with no span stated anywhere still has to be drawn. This is the aspect
# ratio assumed for it, chosen to match the one the unmeasured boats state.
FALLBACK_ASPECT_RATIO = 4.0
FALLBACK_RIG_ASPECT_RATIO = 3.0


def _span_of(section: Mapping[str, Any], fallback_ar: float) -> float | None:
    """Return a span for drawing this foil, from whatever the config gives.

    A span may sit at the section level, or inside the block of whichever model
    wants it, or not exist at all -- the shape of the boat does not depend on
    which model is selected, so all three are searched and an aspect ratio is
    the last resort.
    """
    area = section.get("area")
    if not isinstance(area, (int, float)) or area <= 0:
        return None

    candidates = [section, *(v for v in (section.get("models") or {}).values() if isinstance(v, Mapping))]
    for source in candidates:
        span = source.get("span")
        if isinstance(span, (int, float)) and span > 0:
            return float(span)
    for source in candidates:
        ar = source.get("effective_aspect_ratio")
        if isinstance(ar, (int, float)) and ar > 0:
            return float((ar * float(area)) ** 0.5)
    return float((fallback_ar * float(area)) ** 0.5)


def _geometry(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return the dimensions a renderer needs to draw this boat to scale.

    The catalog's stats are for reading; these are for building. Everything is
    in metres, in the simulator's own frame: +x forward of the centre of
    rotation, spans measured downwards for the foils and upwards for the rig.
    """
    def number(section: str, key: str) -> float | None:
        value = cfg.get(section, {}).get(key)
        return float(value) if isinstance(value, (int, float)) else None

    parts: dict[str, Any] = {
        "hull": {"L": number("hull", "L"), "B": number("hull", "B"), "T": number("hull", "T")},
    }
    for part, fallback in (("keel", FALLBACK_ASPECT_RATIO), ("rudder", FALLBACK_ASPECT_RATIO),
                           ("sail", FALLBACK_RIG_ASPECT_RATIO)):
        section = cfg.get(part, {})
        parts[part] = {
            "area": number(part, "area"),
            "span": _span_of(section, fallback),
            "x_pos": number(part, "x_pos") or 0.0,
        }
    return parts


def _describe_boat(path: Path) -> dict[str, Any] | None:
    """One boat card, or None if the YAML is not a boat config."""
    with path.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file) or {}
    if not all(section in cfg for section in ("boat", "hull", "sail", "keel", "rudder")):
        return None

    defaults: dict[str, str] = {}
    for row, (part, _) in PARTS.items():
        fallback = "flat" if row == "hull" else "basic"
        defaults[row] = _canonical(part, _selected(row, cfg.get(row, {}))) or fallback

    # What each model says it needs, against what this boat gives it. No hub is
    # built: the question is asked of the declaration, not of the object, so the
    # catalog costs a dict lookup per option rather than a boat per option.
    available: dict[str, dict[str, str | None]] = {}
    for row, (part, key) in PARTS.items():
        available[row] = {}
        for option in options(part):
            available[row][option.id] = _reason_unavailable(part, key, option.id, cfg.get(row, {}))

    stats = []
    for section, key, label, unit in BOAT_STATS:
        value = cfg.get(section, {}).get(key)
        if isinstance(value, (int, float)):
            stats.append({"label": label, "value": float(value), "unit": unit})

    return {
        "id": path.name,
        "name": _boat_name(cfg, path.stem),
        "geometry": _geometry(cfg),
        "defaults": defaults,
        "available": available,
        "stats": stats,
    }


def _find_policies(default_model: str | None, default_config: str | None) -> list[Policy]:
    """Trained checkpoints under runs/, plus whatever the CLI was started with."""
    if importlib.util.find_spec("stable_baselines3") is None:
        return []

    policies: list[Policy] = []
    seen: set[str] = set()

    def add(model_path: Path, name: str, config_path: Path | None) -> None:
        key = str(model_path)
        if key in seen:
            return
        seen.add(key)
        policies.append(
            Policy(
                id=key,
                name=name,
                model_path=key,
                config_path=str(config_path) if config_path is not None else None,
            ),
        )

    if default_model is not None and Path(default_model).is_file():
        add(
            Path(default_model),
            f"{Path(default_model).parent.name}/{Path(default_model).name} (--policy-model)",
            Path(default_config) if default_config else None,
        )

    runs = Path(RUNS_PATH)
    if runs.is_dir():
        for run in sorted(p for p in runs.iterdir() if p.is_dir()):
            # The run's own config carries the env scaling the policy was trained
            # with; fall back to the CLI's config when a run did not keep one.
            used = run / "config_used.yaml"
            config = used if used.is_file() else (Path(default_config) if default_config else None)
            best = run / "best_model" / "best_model.zip"
            if best.is_file():
                add(best, f"{run.name} (best)", config)
            final = run / "final_model.zip"
            if final.is_file():
                add(final, f"{run.name} (final)", config)
    return policies


def build_catalog(
    default_boat: str,
    policy_model: str | None = None,
    policy_config: str | None = None,
) -> Catalog:
    """Scan configs/ and runs/ for what the shipyard can offer."""
    boats = []
    for path in sorted(Path(CONFIG_PATH).glob("*.yaml")):
        try:
            boat = _describe_boat(path)
        except Exception:  # noqa: BLE001 - a broken YAML is not a boat, skip it
            continue
        if boat is not None:
            boats.append(boat)
    if not boats:
        msg = f"no boat configs found under {CONFIG_PATH}"
        raise RuntimeError(msg)

    parts = {
        row: [{"id": o.id, "name": o.name, "blurb": o.described()} for o in options(part)]
        for row, (part, _) in PARTS.items()
    }
    policies = _find_policies(policy_model, policy_config)
    default_helm = policies[0].id if policy_model is not None and policies else "manual"
    if default_boat not in {b["id"] for b in boats}:
        default_boat = boats[0]["id"]
    return Catalog(boats=boats, parts=parts, policies=policies, default_boat=default_boat, default_helm=default_helm)


def validate_setup(setup: SetupInputs, catalog: Catalog) -> SetupInputs:
    """Check a setup against the catalog; the hub still has the final say."""
    if setup.boat not in catalog.boat_ids:
        known = ", ".join(sorted(catalog.boat_ids))
        msg = f"unknown boat {setup.boat!r}; expected one of: {known}"
        raise ValueError(msg)
    for part, option in setup.parts.items():
        offered = {o["id"] for o in catalog.parts.get(part, [])}
        if part not in catalog.parts:
            msg = f"unknown part {part!r}; expected one of: {', '.join(sorted(catalog.parts))}"
            raise ValueError(msg)
        if option not in offered:
            msg = f"unknown {part} model {option!r}; expected one of: {', '.join(sorted(offered))}"
            raise ValueError(msg)
    if setup.helm != "manual" and catalog.policy(setup.helm) is None:
        msg = f"unknown helm {setup.helm!r}; expected 'manual' or a listed policy"
        raise ValueError(msg)
    return setup
