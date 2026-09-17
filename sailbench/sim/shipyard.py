"""What the browser's shipyard screen can offer, and how to build what it picks.

The catalog is derived, not declared: boats are whatever YAML files under
``configs/`` describe a boat, and an option is available for a boat only if the
hub can actually construct it on that config. So adding a boat is dropping in a
YAML, and a model that needs keys the config lacks (``jib_area``, ``span``) is
greyed out with the hub's own reason rather than blowing up after launch.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from sailbench.sim.part_config import model_name
from sailbench.sim.sailboat_hub import CONFIG_PATH, KEEL_MODELS, RUDDER_MODELS, SAIL_MODELS, SailboatHub

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sailbench.sim.protocol import SetupInputs

RUNS_PATH = "runs/"

# Which YAML key an override sets to select each part's model, and what to call
# the options. The option ids for the foils are the hub's registry keys; the hull
# has no registry, its "model" is the friction law BasicHullModel reads.
#
# The foils are overridden through `model` rather than `model_type` on purpose.
# Both spellings are accepted, `model` wins, and a config may use either -- so an
# override that wrote `model_type` would be quietly ignored on any boat that had
# already migrated to `model`.
PART_KEY: dict[str, str] = {
    "sail": "model",
    "keel": "model",
    "rudder": "model",
    "hull": "friction_model",
}

PART_OPTIONS: dict[str, list[tuple[str, str, str]]] = {
    "sail": [
        ("basic", "Basic", "Symmetric NACA section polar at a geometric angle of attack"),
        ("orc_main", "ORC main", "ORC VPP coefficient envelope, single mainsail"),
        ("orc_w_jib", "ORC main + jib", "ORC envelope for a sloop, blended by area share (needs jib_area)"),
    ],
    "keel": [
        ("basic", "Basic", "2-D section polar; side force comes almost for free"),
        ("finite_span", "Finite span", "Lifting-line induced drag and post-stall blend (needs span)"),
    ],
    "rudder": [
        ("basic", "Basic", "2-D section polar, coefficients clamped past stall"),
        ("finite_span", "Finite span", "Induced drag and post-stall blend (needs span, no clamps)"),
    ],
    "hull": [
        ("flat", "Flat plate", "Constant skin-friction coefficient 0.004"),
        ("hughes", "Hughes", "ITTC-style Reynolds-dependent skin friction with form factor"),
    ],
}

# Registry aliases: the catalog names one id per model class, so a config that
# says `model_type: sail` shows up as `basic`.
_REGISTRIES: dict[str, dict[str, type]] = {"sail": SAIL_MODELS, "keel": KEEL_MODELS, "rudder": RUDDER_MODELS}

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
    key = str(value).lower()
    registry = _REGISTRIES.get(part)
    if registry is None:
        return key
    cls = registry.get(key)
    if cls is None:
        return key
    for option_id, _, _ in PART_OPTIONS[part]:
        if registry.get(option_id) is cls:
            return option_id
    return key


def overrides_for(parts: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Turn a parts selection into hub overrides."""
    return {part: {PART_KEY[part]: option} for part, option in parts.items() if part in PART_KEY}


def _pretty_name(stem: str) -> str:
    return stem.replace("_", " ").replace("-", " ").title()


def _describe_boat(path: Path) -> dict[str, Any] | None:
    """One boat card, or None if the YAML is not a boat config."""
    with path.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file) or {}
    if not all(section in cfg for section in ("boat", "hull", "sail", "keel", "rudder")):
        return None

    defaults: dict[str, str] = {}
    for part in PART_KEY:
        fallback = "flat" if part == "hull" else "basic"
        defaults[part] = _canonical(part, _selected(part, cfg.get(part, {}))) or fallback

    # Try every option on this boat. Constructing a hub is milliseconds, and the
    # error message is the model's own account of what it is missing.
    available: dict[str, dict[str, str | None]] = {}
    for part, options in PART_OPTIONS.items():
        available[part] = {}
        for option_id, _, _ in options:
            try:
                SailboatHub(path.name, overrides={part: {PART_KEY[part]: option_id}})
            except Exception as exc:  # noqa: BLE001 - any failure means "not on this boat"
                available[part][option_id] = str(exc)
            else:
                available[part][option_id] = None

    stats = []
    for section, key, label, unit in BOAT_STATS:
        value = cfg.get(section, {}).get(key)
        if isinstance(value, (int, float)):
            stats.append({"label": label, "value": float(value), "unit": unit})

    return {
        "id": path.name,
        "name": _pretty_name(path.stem),
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
            )
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
        part: [{"id": option_id, "name": name, "blurb": blurb} for option_id, name, blurb in options]
        for part, options in PART_OPTIONS.items()
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
        options = {o["id"] for o in catalog.parts.get(part, [])}
        if part not in catalog.parts:
            msg = f"unknown part {part!r}; expected one of: {', '.join(sorted(catalog.parts))}"
            raise ValueError(msg)
        if option not in options:
            msg = f"unknown {part} model {option!r}; expected one of: {', '.join(sorted(options))}"
            raise ValueError(msg)
    if setup.helm != "manual" and catalog.policy(setup.helm) is None:
        msg = f"unknown helm {setup.helm!r}; expected 'manual' or a listed policy"
        raise ValueError(msg)
    return setup
