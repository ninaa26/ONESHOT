"""What the browser's shipyard screen can offer, and how to build what it picks.

The catalog is derived, not declared: boats are whatever YAML files under
``configs/`` describe a boat, and an option is available for a boat only if the
hub can actually construct it on that config. So adding a boat is dropping in a
YAML, and a model that needs keys the config lacks (``jib_area``, ``span``) is
greyed out with the hub's own reason rather than blowing up after launch.
"""

from __future__ import annotations

import importlib.util
import json
import re
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

#: What `scripts/score_runs.py` leaves in a run directory.
SCORECARD_NAME = "scorecard.json"

#: A run directory the trainer named itself. Any other name was chosen by a
#: person, and a name a person chose beats one derived from a config diff.
GENERATED_RUN = re.compile(r"^waypoint_ppo_\d{8}_\d{6}$")

#: How a differing config key is said in a chip. Cosmetic only -- a key that is
#: not here falls back to its own name, so a knob that starts differing between
#: runs needs nothing added.
KNOB_NAMES: dict[str, str] = {
    "env.upwind_waypoint_bias": "upwind bias",
    "env.no_go_zone_penalty": "no-go",
    "env.no_go_zone_half_angle_deg": "no-go angle",
    "env.no_go_occupancy_tau_s": "no-go lag",
    "env.jibe_penalty": "jibe penalty",
    "env.include_actuator_state": "actuator obs",
    "env.success_reward": "mark bonus",
    "train.learning_rate": "lr",
    "train.ent_coef": "entropy",
    "train.gamma": "gamma",
}

#: Keys that differ without saying anything about what a run tried: the boat
#: (the helm row reports it already), bookkeeping cadence, and the budget, which
#: every chip carries in front anyway.
KNOBS_IGNORED = frozenset(
    {
        "env.simulator_config",
        "train.total_timesteps",
        "train.checkpoint_freq",
        "train.eval_freq",
        "train.tensorboard_log",
        "train.device",
        "train.seed",
        "train.watch_web",
        "train.watch_stride",
    }
)

#: A policy reaching the mark less often than this has that said about it first;
#: above it, success rate separates nothing on these runs and speed is the story.
SUCCESS_WORTH_SAYING = 0.9

#: Sheet command pinned at an extreme for less than this share of steps means
#: the policy is actually trimming, which one of these runs does and the rest
#: do not.
TRIMS_BELOW = 0.9

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
    #: The boat the run trained against, when its config kept one. A policy can
    #: still be sailed on another boat; the shipyard just says which it learned.
    trained_on: str | None = None
    #: Why this checkpoint cannot be sailed, or None when it can. A checkpoint
    #: that exists is always listed: greyed out with the reason beats vanishing.
    reason: str | None = None
    #: The long form of the name: the path, what the run changed, and what the
    #: scorecard measured. The chip has room for a few words; this has room for
    #: the sentence behind them.
    blurb: str | None = None


@dataclass(slots=True)
class Catalog:
    """Everything the shipyard screen needs to render its choices."""

    boats: list[dict[str, Any]]
    parts: dict[str, list[dict[str, str]]]
    policies: list[Policy]
    default_boat: str
    default_helm: str = "manual"
    #: One line for the helm row when there is something to explain: no trained
    #: runs on disk, or runs that are there but cannot be loaded.
    policy_note: str | None = None
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
            "policies": [
                {
                    "id": p.id,
                    "name": p.name,
                    "trained_on": p.trained_on,
                    "blurb": p.blurb,
                    "disabled": p.reason is not None,
                    "reason": p.reason,
                }
                for p in self.policies
            ],
            "default_boat": self.default_boat,
            "default_helm": self.default_helm,
            "policy_note": self.policy_note,
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
    for part, fallback in (
        ("keel", FALLBACK_ASPECT_RATIO),
        ("rudder", FALLBACK_ASPECT_RATIO),
        ("sail", FALLBACK_RIG_ASPECT_RATIO),
    ):
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


def _backend_reason() -> str | None:
    """Why a checkpoint on disk could not be sailed, or None when it can be.

    Only *loading* a policy needs the RL extras; finding one is a directory
    listing. Keeping the two apart is what lets the shipyard show the boat's
    trained models greyed out with this reason, rather than claiming the runs
    directory is empty.
    """
    for module, extra in (("stable_baselines3", "stable-baselines3"), ("gymnasium", "gymnasium")):
        if importlib.util.find_spec(module) is None:
            return f"needs {extra}: install the RL extras with `uv sync --group rl`"
    return None


def _trained_on(config_path: Path | None) -> str | None:
    """Return the boat config a run trained against, or None when it kept none."""
    if config_path is None or not config_path.is_file():
        return None
    try:
        with config_path.open(encoding="utf-8") as file:
            cfg = yaml.safe_load(file) or {}
    except Exception:
        return None
    boat = (cfg.get("env") or {}).get("simulator_config")
    return str(boat) if boat else None


@dataclass(slots=True)
class _Checkpoint:
    """A trained file on disk, before it has anything to call itself."""

    model: Path
    #: Which checkpoint of its run this is: the best evaluation, or the last.
    kind: str
    #: The run it came from, or None for a path handed in on the command line.
    run: Path | None
    config: Path | None


def _checkpoints(default_model: str | None, default_config: str | None) -> list[_Checkpoint]:
    """Every checkpoint the helm could be handed, in run order."""
    fallback = Path(default_config) if default_config else None
    found: list[_Checkpoint] = []
    seen: set[Path] = set()

    runs = Path(RUNS_PATH)
    if runs.is_dir():
        for run in sorted(p for p in runs.iterdir() if p.is_dir()):
            # The run's own config carries the env scaling the policy was trained
            # with; fall back to the CLI's config when a run did not keep one.
            used = run / "config_used.yaml"
            config = used if used.is_file() else fallback
            for kind, relative in (("best", "best_model/best_model.zip"), ("final", "final_model.zip")):
                model = run / relative
                if model.is_file():
                    found.append(_Checkpoint(model=model, kind=kind, run=run, config=config))
                    seen.add(model)

    # A checkpoint named on the command line that is not under runs/ is still
    # sailable, so it is listed -- it just has no run to be described by.
    if default_model is not None:
        model = Path(default_model)
        if model.is_file() and model not in seen:
            found.append(_Checkpoint(model=model, kind="named", run=None, config=fallback))
    return found


def _read_config(path: Path | None) -> dict[str, Any]:
    """Load a run's config, or an empty one when it kept none."""
    if path is None or not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as file:
            return dict(yaml.safe_load(file) or {})
    except (OSError, yaml.YAMLError):
        return {}


def _flatten(config: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a config to `section.key` entries so two runs can be compared."""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, Mapping):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _number(value: object) -> str:
    """Render a config value the way a label wants it, not the way YAML wrote it."""
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".") or "0"
    return str(value)


def _budget(steps: object) -> str | None:
    """Return `750000` as `750 k`, which is how anyone reads a training budget."""
    if isinstance(steps, bool) or not isinstance(steps, (int, float)):
        return None
    total = int(steps)
    if total >= 1_000_000:
        return f"{total / 1_000_000:g} M"
    if total >= 1_000:
        return f"{total / 1_000:g} k"
    return str(total)


def _unusual(flat: Mapping[str, Any], peers: list[Mapping[str, Any]], limit: int = 2) -> list[str]:
    """Return the knobs that set this run apart from the others on the same boat.

    There is no list of interesting keys, because which key is interesting
    depends on the runs sitting next to it: a knob matters here exactly when
    the runs disagree about it. They come back rarest-value first, so the
    leading token is the thing this run was actually trying.
    """
    ranked: list[tuple[int, str]] = []
    for key, value in flat.items():
        if key in KNOBS_IGNORED:
            continue
        values = [peer.get(key) for peer in peers]
        if all(other == value for other in values):
            continue
        ranked.append((sum(1 for other in values if other == value), key))
    ranked.sort()
    return [
        f"{KNOB_NAMES.get(key, key.rsplit('.', 1)[-1].replace('_', ' '))} {_number(flat[key])}"
        for _, key in ranked[:limit]
    ]


def _score(checkpoint: _Checkpoint) -> dict[str, Any] | None:
    """Return what `scripts/score_runs.py` measured for this checkpoint, if anything.

    A scorecard written against a different file than the one on disk is
    ignored rather than shown: a stale number under a chip is worse than no
    number, because nothing about it looks wrong.
    """
    if checkpoint.run is None:
        return None
    card_path = checkpoint.run / SCORECARD_NAME
    if not card_path.is_file():
        return None
    try:
        with card_path.open(encoding="utf-8") as file:
            card = json.load(file)
        score = dict(card.get("checkpoints", {}).get(checkpoint.kind, {}))
    except (OSError, ValueError, AttributeError):
        return None
    if not score or score.get("size") != checkpoint.model.stat().st_size:
        return None
    return score


def _measured(score: Mapping[str, Any] | None) -> str:
    """Return the short form: what it does, in the terms that separate these runs.

    Success rate leads only when it is bad. Every Flingo run on disk reaches
    the mark essentially every time, so ranking them by it ranks nothing --
    speed and how much of the episode goes on pinching are the difference, and
    `runs/README.md` has the argument for why.
    """
    if score is None:
        return "unscored"
    diverged = float(score.get("diverged_share", 0.0))
    if diverged > 0.0:
        return f"blows the solver up in {diverged * 100:.0f}% of episodes"
    bits = []
    success = float(score.get("success_rate", 0.0))
    if success < SUCCESS_WORTH_SAYING:
        bits.append(f"reaches the mark {success * 100:.0f}%")
    bits.append(f"{float(score.get('mean_speed_m_s', 0.0)):.2f} m/s")
    bits.append(f"{float(score.get('pinch_share', 0.0)) * 100:.0f}% pinch")
    if float(score.get("sheet_pinned_share", 1.0)) < TRIMS_BELOW:
        bits.append("trims")
    return ", ".join(bits)


def _blurb(checkpoint: _Checkpoint, knobs: list[str], score: Mapping[str, Any] | None, boat: str | None) -> str:
    """Return the long form, for the line under the helm row."""
    parts = [str(checkpoint.model)]
    if boat:
        parts.append(f"trained on {boat}")
    if knobs:
        parts.append("changed: " + ", ".join(knobs))
    if score is None:
        parts.append("not scored yet — run `uv run python scripts/score_runs.py` to fill this in")
        return " · ".join(parts)

    episodes = int(score.get("episodes", 0))
    diverged = float(score.get("diverged_share", 0.0))
    measured = [
        f"{float(score.get('success_rate', 0.0)) * 100:.0f}% reached the mark",
        f"{float(score.get('mean_speed_m_s', 0.0)):.2f} m/s",
        f"{float(score.get('pinch_share', 0.0)) * 100:.0f}% of steps inside 25° of the wind",
        f"sheet pinned {float(score.get('sheet_pinned_share', 0.0)) * 100:.0f}% of steps",
    ]
    if diverged > 0.0:
        measured.append(f"{diverged * 100:.0f}% of episodes blew the solver up")
    parts.append(f"over {episodes} seeded episodes: " + ", ".join(measured))
    return " · ".join(parts)


def _best_of_each_boat(scores: Mapping[Path, Mapping[str, Any] | None], boats: Mapping[Path, str | None]) -> set[Path]:
    """Return the fastest checkpoint per boat, among those that can sail one at all."""
    leaders: dict[str | None, tuple[float, Path]] = {}
    for model, score in scores.items():
        if score is None or float(score.get("diverged_share", 0.0)) > 0.0:
            continue
        if float(score.get("success_rate", 0.0)) < SUCCESS_WORTH_SAYING:
            continue
        speed = float(score.get("mean_speed_m_s", 0.0))
        boat = boats.get(model)
        if boat not in leaders or speed > leaders[boat][0]:
            leaders[boat] = (speed, model)
    return {model for _, model in leaders.values()}


def _find_policies(default_model: str | None, default_config: str | None, reason: str | None) -> list[Policy]:
    """Trained checkpoints under runs/, each named for what it tried and what it does.

    `waypoint_ppo_20260919_074021 (best)` is a timestamp and a directory
    listing: it cannot tell you what the run changed or whether the policy is
    any good, which are the only two things you want to know while picking one.
    Both are on disk already -- the run kept the config it trained with, and
    `scripts/score_runs.py` leaves a scorecard next to it -- so the name is
    built from them instead.
    """
    found = _checkpoints(default_model, default_config)
    configs = {cp.model: _flatten(_read_config(cp.config)) for cp in found}
    boats = {cp.model: _trained_on(cp.config) for cp in found}
    scores = {cp.model: _score(cp) for cp in found}

    # Peers are the other *runs* on the same boat, counted once each: a config
    # is a property of the run, so best and final must not vote twice.
    by_run: dict[Path | None, Mapping[str, Any]] = {}
    for checkpoint in found:
        by_run.setdefault(checkpoint.run or checkpoint.model, configs[checkpoint.model])
    peers_for_boat: dict[str | None, list[Mapping[str, Any]]] = {}
    for flat in by_run.values():
        peers_for_boat.setdefault(flat.get("env.simulator_config"), []).append(flat)

    starred = _best_of_each_boat(scores, boats)

    policies: list[Policy] = []
    for checkpoint in found:
        flat = configs[checkpoint.model]
        peers = peers_for_boat.get(flat.get("env.simulator_config"), [flat])
        knobs = _unusual(flat, peers)
        run_name = checkpoint.run.name if checkpoint.run is not None else checkpoint.model.stem

        head: list[str] = []
        if checkpoint.run is not None and not GENERATED_RUN.match(run_name):
            # Somebody named this run on purpose; that beats a config diff.
            head.append(run_name)
        else:
            budget = _budget(flat.get("train.total_timesteps"))
            head.extend(bit for bit in [budget, *knobs] if bit)
        if not head:
            head.append(run_name)
        if checkpoint.kind != "named":
            head.append(checkpoint.kind)

        name = f"{' · '.join(head)} — {_measured(scores[checkpoint.model])}"

        policies.append(
            Policy(
                id=str(checkpoint.model),
                name=name,
                model_path=str(checkpoint.model),
                config_path=str(checkpoint.config) if checkpoint.config is not None else None,
                trained_on=boats[checkpoint.model],
                reason=reason,
                blurb=_blurb(checkpoint, knobs, scores[checkpoint.model], boats[checkpoint.model]),
            )
        )

    # Two runs can be the same experiment twice -- 074021 and 120703 came out
    # bit-identical -- so say which is which rather than offering one chip
    # twice. Done before the star, which would otherwise hide the collision.
    counts: dict[str, int] = {}
    for policy in policies:
        counts[policy.name] = counts.get(policy.name, 0) + 1
    for policy, checkpoint in zip(policies, found, strict=True):
        if counts[policy.name] > 1 and checkpoint.run is not None:
            policy.name = f"{policy.name} ({checkpoint.run.name.rsplit('_', 1)[-1]})"
        if checkpoint.model in starred:
            policy.name = f"★ {policy.name}"
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
        except Exception:
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
    reason = _backend_reason()
    policies = _find_policies(policy_model, policy_config, reason)
    sailable = [p for p in policies if p.reason is None]

    if not policies:
        note = f"no trained runs under {RUNS_PATH} yet — train one with scripts/train_waypoint_sb3.py"
    elif not sailable:
        found = f"{len(policies)} trained model{'s' if len(policies) != 1 else ''}"
        note = f"{found} on disk, none loadable: {reason}"
    else:
        note = None

    # Only hand the CLI's `--policy-model` the helm when it is one we can load.
    # Matching on the path rather than taking the first sailable policy: the
    # checkpoints are listed in run order now, so first is no longer the CLI's.
    preselected = next((p for p in sailable if policy_model and Path(p.model_path) == Path(policy_model)), None)
    default_helm = preselected.id if preselected is not None else "manual"
    if default_boat not in {b["id"] for b in boats}:
        default_boat = boats[0]["id"]
    return Catalog(
        boats=boats,
        parts=parts,
        policies=policies,
        default_boat=default_boat,
        default_helm=default_helm,
        policy_note=note,
    )


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
    if setup.helm != "manual":
        chosen = catalog.policy(setup.helm)
        if chosen is None:
            msg = f"unknown helm {setup.helm!r}; expected 'manual' or a listed policy"
            raise ValueError(msg)
        if chosen.reason is not None:
            msg = f"{chosen.name} cannot take the helm: {chosen.reason}"
            raise ValueError(msg)
    return setup
