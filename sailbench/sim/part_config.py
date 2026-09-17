"""Resolve a component section into the parameters one model actually reads.

A boat's YAML has given each component a single flat block of parameters, which
makes the presence of a key do two jobs at once: it supplies a number, and it
says which model the config means. That is why the models have had to refuse each
other's keys -- `model_type: basic` beside a `span` is ambiguous, so it raises --
and why a boat could offer the plain 2-D keel or the finite-span one, never both.

A section can now name its models instead::

    keel:
      area: 0.1225        # shared: whichever model runs, it reads these
      x_pos: 0.184
      model: finite_span  # which one to build
      models:
        basic: {}         # each model's parameters, in a namespace it owns
        finite_span:
          span: 0.700
          end_plate_factor: 2.0

`compose` layers the selected model's block over the shared keys, so a model is
handed its own parameters and nobody else's. The collision the refusals exist to
catch cannot be written down in this shape at all.

The arrangement is OpenFOAM's run-time selection tables: a `type` keyword names
the model and a `<model>Coeffs` subdictionary holds that model's coefficients.
The subdictionaries are gathered under one `models` mapping here so that its keys
answer, by themselves, which models a given boat offers.

A section with no `models` key comes back untouched -- the flat block is the
selected model's parameters, exactly as before. Every existing config keeps
loading unchanged, including the `config_used.yaml` that each trained policy
carries, and a boat can migrate one section at a time.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sailbench.sim.registry import defaults

# The key that names a component's model. `model_type` came first and stays
# accepted for good; `model` is preferred because it reads better beside `models`
# and says what the section is choosing. `model` wins when both are present, so a
# caller layering an override onto a config -- the shipyard does exactly this --
# need not know which spelling that config happens to use.
SELECTOR_KEYS: tuple[str, ...] = ("model", "model_type")

# Keys of a section that are not parameters of the selected model.
RESERVED_KEYS: frozenset[str] = frozenset({"models"})


def model_name(section: Mapping[str, Any], default: str = "basic") -> str:
    """Return the model a component section selects, lowercased."""
    for key in SELECTOR_KEYS:
        value = section.get(key)
        if value is not None:
            return str(value).lower()
    return default


def offered_models(section: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the models this section carries a block for, or () when it is flat.

    An empty tuple means the section has not been migrated, not that it offers
    nothing: a flat section runs whichever model it names and no other.
    """
    models = section.get("models")
    if not isinstance(models, Mapping):
        return ()
    return tuple(models)


def compose(part: str, section: Mapping[str, Any], default: str = "basic") -> dict[str, Any]:
    """Return the parameters the selected model should be built with.

    The shared keys of `section` with the selected model's own block layered over
    them, and `models` itself dropped, so no model can see another's parameters.
    A section with no `models` key is copied through as it stands.

    Under all of it go the model's own `DEFAULTS`, so a physics constant is
    written once in the model that owns it rather than in every boat that uses
    it. Anything the config says wins over them.

    Raises:
        TypeError: `models`, or the selected model's block, is not a mapping.
        ValueError: the section names a model it carries no block for.

    """
    name = model_name(section, default)
    models = section.get("models")
    if models is None:
        return {**defaults(part, name), **section}

    if not isinstance(models, Mapping):
        msg = f"{part} `models` must map a model name to its parameters, got {type(models).__name__}"
        raise TypeError(msg)

    if name not in models:
        offered = ", ".join(sorted(models)) or "(none)"
        msg = f"{part} selects model {name!r}, but this boat carries no `models.{name}` block. It offers: {offered}"
        raise ValueError(msg)

    # `basic:` with nothing indented under it is an empty block, not a mistake:
    # a model that needs no parameters of its own still has to be offered.
    own = models[name] or {}
    if not isinstance(own, Mapping):
        msg = f"{part} `models.{name}` must map parameter names to values, got {type(own).__name__}"
        raise TypeError(msg)

    composed = {key: value for key, value in section.items() if key not in RESERVED_KEYS}
    composed.update(own)
    return {**defaults(part, name), **composed}
