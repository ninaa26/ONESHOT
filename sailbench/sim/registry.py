"""Which classes a part can be built from, and how a model says it is one of them.

Adding a model has meant editing a dict in the hub and a table in the shipyard,
neither of which is where the model lives. Two places to update, both remote from
the file being written, and nothing checks that they agree.

A model states it for itself instead::

    @register("keel", "finite_span")
    class FiniteSpanKeel(BasicKeel):
        ...

which is OpenFOAM's run-time selection table: the base class holds a table, every
derived model adds itself to it on load, and a solver asks the table for the
model a case named. Nothing central has to know the model exists.

A *part* is a slot in the boat, and everything registered under one part answers
the same contract:

    sail, keel, rudder, hull    a `Model`: built from one mapping of parameters,
                                and computes a force from a state
    friction                    a callable (speed, waterline_length, params)
                                returning a skin-friction coefficient

That is Modelica's constraining type -- what may be swapped in is whatever
satisfies the interface the slot promises, and the slot does not care which.

Registration happens on import, so `sailbench.foils` and `sailbench.dynamics`
import every model module they contain. A model in a file nobody imports is a
model the registry has never heard of.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

T = TypeVar("T")

# part -> {name: model}. Written only by `register`, at import time.
_PARTS: dict[str, dict[str, Any]] = {}

# part -> the options it offers, in the order they registered.
_OPTIONS: dict[str, list[Option]] = {}


@dataclass(frozen=True, slots=True)
class Option:
    """One of the models a part offers, as a screen needs to name it.

    `blurb` is what the model is; `requires` is what a boat must supply to use
    it, phrased for a reader ("span or effective_aspect_ratio"). Both come from
    the model itself, so a new model arrives in the catalog already described.
    """

    id: str
    name: str
    blurb: str
    requires: tuple[str, ...] = ()

    def described(self) -> str:
        """Return the blurb with what the model needs appended, if anything."""
        if not self.requires:
            return self.blurb
        return f"{self.blurb} (needs {'; '.join(self.requires)})"


def _titled(option_id: str) -> str:
    """Return a display name derived from an option id."""
    return option_id.replace("_", " ").capitalize()


def _phrase(group: tuple[str, ...]) -> str:
    """Return one requirement group as a reader would say it."""
    return " or ".join(group)


def register(part: str, *names: str, name: str | None = None, blurb: str = "") -> Callable[[T], T]:
    """Register the decorated model under one or more names for `part`.

    Several names are for aliases a config may already carry: `model_type: sail`
    predates `basic` and still has to resolve to the same class. The first name
    is the id the catalog offers; the rest resolve to the same model silently.

    `name` and `blurb` are what a screen calls this model. They live here rather
    than in a table beside the shipyard so that a model arrives already
    described, and cannot be renamed in one place and not the other.

    Raises:
        ValueError: no name was given, or a name is taken by another model.

    """
    if not names:
        msg = f"a {part} registration needs at least one name"
        raise ValueError(msg)

    def decorate(model: T) -> T:
        entries = _PARTS.setdefault(part, {})
        for alias in names:
            key = alias.lower()
            taken = entries.get(key)
            # Re-registering the same object is a module imported twice, which is
            # fine. Two different models under one name is a collision that would
            # otherwise be decided by import order.
            if taken is not None and taken is not model:
                msg = (
                    f"{part} model {key!r} is already registered to "
                    f"{getattr(taken, '__name__', taken)!r}; two models cannot share a name"
                )
                raise ValueError(msg)
            entries[key] = model
        option_id = names[0].lower()
        offered = _OPTIONS.setdefault(part, [])
        if not any(o.id == option_id for o in offered):
            offered.append(
                Option(
                    id=option_id,
                    name=name or _titled(option_id),
                    blurb=blurb,
                    requires=tuple(_phrase(group) for group in getattr(model, "REQUIRES", ())),
                ),
            )
        return model

    return decorate


def lookup(part: str, name: str) -> Any:  # noqa: ANN401 - a part's contract is its own, not this table's
    """Return the model registered for `part` under `name`.

    Raises:
        ValueError: nothing is registered under that name.

    """
    entries = _PARTS.get(part, {})
    key = name.lower()
    if key not in entries:
        known = ", ".join(sorted(entries)) or "(none registered)"
        msg = f"unknown {part} model {key!r}; expected one of: {known}"
        raise ValueError(msg)
    return entries[key]


def canonical(part: str, name: str) -> str:
    """Return the option id `name` resolves to, following aliases.

    `model_type: sail` and `model: basic` are the same model, so a `models`
    mapping keyed on the id the catalog offers must still be found by a config
    that spells it the old way. Unknown names come back unchanged, for the
    caller to fail on with its own message.
    """
    entries = _PARTS.get(part, {})
    model = entries.get(name.lower())
    if model is None:
        return name.lower()
    for option in _OPTIONS.get(part, ()):
        if entries.get(option.id) is model:
            return option.id
    return name.lower()


def registered(part: str) -> dict[str, Any]:
    """Return the models registered for `part`, by name, as a copy."""
    return dict(_PARTS.get(part, {}))


def parts() -> tuple[str, ...]:
    """Return every part something has registered under."""
    return tuple(sorted(_PARTS))


def options(part: str) -> tuple[Option, ...]:
    """Return what `part` offers, in registration order."""
    return tuple(_OPTIONS.get(part, ()))


def defaults(part: str, name: str) -> dict[str, Any]:
    """Return the parameters this model supplies when a config does not.

    Physics constants belong to the model, boat geometry belongs to the config.
    A separation angle is the same 25 degrees for every foil section until
    someone measures otherwise; a span is not.
    """
    return dict(getattr(lookup(part, name), "DEFAULTS", {}))


def unavailable(part: str, name: str, params: Mapping[str, Any]) -> str | None:
    """Return why this model cannot be built from `params`, or None if it can.

    Answered from what the model declares, so the question can be asked about
    every model of every boat without building any of them. It is key presence
    only: whether the numbers are sane is the model's own business at launch.
    """
    model = lookup(part, name)
    for group in getattr(model, "REQUIRES", ()):
        if not any(key in params for key in group):
            return f"{part} model {name!r} needs {_phrase(group)}, which this boat does not give"
    stray = [key for key in getattr(model, "REFUSES", ()) if key in params]
    if stray:
        return f"{part} model {name!r} does not use {', '.join(stray)}, which this boat sets"
    return None
