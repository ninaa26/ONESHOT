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

from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")

# part -> {name: model}. Written only by `register`, at import time.
_PARTS: dict[str, dict[str, Any]] = {}


def register(part: str, *names: str) -> Callable[[T], T]:
    """Register the decorated model under one or more names for `part`.

    Several names are for aliases a config may already carry: `model_type: sail`
    predates `basic` and still has to resolve to the same class.

    Raises:
        ValueError: no name was given, or a name is taken by another model.

    """
    if not names:
        msg = f"a {part} registration needs at least one name"
        raise ValueError(msg)

    def decorate(model: T) -> T:
        entries = _PARTS.setdefault(part, {})
        for name in names:
            key = name.lower()
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


def registered(part: str) -> dict[str, Any]:
    """Return the models registered for `part`, by name, as a copy."""
    return dict(_PARTS.get(part, {}))


def parts() -> tuple[str, ...]:
    """Return every part something has registered under."""
    return tuple(sorted(_PARTS))
