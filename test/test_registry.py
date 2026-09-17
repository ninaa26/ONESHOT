"""Models registering themselves, and the parts they register under."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import sailbench.dynamics  # noqa: F401  importing is what registers the models
import sailbench.foils  # noqa: F401
from sailbench.dynamics.basic_hull_model import BasicHullModel
from sailbench.dynamics.linear_hydro import LinearHydroModel
from sailbench.dynamics.quadratic_drag_hydro import QuadraticHydroModel
from sailbench.foils.basic_keel import BasicKeel
from sailbench.models.model import State
from sailbench.sim import sailboat_hub
from sailbench.sim.registry import lookup, parts, register, registered
from sailbench.sim.sailboat_hub import SailboatHub


class TestRegister:
    """The decorator, and what it refuses."""

    def test_registers_under_each_name(self) -> None:
        """Aliases exist so a config's historical spelling still resolves."""

        @register("test_part", "one", "two")
        class Model:
            pass

        assert lookup("test_part", "one") is Model
        assert lookup("test_part", "two") is Model

    def test_names_are_case_insensitive(self) -> None:
        """A config is not held to the capitalisation of the source."""

        @register("test_case", "Mixed_Case")
        class Model:
            pass

        assert lookup("test_case", "mixed_case") is Model

    def test_rejects_a_name_taken_by_another_model(self) -> None:
        """Two models under one name would be resolved by import order."""

        @register("test_clash", "taken")
        class First:
            pass

        with pytest.raises(ValueError, match="already registered"):

            @register("test_clash", "taken")
            class Second:
                pass

    def test_reregistering_the_same_model_is_fine(self) -> None:
        """A module imported twice must not be an error."""

        class Model:
            pass

        register("test_twice", "name")(Model)
        register("test_twice", "name")(Model)
        assert lookup("test_twice", "name") is Model

    def test_needs_a_name(self) -> None:
        """A model nothing can name is a model nothing can select."""
        with pytest.raises(ValueError, match="at least one name"):
            register("test_empty")


class TestLookup:
    """Asking the table for a model."""

    def test_unknown_name_lists_what_there_is(self) -> None:
        """The error is the answer to what the user should have written."""
        with pytest.raises(ValueError, match=r"unknown keel model 'nope'; expected one of: basic, finite_span"):
            lookup("keel", "nope")

    def test_unknown_part_says_so(self) -> None:
        """A part nothing registered under is not a silent empty result."""
        with pytest.raises(ValueError, match=r"\(none registered\)"):
            lookup("mast", "carbon")


class TestTheShippedModels:
    """What the boat can actually be built from."""

    def test_every_part_is_populated(self) -> None:
        """Importing the two model packages is enough to fill the table."""
        for part in ("sail", "keel", "rudder", "hull", "friction"):
            assert registered(part), f"{part} registered nothing"
        assert set(parts()) >= {"sail", "keel", "rudder", "hull", "friction"}

    def test_the_hull_slot_holds_three_alternatives(self) -> None:
        """The two hydro models were reachable only from tests before this."""
        hulls = registered("hull")
        assert hulls["basic"] is BasicHullModel
        assert hulls["linear"] is LinearHydroModel
        assert hulls["quadratic"] is QuadraticHydroModel

    def test_the_legacy_sail_spelling_still_resolves(self) -> None:
        """`model_type: sail` is what every pre-existing config carries."""
        assert lookup("sail", "sail") is lookup("sail", "basic")

    def test_a_keel_is_registered_where_it_is_defined(self) -> None:
        """The class says which name selects it; no table elsewhere repeats it."""
        assert lookup("keel", "basic") is BasicKeel


class TestFrictionLaws:
    """The hull's skin-friction choice goes through the same table."""

    def test_both_laws_are_registered(self) -> None:
        """`flat` and `hughes` are the two the hull has always offered."""
        assert set(registered("friction")) == {"flat", "hughes"}

    def test_flat_is_the_default(self) -> None:
        """A hull that names no law gets the constant it always got."""
        hull = BasicHullModel({"L": 1.5, "B": 0.6, "T": 0.05})
        assert hull._friction_coefficient(1.0, 1.5) == pytest.approx(0.004)  # noqa: SLF001

    def test_hughes_varies_with_speed(self) -> None:
        """The point of the law: the coefficient falls as Reynolds number rises."""
        hull = BasicHullModel({"L": 1.5, "B": 0.6, "T": 0.05, "friction_model": "hughes", "nu_water": 1.0e-6})
        assert hull._friction_coefficient(3.0, 1.5) < hull._friction_coefficient(1.0, 1.5)  # noqa: SLF001

    def test_a_typo_no_longer_means_flat(self) -> None:
        """It used to: anything that was not `hughes` silently ran the constant."""
        hull = BasicHullModel({"L": 1.5, "B": 0.6, "T": 0.05, "friction_model": "hugues"})
        with pytest.raises(ValueError, match="unknown friction model 'hugues'"):
            hull._friction_coefficient(1.0, 1.5)  # noqa: SLF001


class TestHullIsSelectable:
    """A config can now name which hull model it means."""

    @staticmethod
    def boat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hull: dict[str, Any]) -> str:
        """Flingo with its hull section updated and pointed at a temp configs dir."""
        cfg = yaml.safe_load(Path("configs/flingo_floty.yaml").read_text(encoding="utf-8"))
        cfg["hull"].update(hull)
        (tmp_path / "hull.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        monkeypatch.setattr(sailboat_hub, "CONFIG_PATH", f"{tmp_path}/")
        return "hull.yaml"

    def test_default_is_unchanged(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A hull section naming no model still builds the one it always built."""
        name = self.boat(tmp_path, monkeypatch, {})
        assert isinstance(SailboatHub(name).hull, BasicHullModel)

    @pytest.mark.parametrize(
        ("model", "expected"),
        [("basic", BasicHullModel), ("linear", LinearHydroModel), ("quadratic", QuadraticHydroModel)],
    )
    def test_each_model_can_be_named(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str, expected: type
    ) -> None:
        """All three alternatives for the slot are reachable from a config."""
        name = self.boat(tmp_path, monkeypatch, {"model": model})
        assert isinstance(SailboatHub(name).hull, expected)

    def test_a_swapped_hull_actually_computes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Wired in, not merely constructed: its coefficients reach the force sum."""
        name = self.boat(tmp_path, monkeypatch, {"model": "linear", "xu1": 40.0, "yv1": 80.0, "nr1": 5.0})
        hull = SailboatHub(name).hull
        force = hull.compute(State(x=0.0, y=0.0, psi=0.0, u=2.0, v=0.0, r=0.0), sailboat_hub.TFTree2D())
        assert force[0] == pytest.approx(-80.0)

    def test_a_hull_without_added_mass_is_accepted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only the geometry-based hull has stations to integrate; the others do not."""
        name = self.boat(tmp_path, monkeypatch, {"model": "quadratic", "xu2": 20.0})
        hub = SailboatHub(name)
        assert not hasattr(hub.hull, "added_mass")
        assert np.isfinite(hub._forces(State(x=0.0, y=0.0, psi=0.0, u=1.0, v=0.0, r=0.0))).all()  # noqa: SLF001

    def test_an_unknown_hull_model_is_refused(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """With the names it could have meant."""
        name = self.boat(tmp_path, monkeypatch, {"model": "sinks"})
        with pytest.raises(ValueError, match="unknown hull model 'sinks'"):
            SailboatHub(name)
