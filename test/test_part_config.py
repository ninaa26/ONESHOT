"""Composing a component section down to the parameters one model reads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from sailbench.foils.basic_keel import BasicKeel, FiniteSpanKeel
from sailbench.foils.basic_rudder import FiniteSpanRudder
from sailbench.sim import sailboat_hub
from sailbench.sim.part_config import compose, model_name, offered_models
from sailbench.sim.sailboat_hub import SailboatHub


class TestModelName:
    """Which model a section selects."""

    def test_defaults_when_unset(self) -> None:
        """A section that names no model gets the caller's default."""
        assert model_name({}) == "basic"
        assert model_name({}, "flat") == "flat"

    def test_reads_either_spelling(self) -> None:
        """`model_type` is the original key and stays accepted."""
        assert model_name({"model_type": "orc_main"}) == "orc_main"
        assert model_name({"model": "orc_main"}) == "orc_main"

    def test_agreeing_spellings_are_fine(self) -> None:
        """Saying the same thing twice is not a disagreement."""
        assert model_name({"model_type": "orc_main", "model": "orc_main"}) == "orc_main"

    def test_two_different_models_is_refused(self) -> None:
        """An override that wrote the old key onto a migrated config must not sail
        a model nobody picked while the screen shows another."""
        with pytest.raises(ValueError, match="names two different models"):
            model_name({"model_type": "basic", "model": "finite_span"})

    def test_lowercased(self) -> None:
        """Selector values are matched case-insensitively."""
        assert model_name({"model": "Finite_Span"}) == "finite_span"


class TestOfferedModels:
    """What a section says it can be sailed with."""

    def test_flat_section_offers_nothing(self) -> None:
        """An unmigrated section has no `models` mapping to report."""
        assert offered_models({"area": 1.0}) == ()

    def test_lists_the_blocks(self) -> None:
        """The keys of `models` are the answer, with no construction needed."""
        section = {"models": {"basic": {}, "finite_span": {"span": 0.7}}}
        assert set(offered_models(section)) == {"basic", "finite_span"}


class TestFlatSectionsAreUnchanged:
    """The shape every existing config is written in still means what it meant."""

    def test_passes_the_section_through(self) -> None:
        """No `models` key, so the flat block is the model's parameters."""
        section = {"model_type": "basic", "area": 0.75, "span": 0.5}
        assert compose("keel", section) == section

    def test_does_not_alias_the_input(self) -> None:
        """The composed parameters are the hub's to mutate, not the caller's dict."""
        section: dict[str, Any] = {"area": 0.75}
        composed = compose("keel", section)
        composed["area"] = 1.0
        assert section["area"] == 0.75


class TestComposition:
    """A section that names its models hands out one model's parameters."""

    @staticmethod
    def section(selected: str) -> dict[str, Any]:
        """A keel carrying both models, with `selected` chosen."""
        return {
            "area": 0.1225,
            "x_pos": 0.184,
            "model": selected,
            "models": {
                "basic": {},
                "finite_span": {"span": 0.700, "end_plate_factor": 2.0},
            },
        }

    def test_shared_keys_reach_every_model(self) -> None:
        """Geometry is true whichever model runs, so it is not repeated per block."""
        for selected in ("basic", "finite_span"):
            composed = compose("keel", self.section(selected))
            assert composed["area"] == 0.1225
            assert composed["x_pos"] == 0.184

    def test_selected_block_is_layered_on(self) -> None:
        """The chosen model's own parameters come through."""
        composed = compose("keel", self.section("finite_span"))
        assert composed["span"] == 0.700
        assert composed["end_plate_factor"] == 2.0

    def test_other_models_are_invisible(self) -> None:
        """The collision the refusal checks guard against cannot be composed."""
        composed = compose("keel", self.section("basic"))
        assert "span" not in composed
        assert "end_plate_factor" not in composed

    def test_models_mapping_is_dropped(self) -> None:
        """A model is handed parameters, not the catalog it was picked from."""
        assert "models" not in compose("keel", self.section("basic"))

    def test_block_overrides_a_shared_key(self) -> None:
        """The more specific value wins where both name the same key."""
        section = {"area": 1.0, "model": "finite_span", "models": {"finite_span": {"area": 2.0}}}
        assert compose("keel", section)["area"] == 2.0

    def test_empty_block_is_a_real_offer(self) -> None:
        """`basic:` with nothing under it means the model needs no parameters."""
        section = {"area": 1.0, "model": "basic", "models": {"basic": None}}
        assert compose("keel", section) == {"area": 1.0, "model": "basic"}


class TestComposeRejections:
    """What a malformed or unofferable selection says."""

    def test_unoffered_model_names_what_there_is(self) -> None:
        """The boat has no block for it, and the error says what it does have."""
        section = {"model": "finite_span", "models": {"basic": {}}}
        with pytest.raises(ValueError, match=r"no `models\.finite_span` block.*offers: basic"):
            compose("keel", section)

    def test_models_must_be_a_mapping(self) -> None:
        """A list of names would lose the parameters each model needs."""
        with pytest.raises(TypeError, match="must map a model name"):
            compose("keel", {"models": ["basic", "finite_span"]})

    def test_a_block_must_be_a_mapping(self) -> None:
        """A block holds parameters, not a bare value."""
        with pytest.raises(TypeError, match="must map parameter names"):
            compose("keel", {"model": "basic", "models": {"basic": 4.0}})


class TestHubComposes:
    """The hub builds a migrated section's model from that model's own block."""

    def test_selected_model_gets_its_block(self) -> None:
        """Flingo ships migrated: its measured span reaches the finite-span keel."""
        keel = SailboatHub("flingo_floty.yaml").keel
        assert isinstance(keel, FiniteSpanKeel)
        assert keel.p["span"] == pytest.approx(0.700)
        assert keel.p["end_plate_factor"] == pytest.approx(2.0)

    def test_shared_geometry_still_reaches_it(self) -> None:
        """Area stayed at the section level and is not repeated in either block."""
        assert SailboatHub("flingo_floty.yaml").keel.p["area"] == pytest.approx(0.1225)

    def test_the_other_model_is_now_reachable(self) -> None:
        """The point of the exercise: this boat can run the plain 2-D keel too.

        On the flat shape it could not. `span` and `alpha_sep_deg` sat beside
        `model_type`, and BasicKeel refuses them rather than ignore them, so the
        boat that measured a span could never be sailed without one.
        """
        hub = SailboatHub("flingo_floty.yaml", overrides={"keel": {"model": "basic"}})
        assert isinstance(hub.keel, BasicKeel)
        assert not isinstance(hub.keel, FiniteSpanKeel)
        assert "span" not in hub.keel.p

    def test_selecting_an_unoffered_model_says_so(self) -> None:
        """A boat with no block for a model refuses it with its own account."""
        with pytest.raises(ValueError, match=r"no `models\.orc_main` block"):
            SailboatHub("flingo_floty.yaml", overrides={"keel": {"model": "orc_main"}})


class TestFlatConfigsStillLoad:
    """The shape everything was written in before the migration still works."""

    @pytest.fixture
    def flat(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
        """Flingo's keel flattened back: finite-span keys beside `model_type`."""
        cfg = yaml.safe_load(Path("configs/flingo_floty.yaml").read_text(encoding="utf-8"))
        keel = cfg["keel"]
        keel.update(keel.pop("models")["finite_span"])
        keel["model_type"] = keel.pop("model")

        (tmp_path / "flat.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        monkeypatch.setattr(sailboat_hub, "CONFIG_PATH", f"{tmp_path}/")
        return "flat.yaml"

    def test_it_builds_the_same_model(self, flat: str) -> None:
        """No `models` key, so the flat block is the model's parameters."""
        keel = SailboatHub(flat).keel
        assert isinstance(keel, FiniteSpanKeel)
        assert keel.p["span"] == pytest.approx(0.700)

    def test_the_legacy_spelling_still_selects(self, flat: str) -> None:
        """`model_type` is what every unmigrated config and checkpoint carries."""
        assert SailboatHub(flat).keel.p["model_type"] == "finite_span"
