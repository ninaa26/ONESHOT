"""The shipyard catalog and the setup handshake behind the pre-sail screen."""

from __future__ import annotations

from pathlib import Path

import pytest

from sailbench.foils.basic_sail import BasicSail
from sailbench.foils.orc_sail import ORCMainSail
from sailbench.sim import shipyard
from sailbench.sim.protocol import SetupInputs, parse_setup_message
from sailbench.sim.sailboat_hub import SailboatHub
from sailbench.sim.shipyard import Catalog, Policy, build_catalog, overrides_for, validate_setup


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    """The catalog the runner would send, built once for the module."""
    return build_catalog("basic_sailbot.yaml")


def boat(catalog: Catalog, boat_id: str) -> dict:
    """One boat entry by id."""
    return next(b for b in catalog.boats if b["id"] == boat_id)


class TestHubOverrides:
    """`overrides` lets a caller swap a model without editing the YAML."""

    def test_no_overrides_is_the_file(self) -> None:
        """The default path is unchanged."""
        assert isinstance(SailboatHub("basic_sailbot.yaml").sail, BasicSail)

    def test_override_swaps_the_model(self) -> None:
        """A model_type override picks a different sail class."""
        hub = SailboatHub("basic_sailbot.yaml", overrides={"sail": {"model_type": "orc_main"}})
        assert isinstance(hub.sail, ORCMainSail)

    def test_override_reaches_the_component(self) -> None:
        """Any key can be overridden, not just the model selector."""
        hub = SailboatHub("basic_sailbot.yaml", overrides={"hull": {"friction_model": "hughes"}})
        assert hub.hull_cfg["friction_model"] == "hughes"

    def test_override_that_the_model_rejects_raises(self) -> None:
        """The hub does not paper over a model refusing its config."""
        with pytest.raises(ValueError, match="jib_area"):
            SailboatHub("basic_sailbot.yaml", overrides={"sail": {"model_type": "orc_w_jib"}})


class TestCatalog:
    """What the browser is offered is derived from the configs."""

    def test_lists_every_boat_config(self, catalog: Catalog) -> None:
        """Every boat YAML under configs/ is a card; the RL configs are not."""
        ids = {b["id"] for b in catalog.boats}
        assert {"basic_sailbot.yaml", "flingo_floty.yaml", "fun_boat.yaml", "real_boat.yaml"} <= ids
        assert "rl_waypoint_sb3.yaml" not in ids
        assert "flingo_rl.yaml" not in ids

    def test_default_boat_is_the_cli_config(self, catalog: Catalog) -> None:
        """--config becomes the preselected card."""
        assert catalog.default_boat == "basic_sailbot.yaml"
        assert catalog.default_helm == "manual"

    def test_unknown_default_falls_back_to_first_boat(self) -> None:
        """A --config the catalog does not know still yields a usable default."""
        assert build_catalog("not_a_boat.yaml").default_boat == "basic_sailbot.yaml"

    def test_legacy_model_type_is_canonicalised(self, catalog: Catalog) -> None:
        """`model_type: sail` in the old configs shows up as the `basic` option."""
        assert boat(catalog, "basic_sailbot.yaml")["defaults"]["sail"] == "basic"

    def test_flingo_defaults_match_its_config(self, catalog: Catalog) -> None:
        """The measured boat arrives rigged the way its YAML says."""
        assert boat(catalog, "flingo_floty.yaml")["defaults"] == {
            "sail": "orc_w_jib",
            "keel": "finite_span",
            "rudder": "finite_span",
            "hull": "hughes",
        }

    def test_availability_comes_from_actually_building(self, catalog: Catalog) -> None:
        """An option is greyed out with the model's own reason, not a guess."""
        basic = boat(catalog, "basic_sailbot.yaml")["available"]
        assert basic["sail"]["basic"] is None
        assert basic["sail"]["orc_main"] is None
        assert "jib_area" in basic["sail"]["orc_w_jib"]
        assert "span" in basic["keel"]["finite_span"]

        flingo = boat(catalog, "flingo_floty.yaml")["available"]
        assert flingo["sail"]["orc_w_jib"] is None
        assert "jib_area" in flingo["sail"]["orc_main"]

    def test_every_default_is_available(self, catalog: Catalog) -> None:
        """A boat's own defaults must never be greyed out."""
        for entry in catalog.boats:
            for part, option in entry["defaults"].items():
                assert entry["available"][part][option] is None, (entry["id"], part, option)

    def test_stats_carry_the_numbers_the_cards_show(self, catalog: Catalog) -> None:
        """Mass and sail area come straight from the YAML."""
        stats = {s["label"]: s["value"] for s in boat(catalog, "flingo_floty.yaml")["stats"]}
        assert stats["Mass"] == pytest.approx(27.0)
        assert stats["Sail area"] == pytest.approx(1.971)

    def test_payload_is_json_shaped(self, catalog: Catalog) -> None:
        """The browser gets what it renders a helm chip from, not filesystem details."""
        payload = catalog.to_payload()
        assert payload["type"] == "catalog"
        assert set(payload["parts"]) == {"sail", "keel", "rudder", "hull"}
        assert "policy_note" in payload
        for policy in payload["policies"]:
            assert set(policy) == {"id", "name", "trained_on", "disabled", "reason"}


class TestPolicies:
    """Trained checkpoints reach the helm row whether or not they can be loaded."""

    def test_finds_the_checkpoints_on_disk(self, catalog: Catalog) -> None:
        """Every run's best and final model is offered as a helm."""
        names = {p.name for p in catalog.policies}
        assert any(name.endswith("(best)") for name in names), names
        assert all(p.model_path.endswith(".zip") for p in catalog.policies)

    def test_reports_the_boat_a_run_learned(self, catalog: Catalog) -> None:
        """A run that kept its config says which boat it trained against."""
        trained = {p.trained_on for p in catalog.policies if p.config_path}
        assert trained, "no run kept a config to read"
        assert all(name is None or name.endswith(".yaml") for name in trained)

    def test_missing_extras_greys_out_rather_than_hides(
        self,
        catalog: Catalog,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without stable-baselines3 the checkpoints are still listed, with the reason.

        Finding a checkpoint is a directory listing; only sailing one needs the
        RL extras. Dropping them from the catalog told the shipyard there were
        no trained models at all.
        """
        monkeypatch.setattr(shipyard, "_backend_reason", lambda: "needs stable-baselines3")
        without = build_catalog("basic_sailbot.yaml")

        assert without.policies, "checkpoints vanished when the extras did"
        assert [p.id for p in without.policies] == [p.id for p in catalog.policies]
        assert all(p.reason == "needs stable-baselines3" for p in without.policies)
        assert without.default_helm == "manual"
        assert without.policy_note is not None
        assert "stable-baselines3" in without.policy_note
        assert all(p["disabled"] for p in without.to_payload()["policies"])

    def test_backend_reason_names_the_missing_package(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The reason a checkpoint cannot sail names what to install."""
        monkeypatch.setattr(shipyard.importlib.util, "find_spec", lambda name: None)
        reason = shipyard._backend_reason()
        assert reason is not None
        assert "stable-baselines3" in reason
        assert "uv sync --group rl" in reason

    def test_no_runs_directory_says_so(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """An empty runs/ is explained, not left as a bare Manual chip."""
        monkeypatch.setattr(shipyard, "RUNS_PATH", str(tmp_path / "runs"))
        empty = build_catalog("basic_sailbot.yaml")
        assert empty.policies == []
        assert empty.policy_note is not None
        assert "train" in empty.policy_note


class TestOverridesFor:
    """A parts pick maps onto the config key each part reads."""

    def test_maps_each_part_to_its_selector_key(self) -> None:
        """Foils use model_type; the hull uses friction_model."""
        assert overrides_for({"sail": "orc_main", "hull": "hughes"}) == {
            "sail": {"model_type": "orc_main"},
            "hull": {"friction_model": "hughes"},
        }

    def test_ignores_parts_it_does_not_know(self) -> None:
        """Unknown parts are dropped rather than written into the config."""
        assert overrides_for({"mast": "carbon"}) == {}


class TestValidateSetup:
    """The runner checks a setup against the catalog before building."""

    def test_accepts_a_known_build(self, catalog: Catalog) -> None:
        """A valid setup passes through unchanged."""
        setup = SetupInputs(boat="flingo_floty.yaml", parts={"hull": "flat"})
        assert validate_setup(setup, catalog) is setup

    def test_rejects_unknown_boat(self, catalog: Catalog) -> None:
        """Boat ids must be catalog entries."""
        with pytest.raises(ValueError, match="unknown boat"):
            validate_setup(SetupInputs(boat="nope.yaml", parts={}), catalog)

    def test_rejects_unknown_part_and_option(self, catalog: Catalog) -> None:
        """Both the part name and the model name are checked."""
        with pytest.raises(ValueError, match="unknown part"):
            validate_setup(SetupInputs(boat="fun_boat.yaml", parts={"mast": "basic"}), catalog)
        with pytest.raises(ValueError, match="unknown sail model"):
            validate_setup(SetupInputs(boat="fun_boat.yaml", parts={"sail": "magic"}), catalog)

    def test_rejects_unknown_helm(self, catalog: Catalog) -> None:
        """Only manual or a listed policy can take the helm."""
        with pytest.raises(ValueError, match="unknown helm"):
            validate_setup(SetupInputs(boat="fun_boat.yaml", parts={}, helm="runs/x.zip"), catalog)

    def test_rejects_a_listed_but_unsailable_helm(self, catalog: Catalog) -> None:
        """A greyed-out policy is refused with the reason it cannot sail."""
        stuck = Policy(
            id="runs/stuck.zip",
            name="stuck (best)",
            model_path="runs/stuck.zip",
            config_path=None,
            reason="needs stable-baselines3",
        )
        grounded = Catalog(
            boats=catalog.boats,
            parts=catalog.parts,
            policies=[stuck],
            default_boat=catalog.default_boat,
        )
        with pytest.raises(ValueError, match="cannot take the helm: needs stable-baselines3"):
            validate_setup(SetupInputs(boat="fun_boat.yaml", parts={}, helm=stuck.id), grounded)


class TestParseSetupMessage:
    """Shape checks on the JSON the browser sends."""

    def test_parses_the_full_shape(self) -> None:
        """All three fields come through."""
        setup = parse_setup_message(
            {"type": "setup", "boat": "fun_boat.yaml", "parts": {"sail": "basic"}, "helm": "manual"},
        )
        assert setup == SetupInputs(boat="fun_boat.yaml", parts={"sail": "basic"}, helm="manual")

    def test_parts_and_helm_are_optional(self) -> None:
        """A bare boat is a valid setup."""
        assert parse_setup_message({"type": "setup", "boat": "fun_boat.yaml"}) == SetupInputs(
            boat="fun_boat.yaml", parts={}, helm="manual",
        )

    def test_rejects_wrong_type_and_missing_boat(self) -> None:
        """A control message is not a setup, and a setup needs a boat."""
        with pytest.raises(ValueError, match="Unsupported setup"):
            parse_setup_message({"type": "control", "boat": "fun_boat.yaml"})
        with pytest.raises(TypeError, match="boat"):
            parse_setup_message({"type": "setup"})

    def test_rejects_malformed_parts(self) -> None:
        """Parts must be a string-to-string mapping."""
        with pytest.raises(TypeError, match="parts"):
            parse_setup_message({"type": "setup", "boat": "fun_boat.yaml", "parts": ["sail"]})
        with pytest.raises(TypeError, match="parts"):
            parse_setup_message({"type": "setup", "boat": "fun_boat.yaml", "parts": {"sail": 3}})
