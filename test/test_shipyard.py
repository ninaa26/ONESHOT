"""The shipyard catalog and the setup handshake behind the pre-sail screen."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from sailbench.foils.basic_sail import BasicSail
from sailbench.foils.orc_sail import ORCMainSail
from sailbench.sim import shipyard
from sailbench.sim.protocol import SetupInputs, parse_setup_message
from sailbench.sim.sailboat_hub import SailboatHub
from sailbench.sim.shipyard import PART_KEY, Catalog, Policy, build_catalog, overrides_for, validate_setup


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
        """A `model` override picks a different sail class."""
        hub = SailboatHub("basic_sailbot.yaml", overrides={"sail": {"model": "orc_main"}})
        assert isinstance(hub.sail, ORCMainSail)

    def test_an_override_in_the_old_spelling_is_refused(self) -> None:
        """Rather than silently sailing the model the config named.

        Every shipped boat now says `model`. An override writing `model_type`
        would lose to it, so the screen would show one model and the boat run
        another; the disagreement is an error instead.
        """
        with pytest.raises(ValueError, match="names two different models"):
            SailboatHub("basic_sailbot.yaml", overrides={"sail": {"model_type": "orc_main"}})

    def test_override_reaches_the_component(self) -> None:
        """Any key can be overridden, not just the model selector."""
        hub = SailboatHub("basic_sailbot.yaml", overrides={"hull": {"friction_model": "hughes"}})
        assert hub.hull_cfg["friction_model"] == "hughes"

    def test_override_of_a_model_the_boat_does_not_offer_raises(self) -> None:
        """The hub does not paper over a boat having no block for a model."""
        with pytest.raises(ValueError, match=r"no `models\.orc_w_jib` block"):
            SailboatHub("basic_sailbot.yaml", overrides={"sail": {"model": "orc_w_jib"}})


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

    def test_a_migrated_boat_offers_every_model_it_carries(self, catalog: Catalog) -> None:
        """What the migration was for: both models of a part, on one boat.

        Flingo measured a span, which used to mean it could never be sailed
        without one, and carries a jib, which used to rule out the mainsail-only
        rig. Every option it has a block for is now available.
        """
        flingo = boat(catalog, "flingo_floty.yaml")["available"]
        for part, offered in flingo.items():
            assert all(reason is None for reason in offered.values()), f"{part}: {offered}"

    def test_a_boat_is_greyed_out_only_where_it_offers_nothing(self, catalog: Catalog) -> None:
        """The single-sail boats decline the sloop rig, and say why in their terms."""
        basic = boat(catalog, "basic_sailbot.yaml")["available"]
        assert basic["keel"]["finite_span"] is None  # from its assumed aspect ratio
        assert basic["rudder"]["finite_span"] is None  # the clamps moved under `basic`
        assert basic["sail"]["orc_main"] is None
        assert "no `models.orc_w_jib` block" in basic["sail"]["orc_w_jib"]

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
            assert set(policy) == {"id", "name", "trained_on", "blurb", "disabled", "reason"}


class TestPolicies:
    """Trained checkpoints reach the helm row whether or not they can be loaded."""

    def test_finds_the_checkpoints_on_disk(self, catalog: Catalog) -> None:
        """Every run's best and final model is offered as a helm."""
        names = {p.name for p in catalog.policies}
        assert any(" best — " in name for name in names), names
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


class TestPolicyNames:
    """A helm chip says what the run changed and what the policy does.

    `waypoint_ppo_20260919_074021 (best)` is a timestamp: picking between six
    of those means opening six directories. Both halves of the name are read
    off disk -- the config the run kept, and the scorecard
    `scripts/score_runs.py` writes beside it.
    """

    BOAT = "flingo_floty.yaml"

    def _run(
        self,
        root: Path,
        name: str,
        *,
        env: dict | None = None,
        steps: int = 750_000,
        best: bytes = b"a checkpoint",
        score: dict | None = None,
    ) -> Path:
        """Write a run directory the catalog can read: config, model, scorecard."""
        run = root / name
        (run / "best_model").mkdir(parents=True)
        (run / "best_model" / "best_model.zip").write_bytes(best)
        config = {
            "env": {"simulator_config": self.BOAT, **(env or {})},
            "train": {"total_timesteps": steps},
        }
        (run / "config_used.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        if score is not None:
            card = {
                "task": {"episodes": 20},
                "checkpoints": {"best": {"size": len(best), "episodes": 20, **score}},
            }
            (run / shipyard.SCORECARD_NAME).write_text(json.dumps(card), encoding="utf-8")
        return run

    SAILS = {
        "success_rate": 1.0,
        "mean_speed_m_s": 1.32,
        "pinch_share": 0.148,
        "sheet_pinned_share": 1.0,
        "diverged_share": 0.0,
    }

    def _names(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        monkeypatch.setattr(shipyard, "RUNS_PATH", str(tmp_path))
        monkeypatch.setattr(shipyard, "_backend_reason", lambda: None)
        return [p.name for p in build_catalog(self.BOAT).policies]

    def test_the_name_is_the_budget_the_change_and_the_measurement(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two runs differing in one knob are named by that knob, not their timestamps."""
        self._run(tmp_path, "waypoint_ppo_20260919_074021", env={"no_go_zone_penalty": 3.0}, score=self.SAILS)
        self._run(
            tmp_path,
            "waypoint_ppo_20260919_073312",
            env={"no_go_zone_penalty": 15.0},
            best=b"another checkpoint",
            score={**self.SAILS, "mean_speed_m_s": 1.19, "pinch_share": 0.276},
        )
        names = self._names(tmp_path, monkeypatch)

        assert any("no-go 15" in name and "1.19 m/s" in name and "28% pinch" in name for name in names), names
        assert any("no-go 3" in name and "1.32 m/s" in name for name in names), names
        assert all("750 k" in name for name in names), names
        assert all("20260919" not in name for name in names), names

    def test_a_knob_every_run_shares_is_not_worth_saying(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A name carries what makes a run unusual, so agreement is silent."""
        shared = {"no_go_zone_penalty": 3.0, "jibe_penalty": 0.5}
        self._run(tmp_path, "waypoint_ppo_20260919_074021", env=shared, score=self.SAILS)
        self._run(tmp_path, "waypoint_ppo_20260919_120703", env={**shared, "gamma": 0.999}, best=b"b", score=self.SAILS)
        names = self._names(tmp_path, monkeypatch)

        assert not any("jibe" in name for name in names), names
        assert sum("gamma 0.999" in name for name in names) == 1, names

    def test_a_name_somebody_chose_is_kept(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """`good_vmg_stable` already says more than a config diff could."""
        self._run(tmp_path, "good_vmg_stable", score=self.SAILS)
        names = self._names(tmp_path, monkeypatch)
        assert names[0].startswith("★ good_vmg_stable"), names
        assert "1.32 m/s" in names[0]

    def test_the_fastest_is_starred(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The question behind the row is which one to sail, so one chip answers it."""
        self._run(tmp_path, "waypoint_ppo_20260919_074021", env={"gamma": 0.99}, score=self.SAILS)
        self._run(
            tmp_path,
            "waypoint_ppo_20260919_120703",
            env={"gamma": 0.98},
            best=b"b",
            score={**self.SAILS, "mean_speed_m_s": 1.10},
        )
        names = self._names(tmp_path, monkeypatch)
        starred = [name for name in names if name.startswith("★")]
        assert len(starred) == 1, names
        assert "1.32 m/s" in starred[0]

    def test_a_policy_that_blows_the_solver_up_says_so(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A mean speed averaged over episodes the integrator lost is not a speed."""
        self._run(tmp_path, "waypoint_ppo_20260919_074021", score={**self.SAILS, "diverged_share": 0.2})
        names = self._names(tmp_path, monkeypatch)
        assert "blows the solver up in 20% of episodes" in names[0], names
        assert not names[0].startswith("★"), names

    def test_reaching_the_mark_is_mentioned_only_when_it_is_a_problem(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Every Flingo run scores ~100%, so the number ranks nothing until it drops."""
        self._run(tmp_path, "waypoint_ppo_20260919_074021", score=self.SAILS)
        self._run(tmp_path, "waypoint_ppo_20260919_120703", best=b"b", score={**self.SAILS, "success_rate": 0.45})
        names = sorted(self._names(tmp_path, monkeypatch))
        assert not any("reaches the mark 100%" in name for name in names), names
        assert any("reaches the mark 45%" in name for name in names), names

    def test_trimming_is_named_because_one_run_does_it(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A policy that works the sheet instead of pinning it is the rare one."""
        self._run(tmp_path, "waypoint_ppo_20260919_074021", score={**self.SAILS, "sheet_pinned_share": 0.65})
        assert "trims" in self._names(tmp_path, monkeypatch)[0]

    def test_an_unscored_checkpoint_says_so_rather_than_guessing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Silence about quality beats a number nobody measured."""
        run = self._run(tmp_path, "waypoint_ppo_20260919_074021")
        names = self._names(tmp_path, monkeypatch)
        assert names[0].endswith("unscored"), names
        assert "score_runs.py" in build_catalog(self.BOAT).policies[0].blurb
        assert not (run / shipyard.SCORECARD_NAME).exists()

    def test_a_scorecard_for_a_different_file_is_ignored(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Retraining into the same path must not leave the old numbers on the chip."""
        run = self._run(tmp_path, "waypoint_ppo_20260919_074021", score=self.SAILS)
        (run / "best_model" / "best_model.zip").write_bytes(b"a retrained checkpoint")
        names = self._names(tmp_path, monkeypatch)
        assert names[0].endswith("unscored"), names

    def test_the_same_experiment_twice_is_still_two_chips(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """074021 and 120703 are the same config, so the label has to disambiguate."""
        self._run(tmp_path, "waypoint_ppo_20260919_074021", score=self.SAILS)
        self._run(tmp_path, "waypoint_ppo_20260919_120703", best=b"b", score=self.SAILS)
        names = self._names(tmp_path, monkeypatch)
        assert len(set(names)) == 2, names
        assert any(name.endswith("(074021)") for name in names), names

    def test_the_blurb_carries_the_path_and_the_full_measurement(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The chip has room for a few words; the line under the row has the rest."""
        self._run(tmp_path, "waypoint_ppo_20260919_074021", env={"no_go_zone_penalty": 3.0}, score=self.SAILS)
        monkeypatch.setattr(shipyard, "RUNS_PATH", str(tmp_path))
        monkeypatch.setattr(shipyard, "_backend_reason", lambda: None)
        policy = build_catalog(self.BOAT).policies[0]

        assert policy.blurb is not None
        assert "best_model.zip" in policy.blurb
        assert f"trained on {self.BOAT}" in policy.blurb
        assert "20 seeded episodes" in policy.blurb
        assert "sheet pinned 100% of steps" in policy.blurb
        assert build_catalog(self.BOAT).to_payload()["policies"][0]["blurb"] == policy.blurb

    def test_the_preselected_checkpoint_takes_the_helm(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`--policy-model` picks a checkpoint by path, not whichever is listed first."""
        self._run(tmp_path, "waypoint_ppo_20260919_061347", score=self.SAILS)
        wanted = self._run(tmp_path, "waypoint_ppo_20260919_121045", best=b"b", score=self.SAILS)
        monkeypatch.setattr(shipyard, "RUNS_PATH", str(tmp_path))
        monkeypatch.setattr(shipyard, "_backend_reason", lambda: None)
        model = wanted / "best_model" / "best_model.zip"

        catalog = build_catalog(self.BOAT, policy_model=str(model))
        assert catalog.default_helm == str(model)


class TestOverridesFor:
    """A parts pick maps onto the config key each part reads."""

    def test_maps_each_part_to_its_selector_key(self) -> None:
        """Foils are overridden through `model`; the hull uses friction_model.

        `model_type` is still read, but an override has to write the key that
        wins, or it would be silently ignored on a boat that had migrated to
        `model` and left a stale `model_type` beside it.
        """
        assert overrides_for({"sail": "orc_main", "hull": "hughes"}) == {
            "sail": {"model": "orc_main"},
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
            boat="fun_boat.yaml",
            parts={},
            helm="manual",
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


class TestDeclaredAvailabilityMatchesReality:
    """The catalog stopped building every option to find out whether it works.

    What a model declares it needs and what it actually refuses at construction
    are written in two places, so they can drift. This walks every boat against
    every option and holds the declaration to what building it really does.
    """

    def test_every_boat_and_option_agrees_with_construction(self, catalog: Catalog) -> None:
        """Declared reason and real outcome agree, option by option."""
        checked = 0
        for entry in catalog.boats:
            for row, offered in entry["available"].items():
                for option_id, declared in offered.items():
                    try:
                        SailboatHub(entry["id"], overrides={row: {PART_KEY[row]: option_id}})
                    except Exception as exc:
                        built: str | None = str(exc)
                    else:
                        built = None
                    assert (declared is None) == (built is None), (
                        f"{entry['id']} {row}.{option_id}: catalog says {declared!r}, building says {built!r}"
                    )
                    checked += 1
        expected = len(catalog.boats) * sum(len(catalog.parts[row]) for row in catalog.parts)
        assert checked == expected, f"expected {expected} boat/option pairs, walked {checked}"


class TestOptionsComeFromTheModels:
    """The catalog's names and blurbs are the models' own."""

    def test_requirement_hints_are_generated(self, catalog: Catalog) -> None:
        """`(needs ...)` used to be typed into a table beside the catalog."""
        by_id = {o["id"]: o for o in catalog.parts["keel"]}
        assert by_id["finite_span"]["blurb"].endswith("(needs span or effective_aspect_ratio)")
        assert "needs" not in by_id["basic"]["blurb"]

    def test_names_are_the_models(self, catalog: Catalog) -> None:
        """A model is named where it is defined, not in a second table."""
        assert {o["name"] for o in catalog.parts["sail"]} == {"Basic", "Hybrid", "ORC main", "ORC main + jib"}
