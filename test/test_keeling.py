from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from sailbench.models.model import State
from sailbench.sim import sailboat_hub
from sailbench.sim.protocol import make_state_message


def _load_cfg() -> dict:
    with Path("configs/basic_sailbot.yaml").open(encoding="utf-8") as file:
        return dict(yaml.safe_load(file))


def _make_state() -> State:
    return State(x=0.0, y=0.0, psi=(1.0, 0.0), u=2.2, v=0.35, r=0.08)


def _prepare_hub(hub: sailboat_hub.SailboatHub, state: State) -> None:
    hub._update_dynamic_frames(  # noqa: SLF001 - test intentionally exercises private physics hook
        state=state,
        sheet_limit_rad=float(np.radians(50.0)),
        rudder_angle_deg=8.0,
        dt=0.02,
    )


def test_keeling_disabled_matches_legacy_forces(tmp_path: Path, monkeypatch) -> None:
    cfg_without_keeling = _load_cfg()
    cfg_without_keeling.pop("keeling", None)

    cfg_disabled = _load_cfg()
    cfg_disabled["keeling"] = {"enabled": False}

    base_path = tmp_path / "base.yaml"
    disabled_path = tmp_path / "disabled.yaml"
    base_path.write_text(yaml.safe_dump(cfg_without_keeling), encoding="utf-8")
    disabled_path.write_text(yaml.safe_dump(cfg_disabled), encoding="utf-8")
    monkeypatch.setattr(sailboat_hub, "CONFIG_PATH", f"{tmp_path}/")

    hub_base = sailboat_hub.SailboatHub(config_file="base.yaml")
    hub_disabled = sailboat_hub.SailboatHub(config_file="disabled.yaml")
    state = _make_state()
    _prepare_hub(hub_base, state)
    _prepare_hub(hub_disabled, state)

    forces_base = hub_base._forces(state)  # noqa: SLF001 - test parity at force-assembly layer
    forces_disabled = hub_disabled._forces(state)  # noqa: SLF001 - test parity at force-assembly layer
    assert np.allclose(forces_base, forces_disabled, atol=1e-9)


def test_keeling_enabled_modifies_forces_and_produces_heel(tmp_path: Path, monkeypatch) -> None:
    cfg_enabled = _load_cfg()
    cfg_enabled["keeling"] = {
        "enabled": True,
        "gm_m": 0.3,
        "heeling_lever_m": 1.0,
        "max_heel_deg": 35.0,
        "smoothing": 1.0,
        "hydro_force_scale_per_deg": 0.02,
        "hydro_min_scale": 0.3,
        "aero_force_scale_per_deg": 0.02,
        "aero_min_scale": 0.3,
    }
    enabled_path = tmp_path / "enabled.yaml"
    enabled_path.write_text(yaml.safe_dump(cfg_enabled), encoding="utf-8")
    monkeypatch.setattr(sailboat_hub, "CONFIG_PATH", f"{tmp_path}/")

    hub = sailboat_hub.SailboatHub(config_file="enabled.yaml")
    state = _make_state()
    _prepare_hub(hub, state)
    fx_total, fy_total, _ = hub._forces(state)  # noqa: SLF001 - validating keeling behavior

    assert abs(hub.last_heel_deg) > 0.01
    assert abs(hub.last_sail_force[0]) > 0.0
    assert np.isfinite(fx_total)
    assert np.isfinite(fy_total)


def test_state_message_carries_heel_angle() -> None:
    state = _make_state()
    msg = make_state_message(
        state=state,
        t=1.2,
        heel_angle_deg=6.4,
    )
    assert msg["boat"]["heel_deg"] == 6.4
