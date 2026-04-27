from __future__ import annotations

import numpy as np

from sailbench.dynamics.basic_hull_model import BasicHullModel
from sailbench.models.model import State
from sailbench.tf.tf_tree import TFTree2D


def _state(u: float, v: float, r: float, roll_deg: float) -> State:
    return State(x=0.0, y=0.0, psi=(1.0, 0.0), u=u, v=v, r=r, roll_deg=roll_deg)


def test_basic_hull_roll_changes_effective_area_scaling() -> None:
    model = BasicHullModel({"L": 1.5, "B": 0.6, "T": 0.05, "rho": 1000.0})
    tf = TFTree2D()

    flat = _state(u=2.0, v=1.0, r=0.4, roll_deg=0.0)
    heeled = _state(u=2.0, v=1.0, r=0.4, roll_deg=30.0)

    fx_flat, fy_flat, mz_flat = model.compute(flat, tf)
    fx_heeled, fy_heeled, mz_heeled = model.compute(heeled, tf)

    # Heel reduces planform contribution -> less surge drag magnitude.
    assert abs(fx_heeled) < abs(fx_flat)
    # Heel increases side exposure -> more sway drag magnitude.
    assert abs(fy_heeled) > abs(fy_flat)
    # Yaw damping remains finite/stable with heel.
    assert np.isfinite(mz_heeled)
    assert np.isfinite(mz_flat)


def test_basic_hull_uses_port_starboard_side_area() -> None:
    model = BasicHullModel({"L": 1.5, "B": 0.6, "T": 0.05, "rho": 1000.0})
    tf = TFTree2D()

    # Same roll, opposite sway direction should sample opposite side draft.
    v_starboard = _state(u=0.0, v=1.0, r=0.0, roll_deg=30.0)
    v_port = _state(u=0.0, v=-1.0, r=0.0, roll_deg=30.0)

    _, fy_starboard, _ = model.compute(v_starboard, tf)
    _, fy_port, _ = model.compute(v_port, tf)

    # Magnitudes differ because starboard/port immersed drafts differ at nonzero roll.
    assert not np.isclose(abs(fy_starboard), abs(fy_port))
