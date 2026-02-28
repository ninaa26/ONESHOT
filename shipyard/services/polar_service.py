from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from sailbench.models.foil import Foil


def _alpha_range(element_config: dict[str, Any], num_points: int = 150) -> NDArray[np.float64]:
    alpha_min = float(element_config.get("alpha_min", -25.0))
    alpha_max = float(element_config.get("alpha_max", 25.0))
    return np.linspace(np.deg2rad(alpha_min), np.deg2rad(alpha_max), num_points)


def _re_from_config(element_config: dict[str, Any], default: float = 5e5) -> float:
    res_value = element_config.get("res")
    if isinstance(res_value, (int, float)):
        return float(res_value)
    if isinstance(res_value, (list, tuple)) and res_value:
        first = res_value[0]
        if isinstance(first, (int, float)):
            return float(first)
    return default


def _foil_curve(
    element_config: dict[str, Any],
    num_points: int = 150,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute CL and CD versus alpha for a single foil element."""
    foil = Foil(element_config)  # type: ignore[abstract]
    re = _re_from_config(element_config)
    alpha_rad = _alpha_range(element_config, num_points=num_points)

    cl_list: list[float] = []
    cd_list: list[float] = []
    for alpha in alpha_rad:
        cl, cd = foil.cl_cd(alpha, re)
        cl_list.append(cl)
        cd_list.append(cd)

    cl: NDArray[np.float64] = np.asarray(cl_list, dtype=float)
    cd: NDArray[np.float64] = np.asarray(cd_list, dtype=float)
    alpha_deg: NDArray[np.float64] = np.degrees(alpha_rad)
    return alpha_deg, cl, cd


def plot_element_polar(element_config: dict[str, Any], num_points: int = 150) -> None:
    """Plot a single element's aerodynamic polar (CL vs CD)."""
    alpha_deg, cl, cd = _foil_curve(element_config, num_points=num_points)

    plt.figure()
    plt.plot(cd, cl)
    plt.xlabel("CD")
    plt.ylabel("CL")
    title = f"{element_config.get('model_type', 'element')} Polar"
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_total_polar(config: dict[str, dict[str, Any]], num_points: int = 150) -> None:
    """Plot an approximate total-boat polar by area-weighting each foil element.

    This combines the keel, sail, and rudder polars into a single CD–CL curve.
    """
    element_keys = ["keel", "sail", "rudder"]
    curves: list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]] = []

    for key in element_keys:
        element_cfg = config.get(key)
        if not isinstance(element_cfg, dict):
            continue
        area = float(element_cfg.get("area", 1.0))
        alpha_deg, cl, cd = _foil_curve(element_cfg, num_points=num_points)
        curves.append((alpha_deg, cl, cd, area))

    if not curves:
        return

    # Assume all curves share the same alpha grid (by construction).
    _, _, _, first_area = curves[0]
    alpha_deg = curves[0][0]

    total_area = sum(area for *_rest, area in curves)
    if total_area <= 0.0:
        return

    cl_total = np.zeros_like(curves[0][1])
    cd_total = np.zeros_like(curves[0][2])

    for _, cl, cd, area in curves:
        weight = area / total_area
        cl_total += weight * cl
        cd_total += weight * cd

    plt.figure()
    plt.plot(cd_total, cl_total)
    plt.xlabel("CD")
    plt.ylabel("CL")
    plt.title("Total Boat Polar (area-weighted)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()