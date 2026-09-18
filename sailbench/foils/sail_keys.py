"""Which keys belong to which family of sail model.

The two sail families read disjoint parameters. A section polar model wants an
airfoil and a Reynolds number; the ORC envelope wants none of that and takes its
coefficients from apparent wind angle instead. Each refuses the other's keys
rather than ignoring them, so a config that names one model and carries the
other's parameters is told, instead of quietly sailing a boat whose numbers do
nothing.

The two lists live here rather than in either model because they describe the
*boundary* between them, and belong to neither. When `basic_sail` imported its
refusal list from `orc_sail`, importing the plain sail pulled the ORC model in
first, so `orc_main` registered ahead of `basic` and the catalog offered the
models in an order that depended on which file happened to import which. A
shared module neither model owns removes both the coupling and the ordering.
"""

from __future__ import annotations

# Read only by the ORC envelope models.
ORC_ONLY_KEYS: tuple[str, ...] = (
    "eff_span_corr",
    "flat_stall_floor",
    "heel_arm_m",
    "heff",
    "heff_model",
    "jib_area",
    "max_heeling_moment_nm",
)

# Read only by the NeuralFoil section-polar models.
FOIL_ONLY_KEYS: tuple[str, ...] = (
    "airfoil_name",
    "alpha_min",
    "alpha_max",
    "alpha_sep_deg",
    "cn_plate",
    "effective_aspect_ratio",
    "end_plate_factor",
    "luff_deg",
    "luff_ramp_deg",
    "oswald_efficiency",
    "polar_step_deg",
    "re",
    "res",
    "span",
)
