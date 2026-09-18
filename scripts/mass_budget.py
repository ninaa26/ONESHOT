"""Mass, CG and inertia tensor from real materials instead of CAD densities.

The CAD carries no materials at all: the whole assembly sits at the SolidWorks
default, so its 29.78 kg is just 29.51 L with a unit change, and its CG and
inertia tensor are volume-weighted rather than mass-weighted.

So this takes geometry from the mesh and density from the Fall 2025 mechanical
report, body by body. Where the CAD's *thickness* is also fiction -- the hull
shell is modelled at 4.44 mm against a resin-infused laminate of about 2 mm --
the body's mass is computed from its skin area and the real layup, then spread
back over the CAD shell as an effective density. That keeps the mass where the
geometry says it is while giving it the magnitude the materials say it has.

Measured values win. `boat.mass: 27.0` is the weighed boat, so equipment that
the report does not specify (batteries, electronics, fasteners) is scaled to
close the gap rather than guessed part by part.

Usage:
    uv run python scripts/mass_budget.py boat.stl
    uv run python scripts/mass_budget.py boat.stl --hull-laminate-mm 1.5 --target-mass 27.0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from hull_analysis import autoscale, load_stl, normals_and_areas, split_components, volume_and_centroid
from numpy.typing import NDArray

Array = NDArray[np.float64]

# Densities [kg/m^3] and areal masses [kg/m^2], from the Fall 2025 report's
# stated materials plus standard values for those materials.
CFRP_LAMINATE = 1550.0  # carbon/epoxy by resin infusion
CARBON_ROD = 1600.0  # pultruded rod, mast and boom
LEAD_EPOXY = 7000.0  # lead shot in epoxy, ~60% lead by volume
ALUMINIUM = 2700.0  # 6061 keel rods
PRINTED_CORE = 220.0  # ASA/PLA at ~20% infill
HD_FOAM = 200.0  # high-density foam, rudder fill
SAILCLOTH_AREAL = 0.15  # kg/m^2

KEEL_ROD_COUNT, KEEL_ROD_DIA, KEEL_ROD_LEN = 2, 0.010, 0.700
SKIN_PLIES_MM = 0.5  # 2 carbon layers on keel and rudder

# The tetrahedron covariance of a unit tetra, used for the inertia tensor.
CANON = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]) / 120.0


@dataclass
class Part:
    """One connected body with a material story attached."""

    tris: Array
    name: str
    volume: float
    area: float  # total mesh area (both skins for a shell)
    mass: float = 0.0
    note: str = ""


def body_moments(tris: Array) -> tuple[float, Array, Array]:
    """Volume, first moment and second-moment matrix about the origin."""
    vol = 0.0
    first = np.zeros(3)
    cov = np.zeros((3, 3))
    for t in tris:
        mat = np.column_stack(t)
        det = float(np.linalg.det(mat))
        vol += det / 6.0
        first += det * t.sum(axis=0) / 24.0
        cov += det * (mat @ CANON @ mat.T)
    return vol, first, cov


def assemble(parts: list[Part]) -> tuple[float, Array, Array]:
    """Total mass, CG and inertia tensor at the CG, for parts with set masses."""
    mass = 0.0
    first = np.zeros(3)
    cov = np.zeros((3, 3))
    for p in parts:
        vol, f, c = body_moments(p.tris)
        if abs(vol) < 1e-12 or p.mass <= 0.0:
            continue
        rho = p.mass / abs(vol)  # effective density for this body
        scale = rho * np.sign(vol)
        mass += p.mass
        first += scale * f
        cov += scale * c
    cg = first / mass
    cov_cg = cov - mass * np.outer(cg, cg)
    return mass, cg, np.trace(cov_cg) * np.eye(3) - cov_cg


def classify(tris: Array, up: int, fwd: int, lat: int) -> list[Part]:
    """Split the mesh and name each body by its geometry."""
    bodies = split_components(tris)
    bodies.sort(key=lambda b: abs(volume_and_centroid(b, np.zeros(3))[0]), reverse=True)

    parts: list[Part] = []
    for i, b in enumerate(bodies):
        vol = abs(volume_and_centroid(b, np.zeros(3))[0])
        _, areas = normals_and_areas(b)
        lo = b.reshape(-1, 3).min(axis=0)
        hi = b.reshape(-1, 3).max(axis=0)
        height = hi[up] - lo[up]

        if i == 0:
            name = "hull+deck"
        elif len(b) < 40 and height > 1.0:
            name = "sail"
        elif hi[up] < -0.88 and (hi[fwd] - lo[fwd]) > 0.2:
            name = "bulb"
        elif height > 0.6 and (hi[lat] - lo[lat]) < 0.05 and lo[up] < -0.5:
            name = "keel fin"
        elif 0.4 < height < 0.7 and lo[up] < -0.5:
            name = "rudder"
        elif height > 0.3 and (hi[lat] - lo[lat]) < 0.05:
            name = "spar"
        else:
            name = "equipment"
        parts.append(Part(tris=b, name=name, volume=vol, area=float(areas.sum())))
    return parts


def assign_masses(parts: list[Part], hull_mm: float, target: float) -> tuple[float, float]:
    """Give every part a mass. Returns (structural mass, equipment mass)."""
    keel_rod_vol = KEEL_ROD_COUNT * np.pi * (KEEL_ROD_DIA / 2) ** 2 * KEEL_ROD_LEN
    keel_rod_mass = keel_rod_vol * ALUMINIUM

    for p in parts:
        one_skin = p.area / 2.0
        if p.name == "hull+deck":
            # Solid infused laminate over the real skin area, not the CAD shell.
            p.mass = one_skin * (hull_mm / 1000.0) * CFRP_LAMINATE
            p.note = f"{one_skin:.2f} m^2 skin x {hull_mm:.1f} mm CFRP"
        elif p.name == "keel fin":
            skin = one_skin * (SKIN_PLIES_MM / 1000.0) * CFRP_LAMINATE
            core = max(p.volume - keel_rod_vol, 0.0) * PRINTED_CORE
            p.mass = skin + core + keel_rod_mass
            p.note = f"carbon skin {skin:.2f} + printed core {core:.2f} + rods {keel_rod_mass:.2f} kg"
        elif p.name == "bulb":
            p.mass = p.volume * LEAD_EPOXY
            p.note = f"{p.volume * 1000:.2f} L lead epoxy at {LEAD_EPOXY:.0f} kg/m^3"
        elif p.name == "rudder":
            skin = one_skin * (SKIN_PLIES_MM / 1000.0) * CFRP_LAMINATE
            fill = p.volume * HD_FOAM
            p.mass = skin + fill
            p.note = f"carbon skin {skin:.2f} + foam {fill:.2f} kg"
        elif p.name == "spar":
            p.mass = p.volume * CARBON_ROD
            p.note = "carbon rod"
        elif p.name == "sail":
            p.mass = one_skin * SAILCLOTH_AREAL
            p.note = f"{one_skin:.2f} m^2 cloth at {SAILCLOTH_AREAL} kg/m^2"

    structural = sum(p.mass for p in parts if p.name != "equipment")
    equipment_vol = sum(p.volume for p in parts if p.name == "equipment")
    equipment = max(target - structural, 0.0)
    for p in parts:
        if p.name == "equipment":
            p.mass = equipment * (p.volume / equipment_vol) if equipment_vol > 0 else 0.0
            p.note = "scaled to close the gap to the weighed mass"
    return structural, equipment


def main() -> None:
    """Print the budget and the resulting mass properties."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stl", type=Path)
    ap.add_argument("--hull-laminate-mm", type=float, default=2.0)
    ap.add_argument("--target-mass", type=float, default=27.0, help="weighed boat mass [kg]")
    ap.add_argument("--up", type=int, default=1)
    ap.add_argument("--fwd", type=int, default=0)
    args = ap.parse_args()

    up, fwd = args.up, args.fwd
    lat = ({0, 1, 2} - {up, fwd}).pop()

    tris, note = autoscale(load_stl(args.stl), "auto")
    parts = classify(tris, up, fwd, lat)
    structural, equipment = assign_masses(parts, args.hull_laminate_mm, args.target_mass)

    print(f"{args.stl.name}: {len(tris)} triangles, units {note}")
    print(f"hull laminate {args.hull_laminate_mm:.1f} mm, target mass {args.target_mass:.1f} kg\n")

    print(f"{'part':<14} {'n':>3} {'vol L':>7} {'mass kg':>8}  how")
    for name in ("hull+deck", "keel fin", "bulb", "rudder", "spar", "sail", "equipment"):
        group = [p for p in parts if p.name == name]
        if not group:
            continue
        m = sum(p.mass for p in group)
        v = sum(p.volume for p in group)
        print(f"{name:<14} {len(group):3d} {v * 1000:7.2f} {m:8.2f}  {group[0].note}")

    print(f"\nstructural subtotal (report materials) {structural:6.2f} kg")
    print(f"equipment to reach the weighed mass    {equipment:6.2f} kg")
    print(f"total                                  {structural + equipment:6.2f} kg")

    mass, cg, inertia = assemble(parts)
    print(f"\nmass {mass:.3f} kg")
    print(f"CG (CAD frame) {cg * 1000} mm")
    print(
        f"CG in sim frame: x {-(cg[fwd] - cg[fwd]):+.4f} m by definition; "
        f"height above keel tip {cg[up] - tris.reshape(-1, 3)[:, up].min():.4f} m"
    )
    print("inertia at the CG [kg*m^2]:")
    print(f"  roll  Lxx {inertia[0, 0]:8.3f}")
    print(f"  yaw   Lyy {inertia[1, 1]:8.3f}   <- boat.inertia_z")
    print(f"  pitch Lzz {inertia[2, 2]:8.3f}")
    print(f"  products  Lxy {inertia[0, 1]:7.3f}  Lxz {inertia[0, 2]:7.3f}  Lyz {inertia[1, 2]:7.3f}")


if __name__ == "__main__":
    main()
