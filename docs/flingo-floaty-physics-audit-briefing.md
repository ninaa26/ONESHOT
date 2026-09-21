# Flingo Floaty — Physics Audit Briefing

2026-09-16 · @Someone

## Bottom line

Flingo Floaty has two confirmed physics defects, both **missing force terms** rather than sign errors or broken code.

1. **No wave-making resistance.** The hull model has skin friction only. Flingo reaches Froude 0.78 — roughly twice hull speed for its 1 m waterline.
2. **No induced drag on keel or rudder.** 2D section coefficients are used raw, so the keel makes side force almost for free. Close-hauled leeway is 0.65–1.5° where it should be 4–6°.

A prototype adding both terms produces a physically sensible boat: no-go at \~35–40°, leeway 4.3–4.7°, Froude capped at 0.59, heading held under 2° from TWA 45 up.

One number blocks the fix and is **not in the repo**: Flingo's actual keel span and rudder span. Only `area` is configured, and induced drag varies 6× across plausible spans.

Separately, `scripts/polar_diagram.py` produces a polar whose x-axis is wrong by 15° or more, worst near the no-go. That needs fixing before any physics change can be measured.

No repo files were changed. All prototypes are in a scratchpad.

## A correction, and a bug it exposed

An earlier pass of this audit claimed Flingo "sails at 20° TWA, which is impossible." **That claim was wrong**, and the reason it was wrong is itself a finding.

The measurement harness held heading with a P+D controller — the same one in `scripts/polar_diagram.py`. A P+D controller cannot null a constant disturbance, and the sail's yaw moment (`sail.x_pos: 0.1`) is exactly that. The boat settled about 15° off the commanded heading, so every data point was filed under the wrong true wind angle.

Re-run with a PI controller and binned on **achieved** TWA, Flingo's real no-go is \~40°, which is entirely reasonable.

The consequence for the repo: `scripts/polar_diagram.py` is producing a polar whose x-axis is wrong by a variable 15° or more. The error is largest near the no-go angle, where the polar matters most. Two fixes, either of which works:

- Add an integral term to the heading hold, or
- Record the TWA the boat actually achieved and bin on that instead of the commanded value.

This is independent of any physics change and should land first — without it, there is no trustworthy way to measure whether the physics work helped.

## Defect 1 — no wave-making resistance

Flingo's waterline is 1.0 m, giving a hull-speed scale of 0.4·√(gL) = **1.25 m/s**. The simulator runs it at 2.44 m/s.

`BasicHullModel` computes skin friction only, with a flat-plate coefficient of 0.004. There is no residuary or wave-making term, so nothing resists the boat as it passes hull speed. For a 1 m displacement hull at Froude 0.78, wave-making should be the dominant resistance by a wide margin.

Baseline measured with a PI heading hold and achieved-TWA binning, TWS 5 m/s:

| Achieved TWA | Speed (m/s) | Vb/Vt | Froude | Leeway |
| --- | --- | --- | --- | --- |
| 44.1° | 1.368 | 0.27 | 0.44 | −1.54° |
| 59.8° | 1.792 | 0.36 | 0.57 | −0.91° |
| 90.0° | 2.361 | 0.47 | **0.75** | −0.65° |
| 135.0° | 2.442 | 0.49 | **0.78** | −0.33° |
| 180.0° | 2.258 | 0.45 | 0.72 | 0.00° |

The three worst rows are all reaching and running, which is where a displacement hull is most speed-limited in reality and least limited here.

## Defect 2 — no induced drag on keel or rudder

`Foil.cl_cd()` returns NeuralFoil's **2D section** coefficients and they are used directly as if the foil had infinite span. Two things are missing:

- **Induced drag**, `CD_i = CL² / (π·AR·e)` — absent entirely.
- **Finite-span lift reduction** — 2D lift over-predicts the real 3D value by roughly 1.7× at AR 3.

Measured on Flingo's NACA0010 keel at Re 5×10⁵:

| AoA | CL (2D) | CD (2D section) | CD\_i at AR 3 | Ratio |
| --- | --- | --- | --- | --- |
| 4° | 0.482 | 0.0091 | 0.0274 | 3.0× |
| 6° | 0.681 | 0.0121 | 0.0547 | 4.5× |
| 8° | 0.859 | 0.0159 | 0.0870 | **5.5×** |
| 10° | 1.027 | 0.0222 | 0.1243 | 5.6× |

At a realistic keel angle of attack the missing term is five times larger than the drag that *is* modelled.

The visible symptom is leeway. Flingo sails close-hauled at 0.65–1.5° of leeway; a real small sailboat makes 4–6°. The keel is producing its side force at almost no cost, so the boat barely slips sideways.

A related trap: `rudder.effectiveness: 0.2` is a hand-tuned fudge that exists to scale down the same over-predicted 2D lift. It is discussed under the fix, because adding the correct term without retuning it double-counts.

## Flingo's numbers, and the one we don't have

Derived from `configs/flingo_floty.yaml` (m = 27 kg, L = 1.0, B = 0.5, T = 0.125):

| Quantity | Value | Note |
| --- | --- | --- |
| Displacement | 0.0270 m³ | block coefficient 0.43 |
| Wetted surface | 1.062 m² | 1.7·L·(B+T) |
| Hull-speed scale | 1.25 m/s | 0.4·√(gL) |
| SA / disp^(2/3) | 14.9 | typical 15–25 |
| Lateral plane / sail area | 13.1 % | — |
| Radius of gyration from `inertia_z: 1.0` | 0.19·L | typical 0.25·L |
| Rigid-body Iz at 0.25·L | 1.69 kg·m² | plus yaw added inertia 2.05 → **\~3.7** |
| Sway added mass | 24.5 kg | vs 27 kg boat → 1.9× effective sway inertia |

Two things stand out. `inertia_z: 1.0` is roughly **3.7× too small** once added inertia is counted. And sway added mass is comparable to the boat's whole mass, yet is not modelled at all.

### The blocking unknown

The configs specify keel and rudder **`area` only**. Aspect ratio is what governs induced drag, and it cannot be derived from area alone. Holding Flingo's keel area at 0.13 m²:

| Keel span | Chord | AR (geometric) | AR (effective, hull end-plate) | CD\_i at CL 0.6 |
| --- | --- | --- | --- | --- |
| 0.25 m | 0.52 m | 0.48 | 0.96 | 0.1324 |
| 0.30 m | 0.43 m | 0.69 | 1.38 | 0.0920 |
| 0.35 m | 0.37 m | 0.94 | 1.88 | 0.0676 |
| 0.40 m | 0.33 m | 1.23 | 2.46 | 0.0517 |
| 0.50 m | 0.26 m | 1.92 | 3.85 | 0.0331 |

Induced drag varies **6×** across that range. Everything downstream depends on it.

**Action: someone needs to measure Flingo's keel span (depth below the hull) and rudder span.** The figures used throughout this briefing assume 0.35 m and 0.22 m. Once measured, it is a one-line config change.

## The validated fix

Prototyped as a monkeypatch and measured against the same harness. Two terms added:

- **Lifting-line correction on every foil**: lift reduced by the finite-span factor, then `CD += CL²/(π·AR·e)` with e = 0.9.
- **Residuary resistance on the hull**: `F_wr = −sign(u)·c_wave·q·(L·T)·(|u|/v_hull)⁴`, with `v_hull = 0.4·√(gL)`.

Result, with keel AR\_eff 1.88, rudder AR\_eff 1.05, c\_wave 0.01, **rudder effectiveness 0.6**:

| Achieved TWA | Speed (m/s) | Vb/Vt | Froude | Leeway | Heading error |
| --- | --- | --- | --- | --- | --- |
| 37.2° | 0.888 | 0.18 | 0.28 | −3.72° | 9.4° |
| 43.8° | 1.044 | 0.21 | 0.33 | −4.71° | 6.6° |
| 46.1° | 1.195 | 0.24 | 0.38 | −4.29° | 2.0° |
| 59.9° | 1.489 | 0.30 | 0.48 | −2.61° | 0.1° |
| 90.0° | 1.748 | 0.35 | 0.56 | −2.41° | 0.0° |
| 135.0° | 1.841 | 0.37 | 0.59 | −1.38° | 0.0° |

No-go at \~35–40°, leeway 4.3–4.7° close-hauled, Froude capped at 0.59, heading held under 2° from TWA 45 up. That is a physically sensible boat.

Usefully, the two knobs are **independent**: keel span sets upwind ability and leeway, `c_wave` sets top speed. Reaching speed barely moves across the whole plausible keel-span range because it is wave-limited.

### The effectiveness trap

`rudder.effectiveness: 0.2` is a hand-tuned stand-in for exactly the 3D correction being added. Keeping it at 0.2 double-counts and the rudder loses authority — the no-go pushes out to 57°, worse than reality. At 1.0 the boat over-steers and occasionally blows out past 100° of heading error.

**0.6 is the value.** Do not add the lifting-line correction without retuning this.

## Config changes for flingo\_floty.yaml

```yaml
hull:
  c_wave: 0.01                   # NEW: residuary resistance, caps Fn near hull speed

keel:
  span: 0.35                     # NEW - MEASURE THIS
  effective_aspect_ratio: 1.88   # 2*span^2/area (hull acts as end plate)

rudder:
  span: 0.22                     # NEW - MEASURE THIS
  effective_aspect_ratio: 1.05   # span^2/area (surface-piercing, no image)
  effectiveness: 0.6             # was 0.2 - retune, it stood in for the 3D correction

boat:
  inertia_z: 3.7                 # was 1.0 - rigid body 1.69 + added inertia 2.05
```

### Dead config and silent fallbacks

These are not cosmetic — they mislead anyone trying to tune the boat.

- **`xu1`, `yv1`, `nr1`, `xu2`, `yv2`, `nr2` are never read.** `SailboatHub` instantiates `BasicHullModel`, which uses only `L`, `B`, `T`. `QuadraticHydroModel` is imported but unused; `LinearHydroModel` is not imported at all. Anyone "tuning damping" today is changing nothing.
- **`BasicHullModel` reads `rho`, the config sets `rho_water`.** It falls back to 1000, so the answer is right by luck.
- **`BasicKeel` reads `water_density`, which no config defines.** Same silent fallback.
- **`BasicHullModel` yaw damping is \~8× strip theory**: it uses `k_r = (1/8)·ρ·T·L⁴` where strip theory gives `ρ·T·Cd·L⁴/64`.
- **Keel and hull generate no yaw moment** in Flingo — both sit at `x_pos: 0.0`, which deletes the centre-of-effort / centre-of-lateral-resistance lead that creates weather helm.

Either wire these up or delete them. As they stand they are a trap.

## Performance — the cheapest win available

Flingo runs at **64 steps/s, or 1.3× realtime**. NeuralFoil is invoked 12 times per RK4 step (3 foils × 4 stages), and that dominates everything else.

At that rate a 1M-step RL run takes **4.3 hours** on a single environment. The checkpoints in `runs/` go to 1.5M steps.

Tabulating each foil's polar once over angle of attack and interpolating gives:

| Metric | Value |
| --- | --- |
| Speedup | **308×** |
| Table build time | 0.04 s |
| Max lift-coefficient error | 0.003 |
| Max drag-coefficient error | 0.0005 |

The error is far below the uncertainty in every other term in the model. `sailbench/models/constants.py` still carries an unused `FOIL_CACHE` constant, and `Foil`'s docstring says "no cache dependency anymore" — so this existed once and was removed.

This is a pure speed change with no effect on physics, which makes it the safest thing to land first and the thing that makes everything else practical to measure.

## Recommended order

1. **Tabulated foil polars.** Pure speed, no physics change. Makes everything after it measurable.
2. **Fix `scripts/polar_diagram.py`** (PI heading hold, or bin on achieved TWA). Without this there is no trustworthy way to evaluate steps 3 and 4.
3. **Measure Flingo's keel and rudder span**, then add induced drag and finite-span lift — *with* the `rudder.effectiveness` retune to 0.6.
4. **Add wave resistance** (`c_wave: 0.01`).
5. **Delete or wire up the dead hull coefficients**, and fix the `rho` / `rho_water` and `water_density` key mismatches.

Later, and a larger change: added mass and the Munk moment. Sway added mass is 1.9× the boat's mass, and the missing Munk moment removes the hull's natural directional instability, which matters for how RL learns yaw control.

### Risk

Steps 3 and 4 change the polar substantially — top speed drops by roughly a third and leeway triples. **The trained checkpoints in `runs/` will not transfer.** That is a real cost and it is a judgement call how to sequence it against the RL work on `dev/nina-flingo-rl`.

### Relation to the in-flight branches

`fix/foil-force-signs` is good and none of this duplicates it. It already lands two genuine fixes: the rudder drag-sign bug (drag was rotated by R(−track) instead of R(track)) and the stale-`fluid`-frame hazard, where the test fixture pinned `fluid` to identity so the keel was never exercised in its real frame. Its `fx*u + fy*v < 0` power assertion is the right invariant.

The physics work here should branch off `fix/foil-force-signs` rather than off `main`, so it builds on those sign fixes instead of colliding with them.

### Checked and found correct

Worth recording so nobody "fixes" them: the sail's angle-of-attack sign convention, the boom side, the rigid-body Coriolis terms, and the RK4 integrator are all correct. The sail convention is confusing but sound — the reported `sail.angle_deg` is the chord-forward bearing, which is **180° from the boom**, and nothing in the code says so. Worth a comment.

## References

The model that fits sailbench's per-component architecture almost exactly is Buehler et al. It decomposes forces per component (sail, keel, rudder, hull) the same way, and supplies precisely the terms Flingo is missing.

| Source | Date | What it gives us |
| --- | --- | --- |
| [Buehler, Heinz & Kohaut, "Dynamic Simulation Model for an Autonomous Sailboat"](https://ceur-ws.org/Vol-2331/paper3.pdf) — International Robotic Sailing Conference, Southampton | 31 Aug 2018 | `c_D = c_f + c_L²/(πΛ)`; wave resistance `F_wr ∝ (v/v_hull)⁴` with `v_hull = 0.4√(g·L_wl)`; blended stall via `s = 1 − exp[−(α/α_sep)²]`, α\_sep = 25° |
| [ORC VPP Documentation 2023](https://orc.org/uploads/files/ORC-VPP-Documentation-2023.pdf), §5.4.3 and §6.5 | 2023 | Professional-grade form of the same core: efficiency coefficient `CE = KPP + A_ref/(π·heff²)`; hydrodynamic induced drag from simplified lifting-line theory over hull plus combined appendages |
| Delft Systematic Yacht Hull Series (Keuning & Katgert bare-hull resistance regression) | data public since 2010 | The hull-resistance basis underlying the ORC residuary model; \~70 systematically varied hulls |

The through-line: every serious implementation, from a student robotic sailboat to the ORC racing rule, models induced drag as `CL²/(π·AR·e)` and caps speed with a residuary term. Flingo currently has neither.

One caveat on Buehler's stall model: sailbench currently hard-clamps CL and CD at the stall limit rather than blending between attached and separated flow. That is a smaller issue than the two headline defects, but it is why foil behaviour is discontinuous at high angles of attack.
