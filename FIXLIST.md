# Sailbench — fix list

Working list, ordered by what makes sense to do first. Each item carries enough explanation to
act on it cold.

> **Update, 2026-09-17.** Items 1, 2 and 3 have landed on `flingo/integration`;
> the table below is the original 2026-09-12 snapshot and is kept as the record of
> what the state was. The suite is green at 381 tests, `ruff check` and
> `ruff format --check` both pass at zero, and `.github/workflows/ci.yml` runs
> lint and tests on every PR. **CI still only reports** until branch protection
> is switched on -- see item 3, step 2. Items 4-7 are open.

Status as of 2026-09-12, against commit `27b6e38`:

| Metric | Now | Target |
|---|---|---|
| Unit tests passing | 41 / 48 | 48 / 48 |
| Line coverage (`sailbench/`) | 64% | 85%+ |
| Ruff violations | 342 | 0 (against an agreed rule set) |
| PR gating | none | tests + lint must pass |
| ROS 2 integration | none | deterministic algorithm runs in sim |

## The order

These are not independent. Some strictly block others; some are semester-long lanes that should
start now and run alongside.

| # | Item | Blocked by | Shape of the work |
|---|---|---|---|
| 1 | Fix the 7 failing tests | — | Days. One person. **Blocks 2 and 3.** Contains a physics decision, not just a fix. |
| 2 | Ruff config → zero violations | 1 | Days. Same person. Mostly config plus autofix. |
| 3 | CI + branch protection | 1, 2 | Hours once the suite is green. Cannot gate on red checks. |
| 4 | Coverage 64% → 85% | 3 | Continuous. **Everyone**, on the code they own. |
| 5 | ROS 2 integration | — | Semester lane. **Start now, in parallel.** |
| 6 | Sail model | — | Semester lane, parallel. One overlap with 1. |
| 7 | Simulator UI | — | Parallel. Tier 1 is wiring, not building — a day's work. |

**Two tracks, not one queue.** Items **1–3 are a short sequential chain** best given to a single
person — splitting them across people creates handoffs costlier than the work, and it is
realistically a week or two that makes everyone else's work safer immediately. Items **5 and 6
are semester lanes** with different owners and should begin straight away rather than waiting for
the chain to finish.

The one genuine overlap: deciding which sail model is canonical (item 6) also settles the failing
sail test (item 1c), so agree that before fixing it.

**One cheap thing to pull forward into the chain:** log a **success rate** in the eval callback.
Nobody can currently answer "does the boat reach the mark?" — episode length is a proxy, not an
answer. Until that exists, every judgement about items 5 and 6 is guesswork.

---

## 1. Fix the 7 currently failing unit tests

`uv run pytest -q` on a clean clone gives **7 failed, 41 passed**. These fail before anyone
touches anything, which is corrosive: a permanently red suite trains everyone to ignore red, and
then it cannot be used as a gate at all. **Nothing else in this list is worth doing first.**

### What a unit test is

A small function that calls one piece of code with known inputs and asserts something about the
result. Run the suite in a couple of seconds and you learn whether a change broke anything,
without launching the simulator and squinting at a boat.

The seven failures are three separate problems — and **mostly bugs in the tests, not the physics.**

### 1a. Keel: the test fixture never rotates the fluid frame (3 failures)

`test_keel.py::test_forward_flow` and both `test_quadrant_flow` cases.

```
assert fx < -10.0    # drag must oppose forward motion
E    assert np.float64(59.41) < -10.0
```

Drag comes back **positive** — the keel appears to push the boat along.

The cause is `test/conftest.py:29`. The shared `tf_tree` fixture registers the `fluid` frame as
identity, aligned with the boat:

```python
tf.add_frame(name="fluid", parent="boat",
             transform=Transform2D(x=0.0, y=0.0, c=1.0, s=0.0))
```

But in production, `SailboatHub._update_dynamic_frames()` points that frame **opposite** the
velocity:

```python
c_fluid, s_fluid = -state.u / speed, -state.v / speed
```

`BasicKeel` returns `[drag, lift]` in the fluid frame and relies on that 180° rotation to come out
negative in the boat frame. The fixture skips the rotation, so the sign never flips.

**Fix:** make the fixture derive the fluid frame from the state, as the hub does — ideally by
calling the hub's own frame-update code so the two cannot drift apart again. **A test fixture
that hand-rolls a copy of production logic is exactly what broke here.**

### 1b. Rudder: a genuine sign bug in production code (3 failures)

Same symptom as the keel, **different cause** — and this one is not the test's fault.

`BasicRudder` does not use the `fluid` frame at all. It builds its own rotation from angle of
attack:

```python
f_fluid = np.array([drag, lift], dtype=float)
c, s = np.cos(aoa), np.sin(aoa)
f_rudder = np.array([[c, -s], [s, c]]) @ f_fluid
return tf_tree.vector_to_frame(f_rudder, "rudder", "boat")
```

With straight-ahead flow `aoa = 0`, so the rotation is the identity and it returns `+drag` along
the boat's **+x** axis. A rudder that speeds you up. `basic_sail.py` uses the identical pattern and
may share the error uncaught.

**Fix — this is a decision, not a guess.** Either negate the drag component, or route the rudder
through the `fluid` frame the way `BasicKeel` does. Pick one convention, **write it into a
docstring**, and apply it to all three foils. The underlying problem is that *"fluid frame +x"*
means the flow direction in one file and the direction the flow comes **from** in another.

### 1c. Sail: an assertion comparing zero to zero (1 failure)

`test_sail.py:91::test_forward_motion_reduces_apparent_wind`

```
assert abs(fy1) < abs(fy0)
E    assert np.float64(0.0) < np.float64(0.0)
```

Pure tailwind, boat heading the same way, sail on the centreline. Angle of attack is exactly 0,
`HybridSail` zeroes lift below its luff threshold, and lateral force is **identically zero in both
cases**. A strict `<` between two structural zeros can never pass. The `fx` half is fine and tests
something real.

**Fix:** drop the `fy` assertion, or build an off-axis case where lateral force is genuinely
non-zero and expected to shrink. Do **not** relax it to `<=` — that passes while testing nothing.

**Note:** this test exercises `HybridSail`, which the simulator never runs. See item 6.

### 1d. While you are in there

`test_rudder.py::test_angle90_print_test` computes the same value twice, has its prints commented
out, and makes **no assertions at all**. It passes unconditionally and counts toward the green
total. Give it a real assertion or delete it.

---

## 2. Ruff config and lint cleanup

### What a linter is

A tool that reads code without running it and flags style problems and likely-bug patterns —
unused imports, missing annotations, shadowed variables. `ruff` is the one this project already
declares.

### Why this cannot be skipped straight to CI

**Switching ruff on today would fail every PR.** Two problems:

1. **The config sits in a deprecated location.** `pyproject.toml` has `select = ["ALL"]` under
   `[tool.ruff]`, but modern ruff wants `[tool.ruff.lint]`. Ruff warns on every run:
   ```
   warning: The top-level linter settings are deprecated in favour of their
   counterparts in the `lint` section:  'select' -> 'lint.select'
   ```

2. **`"ALL"` means literally every rule**, including mutually contradictory ones. **342
   violations**, and most are noise:

   | Count | Rule | What it is |
   |---|---|---|
   | 78 | `S101` | Use of `assert` — flagged by the *security* ruleset, but asserts are how pytest works |
   | 37 | `PLR2004` | "Magic value in comparison" — every `assert fx < -10.0` in the tests |
   | 27 | `E402` | Import not at top of file — mostly the deliberate `sys.path` shuffle in `scripts/` |
   | 23 | `D103` | Missing docstring on a public function |
   | 23 | `T201` | `print()` left in code. Some of these are real |
   | 18 | `COM812` | Missing trailing comma. Auto-fixable |

### Step 1 — make the config honest

```toml
[tool.ruff]
line-length = 120
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "SIM", "RUF", "ANN", "D"]
ignore = ["D203", "D213", "COM812"]

[tool.ruff.lint.per-file-ignores]
"test/*"    = ["S101", "PLR2004", "D100", "D103", "ANN"]
"scripts/*" = ["E402", "T201"]
```

### Step 2 — get to zero

```bash
uv run ruff check . --fix
uv run ruff format .
uv run ruff check .          # must print "All checks passed!"
```

---

## 3. CI pipeline — gate PRs on tests and lint

### What CI is

Every time someone opens a pull request, a fresh machine checks out the branch, installs the
dependencies, and runs the tests and linter. If anything fails, GitHub marks the PR red and —
once configured — refuses to let it merge.

The point is not automation for its own sake. It is that **"it works on my machine" stops being
an argument.** Right now nothing stops someone merging a branch that breaks the physics, because
nobody is obliged to run `pytest` first. CI makes that obligation automatic and impersonal: the
robot says no, not a teammate.

### Step 1 — add the workflow

`.github/workflows/ci.yml`:

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true

      - name: Install dependencies
        run: uv sync --group dev --group rl

      - name: Lint
        run: uv run ruff check .

      - name: Format check
        run: uv run ruff format --check .

      - name: Unit tests
        run: uv run pytest -q
```

**Reading it:** `on:` says when to run. `runs-on:` picks the machine. Each `- name:` is one step,
run in order; **if any step exits non-zero the job fails and the PR goes red.**
`enable-cache: true` reuses downloaded packages so it takes seconds, not minutes.

### Step 2 — make it actually block merging

The workflow alone only *reports*. In GitHub: **Settings → Branches → Add branch protection
rule** for `main`, tick *Require status checks to pass before merging*, and select the `check`
job. Without this, a red mark is a suggestion anyone can ignore.

### Notes

- Add `mypy` last. `pyproject.toml` sets `strict = true`, which will surface a lot — worth doing,
  but do not let it block the tests-and-lint gate landing.
- `--group rl` pulls in torch, a large download. If CI feels slow, split into a fast lint job
  with `--group dev` only and a separate test job.

---

## 4. Test coverage

### What coverage is

The percentage of source lines that executed at least once while the suite ran:

```bash
uv run --with pytest-cov pytest --cov=sailbench --cov-report=term-missing
```

`term-missing` is the useful half — it prints the line numbers that never ran, so you get a
to-do list rather than a grade.

**What coverage is not:** proof of correctness. A line can execute and still be wrong — every one
of the seven failures in item 1 sits in code that is 100% "covered." Coverage tells you where you
are definitely *not* looking. It says nothing about how carefully you looked at the rest. Treat it
as a floor, never a target to game.

### Where it stands: 64%

| Module | Coverage | Note |
|---|---|---|
| `sim/web_runner.py` | **0%** | 205 statements, entirely untested |
| `rl/live_vis.py` | **0%** | 96 statements, dev-only visualisation |
| `utils/coordinate_helper.py` | **38%** | pure functions — the easiest wins in the repo |
| `dynamics/quadratic_drag_hydro.py` | **38%** | wired up but never selected |
| `sim/protocol.py` | 82% | message-parsing paths untested |
| `rl/envs/waypoint_env.py` | 88% | most reward-penalty branches never fire |
| `tf/tf_tree.py`, `solvers/rk4.py`, `foils/basic_keel.py` | 100% | already solid |

### Priority order

1. **`utils/coordinate_helper.py`, 38% → 100%.** Best effort-to-value ratio available. Pure
   functions — same input, same output, no setup. Test the ±180° wrap in `get_local_track`, and
   note that `get_local_track_vector` **divides by zero magnitude with no guard**. Write the test
   that catches that first.
2. **`rl/envs/waypoint_env.py`, 88% → ~98%.** The uncovered lines are the penalty functions, which
   early-return zero when their weight is zero — and the default config zeroes several.
   **Reward bugs are the most expensive kind here:** they do not crash, they quietly train the
   boat to do the wrong thing for six hours.
3. **`sim/protocol.py`, 82% → 100%.** Pure serialisation, fed straight off a network socket. Test
   that malformed and missing fields do not raise.
4. **`sim/web_runner.py`, 0%.** Hard because it is async and socket-bound. Do not test the server —
   `apply_controls()` and `step()` are plain synchronous methods testable with no WebSocket at
   all. Leave the asyncio plumbing uncovered and say so deliberately.
5. **`rl/live_vis.py`, 0%.** Dev-only. Reasonable to deprioritise, but make that a stated decision.

**Spread this item across people.** Coverage is best written by whoever owns the code being
covered — the reward branches belong to the RL lane, the sensor/actuator models to Simulator
Realism. Like documentation, it is carried by everyone rather than assigned to one person.

### Set a floor

```bash
uv run pytest --cov=sailbench --cov-fail-under=85
```

Move `pytest-cov` into the `dev` dependency group so it is not a `--with` flag every time.

---

## 5. ROS 2 integration

Everything above is hygiene — making the repository safe to work in. This is the stated objective
for the semester, and the item that changes what sailbench **is for**. It does not depend on the
chain above, so it starts now.

### What ROS 2 is

The framework the boat runs on — the standard robotics middleware, where each component is a
separate program (a *node*) and they communicate by publishing and subscribing to named channels
(*topics*). The Jetson runs ROS 2 Humble, and every major navigation component is a node on it.

### The problem

Sailbench is not one of those nodes. It is a standalone Python process talking to a browser over
its own WebSocket. So today the simulator **cannot test the code that actually sails the boat** —
only its own internal RL environment.

### What it would enable

The real navigation algorithm driving a simulated boat. That is how you test a controller against
gusts, wind shifts, sensor dropouts and actuator failures that are unsafe or impractical to stage
on a lake.

Concretely: sailbench publishes what the boat's sensors publish — GPS position, heading, wind,
actuator telemetry — and subscribes to the rudder and sail commands the navigation nodes emit, so
that from the algorithm's point of view the simulator is indistinguishable from hardware.

### Why it comes before the RL work

The end goal is a fair comparison between the learned policy and the existing deterministic
algorithm, on identical scenarios with identical metrics. **That comparison is impossible until
the deterministic algorithm can run inside sailbench at all** — and it cannot, today. Every RL
improvement is worth less until there is something honest to measure it against.

---

## 6. Sail model: stop treating the sail as a rigid airfoil

### The problem

`BasicSail` asks NeuralFoil for the lift and drag of a **NACA0012** section — a rigid, symmetric
aircraft wing, 12% as thick as it is long. A sail is none of those things:

| NACA0012 assumes | A real sail is |
|---|---|
| 12% thickness | A membrane, effectively zero thickness |
| Symmetric, no camber | Strongly cambered — typically 8–15% |
| Fixed shape | Shape changes constantly with sheet tension, outhaul, mast bend |
| Pushes or pulls either way | Can only **pull**. Reverse the pressure and it collapses |
| Works to ~15° before stalling | Routinely sails at 20–30°+ |

The symmetry point matters most: a symmetric section makes **zero lift at zero angle of attack**,
while a real cambered sail makes plenty. So the model is wrong in exactly the regime — close-hauled,
small angles — that the RL agent is trained on.

### The tell: luffing is bolted on, not emergent

```python
cl_scale = np.clip((aoa_deg_abs - luff_deg) / luff_ramp_deg, 0.0, 1.0)
cl *= cl_scale
```

A real sail luffs because the membrane cannot carry a reversed pressure difference — it is a
consequence of the physics. Here it is a correction applied *after* the fact, with two magic
numbers. **When a model needs a bolted-on correction for behaviour it should produce naturally,
that is the model telling you it has the wrong shape.**

### The better model is already in the repo and not wired up

`hybrid_sail.py` uses an analytic form instead of a foil lookup:

```python
cl = cl_max * np.sin(2 * alpha)
cd = cd0 + cd1 * (1.0 - np.cos(2 * alpha))
```

`sin(2α)` is `2·sin(α)·cos(α)` — the classic **flat-plate** result, a far better starting point for
a thin membrane than a thick airfoil, and it stays sensible at the high angles sails actually work
at. **But `SailboatHub` instantiates `BasicSail`.** `HybridSail` is imported at
`sailboat_hub.py:14` and never used.

### And the running sail model is completely untested

- The simulator runs **`BasicSail`** (`sailboat_hub.py:51`).
- Every test in `test_sail.py` exercises **`HybridSail`**, via the fixture in `conftest.py:42`.

The sail model that drives the boat has **no test coverage at all**, and the one the suite
validates never runs. This also explains the failing sail test in item 1c.

### One more: HybridSail ignores its own config

```python
LUFF_DEG = 5.0   # module constant in hybrid_sail.py
```

The YAML sets `luff_deg: 14`. `HybridSail` ignores it. Two sources of truth, and the config loses.

### What needs to be done

**First decide which model is canonical** and delete or clearly mark the other. Two sail models
where the tested one is not the running one is worse than having either alone. Then improve the
survivor:

1. **Add camber.** The single biggest accuracy win — lift at zero angle, asymmetric about it, which
   is how sails actually behave. Make it a config parameter.
2. **Make luffing emergent.** Collapse the lift when the pressure difference reverses, not when an
   angle crosses a threshold. The magic numbers then disappear.
3. **Blend two regimes.** Thin-airfoil theory (`CL ≈ 2π·sin(α + camber)`) at small angles into the
   flat-plate form at large ones. Sails spend real time in both.
4. **Compute Reynolds from actual apparent-wind speed** rather than the hardcoded `res: [1e5]`.
5. **Later: model mainsail and jib separately.** The real boat has a mainsail and *two jib servos*;
   the RL action vector has one sheet command. Prerequisite for anything that transfers to hardware.

### How to know whether it got better

`scripts/polar_diagram.py` already exists. A **polar diagram** — boatspeed against wind angle — is
how sailors have always characterised a boat, and it makes model errors obvious: the no-go zone
should be a clean notch, speed should peak on a reach, with a visible dip dead downwind. Generate
one before and after any change.

**Do not judge this by the RL score.** A better sail model will likely *lower* scores at first,
because the policy was trained against the old one and has learned to exploit its specific
wrongness. That is the sim-to-real gap in miniature, and the whole reason this matters.

---

---

## 7. Simulator UI

Not blocking anything, which is why it sits last — but the first tier is close to free, and two of
the items directly serve the RL-vs-deterministic comparison that items 5 and 6 are building toward.

**The UI currently has zero inputs, buttons or sliders.** The only interactive element in
`web/index.html` is the forces-panel header. Keyboard input is the entire control surface.

### Tier 1 — the backend already supports these, the UI just never sends them

`apply_controls()` in `web_runner.py` handles `wind_speed`, `wind_dir_deg`, `paused` and `reset`.
`main.js` sends **none** of them. This tier is wiring, not building.

- **Wind controls.** A speed slider and a compass dial you can drag mid-sail. This single control
  turns the sim from a demo into a test rig — and it is the manual version of the domain
  randomisation on the mid-fall research list.
- **Pause / reset / single-step.** Add **step** while you are there: advancing one 20 ms tick at a
  time is how you actually debug a force sign error like item 1b.
- **VMG readout.** It is what the RL reward optimises and the HUD does not show it. Computable
  client-side from the waypoint and velocity already being sent.

### Tier 2 — highest value for the comparison work

- **A boat track.** You cannot currently see where the boat has been. A persistent trail with tack
  points marked is the single most useful addition for comparing RL against the deterministic
  algorithm — two tracks overlaid, same wind, same waypoint, **is** the comparison.
- **Rolling strip charts.** Speed, heading and rudder over the last ~30 s. Recall
  `claude_run_5_lr1e4_oscillating`: the oscillation was invisible in the score but would be
  unmistakable as a sawtooth. Instantaneous bars cannot show a *behaviour*.
- **True and apparent wind as separate arrows.** Given the true/apparent mismatch in the
  observation vector, making both visible turns an invisible bug class into something you can see.
  The sim computes apparent wind internally and throws it away.
- **The no-go zone drawn on the water.** A translucent wedge anchored to the boat, rotating with
  the wind. The constraint the whole task is built around is currently invisible.

### Tier 3 — build this one first anyway

**A live reward-decomposition panel.** `_build_info()` already computes `reward_vmg`,
`penalty_no_go_zone`, `penalty_jibe`, `penalty_joint` and `penalty_stagnation` — every term, every
step — and none of it reaches the browser. It is thrown away.

Show it as a live stacked bar: positive contributions up, penalties down, net in the middle. When
the boat does something stupid you can see **which term paid for it, in the moment**. Most RL time
goes into the reward function rather than the network; this is the instrument for that work, and
the data already exists.

### Tier 4 — the sailing-native one

**Telltales.** Real sailors trim by watching ribbons taped to the sail: streaming aft means
attached flow, fluttering means stalled or luffing. The angle of attack is already computed every
step — drive a few animated telltales from it.

It is charming, but it is also functional in a way a number is not: it makes **luffing visible as
behaviour rather than a HUD value**. It is the natural companion to item 6 — as camber and
emergent luffing go in, the telltales become the readout for whether the new model behaves like a
sail.

### Tier 5 — bigger builds

- **Ghost replay.** Run a recorded episode as a translucent boat alongside a live one.
  Best-run-vs-current is how you see whether a change actually helped.
- **Live polar diagram.** `scripts/polar_diagram.py` already generates these offline. Put one in
  the corner with a dot for the current operating point against the theoretical curve. A sailor
  reads that instantly: am I fast for this angle, or leaving speed on the table?
- **Scenario picker with a fixed seed.** The late-fall deliverable asks for repeatable evaluation
  scenarios. A dropdown that loads a named scenario with fixed wind and seed makes "run both
  algorithms on scenario 3" a UI action instead of a code edit.

### Tier 6 — real water bodies

Import a real lake's shoreline and sail in it. Worth taking seriously, with two caveats.

**Use OpenStreetMap, not Google Maps.** You want a *shoreline polygon*, not imagery. OSM tags
lakes as `natural=water` and the Overpass API returns the coordinates free under ODbL:

```
[out:json];
way["natural"="water"](42.44,-76.52,42.50,-76.46);
(._;>;);
out geom;
```

Google's terms prohibit extracting geometry or deriving datasets from their imagery. For a student
project that gets published, that difference matters.

**Split it in two, because the halves cost very differently:**

- **Phase 1 — geometry as scenery and boundary.** Load the polygon, project lat/lon to local
  metres about a reference point, render it, ground the boat, end the episode on grounding. No
  observation change. The RL policy still cannot see land, but *you* can, and the deterministic
  algorithm can be tested against real geometry immediately. Days of work, and it slots into the
  repeatable-scenarios deliverable.
- **Phase 2 — make the policy aware of it.** A raycast fan (8–16 rays, each returning normalised
  distance to shore) is the standard approach, and maps onto something the real boat could
  plausibly have. But it roughly doubles the observation size, changes the network shape, and
  **invalidates every existing checkpoint.** Retraining from scratch. A semester-scale project.

**The trap:** accurate geometry with a uniform wind field can be *worse* than no geometry. Real
wind near a shoreline shadows behind terrain, bends along the shore, and gusts unpredictably —
sailors know a shoreline as the place wind becomes unreliable. Render a lake precisely and blow a
constant 5 m/s across it and the picture looks authoritative while the physics has become less
representative exactly where the new constraint lives. If you do this, pair it with at least a
crude rule reducing speed and adding directional noise near land.

Park Phase 2 until items 5 and 6 are settled. Retraining against a better sail model *and* a new
observation space at once means you will not know which change caused what.

### If you only do three

**Wind controls** (nearly free, immediately useful), **boat track** (directly serves the comparison
you have to deliver), and **the reward panel** (the data exists and you are discarding it).

---

## Quick reference

```bash
uv run pytest -q                                    # run the suite
uv run pytest test/test_keel.py -v                  # one file, verbose
uv run pytest -k "rudder" -v                        # tests matching a name
uv run pytest --lf                                  # re-run only last failures
uv run pytest -q --tb=long test/test_sail.py        # full traceback

uv run ruff check .                                 # lint
uv run ruff check . --fix                           # lint + autofix
uv run ruff format .                                # format
uv run mypy sailbench                               # type check

uv run --with pytest-cov pytest --cov=sailbench --cov-report=term-missing
```

Full write-up with diagrams: the **Sailbench Field Guide** artifact, sections 10 and 11.
