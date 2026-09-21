# Trained waypoint policies

Four PPO runs on Flingo Floaty, all from 2026-09-19. They are kept because each
one answers a question the others do not, not because they are all good.

The boat is `configs/flingo_floty.yaml` throughout: `BasicHullModel` on the
Hughes friction line with the measured wetted surface and seven stations,
`FiniteSpanKeel` (NACA0010, effective AR 8.0), `ORCWithJibSail` on the 2022
kheff curve, `FiniteSpanRudder`, windage active. 27 kg, wind 5 m/s. Nothing
about the boat changes between runs; only the task and the reward do.

What is stored: `best_model/best_model.zip`, the config the run actually used,
the evaluation history and `summary.json`. The intermediate checkpoints are not
committed -- they are resume points, and there were 140 of them across these
four runs, about 22 MB. If you need one, rerun from `config_used.yaml`; the runs
are reproducible, seed 7 (two of these runs came out bit-identical from the same
config, which is how we know).

## Reading the numbers

Returns are **not** comparable between runs. The reward function itself changes
-- the no-go weight goes 0.3, 15.0, 3.0 -- so a higher return can just mean a
gentler penalty. Compare the behaviour columns instead.

"Pinch" is the share of steps sailed inside 25 degrees of the true wind. It
matters because the simulator lets Flingo sail at 22 degrees and hold 1.46 m/s
there, which a boat of her type cannot do; a policy that lives in that band has
learned something that will not work on the water. "Sheet" is how often the sail
command sits pinned at an extreme -- 100% means the policy gave up on trimming
and steered with the rudder alone.

| run | task | success | speed | pinch | steps | sheet pinned |
|---|---|---|---|---|---|---|
| `121045` | 25 M, bias 0.5, no-go 3.0 | 100% | 1.319 m/s | 14.8% | 883 | 100% |
| `073312` | 750 k, bias 0.5, no-go 15.0 | 100% | 1.198 | 27.6% | 1058 | 100% |
| `074021` | 750 k, bias 0.5, no-go 3.0 | 97.5% | 1.147 | 36.5% | 1009 | 65% |
| `061347` | 750 k, bias 1.0, no-go 0.3 | 100%* | 0.972 | 49.6% | -- | 100% |

\* on an all-upwind task, and audited with a different script and different
episode seeds. Do not read it against the other three.

For reference, the boat does 1.68 m/s close-hauled and 1.80 m/s at its best
angle, so even the fastest of these is sailing at about four fifths of what the
hull will give.

## Scorecards

`scripts/score_runs.py` writes `scorecard.json` into each run directory, and the
shipyard reads it: the helm row no longer offers `waypoint_ppo_20260919_074021
(best)` but `750 k · no-go 3 · no-go lag 4 · best — 1.20 m/s, 30% pinch, trims`,
with the left of the dash diffed from the run's own `config_used.yaml` against
the other runs on the same boat. Twenty fixed-seed episodes per checkpoint,
`upwind_waypoint_bias` forced to 0.5 because it is the one knob that changes
what a seed produces, and every other setting -- the observation scales
especially -- left as the run trained with it.

It reproduces the table above, run for run:

| run | speed | pinch | sheet pinned | success |
|---|---|---|---|---|
| `121045` best | 1.34 m/s | 14.0% | 100% | 100% |
| `073312` best | 1.22 | 27.2% | 100% | 100% |
| `074021` best | 1.20 | 30.5% | 57% | 95% |
| `063537` best | 1.13 | 34.6% | 75% | 85% |
| `061347` best | 1.13 | 32.8% | 99% | 100% |

Two things to read carefully. `061347` comes out at 1.13 m/s and 32.8% here
against the 0.972 and 49.6% above, because it is being scored on the mixed task
rather than the all-upwind one it trained on -- it is still the slowest of the
five, but the two numbers are not the same measurement. And `120703` scores
identically to `074021` on all five columns, which is the same-policy claim
above arrived at from the other end.

The three `basic_sailbot` runs in this directory -- `claude_run_4`,
`claude_run_5`, `good_vmg_stable` -- blow the integrator up in 20-35% of
episodes on the current physics: the boat runs away, and the wave drag term's
`(|u|/v_hull)**4` overflows a few steps later. The scorer counts those episodes
and leaves them out of the averages, and the shipyard chip says so instead of
showing a speed. Whether it is the policies or the boat has not been looked at.

---

## `waypoint_ppo_20260919_121045` -- the one to use

25 million steps, about an hour and three quarters. Best checkpoint at 15.4 M.

This is the best policy we have. It reaches the mark every time, averages
1.319 m/s, and spends 14.8% of its time pinching, down from 36.5% on the
identical 750 k run. It is quick upwind in particular: 1.305 m/s in the 0-35
degree band against 0.903 for the earlier trimming policy.

The long budget was worth less than it looks. Success rate hits 100% at 200 k
and then tells you nothing; return keeps improving until roughly 3 M and is flat
after that. Best return landed at 15.4 M but 3-5 M would have given the same
policy in about fifteen minutes. Run that instead unless you have a reason.

It does not trim. The sheet is pinned at one end for every step of every
episode, boom on the centreline in all five wind-angle bands including 140-180
degrees where the measured optimum is 40-80 degrees of sheet. That is not
undertraining -- 25 M steps did not move it. It is that the task only spends 5%
of its time deep downwind, so trimming is not worth learning. Deep downwind it
is slower than `074021` for exactly this reason.

## `waypoint_ppo_20260919_074021` -- the only one that trims

Same 750 k budget, no-go weight 3.0.

Worse than `121045` on every headline number, and still the one to start from if
you want a policy that uses both controls. It is the only run where the sail
command is not saturated (65%, not 100%), the boom is never exactly centred, and
boom angle tracks true wind angle at +0.76 -- it goes 1.1, 2.1, 5.2, 7.4, 13.0
degrees across the bands, which is the right shape even if the downwind end is
far short of the 40-80 it should reach.

Note this run and `waypoint_ppo_20260919_120703` are the same policy, bit for
bit. Same config, same seed. Only one is kept.

## `waypoint_ppo_20260919_073312` -- no-go weight too high

Same as `074021` but with the no-go penalty at 15.0 instead of 3.0.

Kept as the record of what over-weighting that penalty does. It works on its
target: pinching falls to 27.6%, the best of the 750 k runs, and speed rises to
1.198. But at roughly 1.3 per step the penalty matches the entire VMG term, the
sail's contribution to reward becomes a rounding error, and the policy goes
straight back to strapping the main flat for 100% of its steps.

Footing off only has to beat pinching by the VMG difference between 25 and 40
degrees, which is 0.04. 15.0 was sized against the wrong quantity. 3.0 is the
value that stuck.

## `waypoint_ppo_20260919_061347` -- the baseline, and a warning

750 k steps, every mark dead upwind (`upwind_waypoint_bias: 1`), no-go penalty
at the old 0.3, which is inert -- it peaks at 0.054 per step against a VMG term
reaching 1.28. This run also predates the sail trim fix and the no-go occupancy
work, so it is not on the same simulator as the other three.

It scores 100% on its evaluation set and is the worst policy here. That is the
point of keeping it. Half its steps are inside 25 degrees of the wind, it
averages 0.972 m/s against the 1.68 the boat has close-hauled, and it holds the
sail strapped amidships throughout -- boom under 0.11 degrees for the whole
episode. It solved the task by crawling straight at the mark through a no-go
zone the real boat does not have.

If you take one thing from this directory: a perfect success rate in the
simulator is not evidence the policy can sail. None of these four have been
checked against the water, and the pointing gap that lets all of them pinch is
still open -- see the note on `no_go_zone_penalty` in `configs/flingo_rl.yaml`.
A GPS log of Flingo actually beating would settle it.
