# Actuator-space optimal controller

This document describes the direct actuator controller and the exact scope of its
optimality claim. It is deliberately narrower than "optimal driving".

## Claim

When the controller returns `certified = true`, it executed the first action of a
plan that is globally optimal, among all plans of the **declared finite actuator
model**, for the declared integer objective. The accompanying certificate is
accepted by both:

* `autocross.actuator.certificate.check_certificate` (independent Python
  re-encoding and Bellman recomputation), and
* the compiled Lean checker `actuatorcheck`, whose acceptance implies
  `Corpus.Actuator.certificate_optimal` (kernel-checked).

The claim is conditional on the certificate belonging to the same observation,
model instance, objective, and first action that the controller executed. The
controller binds those fields when `emit_certificates` is enabled; a certificate
is not evidence for an unrelated step.

It is **not** a claim of continuous-vehicle, CARLA, whole-course, or
global-physical optimality. It is also not a claim about fallback commands.

## State, actions, transition

State at station $i$:

$$
s_i=(i,d_i,\psi_i,b_i,\delta_i)
$$

with lateral offset $d$ (mm), heading error $\psi$ (µrad), squared speed
$b=v^2$ (mm²/s²), and road-wheel angle $\delta$ (µrad). The action is a
normalised steering request $u$ and an acceleration request $a$ (mm/s²). The
transition is the integer path-relative bicycle model:

* first-order steering lag with rate limit, then nearest-grid snapping of
  $\delta$;
* understeer denominator $L + 25b/10^7$;
* path curvature $\kappa=\delta\cdot1000/\text{den}$;
* heading and offset integration by the midpoint rule, snapped to the declared
  grids;
* squared-speed integration $b'=b+2a\,ds$, rejected below zero, snapped;
* corridor and inflated-obstacle checks at arrival and midpoint;
* lateral-acceleration bound and friction ellipse;
* a terminal set at the horizon, with a braking-envelope speed cap over the
  visible stations beyond the horizon.

The declared grids, constants, preview, terminal distance, and optional
reference offsets are part of the certificate instance. The objective is

$$
1000\cdot\operatorname{cdiv}(2\,ds\,10^6,\lfloor\sqrt b\rfloor+\lfloor\sqrt{b'}\rfloor)
+| \delta'-\delta |/1000
+5000\,|d'-\text{ref}_d|
+5000\,|\psi'-\text{ref}_\psi|/100 .
$$

`ref_d`/`ref_psi` are the optional fast-reference prior (normally `certline`'s
published path). They change the declared objective; they do not turn the
controller into a tracker.

## Certificate

The JSON certificate contains:

* `instance`: every model field, including grids, constants, preview, terminal
  condition, and reference arrays;
* `roots`: `[[b, isqrt(b)], ...]`;
* `states` and `values`: reachable states and optimal option-valued values per
  stage;
* `plan`, `planActions`, `firstAction`, `claimed`, `planCost`;
* `certificate_sha256`.

The Python checker rejects a wrong hash, malformed tables, a non-singleton start
table, duplicate rows, successors outside the next table, wrong terminal values,
any Bellman row mismatch, an inadmissible first action or plan step, a wrong plan
cost, and a claimed value on an infeasible instance. The Lean checker recomputes
the same recurrence; the SHA and metadata are not part of its theorem.

## Reproduce

```bash
python -m pytest -q tests/test_actuator_optimal.py tests/test_actuator_parity.py
python tools/actuator_certificate.py --track tracks/practice_basic.yaml \
  --station 20 --speed 8 --out tmp/actuator_certificate.json
powershell -NoProfile -Command \
  "& '.\lean\.lake\build\bin\actuatorcheck.exe' 'tmp\actuator_certificate.json'"
```

The demo prints the Python report, the first successor/action, and the Lean
checker's JSON. The real certificate used on `practice_basic` at station 20 has
`claimed = 1222912000`, `horizon = 4`, and is accepted by both checkers.

## Controller modes and measurements

`teams/actuator_exact.py` runs the exact DP on a reduced real-time declaration
(horizon 4, 9 offsets, 5 headings, 9 speed buckets, 5 steering angles, 5×5
actions) and explicitly falls back to `certline` when the observation is outside
the actuator envelope, near an obstacle, in tight curvature, or when no exact
plan is admissible. Fallback commands are labelled `certline-fallback` or
`actuator-recovery` and are never labelled optimal.

Bicycle selftest, all five public courses, `compute_budget.mode=measure`:

| course | time (s) | cones | off | gates | p99 (ms) | tech |
|---|---:|---:|---:|---:|---:|---:|
| hackathon_test | 49.8 | 0 | 0 | 0 | 21.5 | 67.0 |
| practice_basic | 31.9 | 0 | 0 | 0 | 27.9 | 66.2 |
| practice_loop | 85.3 | 0 | 0 | 0 | 22.5 | 66.9 |
| practice_obstacles | 53.4 | 0 | 0 | 0 | 21.1 | 66.9 |
| practice_technical | 39.3 | 0 | 0 | 0 | 29.3 | 61.3 |
| **mean** | | **0** | **0** | **0** | | **65.7 / 75** |

The existing `certline` baseline is 69.9/75. The actuator controller is therefore
not yet the fastest public-course controller; it is a formally checked actuator
optimizer whose exact steps are a minority of control steps. On `practice_basic`
the exact DP ran on 132 of 719 control steps (~18%); the rest were explicit
fallbacks. The exact-only mode is not competitive at this grid and horizon.

The exact DP itself is tens of milliseconds per solve on the declared real-time
grid in CPython/NumPy, with observed per-step p99 21–29 ms including fallback and
reference construction. Those are measurements on the development machine, not
a real-time proof for the 1-core/1 GB target; the certificate checker is an
offline verifier and is not run inside the control loop.


### Optional performance governor

`performance_guard: true` falls back to `certline` whenever the exact plan's
requested acceleration is more than 0.4 m/s² below the baseline's. It recovers
the baseline lap times (mean 71.5/75 on the same five courses, zero cones/off),
but it reduces certified exact steps to ~2% of control steps. That mode is a
performance envelope around `certline`, not evidence that the actuator
controller is driving optimally; it is off by default and is reported only as a
measured trade-off.
## Remaining gaps

* The 50 ms budget is approached but not proved; the exact DP is CPython/NumPy,
  and the controller falls back rather than exceeding the envelope.
* The model is integer and path-relative. It does not establish that the bicycle
  backend or CARLA follows the transition.
* The fast-reference prior is a heuristic path; it is part of the declared
  objective, so the theorem is optimality for that objective, not for a
  physical minimum-time objective.
* The exact fraction is low. Increasing it without cones is open work.
