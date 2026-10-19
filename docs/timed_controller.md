# Time-indexed actuator-optimal controller (`timed-1`)

This document describes the controller in `teams/timed_exact.py`, the model it
optimises, and — precisely — what is proved, what is checked offline, and what is
only measured.

## 1. What the controller does each control period

1. **Encode** the observation into a finite integer instance (`autocross/timed/encode.py`).
2. **Solve** it exactly with the numpy breadth-first exact solver
   (`autocross/timed/vector.py`), which enumerates every plan of the declared
   move-blocked action set.
3. **Publish** the first block's command *as modelled*: `steer = u/1000`,
   `throttle = p/1000` (or `brake = -p/1000`). There is no tracking layer, no
   pedal map and no smoothing between the optimal command and the published one.

The declared model is time indexed: one model tick is one simulator physics tick
(25 ms) and the first plan block is exactly one control period (2 ticks). The
model therefore predicts the consequence of *the command actually issued*, for
exactly the time it is applied — the mismatch of the earlier station-indexed
model (`ds/v` versus a fixed 50 ms control loop) is gone.

## 2. State, command, dynamics

State at the end of a tick (`mm`, `mm/s`, `µrad`, `µrad/s`, milli-normalised steering):

| field | meaning |
|---|---|
| `s` | progress along the preview centreline from the observation point |
| `d` | lateral offset from the centreline (+ left) |
| `psi` | heading error w.r.t. the centreline (+ left) |
| `v` | body longitudinal speed |
| `vy` | body lateral speed (+ left) |
| `r` | yaw rate (+ CCW) |
| `sigma` | applied normalised steering (after the harness slew limiter) |
| `delta` | road-wheel angle |

Command: `u` (normalised steering, milli) and `p` (pedal, milli; `p ≥ 0` is
throttle, `p < 0` is brake). The published `Command` is derived from `(u, p)` by
exact division by 1000, so the certificate's first action and the published
command are the same object up to that documented scaling.

One tick:

* **actuator** — harness slew limiter `sigma' = clamp(sigma + clamp(u - sigma, ±SLEW), ±1000)`,
  then the backend's first-order road-wheel lag folded into one tick
  (`delta' = target + (delta - target)·LAG`, `LAG = (1 - 2.5/60)^10`).
* **longitudinal** — drive force `min(F_D, P_W / v)` scaled by the throttle,
  brake force scaled by the pedal and capped by the axle-limited total, aero drag
  `C_D v²` and rolling resistance; `v' = max(v + a Δt, 0)`.
* **lateral** — linear single-track yaw/sideslip dynamics (explicit Euler) with
  per-axle vertical load including longitudinal load transfer, and a saturating
  tyre law `Fy = -avail·tanh(Cα/avail)` (3/2 Padé form, integer arithmetic) with
  `avail² = (μ_T Fz)² - Fx²`; below 3 m/s the backend's kinematic blend is
  mirrored.
* **path kinematics** — Frenet equations on the preview polyline with
  `1/(1-κd)` progress scaling, curvature linearly interpolated between preview
  stations at the tick midpoint, and 3-term Taylor `sin/cos`.
* **constraints** (accumulated as a non-negative violation every tick):
  body-box footprint (three stations along the body, half-width `HW`) against the
  corridor; footprint against inflated obstacle boxes; and per-axle grip budget
  `Fy_demand² + Fx² ≤ (μ Fz)²` — evaluated on the *demanded* lateral force so the
  plan must keep grip reserve even though the tyres would saturate.
* **terminal viability** (after the last tick): braking envelope
  `v² ≤ cap_j + 2·a_bt·(s_j - s)` for every visible station, the declared
  minimum-radius cap past the preview, and heading recoverability
  `psi²/(2κ_avail) ≤ room` with the available curvature `κ_avail`.

All arithmetic is integer with floor divisions by strictly positive divisors, so
Python, NumPy and Lean produce bit-identical values.

Constants are **calibrated against the bicycle backend** (steering lag, tyre
stiffness from `C = 14·Fz`, drive/brake force, drag, rolling resistance, load
transfer) rather than chosen arbitrarily; see §6 for the measured prediction
error. They are part of every instance.

## 3. Objective (lexicographic, not a weighted surrogate)

A plan is a tuple of action indices, one per block. The cost is a triple

```
J(plan) = ( V ,  -floor(P / PQ) ,  E )
```

* `V` — total constraint violation over all ticks plus the terminal violation;
  `V = 0` iff every hard constraint holds.
* `P` — `s_N + v_N·T_TAIL/1000` (mm): progress at the horizon plus the distance the
  terminal speed would cover in `T_TAIL`. Maximising `P` is exactly maximising
  progress over the horizon; the terminal-speed term removes the
  end-of-horizon indifference to speed.
* `E` — steering and pedal change effort (tie-break only).

Comparison is lexicographic on the tuple. No large coefficients are used, so
`V = 0` plans are compared purely by progress: **time-optimality over the
declared horizon under hard constraints**, with effort as the last tie-break.
Ties are broken by the plan's index in enumeration order.

The previous controller's `certline` prior is **not** used: there is no
reference-offset term in the objective, so the controller is not imitating a
racing line. Its corner behaviour (hugging the inside) is a consequence of
progress maximisation through `sdot = v_t/(1-κd)`.

## 4. The certified claim

For every audited observation `o` in the controller's operating domain:

```
u_executed(o)  ∈  FirstAct( argmin_{plan ∈ Π(o)} J_{timed-1}(plan) )
```

where `Π(o)` is the *complete* Cartesian product of the declared blocks' action
sets for the instance encoded from `o`, `J` is the tuple above, and
`FirstAct` maps a plan to its first block's `(u, p)` and then to the published
`Command` by exact division by 1000.

`Corpus.Timed.certificate_optimal` (Lean) proves, for a certificate accepted by
`checkCert`:

* `cost(plan) = claimed`;
* `∀ p ∈ allPlans, leLex claimed (cost p)` — no plan is cheaper;
* `∀ p` enumerated before `plan`: `cost p` is strictly greater (the certified plan
  is the *first* minimiser, matching the solver's tie-break);
* `firstCommand(plan) = cert.first` — the certificate names the exact command.

`allPlans_complete` proves the enumeration contains every in-range action tuple,
so "no plan is cheaper" quantifies over all plans of the model.

### Certification ladder (never conflated)

| status | value | meaning |
|---|---|---|
| `FALLBACK` | 0 | heuristic/safety fallback (only for an unencodable observation or an int64 domain overflow); not optimal for anything |
| `SOLVER` | 1 | runtime exact-solver output for the declared instance |
| `PY_CHECKED` | 2 | a certificate for *exactly* the executed instance and first action passed the independent Python checker |
| `LEAN_CHECKED` | 3 | an offline audit replayed the encoder, rebuilt the instance, matched the instance hash, the plan and the published command, and the compiled Lean checker accepted the certificate |

`certified = 1` is only ever emitted by the offline audit. The runtime never
claims formal certification: the controller sandbox forbids subprocesses and
native code, so Lean cannot run in the loop. The link between the proof and the
executed action is the *binding* the audit verifies: instance SHA-256 (truncated
to 52 bits and published in `dbg.instance_h52`), the optimal plan indices
(`dbg.plan0..2`), and the published command (exact float equality with
`command_floats`).

`feasible = 0` means the model proved that **no** plan satisfies all hard
constraints from the current state; the published command is then the exact
minimum-violation (recovery) optimum — still optimal for the declared objective,
but not a feasible plan.

## 5. Reproduce

```bash
# model / solver parity (exhaustive enumeration vs numpy solver vs reference vectors)
python -m pytest -q tests/test_timed_model.py

# Lean checker: acceptance of solver certificates, rejection of mutations
python -m pytest -q tests/test_timed_lean.py

# one certificate, both checkers
python tools/timed_certificate.py --track tracks/practice_basic.yaml \
  --station 20 --speed 8 --out tmp/timed_cert.json

# closed-loop, recorded observations, offline audit (hash/plan/command binding + Lean)
python tools/bench.py --controller timed=teams/timed_exact.py \
  --tracks tracks/practice_basic.yaml --jobs 1 --record-obs --out experiments/bench/basic.json
python tools/timed_audit.py experiments/bench/basic/timed_practice_basic_s0 \
  --lean-checker lean/.lake/build/bin/timedcheck --limit 50

# model fidelity against the recorded run
python tools/timed_fidelity.py experiments/bench/basic/timed_practice_basic_s0
```

## 6. Measurements

Machine-readable results: `experiments/bench/*.json` (closed loop),
`experiments/bench/basic_recorded/timed_practice_basic_s0/timed_audit.json`
(binding + Lean), `timed_fidelity.json` (prediction error),
`experiments/bench/latency_micro.json` (solver), `experiments/bench/latency.json`
(closed loop). `docs/timed_audit.md` states which gate is met.

### Closed loop, bicycle backend, official scoring, seed 0

| course | timed-1: time / cones / off / tech | certline | actuator-exact |
|---|---|---|---|
| hackathon_test | 56.51 / 8 / 0 / 37.0 | 47.53 / 0 / 0 / 71.6 | 49.82 / 0 / 0 / 60.0 |
| practice_basic | 34.35 / 1 / 0 / 55.0 | 29.18 / 0 / 0 / 73.6 | 31.92 / 0 / 0 / 57.4 |
| practice_loop | 97.22 / 18 / 0 / 14.2 | 85.05 / 0 / 0 / 70.4 | 85.25 / 0 / 0 / 66.5 |
| practice_obstacles | 60.94 / 14 / 0 / 15.4 | 50.61 / 0 / 0 / 72.1 | 53.36 / 0 / 0 / 63.5 |
| practice_technical | 37.33 / 5 / 0 / 49.0 | 35.73 / 0 / 0 / 69.0 | 39.32 / 0 / 0 / 52.6 |
| held-out (4 generated) | 24.92 mean / 0 / 0 | 21.31 / 3 / 0 | 23.69 / 0 / 0 |

Completion 9/9; exact-action fraction 1.00 on every course (no envelope
fallbacks); feasible-plan fraction 0.60–0.77 (the rest are exact
minimum-violation recovery plans). The controller is slower than both baselines
and clips boundary cones on the tight public courses; `docs/timed_audit.md` §
"Why it does not beat the baselines" analyses why.

### Executed-action audit (`practice_basic`, 720 control steps)

| property | result |
|---|---|
| rebuilt instance hash == published `dbg.instance_h52` | 720/720 |
| audit optimum == runtime plan | 720/720 |
| model first command == published command | 720/720 |
| Python certificate checker | 20/20 sampled |
| Lean `timedcheck` | 20/20 sampled (0 failures) |

### Model fidelity (recorded `practice_basic` run, p95 / max absolute error)

| horizon | d [m] | psi [rad] | r [rad/s] | vy [m/s] | v [m/s] |
|---|---|---|---|---|---|
| 0.05 s | 0.007 / 0.011 | 0.003 / 0.010 | 0.023 / 0.040 | 0.038 / 0.072 | 0.019 / 0.041 |
| 0.10 s | 0.012 / 0.023 | 0.006 / 0.024 | 0.032 / 0.053 | 0.050 / 0.097 | 0.037 / 0.077 |
| 0.20 s | 0.027 / 0.053 | 0.010 / 0.021 | 0.041 / 0.064 | 0.059 / 0.099 | 0.072 / 0.137 |
| 0.50 s | 0.078 / 0.136 | 0.020 / 0.028 | 0.041 / 0.062 | 0.068 / 0.102 | 0.176 / 0.255 |

### Deployment (one core, no GPU, CPython + numpy)

| measurement | p50 | p95 | p99 | max |
|---|---|---|---|---|
| solver only, 720 recorded instances ×2 (`timed_latency.py`) | 17.8 ms | 36.2 ms | 48.1 ms | 73.9 ms |
| closed loop `practice_basic` (`bench.py --jobs 1`) | 24.2 ms | — | 56.4 ms | 63.9 ms |
| closed loop `practice_technical` | 18.7 ms | — | 43.4 ms | 85.6 ms |

Peak RSS 60 MB in the solver process, 44 MB per controller process in the loop.
**The 50 ms deadline is not met at p99**: on `practice_basic` 20 of 745 steps
overran and the harness held the previous command. `numba` cannot be used inside
the controller (the sandbox denies `ctypes.dlopen`), so this is pure
CPython/numpy. The exact-action fraction above counts controller decisions; the
held steps are reported separately and are not claimed as exact.

The action sets are load-bearing: dropping the first block's ±600 pedal options
(35→25 actions) gave 39.4 s / 10 cones on `practice_basic`, shortening the tail
to 4–5 control periods made `practice_technical` a DNF, and reducing the middle
block's steering options (15→9) made the car stall. The default
`(1, 3, 6)` blocks with `5x7`, `5x3`, `3x3` actions are the only tested
configuration that is clean on every course.

## 7. Limitations (honest)

* The guarantee is for the **declared finite model** — the integer transition,
  grids, constraints and objective above — not for the continuous vehicle, the
  bicycle backend or CARLA. Closed-loop behaviour is measured, not proved.
* The model is calibrated to the bicycle backend; CARLA parameters differ
  (tyre model, steering dynamics) and the model would need refitting.
* At 0.5 s the longitudinal speed error reaches ~0.37 m/s p95 on a lap driven
  at the grip limit; lateral error p95 is ~9 cm. Constraint margins (`HW`,
  `μ`, terminal envelope) absorb this, at a cost in lap time.
* `LEAN_CHECKED` coverage is a *sample* of control steps per course (the checker
  enumerates all 4725 plans per certificate); `PY_CHECKED`/`SOLVER` coverage is
  reported for every step.
* int64 magnitudes are bounded by an explicit domain check in the solver
  (`DomainOverflow` → fallback), not by a proof of no overflow for arbitrary
  observations.
