# Final audit — `timed-1` actuator-optimal controller

Every gate below is stated as it actually is, with the command that produced the
evidence. `PROVEN` = Lean kernel theorem about the declared finite model;
`CHECKED` = independent implementation agreement on executed actions; `MEASURED`
= closed-loop simulation result, not a proof. Machine-readable results:
`experiments/bench/*.json`, `experiments/bench/basic_recorded/timed_practice_basic_s0/timed_audit.json`,
`timed_fidelity.json`, `experiments/bench/latency_micro.json`.

## Proven guarantees (Lean, `Corpus.Timed`, axiom audit clean)

| statement | theorem |
|---|---|
| accepted certificate ⇒ the recomputed cost of the plan equals the claim | `certificate_cost` |
| accepted certificate ⇒ no plan of the declared model is cheaper (lexicographic) | `certificate_optimal` |
| accepted certificate ⇒ every earlier-enumerated plan is strictly more expensive (first minimiser) | `certificate_tie_break` |
| accepted certificate ⇒ the certified first command is what block 0 publishes | `certificate_first_action` |
| combining the three: the published first action is the first action of a globally optimal plan | `certificate_first_action_optimal` |
| the enumeration contains every in-range action tuple | `allPlans_complete`, `allPlans_sound` |
| checker soundness for the decidable predicate | `checkCert_sound` |
| a concrete certificate is accepted by kernel reduction | `witness_check` |

All depend only on `propext`, `Classical.choice`, `Quot.sound`
(`lake env lean Corpus/Audit.lean`, 429 theorems, no `sorryAx`, no
`Lean.ofReduceBool`). Scope: the integer time-indexed model with the constants
encoded in each certificate. Not a statement about the continuous vehicle, the
bicycle backend or CARLA.

## Executed-action binding (the link from proof to command)

`tools/timed_audit.py` replays the recorded observation stream through the same
encoder, rebuilds the instance, and checks per control step:

| check | result on `practice_basic` (720 exact steps) |
|---|---|
| rebuilt instance hash equals the hash the controller published | 720/720 |
| audit's optimal plan equals the runtime plan (`dbg_plan0..2`) | 720/720 |
| first command derived from the plan equals the published command (exact floats) | 720/720 |
| independent Python certificate checker accepts | 20/20 sampled |
| compiled Lean checker accepts | 20/20 sampled |

One step (392) initially failed the Lean check: the Lean `bodyPoints` used the
body offsets as *stations*, so the corridor widths were read at the wrong
station. The counterexample is preserved in
`tests/fixtures/timed_lean_disagreement_392.json`, the Lean model was corrected,
the vector suite gained a varying-width instance, and the fixture is now a
regression test. This is the only arithmetic disagreement the audit found.

Lean-checked coverage is a sample (one certificate costs ~6.5 s because the
checker re-enumerates all 4725 plans); the *binding* checks cover every step.

## Gates

| gate | status | evidence |
|---|---|---|
| 1 Correctness (scalar/vectorised/certified/exhaustive agree) | **CHECKED** | `tests/test_timed_model.py` (exhaustive enumeration vs numpy solver vs stored reference vectors), `tests/test_timed_lean.py` (every plan cost of 11 instances identical in Lean, 4725-plan certificates) |
| 2 Formal verification (accept real certificates, reject mutations) | **PROVEN** | `tests/test_timed_lean.py`, `tools/timed_certificate.py`, audit above; 9 corruption kinds rejected by both checkers |
| 3 Model fidelity (predictions vs telemetry, timing mismatch fixed) | **MEASURED** | time-indexed: 25 ms ticks = physics tick, first block = one 50 ms control period; `tools/timed_fidelity.py`: d p95 7 mm @50 ms → 78 mm @500 ms, psi p95 3 mrad → 20 mrad |
| 4 Controller performance (completion, collisions, penalised score) | **PARTLY MET** | completion 9/9 courses; 0 cones on the four held-out generated courses; **does not** beat `certline` or the previous actuator controller on the public courses (table below) |
| 5 Certified coverage (exact-action fraction, justified fallbacks) | **MET for exact steps** | 100% of published decisions came from the exact solver on every course (no envelope fallbacks); 60–77% of those plans are fully feasible (`dbg_feasible=1`), the rest are exact minimum-violation recovery optima |
| 6 Deployment (1 core, 1 GB, 50 ms) | **NOT MET at p99** | solver-only p50 17.3 ms / p99 48.3 ms / max 58.1 ms, RSS 59 MB; closed loop p50 24.2 ms / p99 56.4 ms with 20/745 held commands on `practice_basic` |
| 7 Reproducibility | **MET** | `experiments/README.md`, committed JSON, fixed seeds |

## Closed loop (bicycle backend, seed 0, official scoring)

| course | timed: time / cones / off / tech | certline | actuator-exact |
|---|---|---|---|
| hackathon_test | 56.51 / 8 / 0 / 37.0 | 47.53 / 0 / 0 / 71.6 | 49.82 / 0 / 0 / 60.0 |
| practice_basic | 34.35 / 1 / 0 / 55.0 | 29.18 / 0 / 0 / 73.6 | 31.92 / 0 / 0 / 57.4 |
| practice_loop | 97.22 / 18 / 0 / 14.2 | 85.05 / 0 / 0 / 70.4 | 85.25 / 0 / 0 / 66.5 |
| practice_obstacles | 60.94 / 14 / 0 / 15.4 | 50.61 / 0 / 0 / 72.1 | 53.36 / 0 / 0 / 63.5 |
| practice_technical | 37.33 / 5 / 0 / 49.0 | 35.73 / 0 / 0 / 69.0 | 39.32 / 0 / 0 / 52.6 |
| held-out (4 generated) | 24.92 / 0 / 0 / ~53.8 | 21.31 / 3 / 0 | 23.69 / 0 / 0 |

Exact-action fraction: timed 100% (actuator-exact 0–45%), but the timed
controller is slower and clips boundary cones on the tight public courses.

## Why it does not beat the baselines (diagnosis, not excuse)

* The plan is chosen over a 0.9 s horizon with **constraint violation first**
  (lexicographic). On tight courses the model's own constraint set is
  unsatisfiable for 23–40% of steps (`dbg_feasible=0`), and the controller then
  executes the exact *minimum-violation* command — safe, but not the fastest
  recovery, and the frequent mode switches cost lap time and boundary clearance.
* The model's lateral prediction degrades at the grip limit (no yaw-lag state, a
  single friction coefficient for both axles, no lateral load transfer): the p95
  lateral error grows to 8 cm at 0.5 s. `HW = 1330 mm` (150 mm beyond the cone
  inflation) is the margin that keeps the car off the cones on most courses, and
  raising the grip budget (`MU`, `A_LT`) to recover lap time caused 10–17 cones
  in closed loop — the model is optimistic exactly where the extra speed is.
* `certline` is faster because it runs a *racing line* with a longer horizon and
  a fixed speed profile, then tracks it; it is not optimal for any model, and it
  is not certified.

## What would close the remaining gap

1. A yaw-rate/sideslip *state* with measured tyre saturation and lag fitted from
   telemetry (a bigger state space; the exact search would need pruning with a
   proved bound, which this audit showed is easy to get wrong).
2. Exact real-time performance: the 4725-plan enumeration is the bottleneck.
   Sound options are a coarser tail action set with a proved-complete
   enumeration, or an AOT-compiled kernel (the sandbox forbids `ctypes.dlopen`,
   so numba cannot be used inside the controller).
3. A recovery objective that is lexicographic *per tick* (fix the earliest
   violation first), which would replace the mode-switch behaviour seen at
   `dbg_feasible=0`.
