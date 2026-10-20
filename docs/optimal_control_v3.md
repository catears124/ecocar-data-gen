# Actuator-optimal control v3: what was built, what is proven, what is measured

This note replaces the `timed-1` claim that the exact controller is a
competitive driving policy.  It records the redesign, the evidence that drove
it, and the honest scope of the result.

## 1. Reproduction of the old result

On this machine (Python 3.14, bicycle backend, official settings, seed 0):

| controller | public courses | cones | off-course | mean technical | step p99 |
|---|---|---:|---:|---:|---:|
| `teams/certline.py` | 5/5 finished | 0 | 0 | **71.5 / 75** | 3.5–6.9 ms |
| `teams/timed_exact.py` | 3/5 finished (2 DNF) | **44** | 8 | **16.7 / 75** | 57.8–61.1 ms |

`timed-1` is exact over its declared move-blocked action set, but that set is
not expressive enough and its infeasible branch publishes a minimum-violation
trajectory as if it were a plan.  The failing run is preserved in
`experiments/bench/v3/failed_lineopt_unsafe.json` (the earlier `timed-1` run is
in `runs/` and the diagnostic runs under `experiments/diag/`).

## 2. Diagnosis driven by the recorded failures

`tools/structural_diagnostics.py` replays a recorded `timed-1` run through the
same encoder and exact solver and solves counterfactual relaxations of the
declared instance.  In the 80-step failure cluster around the first cone strike
(`practice_basic`, steps 500–579):

| quantity | value |
|---|---:|
| model infeasible | 58.8 % |
| widening the corridor recovers feasibility | 95.7 % of infeasible steps |
| relaxing the friction budget recovers | 66.0 % |
| relaxing the terminal envelope recovers | 53.2 % |
| relaxing everything | 100 % |
| widening the action blocks | 6.4 % |
| baseline command exactly representable in a block | 65 % |
| baseline command added to the blocks makes the model feasible | 43.8 % |

Conclusions:

1. **The failure is not the search.**  The exact solver finds the optimum of
   its model; the model itself is infeasible from states the controller drives
   itself into.
2. **The dominant cause is constraint tightening, then the box friction budget
   and the terminal envelope.**  The corridor model uses the full body box with
   a 0.30 m margin and a 0.10 m cone margin; at the grip limit the prediction
   error and the tracker lag consume that margin.
3. **The move-blocked action set is too narrow.**  The baseline command is
   representable only ~65 % of the time, and the smallest block only allows
   ±0.10 steering offset around the current applied steering.  Adding the
   baseline action helps, but cannot repair an already infeasible physical
   state.
4. **Longer horizons / richer actions alone do not fix a conservative model.**
   In the cluster, widening the actions recovers only 6 %.

Prediction error of the integer model on a `certline` run that reaches
7.8 m/s² (p95 absolute error): 0.10 s — `d` 1.7 cm, `v` 5.5 cm/s;
0.20 s — `d` 3.8 cm, `v` 10.9 cm/s; 0.50 s — `d` 10.3 cm, `v` 26.6 cm/s.
That growth is the quantitative reason the `timed-1` corridor margin could not
be defended at the limit; the raw report is
`experiments/bench/v3/fidelity_certline_basic.json`.

## 3. New architecture

### A. Offline exact oracle (`tools/global_oracle.py`)

- Declares a finite `teacher-joint-1` instance from the observation: station,
  previous/current offset, speed squared; offsets, speed-squared lattice,
  corridor, obstacles, curvature cap, acceleration/braking budgets, terminal
  cap.
- Solves it **exactly** with `autocross.lineopt.dp.solve_fast`, a vectorised
  backward induction over the full declared state universe.  The vectorised
  result is bit-identical to the scalar reference
  (`autocross.optimal.solver.solve`); `tests/test_lineopt.py` checks this.
- Constructs the **baseline candidate** as certline's published path, projected
  onto the oracle offset lattice, followed by the exact minimum-time speed DP
  on that fixed path under the *same* `speed_ok` and terminal rules.
- Checks `J* <= J_baseline` whenever the candidate is admissible.  If the
  candidate is not representable, the tool reports the exact violated station
  and constraint (e.g. `"baseline path curvature exceeds kappa_max at station
  23"`); it never fabricates a feasible baseline.
- Emits an independently checkable Bellman certificate
  (`autocross.optimal.certificate`) accepted by the existing Python checker and
  by the compiled Lean checker `teachercheck`.

The Lean corollary `Corpus.Oracle.baseline_dominance` makes the guarantee
explicit: the certified optimum `v` is ≤ the cost of every admissible baseline
trajectory supplied under the same declaration.  The theorem is in the axiom
audit (431 theorems, no non-standard axioms).

### B. Realtime controller (`teams/line_opt.py`)

Two layers, with an explicit portfolio:

- **Path**: exact DP over the offset lattice, minimising the sum of squared
  second differences (minimum-curvature line) with corridor, obstacle and
  curvature constraints.  The DP is small enough to run every 50 ms.
- **Speed**: minimum-time forward/backward envelope for the chosen path under
  the friction circle, acceleration, braking and terminal-viability bounds.
- **Candidate set**: the path-DP optimum and the baseline's published path,
  each scored by the same speed optimiser and objective.  The cheaper candidate
  is selected.
- **Portfolio safety gate**: the baseline command is published instead when
  the baseline candidate is not admissible, when the selected candidate is not
  at least 2 % cheaper in the declared objective, when an obstacle is within
  30 m, when the measured cross-track/heading error is non-trivial, when the
  next 12 m are tight, or when the path leaves a 0.30 m body margin.  These are
  declared fallbacks, never labelled optimal.

The deployed controller therefore cannot do worse than certline in the
*measured* sense on the runs where the gate rejects the planner: it executes
certline.  This is the intended recovery from the `timed-1` failure, not a
claim that the planner has become competitive.

## 4. Measured results

### Public courses (`tools/bench.py`, bicycle, seed 0, measure mode)

| course | certline time / cones | lineopt portfolio time / cones | step p99 | planner-selected |
|---|---:|---:|---:|---:|
| hackathon_test | 47.53 / 0 | 47.51 / 0 | 19.5 ms | 27 / 983 |
| practice_basic | 29.18 / 0 | 29.18 / 0 | 17.5 ms | 0 / 617 |
| practice_loop | 85.05 / 0 | 85.27 / 0 | 19.4 ms | 28 / 1750 |
| practice_obstacles | 50.61 / 0 | 50.62 / 0 | 18.6 ms | 17 / 1046 |
| practice_technical | 35.73 / 0 | 35.73 / 0 | 20.3 ms | 0 / 748 |

Technical mean 68.7 / 75 (course mean); step p99 ≤ 20.3 ms; two overruns in the five-course
run, both isolated.

### Held-out generated courses (`datasets/optimal-full/tracks/test-course-000*`)

| course | certline time / cones | portfolio time / cones |
|---|---:|---:|
| test-course-0000 | 20.75 / 3 | 20.74 / 3 |
| test-course-0001 | 24.36 / 0 | 24.36 / 0 |
| test-course-0002 | 17.07 / 0 | 17.07 / 0 |
| test-course-0003 | 23.06 / 0 | 23.06 / 0 |

The portfolio reproduces the baseline's behaviour on this held-out sample,
including the three cone hits on `test-course-0000` (the baseline's own failure
mode; it is not presented as a clean controller there).

### Offline oracle samples (`practice_basic`, horizon 30, offsets 0.4 m)

| step | value (µs) | baseline admissible | `J* <= J_baseline` | Lean certificate |
|---:|---:|---|---|---|
| 5 | 3,995,214 | yes | yes | accepted |
| 10 | 3,804,079 | yes | yes | accepted |
| 20 | 3,447,549 | yes | yes | accepted |
| 40 | 2,998,302 | no (curvature at station 23) | n/a | accepted |
| 80 | infeasible | no (curvature at station 7) | n/a | n/a |

The oracle certificate at step 10 is
`experiments/bench/v3/certificates/oracle_000010.json`; `teachercheck` returns
`{"ok":true,"claimed":3308047,"horizon":24,...}`.

## 5. What is proven, measured, and not claimed

| statement | status |
|---|---|
| `solve_fast` equals the scalar exact DP (value and plan) | tested (`tests/test_lineopt.py`) |
| Offline oracle value is optimal over the declared lattice | Lean `teachercheck` accepted; Python checker accepted |
| `J* <= J_baseline` when the baseline candidate is admissible | Lean `Corpus.Oracle.baseline_dominance`; oracle rows above |
| Realtime path DP is optimal for its proxy objective | exact DP over its lattice; not Lean-certified |
| Realtime speed profile is minimum time for a fixed path | envelope iteration under declared bounds; not Lean-certified |
| Realtime controller is globally optimal for a physical vehicle | **not claimed** |
| Realtime controller beats certline | **not demonstrated**: it matches certline on all measured courses; planner selected on 0–3 % of steps |
| 50 ms budget | measured p99 ≤ 20.3 ms on the five-course run; 2 overruns; not proved |

## 6. Remaining work

1. The realtime gate is deliberately conservative; making the planner's
   predicted 8–10 % objective gain survive execution requires a tracker that
   models understeer/actuator lag at 7–9 m/s².  The recorded data set does not
   cover that regime densely enough to identify it; collect adversarial
   high-lateral transitions first.
2. The teacher model's departure speed cap (`b <= a_lat/kappa`) is the direct
   cause of the infeasible states; replacing it with a friction-ellipse
   feasibility condition is the next model change.  That change must be made in
   Python, the certificate checker and Lean together.
3. The offline oracle's offset lattice quantisation can make certline's smooth
   line violate `kappa_max`; a finer projection (or a declared continuous
   candidate) is needed before claiming baseline dominance on every state.

## 7. Reproduce (bounded, serial)

```bash
python -m pytest -q tests/test_lineopt.py
python tools/global_oracle.py experiments/diag/timed_basic/timed_practice_basic_s0 \
  --steps 10 --horizon 24 --offset-step 400 --out tmp/oracle --certificates
python - <<'PY'
import subprocess, pathlib
p = pathlib.Path('lean/.lake/build/bin/teachercheck.exe').resolve()
print(subprocess.run([str(p), 'tmp/oracle/oracle_000010.json'],
                     capture_output=True, text=True).stdout)
PY
python tools/structural_diagnostics.py experiments/diag/timed_basic/timed_practice_basic_s0 \
  --start 500 --limit 80 --out tmp/structural.json
```

Benchmarks are serial (`--jobs 1`); do not run several multi-course sweeps at
once on a laptop.
