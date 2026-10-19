# Proof-carrying optimal teacher

This document describes the offline chain

```text
observation -> exact finite model -> sparse dynamic program -> certificate
            -> independent checkers -> label
```

The claim is deliberately finite: the teacher is optimal for the declared preview
model, grids, horizon, constraints, and integer objective. It is **not** a claim of
optimality for a continuous vehicle, a physical vehicle, CARLA, or an entire course.

## Model

`autocross/optimal/model.py` constructs a model from the quantised policy
observation. The preview contains stations `s`, road curvature `k`, left/right
widths, obstacle triples, and the current speed. Units are integer millimetres,
`mm^2/s^2` for speed squared, and integer micro-curvature. The model also carries
declared constants (`consts`), offset and speed-squared grids, a finite horizon,
and the initial state.

`model_from_observation` declares the initial stencil history straight:
`previous = current`, both set to the nearest declared offset-grid point.
Heading error is not used to invent a previous-station offset. That earlier
construction could make the finite model infeasible under a small tracking
error because its first-step curvature demand exceeded `kappa_max`; treating
that artefact as a vehicle property would be unsound. Tracking history remains
outside this finite declaration.

A state is

```text
(station i, previous offset p, current offset x, speed squared b).
```

Offsets are integer millimetres. A transition `s -> t` advances one station,
sets `t.previous = s.current`, chooses `t.current` from the offset grid, and
chooses `t.b` from the speed-squared grid. A transition is admissible only when
all of the following hold:

* the destination offset is in the declared corridor
  `lo_i <= t.current <= hi_i`, where the corridor accounts for vehicle
  half-width and the declared margin;
* at the destination station, the offset is outside every obstacle whose
  longitudinal hold window contains that station. Obstacle lateral extents are
  inflated by the declared vehicle half-width plus margin;
* the integer path-curvature demand
  `abs(k_i) + ceil(10^9 * abs(p - 2*x + n) / step^2)` is at most
  `kappa_max`, where `n = t.current`;
* the departure speed squared is at most the curvature cap
  `min(v_max^2, floor(a_lat * 10^9 / curvature))` (with `v_max^2` when the
  curvature is non-positive);
* both speed-squared values are non-negative, the arrival value is on the
  declared grid, and they are not both zero;
* acceleration and braking budgets hold:
  `w <= b + 2*a_acc*d` and `b <= w + 2*a_brk*d`, where `d` is the station
  gap.

The terminal condition accepts a state exactly when

```text
0 <= b <= min(v_max^2, a_lat * r_h).
```

That terminal cap is the only information supplied about the road beyond the
preview. The model reads no future course or continuous geometry beyond the
quantised preview and this declared cap.

For a segment of length `d` and endpoint speed-squared values `b,w`, the declared
charge is

```text
cdiv(2*d*1_000_000, isqrt(b) + isqrt(w))
```

where `cdiv` rounds up. The objective is the sum of these integer charges. It is
the constant-acceleration expression using floored integer speeds, expressed in
microseconds and rounded up. This is an objective of the finite model; it is not
an assertion that the resulting number is measured physical traversal time.

## Sparse DP and tie-breaking

`autocross/optimal/solver.py::forward_universe` starts with only the observed
start state at station zero. At each station it takes the union of all
admissible successors of the preceding sparse set. `solve` then performs
backward Bellman induction over those sets, assigning `None` to states with no
admissible terminal continuation.

The sparse restriction is exact for the start problem. By induction on station,
every admissible plan from the start is in the universe: the start is included,
and an admissible next state is included because the next set is the union of
all recomputed successor sets. Thus no start-to-terminal plan is removed by the
sparse representation. The certificate checker independently recomputes every
successor and checks closure (`closedUnder` in Lean), so an omitted successor
cannot silently change a Bellman minimum.

Actions are visited in declared offset order (outer loop), then declared speed
order (inner loop). A strict improvement replaces the incumbent; an equal-cost
action does not. The exported plan is therefore the first optimal plan in that
order. The certificate records this as `first_min_offset_then_speed`, and both
checkers verify that each exported step is the first optimal successor.

The exhaustive `enumerate_plans` function is a separate small-instance reference
used by tests; it is not the production solver or a physical simulator.

## Certificate schema

`certificate_from_solution` emits JSON with the following metadata and checker
fields:

* `schema` (`autocross.optimal.teacher_certificate`) and `schema_version`;
* `instance`, containing the declared model, constants, quantised preview,
  grids, horizon, start state, and terminal cap;
* `claimed`, the optional start value (`null` means infeasible);
* `tables`, with one table per station. A row is
  `[station, previous, current, b, value]`, where `value` is an integer or
  `null`;
* `states`, the state universe per station, and Lean-facing `values`, the
  corresponding option-valued rows;
* `plan`, the state sequence (empty for an infeasible claim), and `plan_cost`;
* `tie_break`, `cost_definition`, and `information`, which describe the
  deterministic order, integer cost formula, preview-only information, and
  terminal-cap treatment;
* `consts`, `q`, `offsets`, `speeds`, `roots`, `horizon`, `start`, `terminalCap`,
  and `planCost`, the fields decoded by `Corpus.Teacher.Cert`;
* `certificate_sha256`, a hash of the canonical JSON representation.

`roots` supplies a declared integer square root for the start and every speed-grid
value. The Lean checker verifies the inequalities that make each supplied root
unique; it need not trust an opaque square-root computation.

## Python checker

`autocross.optimal.certificate.check_certificate` rebuilds the model from the
serialized instance and checks:

1. schema/version and root-table consistency;
2. table count, state/value shape, duplicate rows or states, declared-universe
   membership, and station-zero equality with the start;
3. closure under every recomputed admissible successor;
4. every terminal row against the terminal condition;
5. every nonterminal row against the minimum of `cost + next value` over the
   recomputed successor set, including dead ends;
6. equality of the start row and `claimed`;
7. for a feasible claim, start, length, admissible transitions, terminal
   feasibility, summed plan cost, `plan_cost`, and the first-minimum tie rule;
8. for an infeasible claim, absence of an exported plan.

The Python checker is an independent executable translation of the model and
certificate rules. It is not the formal proof; it supplies a separate
implementation check before a certificate is persisted.

## Labels and corpus records

`autocross/optimal/teachers.py::optimal_label` is the label boundary. It
quantises the observation, builds the finite model, solves it, emits the
certificate, and runs `check_certificate` before returning. A feasible label
contains the first selected offset/speed action, the declared `claimed` cost,
the state `plan`, the certificate, `method="exact-dp"`, and `optimal=True`.
An out-of-domain or infeasible model returns an explicit non-feasible label
instead of fabricating an action.

`autocross/datagen/optimal_corpus.py` writes the serialised examples and, for
the exact `optimal` condition, writes the checked certificate under
`certificates/optimal/` and records its relative path and SHA-256 in the label.
Those labels remain labels for the finite teacher model; they are not physical
expert demonstrations or a proof of controller optimality.

## Lean checker and theorems

`lean/Corpus/Teacher.lean` defines `TState`, `Cert`, the transition relation
`stepOKT`, actions `actionsT`, terminal condition `terminalT`, and the decidable
predicate `CheckCert`. `checkCert` is `decide (CheckCert cert)`. `CheckCert`
rechecks the model domain and terminal cap, horizon and start, grid membership and
non-negativity, root certificates, table shape and uniqueness, sparse-universe
closure, terminal rows, every Bellman recurrence, the claimed start value, and
plan admissibility/cost/tie-breaking.

The exact soundness results are:

* `Corpus.Teacher.checkCert_sound`: a boolean result `checkCert cert = true`
  produces a proof of `CheckCert cert`.
* `Corpus.Teacher.table_value_cert`: for every state row accepted at station
  `i`, its table value equals the Bellman `valueO` for the remaining replicated
  action stages.
* `Corpus.Teacher.certificate_value`: the accepted claimed value equals the
  Bellman optimum from the encoded start state.
* `Corpus.Teacher.certificate_lower_bound`: when the claim is `some v`, every
  declarative `PlanOKT` plan has declared cost at least `v`.
* `Corpus.Teacher.certificate_plan`: a non-`none` claim has an exported plan
  beginning at the start, satisfying `PlanOKT`, reaching the horizon, and having
  declared cost equal to the claim.
* `Corpus.Teacher.certificate_infeasible`: a `none` claim rules out every
  declarative plan of the encoded horizon model.
* `Corpus.Teacher.certificate_optimal`: packages the preceding conclusions in
  `TeacherGuarantee`.
* `Corpus.Teacher.actionsT_complete` and `Corpus.Teacher.sparse_value` support
  successor completeness and the sparse Bellman induction used by the checker.
* `Corpus.Teacher.timeCost_eq_isqrt`: under the supplied `rootOK` witnesses,
  the declared charge equals the corresponding integer-square-root expression.

These theorems establish optimality only relative to the finite definitions in
`Corpus.Teacher`. They do not establish continuous-vehicle optimality, CARLA
optimality, tracking safety, or whole-course optimality.

## Running and reproducing

From the repository root, build the Lean executable from the Lean project:

```text
cd lean
lake build teachercheck
```

The resulting executable accepts one certificate path:

```text
lean/.lake/build/bin/teachercheck.exe CERT.json
```

It prints a compact JSON report containing `ok`, `claimed`, `horizon`, `states`,
and `model`; it exits zero exactly when the certificate is accepted, and nonzero
for malformed, rejected, or incorrectly used input. The Python corpus smoke
command (the generator requires `--out`) is:

```text
python -m autocross.datagen.optimal_corpus --out artifacts/optimal_teacher_smoke --smoke
```

The focused teacher tests are:

```text
pytest tests/test_teacher_optimal.py tests/test_teacher_parity.py
```

The first file checks the solver, exhaustive/reference agreement, certificate
mutations, tie-breaking, and objective arithmetic. The parity file sends Python
certificates to the compiled `teachercheck` executable and checks rejection of a
corrupted certificate. A full repository run is `pytest` when the surrounding
optional dependencies are available.

For the full IL comparison, the corpus and held-out track YAMLs were
reproduced as follows. First generate the corpus:

```text
python -m autocross.datagen.optimal_corpus --out datasets/optimal-full --train-courses 24 --val-courses 6 --test-courses 8 --samples-per-course 30 --seed 20261019
```

Then materialize each held-out YAML from its manifest course `spec` object with
`autocross.track.save`:

```text
python -c "from pathlib import Path; import json; from autocross.track import save; r=Path('datasets/optimal-full'); m=json.loads((r/'manifest.json').read_text()); (r/'tracks').mkdir(parents=True, exist_ok=True); [save(c['spec'], r/'tracks'/(str(c['scenario_id'])+'.yaml')) for c in m['courses']]"
```

The training, evaluation, full experiment, and reanalysis commands were:

```text
python -m experiments.il.train_bc --corpus datasets/optimal-full --condition optimal --budget 100 --seed 0 --out experiments/il/results/weights/optimal_b100_s0.npz
python -m experiments.il.evaluate --corpus datasets/optimal-full --weights experiments/il/results/weights --tracks datasets/optimal-full/tracks --conditions optimal --budgets 100 --seeds 0 --output experiments/il/results/evaluate-optimal.json
python -m experiments.il.run_experiment --corpus datasets/optimal-full --tracks datasets/optimal-full/tracks --budgets 100 300 720 --out experiments/il/results
python -m experiments.il.reanalyze --results experiments/il/results/results.json
```

The corpus had 24 train, 6 validation, and 8 test courses, with 1,140 examples
per condition. The comparison used budgets 100, 300, and 720, three seeds, and
four held-out courses, for 36 runs and zero infrastructure failures. On the
bicycle backend, the observed aggregate was 12/12 finishes for `optimal`
versus 8/12 for `heuristic` (42 off-course events); `approx` also finished
12/12 but was slower and incurred more cone hits. These are simulator
observations only and make no CARLA claim.

Exact offline-controller selftests on the bicycle backend were run repeatedly.
Observed completions ranged from 2/5 to 4/5 courses after explicit
uncertified recovery (the best run completed `hackathon_test`,
`practice_basic`, `practice_obstacles`, and `practice_technical`; the latest
run completed `practice_basic` and `practice_technical`). Step p99 ranged from
about 20 ms to about 70 ms across runs, so exact offline mode is not a 50 ms
guarantee and is not claimed to be real-time.

Realtime mode uses a reduced horizon/grid. It is exact for that reduced
declaration, but reports `realtime_exact=False`,
`fallback_reason='realtime reduced horizon/grid (not full optimum)'`, and
`certified=False` relative to the requested full model. The exact full-model
optimum is not preserved in realtime.

## Integration and observed runs

The exact teacher is an offline labeler. Runtime integration may use the
reduced realtime declaration above, or may enter explicit uncertified recovery;
neither path upgrades the runtime controller to the full finite-model optimum.

## Complexity and runtime

Let `H` be the horizon, `U_i` the sparse universe at station `i`, and `A(s)`
the recomputed admissible action list. Forward reachability costs action
enumeration over `sum_i |U_i|`; backward induction costs
`sum_{i < H} sum_{s in U_i} |A(s)|` time and stores
`O(sum_i |U_i|)` values and choices. With offset-grid size `O` and speed-grid
size `B`, one state's unfiltered action enumeration is bounded by `O*B`, and a
full noninitial layer is bounded by `O^2*B`; sparse reachability can be much
smaller. Certificate checking has the same successor-enumeration order and adds
JSON/table storage. Runtime therefore grows with horizon and declared grids; no
runtime benchmark is part of the correctness claim.

## Claim status

| claim | status | scope/evidence |
|---|---|---|
| Bellman value of an accepted certificate equals the encoded finite-model optimum | **formally proven** | `checkCert_sound`, `table_value_cert`, `certificate_value` in `Corpus.Teacher` |
| Every accepted feasible plan is admissible and reaches the declared horizon | **formally proven** | `certificate_plan`, with `PlanOKT` and terminal checks |
| Every accepted infeasible claim has no finite-model plan | **formally proven** | `certificate_infeasible` |
| Python solver output passes the independent Python checker | **computationally checked** | `tests/test_teacher_optimal.py`, including adversarial mutations |
| Python certificates are accepted by compiled Lean and corrupted values are rejected | **computationally checked** | `tests/test_teacher_parity.py` |
| Declared integer cost is the stated floored-speed, ceiling formula | **formally proven** | Lean `timeCost_eq_isqrt`; Python arithmetic tests |
| Declared integer cost is physical traversal time | **unresolved** | It is a quantised constant-acceleration proxy, not a plant theorem |
| Quantised observation, grids, constants, horizon, obstacle inflation, and terminal cap are the intended model | **assumed** | Explicit modelling choices; not learned or physically identified by the theorem |
| Runtime tracking/actuation follows every certified transition | **unresolved** | Tracking and actuation are outside this teacher certificate |
| Continuous, CARLA, or full-course global optimality | **unresolved** | Not claimed by this finite preview construction |
| Label provenance for generated records | **computationally checked** | Record carries the generated certificate; Python and, where run, Lean checker results are reproducible |
| Real-world performance of labels or downstream controller | **empirically observed** | Bicycle-only IL comparison: 36 runs, 0 infrastructure failures, `optimal` 12/12 finishes, `heuristic` 8/12 with 42 off-course events, and `approx` 12/12 but slower/more cone hits; no CARLA claim |
