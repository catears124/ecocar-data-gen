# Proof closure report

Branch `veridrive-proof-closure` (from `main` @ `5ba53e0`), local worktree, uncommitted.
Lean 4.35.0, core only. Companion documents: `docs/verification_matrix.md` (status of every link
in the chain), `docs/formal_guarantees.md` (exact statements), `docs/trust_boundary.md`.

## 1. Theorem inventory

| | `main` @ 5ba53e0 | this branch |
|---|---|---|
| theorems declared in `lean/Corpus/` | 116 | 370 |
| listed in `Audit.lean` (`#print axioms`) | 88 (28 unlisted, unaudited) | 370 |
| audit fails on an unlisted theorem | no | yes (`axiom_audit` reports `unlisted`) |
| axioms used | `propext`, `Classical.choice`, `Quot.sound` | same; no `sorry`, `native_decide`, new `axiom` or `opaque` |

Per module (this branch): Bellman 8, BellmanOpt 36, Cert 32, CertProofs 10, Closed 19,
Controller 33, ControllerLong 13, Dp 19, Joint 28, Levers 43, Mirror 11, Properties 7, Quant 6, Steer 13,
SteerClear 14, SteerFeas 12, SteerProofs 18, SteerRamp 9, SteerShape 28, Track 11.

Weak pre-existing statements found during the audit (vacuous hypotheses, tautologies, assumed
selection facts, toy recursions) are listed with their replacements at the end of
`docs/verification_matrix.md`. None was deleted; none is cited as evidence any more.

## 2. New high-level properties

- **Bellman (`BellmanOpt`)**: option-valued value function with explicit infeasibility. For any
  state type and finite horizon: value `none` ⇔ no admissible trajectory; finite value is a lower
  bound on every admissible trajectory's cost **and is attained**; executable extraction returns an
  optimal admissible trajectory; extracted policy optimal; uniqueness of the Bellman solution.
  Kernel counterexamples show the old `Bellman.value` returns 1 where the optimum is 6 and its greedy
  plan enters a dead end.
- **Joint offset/speed (`Joint`)**: optimality over all joint plans of the discrete reference model,
  exact infeasibility, extraction; every plan's speed sequence is `Cert.Feasible` for the caps of
  its own path (composition with the longitudinal certificate), hence pointwise below the
  certificate's greatest profile. Concrete instance with obstacle, curvature cap and braking all
  binding, cross-checked against an independent Python enumeration.
- **Lateral (`SteerProofs`, `SteerClear`, `SteerShape`, `SteerRamp`)**: hold-window supplier
  selection derived from the algorithm; hold-window clearance and nonnegative clearance of every
  `plan_ok` plan under decidable domain predicates; exact ramp bounds (slope, second difference,
  monotonicity); mirror equivariance under `noTies`; counterexamples for the `steer-1` selector and
  for each dropped hypothesis.
- **Longitudinal (`CertProofs`, `Quant`)**: unconditional four-way classification, exact
  infeasibility detection, nonnegativity, robustness under in-domain continuations, quantiser error
  bounds.
- **Controller (`Controller`, `ControllerLong`)**: clearance interval sound and complete for any
  obstacle memory; projection and publication inside every nonempty interval; speed decision
  greatest-safe for the controller's own caps; steering and pedal invariants of the actuation model.
- **Closed loop (`Closed`)**: receding-horizon persistence of the greatest-safe decision; delayed
  small-gain tracking bound; clearance of the actual position under the bound; braking and bias
  estimator facts.
- **Planning levers (`Levers`)**: the joint model for an arbitrary step relation with its Bellman
  facts; the lever theorem (relaxing constraints never worsens the declared optimum); monotonicity
  of the joint optimum in `a_lat`, `a_acc`, `a_brk`, `r_h`, `v_max`, `kappa_max`, in extra speed
  caps (gap rule, sidestep gate), and box ⊆ friction circle; kernel-checked toy instances showing
  which levers bind (cornering does, braking does not, the circle does) — cross-checked by an
  independent Python enumeration.

## 3. Bellman attainment and extraction

Proved, for the option-valued formulation, for every finite horizon and state type:
`BellmanOpt.valueO_attained`, `extract_spec`, `extract_none_iff`, `extracted_policy_optimal`.
For the old `Bellman.value`: equal to `valueO` only under `NoDeadEnds` (`valueO_eq_old`); in general
only a lower bound (`old_value_le_valueO`) with a kernel counterexample (`dead_mid_old_value`).

## 4. Joint composition

Complete **for the discrete reference model** `Joint`: `joint_value_spec`, `joint_infeasible_iff`,
`joint_extract_spec`, `plan_cert_feasible`, `plan_le_fpass`, `plan_start_le_bpass`. Not complete
for the submitted controller: the controller does not run the joint DP; its path is a heuristic
QP line inside the certified intervals, and only its speed decision for that path is proved
(`ControllerLong.decideLong_gs`). The float curvature estimate `np.gradient` is not proved to bound
the driven path's curvature (U).

## 5. Controller equivalence

- Integer decision layer: `decideLong_eq_cert` (equals `Cert.decide` when the caps coincide) and
  `decideLateral_*_eq` (equals `Steer.offs/curvs/steers` on the signed observation) are kernel
  theorems about the Lean models of the controller functions.
- Python ⇄ Lean: refinement evidence (R), not proof. `tests/test_controller_refinement.py` runs the
  Python functions `decide_long`, `decide_lateral`, `_spare_boxes`, `_project_plan` on random
  in-domain inputs and has the Lean kernel check the identical outputs (`decide`), and checks that a
  perturbed output is rejected. `tests/test_controller_conformance.py` checks the projection
  contract, signed lateral input, cap scale and sidestep caps.
- Not covered: the float racing-line QP, `np.gradient` curvature, tracker trigonometry, path
  projection `_path_errors`, delay roll-out, obstacle memory.

## 6. Closed-loop safety: model and assumptions

Proved in `Closed` for the first-order delayed lateral-error model
`D(e_{k+1} − e_k) = −P·e_{k−d} + D·f_k` with `|f_k| ≤ F` and the small-gain inequality, together
with `RealisedWithin`, `PreviewConsistent`, `NextValid` for the longitudinal side and a plan margin
of at least `M` inside `spareBox`: the error stays below `M`, the actual offset stays inside the
corridor and beyond every covering remembered obstacle's edge, steering demand and lateral
acceleration stay in budget, and every longitudinal decision along the run is greatest-safe.
Non-vacuity: concrete instance `v = 10 m/s`, `dt = 0.05 s`, `l_d = 8 m`, delay 2, `F = 6 mm/step`
⇒ `M = 124 mm`. The bicycle backend and CARLA are **not** shown to satisfy these assumptions,
and the runtime tracker (second-order path-error feedback) is not the first-order model.

## 7. Remaining obligations (U)

| obligation | attempted / missing ingredient |
|---|---|
| bicycle/CARLA plant satisfies `DelayedEquation` with the runtime tracker | needs a discretised nonlinear tyre model in Lean and a bound on its linearisation error; no real-number library (core only) |
| runtime margin `M ≤ line_shrink` | `line_shrink = 0.15 m` vs. concrete `M = 0.124 m` at one operating point only; no per-step runtime check of the small-gain condition |
| float `np.gradient` curvature bounds the driven path's curvature | float semantics not formalised; only the integer second difference is modelled (`pathCurv`) |
| racing-line QP optimality | not attempted; safety does not depend on it (final interval clamp) |
| `_predict_pose` delay compensation | kinematic roll-out; no model |
| obstacle memory `_remember` (matching tolerance, aging, side kept) | `spareBox` is proved for *any* memory, but correctness of the memory itself w.r.t. true obstacles is unproved |
| full `Steer.decide` mirror equivariance | list-level theorems proved under `noTies`; invariance of `wellFormed`/corridor validity under mirroring not proved; tie counterexample `mirror_tie_counterexample` |
| per-sample `holdTargetsOK` / `clearanceDomain` in `corpuscheck` | both decidable; not wired into the compiled checker, so clean corpus labels are not shown to be inside the clearance theorem's domain |
| amplitude bound `|offset| ≤ Cert.stationY`, curvature closure `curv ≤ Cert.kbarOf` | not attempted |
| Python ⇄ Lean equivalence for all inputs | only sampled refinement evidence |
| physical constants (`mu`, budgets, `max_steer`, wheelbase, domain radius, time table) | trust items by design; never claimed proved |
| `v1.0.1` corpus re-validation | it carries `steer-1` labels; must be regenerated under `steer-2` |

## 8. Commands run on this branch (final state)

| command | result |
|---|---|
| `cd lean && lake build` | `Build completed successfully (48 jobs)` |
| `axiom_audit(lake_path())` | `ok: True, theorems: 370, missing: 0, unlisted: 0, non_standard: 0` |
| `REQUIRE_LEAN=1 python -m pytest -q -p no:cacheprovider --basetemp=…` | **203 passed** (baseline on `main`: 102 passed, 1 failed, 77 errors from the temp-dir permission) |
| `python -m autocross data-manifest --profile smoke` ×2 | byte-identical manifests |
| `data-check-manifest` | `manifest OK` |
| `data-generate --backend bicycle --workers 4` | 22 runs, realised coverage 18/18; run failures `boundary_cone_hit: 4`, `obstacle_cone_hit: 2` (excluded, not deleted) |
| `data-validate` | `OK: 0 errors, 201 warnings` |
| `data-lean` | all checks `[ok]`; axiom audit OK (370); kernel witnesses 2679/2679 clean samples; 4722 lateral and 4722 longitudinal certificates recomputed, 0 mismatches |
| `python -m autocross selftest -c teams/certline.py` | 5/5 finished, 0 cones, 0 off, 0 gates, mean **71.6 / 75**, step p99 ≤ 4.0 ms (main: 58.1) |
| `selftest -c controllers/baseline_pure_pursuit.py` | mean 62.3 / 75, 7 cones |
| `selftest --carla` | **not run**: no CARLA server available |

## 9. Runtime behaviour changes

Data generation / certificates:
- `steer-2`: ramp-out obstacle weight capped at 999 (`Steer.wOb`, `autocross/datagen/steer.py`,
  `teams/certline.py::_w_ob`); model id `steer-1` → `steer-2` in datagen, `Main.lean` and tests.
- `Main.lean` delegates `qfix` to `Corpus.Quant`; `lean_bridge.axiom_audit` reports unlisted
  theorems and parses primed names.

Controller (`teams/certline.py`):
- publication order: cap to `±max_offset` first, certified interval clamp last, in every mode;
  empty intervals are kept as `lo > hi` and flagged (`cert_clear`, `uncertified_stations`);
  centreline mode 4 is explicit uncertified recovery;
- `decide_long` checks a negative cap (invalid) before the no-gap branch, as `Cert.decide` does;
- sidestep speed caps `v² ≤ a_lat·D²/(4Δy)` passed to `decide_long`;
- tracker: path-error state feedback at the predicted pose (`trk_mode = 1`, `trk_omega = 3.25`,
  `trk_zeta = 0.9`, `trk_lag = 0.10`); legacy pure-pursuit blend kept as `trk_mode = 0`
  (with the `k_obs_track` obstacle-hold correction);
- racing line anchored on the previous plan in course coordinates (`line_anchor = 1`,
  `line_w_prev = 1.0`, previously 3.0 in vehicle-frame stations);
- `brk_frac` 0.7071 → 0.50 (declared braking budget below the friction box);
- racing-line weights `line_w_center` 0.002 → 0.001, `line_w_prev` → 0.7; gap-speed rule
  `gap_base` 5 → 7 m/s, `gap_per_m` 6 → 8; mid-corner throttle cap `thr_reserve = 0.5`;
- `qfix` computes `floor(M·scale·10^E + 1/2)` from the decimal repr with integer arithmetic instead
  of `Fraction` (identical on 599,954 differential cases; step time p99 ≈ 6 ms → ≈ 4 ms);
- unused `_menger_curvature` removed. Tried and rejected after measurement: a friction-circle braking
  envelope with higher `brk_frac`, `lat_frac` ≥ 0.75, a parallel-curve curvature factor, terminal
  relaxation once the finish is visible;
- debug fields `e_y`, `e_psi`, `cert_clear`, `uncertified_stations`, `obs_track`.

Tests: `braking_with_rounding` retargeted (v0 20291 → 20171) after the domain constants tightened;
`metrics.compute.errors` key; `lateral_corpus` fixture uses the generated corpus; new tests listed
in `docs/verification_matrix.md` and `tests/test_controller_certline.py`
(`test_tracker_steers_back_toward_the_published_path`, `test_plan_start_follows_previous_plan_in_course_frame`).

## 10. Repository state

During the work nothing was committed, pushed or released. At the user's explicit request at the
end, the result was committed on top of `origin/main` (`4e649a5`) and pushed to `main`, and the
release `veridrive-v2.0` was created with the controller and the race videos. The branch's former
base `5ba53e0` (a sibling of `4e649a5` with an older controller variant) is kept as the local
branch `backup/pre-proof-closure-base`; its `line_smooth` controller change is superseded by this
controller (66.1 vs 71.8 technical points on `gen_open_13`).
