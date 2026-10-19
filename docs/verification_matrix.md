# Verification matrix

Every runtime operation of the chain **Observation → Quantization → Lateral planning → Path
selection → Curvature bounds → Speed optimization → Tracking → Actuation → State transition**,
mapped to its specification, its Lean evidence and its status.

Status categories (never mixed):

| code | meaning |
|---|---|
| **K** | kernel-checked Lean theorem about the Lean definition (standard axioms only, listed in `lean/Corpus/Audit.lean`) |
| **R** | machine-checked refinement evidence connecting the executable code to a Lean definition: the Lean kernel evaluates the specification on inputs produced by the Python code (`by decide` theorems), or per-sample kernel witnesses for corpus records. Evidence on the checked inputs only — **not** a proof about arbitrary Python executions |
| **A** | explicit, narrowly scoped assumption (named predicate in a theorem, or declared trust item), with its consequences proved |
| **U** | unresolved obligation: counterexample, failed proof attempt, or missing modelling ingredient |

Inventory: `lean/Corpus/Audit.lean` lists **every** `theorem` declared in `lean/Corpus/*.lean`
(373 at the time of writing); `autocross.datagen.lean_bridge.axiom_audit` fails if one is
unlisted (`unlisted`), not reported (`missing`) or uses a non-standard axiom
(`non_standard`); `tests/test_longitudinal_proofs.py` checks the inventory without Lean.

## 1. Observation and quantization

| operation | specification | evidence | assumptions | implementation | status | open obligations / counterexamples |
|---|---|---|---|---|---|---|
| decimal quantisation `qfix(x, scale)` (corpus checker) | `q = ⌊x·scale + ½⌋` for the exact decimal `x = m·10⁻ᵉ` | `Quant.qfix_floor_bounds`, `qfix_eq_floor`, `qfix_error_bound` (`|q − x·scale| ≤ ½`), `qfix_monotone_mantissa`, `qfix_int_exact` | JSON text is the shortest repr of the float (Lean's JSON parser) | `lean/Main.lean` → `Corpus.Quant.qfix`; Python `cert.qfix` (`Fraction(repr(x))`) | **K** (Lean side), **R** (Python = Lean via per-sample recomputation in `corpuscheck` and `tests/test_certification.py::test_qfix_is_exact_decimal_round_half_up`) | Python `Fraction(repr(float))` vs JSON decimal text equality is a CPython property (trust) |
| malformed observation → `invalid_input` | `Cert.decide` invalid iff ¬(wellFormed ∧ constsOK ∧ capsNonneg) | `CertProofs.decide_classification` | — | `cert.quantise*`, `certline.quantise` (returns `None` on NaN/inf/None/bool, short preview) | **K** + **R** (`tests/test_certification.py` edge cases kernel-checked; controller full-brake tests) | — |
| controller input guard | invalid/short/None road ⇒ full brake, no exception | `Controller.invalid_full_brake` (integer pedal model) | integer model of the float pedal code | `certline._step` | **R** (`tests/test_controller_certline.py::test_invalid_road_full_brake`) | float pipeline not formalised |

## 2. Lateral planning (`steer-2`)

| operation | specification | evidence | assumptions | implementation | status | open / counterexamples |
|---|---|---|---|---|---|---|
| obstacle weight and supplier selection | in an obstacle's hold window, that obstacle supplies the plan | `SteerProofs.wOb_hold`, `wOb_le_999_outside_hold`, `go_stable`, `go_prefix`, `hsep_of_pairwise`, `rawPlanAt_hold` | `pairwiseDisjoint` holds (part of `wellFormed`), `0 < ramp_in` | `steer.py::rawPlanAt`, `certline._raw_off` | **K** | **fixed defect** (`steer-1 → steer-2`): `SteerClear.old_selector_counterexample` — the old ramp-out weight floored to 1000 and kept the wrong supplier; offset `-1250` overlapped obstacle B while every certified clearance was ≥ 0 |
| gap side / amplitude | chosen side has the wider gap; target beyond the obstacle edge | `SteerClear.sideAmp_edge`, `SteerFeas.target_in_corridor_left/right` | obstacle inside the road at its nearest station; even vehicle width | `sideAmp`, `_side_amp` | **K** | tie rule is asymmetric (see mirror) |
| smoothstep ramp | `ramp(0)=0`, `ramp(L)=A`, `0 ≤ ramp ≤ A`, monotone, `ramp 1000 x L ≤ 999` for `x<L`; slope `2L·Δramp < 3Ah + 2L`; `L²·|Δ²ramp| < 6Ah² + 2L²` | `SteerRamp.ramp_zero/ramp_full/ramp_bounds/ramp_mono/ramp_1000_lt`, `SteerShape.ramp_slope_bound`, `ramp_second_bound`, `n_diff_bound` | `0 ≤ A`, `0 < L` (and `0 ≤ x ≤ L` for the second difference) | `ramp`, `_ramp` | **K** | — |
| offset bounds / corridor | every published `steer-2` offset in `[lo_i, hi_i]` when the corridor is nonempty | `SteerFeas.clamp_in_corridor`, `SteerProofs.offs_in_corridor` | `lo_i ≤ hi_i` | `offAt`, `_clamp_at` | **K** | empty corridor ⇒ midpoint (no claim) |
| obstacle clearance through the hold region | `plan_ok` ⇒ every station: `clearAt ≥ 0`; hold stations: offset beyond the edge | `SteerClear.plan_ok_hold_clearance`, `plan_ok_clearance`, `plan_ok_clears_nonneg` | `clearanceDomain` (wellFormed, even width, `0<ramp_in`, nonempty corridors), `holdTargetsOK` (obstacle in road, target fits corridor at each hold station) — both decidable | `steer.decide` | **K** (non-vacuity: `ab_hypotheses_hold`) | `narrowing_counterexample`: without `holdTargetsOK`, `plan_ok` holds but the clamped offset overlaps the obstacle and the certificate records negative clearance |
| ramp entry / exit | outside hold windows only corridor clearance is claimed | `SteerProofs.outside_hold_clearance`, `SteerClear.plan_ok_clearance` | obstacle footprint (inflated by car half-length) lies within `[d − hold, d + hold]`; car is on plan at hold entry | — | **A** + **K** | footprint containment is a geometric trust item |
| mirror equivariance | mirrored input ⇒ negated offsets, flipped sides, equal amps, equal clearance, curvature and steering | `SteerShape.sideAmp_mirror`, `rawPlanAt_mirror`, `offAt_mirror`, `offs_mirror`, `clearAt_mirror`, `curvs_mirror`, `sides_amps_mirror` | `noTies` (no obstacle with equal gaps), nonempty corridors | `steer.py`, `certline` | **K** | `mirror_tie_counterexample` (tie rule not equivariant); full `decide` equivariance (that `wellFormed` is mirror-invariant) not proved — **U** |
| curvature demand | `curv_i = |k_i| + 1000·|Δ²off_i|` | `Steer.curvAt` (definition); single-ramp bound via `ramp_second_bound` | — | `curvs`, `curv_bound_micro` | **K** | `SteerClear.supplier_switch_curvature`: supplier switching between opposing obstacles makes `Δ² = 2500 mm` (demand 2.5·10⁶ µ/m > `kappa_max`) — no general smoothness bound exists for `steer-2` offsets |
| steering feed-forward | `steerAt ∈ [−1000, 1000]`; exact (unsaturated) iff `0 ≤ κ`, `1000κ ≤ 999·κ_max` | `SteerProofs.steerAt_range`, `steerAt_unsaturated_of_condition` | `0 < κ_max` | `steerAt` | **K** | — |
| feasible lateral plan exists | `plan_ok ⇔ wellFormed ∧ ¬hasNoGap`; with the domain predicates, `plan_ok` ⇒ clear | `Steer.decide_plan_ok_iff`, `SteerProofs.plan_ok_iff_wellFormed_noGap`, `SteerClear.plan_ok_clearance` | as above | | **K** | — |

## 3. Joint lateral/longitudinal planning (reference optimizer)

| operation | specification | evidence | assumptions | implementation | status | open / counterexamples |
|---|---|---|---|---|---|---|
| Bellman recursion with infeasibility | `valueO = none ⇔` no admissible trajectory | `BellmanOpt.valueO_none_iff` | — | Lean only (offline reference) | **K** | **fixed defect**: `dead_mid_old_value`/`dead_mid_old_greedy_dead_end` — old `bestCost [] = 0` value 1 vs true optimum 6; old greedy picks a dead end |
| lower bound / attainment / extraction / uniqueness | all admissible trajectories (and policies) cost ≥ value; value attained; executable extraction admissible, one action per stage, cost = value; unique Bellman solution | `valueO_le_cost`, `valueO_attained`, `extract_spec`, `extract_none_iff`, `trajOK_length`, `policy_lower_bound`, `extracted_policy_optimal`, `optimal_value_unique`, `bellmanO_unique`, `valueFns_bellman` | arbitrary finite horizon, arbitrary state type | | **K** | — |
| relation to the old formulation | old value ≤ true optimum; equal iff no dead ends | `old_value_le_cost`, `old_value_le_valueO`, `valueO_eq_old` | `NoDeadEnds` for equality | | **K** | old `Bellman.no_policy_beats` needs actions at **all** states (vacuous for horizon-limited models); old `Dp.dp_optimal`/`speed_dp_optimal` are lower bounds only |
| joint offset/speed model | steps: station+1, history carried, grid, corridor, `clearAt ≥ 0`, `pathCurv ≤ κ_max`, departure speed under the stencil's cap, `Cert.Feasible` acceleration/braking; terminal `≤ lastCap` | `Joint.mem_actions`, `actions_complete`, `traj_plan`, `plan_traj`, `plan_steps`, `plan_progress` | declared offset grid, declared speed² grid (mm²/s²), declared cost table | Lean only | **K** | **fixed model defects** of `Dp.vModel`: `old_speed_cap_inert` (bucket units vs mm²/s²), `old_stencil_undercount` (lateral stencil 1000× too small), arrival-speed/departure-cap index mismatch |
| joint optimality | DP value = min declared cost over the declarative plan set; attained; extraction optimal; none ⇔ no plan | `joint_value_spec`, `joint_extract_spec`, `joint_infeasible_iff` | as above | | **K** | concrete instance: `toy_value`, `toy_value_eq_enumeration`, `toy_extract`, `toy_curvature_binds`, `toy_infeasible`; Python enumeration agrees (`tests/test_joint_dp.py`, 5 start speeds) |
| composition with longitudinal certificate | every joint plan's speeds are `Cert.Feasible` for the chosen path's caps; ≤ the certificate's greatest profile; start within `bpass` | `plan_cert_feasible`, `plan_le_fpass`, `plan_start_le_bpass`, `pathCap_le_capAt` | — | | **K** | — |
| curvature-based speed restriction | `b·κ ≤ a_lat·10⁹` for the declared curvature; physical form if true curvature ≤ declared | `pathCap_lateral_sound`, `pathCap_physical_sound` | true path curvature ≤ declared bound (geometry + tracking) | | **K** + **A** | — |
| runtime vs joint DP | the controller does **not** run the joint DP; it runs `steer-2` + racing-line heuristic + `decideLong` on the published path's caps | `ControllerLong.decideLong_gs` (speed plan optimal for the heuristic path's caps) | — | `certline` | **U** (no joint-optimality claim for the runtime) | heuristic path is not claimed optimal |
| planning levers (generic) | relaxing the step relation (on reachable states) and the terminal cap never worsens the optimum and keeps feasibility | `Levers.valueP_lower`, `valueP_attained`, `valueP_none_iff` (joint model for any step relation; `Joint` is the instance: `joint_value_eq`, `joint_plan_iff`), `lever_value_le`, `planP_mono` | `Shaped` step relation | Lean only | **K** | — |
| budget levers `a_lat`, `a_acc`, `a_brk`, `r_h`, `v_max`, `κ_max` | raising any of them (singly or jointly) lowers the joint optimum or leaves it unchanged | `budget_lever`, `cornering_lever`, `braking_lever`, `accel_lever`, `horizon_lever` (via `capAt_raise`, `pathCap_raise`, `terminalCap_raise`, `stepOK_raise`) | `0 ≤ a_lat, r_h, v_max`; nonnegative segment lengths (`GapsNonneg`) | runtime `lat_frac`, `acc_frac`, `brk_frac`, `horizon_r` | **K** | instances: `toy_cornering_binds` (5 m/s² infeasible, 7 m/s² no change, 8 m/s² 82491 → 51417), `toy_braking_slack` (3 → 9 m/s²: no change) |
| extra speed caps (gap rule, sidestep gate) | an extra cap never lowers the optimum; monotone in the cap | `extra_caps_cost`, `extra_caps_mono` | — | `gap_*`, `_sidestep_caps` | **K** | `toy_extra_cap_infeasible` |
| box vs friction circle | if `a_lat² + max(a_acc,a_brk)² ≤ A²`, every certified box step (from `b ≥ 0`) is a circle step; the circle optimum ≤ the certified optimum | `box_step_circle`, `circle_value_le_box` | fit condition | not used at runtime (measured: circle braking envelope worse on the bicycle backend) | **K** | `toy_circle_gain` (radius 7: 82491 → 74074); circle plans are **not** certified by `Cert` |
| what the levers do on the real plant | measured lap time / cones vs budget | bicycle-backend sweeps (`docs/proof_closure_report.md` §9) | — | `certline` defaults | **R** (measurement) | the theorems order *declared* optima; they say nothing about tracking error or cone hits at higher budgets — **U** |

## 4. Path selection in the submitted controller

| operation | specification | evidence | assumptions | implementation | status | open / counterexamples |
|---|---|---|---|---|---|---|
| certified clearance interval `_spare_boxes` | `x ∈ [lo, hi]` ⇔ inside corridor and beyond every covering remembered obstacle's edge | `Controller.spareBox_sound`, `spareBox_complete`, `spareBox_empty_iff` | arbitrary memory list | `certline._spare_boxes` | **K** + **R** (`tests/test_controller_refinement.py`, kernel-evaluated on random memories) | memory correctness (matching/aging of obstacles) not formalised — **A** |
| shrink / empty boxes | shrunk box ⊆ box; empty box flagged | `planBox_inside`, `planBox_empty_flag` | — | `_plan_boxes` (now returns an `uncertified` mask instead of pinning to an uncleared steer point) | **K** | — |
| rate-limit projection `_project_plan` | output in every nonempty interval for any input and ≥1 iteration | `projectPass_boxed`, `projectPlan_boxed` | `≥ 2` stations, all intervals nonempty | `_project_plan` | **K** + **R** (kernel-evaluated on random integer data) | floats: exact for integer mm data (< 2⁵³); general float inputs rely on IEEE min/max exactness (**A**) |
| publication (all modes) | absolute cap first, interval clamp LAST ⇒ published offset in every nonempty interval | `publishOffset_inside`; test `test_publish_interval_wins_after_absolute_clip`, `test_plan_stays_inside_certified_interval` (no slack) | — | `_publish_offsets` | **K** + **R** | **fixed defect**: the old order clipped to ±`max_offset` AFTER the interval, e.g. centred cone needing 1.88 m was published at 1.5 m |
| fallback modes | modes 1–3 published through the interval clamp; mode 4 (centreline) and any station with an empty interval are **uncertified recovery** (`debug.cert_clear = 0`, `uncertified_stations`) | publication theorem above | — | `_step` | **K** for clamped modes, **U/A** for recovery | old mode 3/4 published raw/centreline offsets outside the interval |
| racing-line QP | minimum-curvature objective | none (float NumPy active set) | — | `_box_qp`, `_racing_line` | **U** (optimality unverified; safety does not depend on it because of the final clamp) | — |

## 5. Speed optimization (longitudinal)

| operation | specification | evidence | assumptions | implementation | status | open / counterexamples |
|---|---|---|---|---|---|---|
| backward pass | upper envelope of every feasible profile | `Cert.le_bpass`, `CertProofs.bpass_greatest`, `bpass_head_nonneg` | nonneg caps/gaps/budgets | `cert.backward`, `certline.backward` | **K** | — |
| forward pass | feasible, pointwise greatest, nonneg, under every cap and terminal cap | `Cert.fpass_feasible`, `fpass_greatest`, `CertProofs.feasible_nonneg_caps`, `decide_greatest_safe_caps` | same | `forward` | **K** | — |
| integer acceleration | `a ∈ [−a_brk, a_acc]`, end point feasible, greatest integer; floor rounding `b0+2ad ≤ b1 < b0+2(a+1)d` | `Cert.accel_greatest`, `decide_accel_greatest`, `CertProofs.accel_floor_rounding` | `d > 0` | `decide` | **K** | test `braking_with_rounding` re-targeted (v0 20291 → 20171) after the legitimate domain tightening (`DOMAIN_R_MIN_M` 9.0 → 6.0); the old value is now correctly `min_violation` |
| classification / infeasible start | four kinds ⇔ four disjoint cases; feasible profile exists ⇔ `v0² ≤ budget` ⇔ `greatest_safe` | `CertProofs.decide_classification`, `feasible_exists_iff`, `decide_gs_iff_feasible`, `Cert.decide_min_violation_infeasible` | — | `decide` | **K** | — |
| optimality for antitone objectives; min-violation braking | greatest profile minimises any speed-antitone objective; full braking pointwise least, minimises cap excess | `Cert.greatest_minimises`, `decide_greatest_safe_minimises`, `brake_least`, `brake_minimises_excess`, `decide_min_violation` | — | | **K** | not a minimiser of arbitrary violation measures |
| robust continuation | safe on every in-domain continuation; no same-information planner does better | `Cert.decide_robust_extend`, `decide_robust_greatest`, `decide_min_violation_robust`, `CertProofs.decide_robust_under` | road continuation caps ≥ `a_lat·r_h` (`InDomainContinuation`) | | **K** + **A** | domain constants (`DOMAIN_R_MIN_M` …) are measured trust items |
| controller speed plan (its OWN caps: path curvature, gap rule, sidestep caps, terminal) | same properties for `decideCaps`/`decideLong` | `ControllerLong.decideCaps_gs_feasible`, `decideCaps_gs_greatest`, `decideCaps_accel`, `decideCaps_mv_infeasible`, `decideLong_gs`, `decideLong_mv`; `Controller.decideLong_eq_cert` (equals `Cert.decide` when caps coincide) | `cap_scale = 1` | `certline.decide_long` | **K** + **R** (`tests/test_controller_refinement.py`) | branch order aligned with the spec (negative cap ⇒ invalid before the gap check) |
| curvature caps soundness | `b ≤ cap ⇒ b·κ ≤ a_lat·10⁹` | `Controller.curvature_cap_sound`, `Joint.pathCap_lateral_sound` | κ computed from the published polyline by `np.gradient` bounds the true curvature | `k_cap` | **K** + **A** | float curvature estimate not proved to bound the driven path's curvature — **U** |
| physical constants | `a_lat`, `a_acc`, `a_brk`, `r_h`, budgets | none | declared model constants | `lon_constants`, `model_constants` | **A** (never claimed physically correct) | — |

## 6. Tracking, actuation and state transition

| operation | specification | evidence | assumptions | implementation | status | open / counterexamples |
|---|---|---|---|---|---|---|
| longitudinal receding horizon | greatest-safe forever along consistent runs | `Closed.longitudinal_step`, `longitudinal_forever`, `realised_under_cap` | `RealisedWithin` (no overshoot of `F₁`), `PreviewConsistent`, `NextValid` | runner + controller | **K** + **A** | perception consistency and actuator tracking are assumptions |
| lateral tracking with delay, disturbance, sensor error, bias | `|e_k| ≤ M` for all `k` | `Closed.delayed_increment`, `delayed_window`, `delayed_small_gain`; instance `concrete_small_gain`, simulated run `concrete_run_bounded` | `DelayedEquation` (first-order lateral-error model with delay), `SmallGain` | path-error state feedback (`trk_mode = 1`: `atan(L·κ + u)`, `u = −(L/v²)(ω²e_y + 2ζωv·e_ψ)`) | **K** + **A** | the runtime tracker is second-order (cross-track + heading) and the bicycle/CARLA plant is not proved to satisfy `DelayedEquation`; the Lean result bounds the model, not this tracker — **U** |
| safe corridor / collision avoidance in closed loop | actual offset in corridor and beyond every remembered obstacle edge | `Closed.closed_loop_clearance`, `corridor_forward_invariant` | plan margin ≥ `M` inside `spareBox`, small-gain hypotheses | | **K** + **A** | runtime margin is `line_shrink` (0.15 m); `M ≤ margin` is not checked at runtime — **U** |
| steering saturation / lateral acceleration under the error bound | demand ≤ `κ_max`; `v²κ ≤ a_lat` | `Closed.curvature_demand_safe`, `lateral_accel_safe`, `concrete_curvature_instance` | `CurvatureSafe`, `LateralAccelSafe` | | **K** + **A** | — |
| steering command | `|steer| ≤ 1`, `|Δsteer| ≤ rate` at every step (rate limit applied last) | `Controller.steerAct_saturated`, `steerAct_rate`, `steerRun_invariant` | integer model of the float clip order | `_step` | **K** (model) + **R** (property tests on the real controller in `tests/test_controller_certline.py`) | float trigonometry (`atan`, path projection `_path_errors`) not formalised |
| pedals | never overlapping; invalid ⇒ full brake | `pedalAct_nonoverlap`, `invalid_full_brake` | integer model | `_pedals`, `_step` | **K** (model) + **R** | — |
| grip adaptation | budgets never above declared | `gripBudget_not_above` | — | `_adapt_grip` | **K** (model) | saturation detector heuristic — **A** |
| fallback braking | full braking reaches zero in `⌈b₀/(2ad)⌉` stations, pointwise least | `Closed.full_brake_le`, `full_brake_reaches_zero`, `full_brake_pointwise_least` | — | `KIND_INVALID` branch | **K** | — |
| bias estimator | estimate stays in `[−lim, lim]` | `Closed.bias_estimator_bounded` | — | bias estimator | **K** (model) | residual bias enters `F` |
| delay compensation | — | none | kinematic roll-out | `_predict_pose` | **U** | — |

## 7. Corpus and specification integrity (unchanged pipeline, re-run)

| property | evidence | status |
|---|---|---|
| manifest schema, spec equality, formal verdicts, bins, ids/hashes unique, pairs, split rule, coverage | `Main.manifestChecks` decided in Lean; `Properties.pairs_within_split`, `full_coverage_of_exact`, `adjCheck_sound`, `noCrossSplit_of_check` | **K** + compiled checker |
| every clean sample's longitudinal and lateral certificate recomputed | `corpuscheck` (compiled Lean) + `validate.py` (Python), per-sample kernel witnesses `Cert.decide c q = d`, `Steer.decide c q = d` | **R** |
| clean certified labels satisfy the clearance theorem's domain | `holdTargetsOK`/`clearanceDomain` are decidable but are **not yet** checked per sample by `corpuscheck` | **U** (see report) |
| mirror transform | `Mirror.mirrorObs_involutive`, checker equality on exported samples | **K** + **R** |

## Pre-existing statements found weak during the audit

| theorem | issue | resolution |
|---|---|---|
| `Bellman.value_attains`, `Bellman.no_policy_beats` | require actions at every state; vacuous for any horizon-limited model | superseded by `BellmanOpt` |
| `Dp.dp_optimal`, `Dp.speed_dp_optimal` | lower bound only; dead ends get value 0; `vModel` units and index wrong | superseded by `Joint`; counterexamples kernel-checked |
| `Dp.transition_certificate` | `(h : P) : P` tautology | retained (harmless), not cited as evidence |
| `SteerFeas.off_clearance_hold`, `clearAt_plan_ok_nonneg` | assume the plan-selection fact (`hplan`/`hsource`) — which was false for `steer-1` | superseded by `SteerClear.plan_ok_clearance` (selection derived) |
| `Steer.steer_unsaturated` | hypothesis about the result's `cdiv` | superseded by `SteerProofs.steerAt_unsaturated_of_condition` |
| `Track.error_bound_steady`, `corridor_invariant`, `concrete_margin_*` | toy halving recursion; assumes `|e₀| ≤` its own fixed point | superseded by `Closed` |
| `Mirror.chooseSide_mirror` | about a separate `chooseSide`, not the decision function | superseded by `SteerShape.sideAmp_mirror` and the list-level theorems |

## 8. Proof-carrying optimal teacher

The rows below describe the finite preview teacher in `autocross/optimal` and
its independent Lean checker. They do not assert continuous-vehicle, CARLA, or
full-course optimality.

| operation | specification | evidence | assumptions | implementation | status | open obligations / counterexamples |
|---|---|---|---|---|---|---|
| model encoding | state `(station, previous offset, current offset, speed²)`; declared offset/speed grids; corridor; inflated obstacle hold rule; integer curvature; acceleration/braking; terminal cap | `Corpus.Teacher.TState`, `stepOKT`, `terminalCap`, `CheckCert` | quantised observation and all grids/constants are the declared finite model | `autocross/optimal/model.py`; `lean/Corpus/Teacher.lean` | **K** + **R** (Lean definition and Python/Lean certificate inputs) | quantisation and physical interpretation are trust/model choices |
| DP optimality | Bellman minimum over every recomputed admissible successor, with terminal rows | `table_value_cert`, `certificate_value`, `certificate_lower_bound`; supporting `sparse_value` | accepted `CheckCert`; finite preview only | `optimal/solver.py::solve` | **K** | no claim beyond the encoded horizon/model |
| certificate checking | shape, uniqueness, roots, closure, terminal rows, Bellman rows, claimed start value, and exported plan checks | `checkCert_sound`; `CheckCert`; Python `check_certificate` | certificate JSON is parsed as the declared schema | `autocross/optimal/certificate.py`; `Corpus.Teacher.checkCert` | **K** + **R** | Python execution itself is not quantified by a Lean theorem |
| plan extraction | accepted feasible plan starts at the start, follows admissible steps, reaches the horizon, attains the claimed cost, and follows first-minimum tie order | `certificate_plan`; `claimOK`; `planTieOK` | a non-`none` claim | `solve` choices and `certificate_from_solution` | **K** + **R** | tie order is model order, not a physical preference |
| infeasibility | an accepted `claimed = none` rules out every admissible plan of the encoded finite model | `certificate_infeasible`; `BellmanOpt.valueO_none_iff` | accepted certificate and declared finite horizon | `solve` returns `cost=None`; Python checker rejects a plan on `none` | **K** + **R** | only finite-model infeasibility; no physical impossibility claim |
| Python/Lean parity | Python-emitted certificate represents the same encoded instance and is accepted by compiled Lean; corruption is rejected | `tests/test_teacher_parity.py`; `timeCost_eq_isqrt`; `rootsOK` | compiled `teachercheck` is built and invoked on the test certificate | Python certificate writer; `lean/TeacherCheck.lean` | **R** + **K** | parity is checked on supplied/generated certificates, not arbitrary Python executions |
| objective versus physical time | exact declared charge `cdiv(2*d*1_000_000, isqrt(b)+isqrt(w))`; interpretation as a floored-speed constant-acceleration proxy | `timeCost_eq_isqrt`; `tests/test_teacher_optimal.py::test_time_cost_is_exact_ceiling_and_physical_tolerance` | integer speed roots and positive segment denominator | `model.time_cost` / `Teacher.timeCost` | **K** + **A** | no identity with continuous or CARLA traversal time |
| controller integration | exact offline mode uses the full finite declaration; realtime mode is exact only for a reduced horizon/grid and reports `realtime_exact = false`, `fallback_reason = "realtime reduced horizon/grid (not full optimum)"`, `certified = false` relative to the requested model | `TeacherResult.realtime_exact`; `TeacherResult.fallback_reason`; runtime selftest | tracking, actuation, and explicit uncertified recovery remain outside the teacher theorem | `autocross/optimal/control.py`; `autocross/optimal/teachers.py`; `certline` | **U** | full-model realtime optimality is not preserved; recovery is not certified |
| dataset/label provenance | persisted exact labels retain the model/certificate needed to recompute the declared value and plan | `optimal_corpus.py`; Python certificate checks; parity tests when Lean executable is available | generated record and certificate are not altered after checking | `certificate_from_solution`, `check_certificate`, `teachercheck` | **R** | data handling outside the checked certificate and physical label meaning remain trust items |
| dataset/BC comparison | 36 runs, zero infrastructure failures; 24 train/6 val/8 test course-disjoint; 1,140 examples per condition; budgets 100/300/720; three seeds; four held-out courses | observed bicycle-only comparison: `optimal` 12/12 finishes; `heuristic` 8/12 with 42 off-course events; `approx` 12/12 but slower and with more cone hits | simulator configuration, bicycle backend, held-out track set, and run seeds | `experiments/il/run_experiment.py`; `experiments/il/reanalyze.py` | **R**; **empirically observed (bicycle only)** | no CARLA claim; no safety or continuous-optimality inference |

## 9. Direct actuator optimal controller

The rows below describe `autocross/actuator` and `teams/actuator_exact.py`. The
theorem is for the finite actuator model actually encoded in the certificate; it
is not a continuous-vehicle or whole-course claim.

| property | statement | Lean theorem | hypotheses | runtime | status | notes |
|---|---|---|---|---|---|---|
| actuator model | state `(station,d,psi,b,delta)`, declared grids, rate/lag-limited steering, understeer, midpoint integration, corridor/obstacle, lateral and friction bounds | `Corpus.Actuator.ActModel`, `nextStateA`, `actionOKA`, `actionsA` | quantised instance, finite horizon, declared constants | `actuator/model.py` | **K** + **R** | integer model only |
| terminal condition | terminal corridor/heading/steer/speed and braking-envelope cap over visible stations | `terminalOKB`, `terminalCapA` | nondegenerate grid, positive gaps | `terminal_ok`, `terminal_cap` | **K** | no claim beyond the preview |
| objective | `1000*time + effort + 5000*|d-ref_d| + 5000*|psi-ref_psi|/100` | `actionCostA` | declared reference arrays | `action_cost` | **K** | reference prior is model input |
| DP optimality | accepted table rows equal the option-valued Bellman optimum over every recomputed action | `sparse_value`, `table_value_cert`, `certificate_value`, `certificate_lower_bound` | accepted `CheckCert` | `solver.py::solve`, `solve_sparse` | **K** | no pruning in the certificate universe |
| first action | accepted `firstAction` produces plan[1], is admissible, and reaches the claimed value | `certificate_plan_some`, `certificate_optimal` | non-`none` claim | `command_from_first` | **K** + **R** | binds a command to the checked plan |
| infeasibility | `claimed = none` rules out every admissible plan of the encoded horizon | `certificate_infeasible` | accepted certificate | `solve` returns `cost=None` | **K** | finite-model infeasibility only |
| Python/Lean parity | a real H=4 observation certificate is accepted by Python and by compiled `actuatorcheck`; corrupt value/action/plan/successor certificates are rejected | `tests/test_actuator_parity.py`; compiled checker | `actuatorcheck` built | `tools/actuator_certificate.py` | **R** + **K** | parity checked on generated certificates |
| real-time controller | exact DP on a reduced H=4 grid, explicit `certline` fallback outside the envelope; all five public courses finish | `Corpus.Actuator.certificate_optimal` only for certified exact steps | fallback commands are not certified | `teams/actuator_exact.py` | **R** observed; **U** for fallback | 65.7/75 mean, 0 cones/off, p99 21–29 ms; exact steps are a minority |
