# Formal guarantees

Exact statements of the main kernel-checked results and their hypotheses. Every theorem below
is listed in `lean/Corpus/Audit.lean` and depends only on `propext`, `Classical.choice`,
`Quot.sound` (checked by `autocross.datagen.lean_bridge.axiom_audit`). Integer units: mm,
mm/s, mm²/s², mm/s², 10⁻⁶/m (µ/m). Nothing here is a statement about CARLA, the bicycle
backend or a physical car unless a named assumption says so.

## A. Bellman optimality with explicit infeasibility (`Corpus.BellmanOpt`)

Model: a finite list of stages `st : σ → List (σ × Int)` (successor, step cost) and a terminal
`terminal : σ → Option Int` (`none` = infeasible). `valueO` is backward induction with an
option-minimum; `TrajOK` = one offered action per stage and a feasible terminal; `trajCost`
sums the step costs and the terminal value. For **any** state type and **any** finite horizon:

| theorem | statement |
|---|---|
| `valueO_none_iff` | `valueO ss s = none ↔ ¬ ∃ xs, TrajOK ss s xs` — empty action lists and infeasible terminals can never yield a finite value |
| `valueO_le_cost` | `TrajOK ss s xs → trajCost = some w → valueO = some v → v ≤ w` |
| `valueO_attained` | `valueO = some v → ∃ xs, TrajOK ∧ trajCost = some v` |
| `extract_spec` | `valueO = some v → ∃ xs, extract ss s = some xs ∧ TrajOK ∧ trajCost = some v ∧ xs.length = ss.length` (executable greedy extraction, one action per stage) |
| `extract_none_iff` | `extract = none ↔ valueO = none` |
| `policy_lower_bound`, `extracted_policy_optimal` | the same for policies (functions per stage); the extracted plan replayed as a policy is admissible and optimal |
| `optimal_value_unique`, `bellmanO_unique`, `valueFns_bellman` | the least admissible cost is unique; every solution of the option-valued Bellman equations equals `valueO`; the backward-induction functions are a solution |
| `valueO_eq_old`, `old_value_le_valueO` | with no empty action lists the old `Bellman.value` is exact; in general it is only a lower bound |
| `dead_mid_*` (instances) | old value 1, true optimum 6, old greedy picks a dead end with no continuation; extraction returns `[(2,5),(3,1)]`; infeasible start detected |

## B. Joint offset/speed optimality (`Corpus.Joint`)

State `(i, previous offset, current offset, b)` with `b` the speed² at station `i`.
`stepOK c q grid s t` requires: `t.station = s.station+1`; `t.previous = s.current`;
`t.current ∈ Dp.bucketGrid`; corridor and `Steer.clearAt ≥ 0` at the arrival station;
`pathCurv = |kbar_road(i)| + ⌈10⁹·|n − 2x + p| / step²⌉ ≤ κ_max`; `s.b ≤ pathCap` (cap of the
departure stencil, `≤ Cert.capAt`); `t.b ∈ grid`, `0 ≤ t.b`, `t.b ≤ s.b + 2·a_acc·d_i`,
`s.b ≤ t.b + 2·a_brk·d_i`; terminal `b ≤ Cert.lastCap`. `PlanOK` is this relation for a
whole plan; `planCost time` sums an arbitrary declared cost `time i b_i b_{i+1}`.

| theorem | statement |
|---|---|
| `joint_value_spec` | `valueO (terminal c q) (replicate m (actions …)) s = some v →` (∀ plans `ts` of length `m`, `v ≤ planCost ts`) ∧ (∃ such a plan with cost `v`) |
| `joint_infeasible_iff` | value `none` ⇔ no joint plan of length `m` exists |
| `joint_extract_spec` | extraction returns a plan in the declarative set, of length `m`, with cost `v`, and no plan is cheaper |
| `plan_steps`, `plan_progress` | every step of a plan satisfies `stepOK`; the last station is `start + m` and its speed² is under the terminal cap |
| `plan_cert_feasible` | **composition**: for every plan from a state with `0 ≤ b`, the speed² sequence is `Cert.Feasible a_acc a_brk (segsOf plan) terminalCap` — the longitudinal certificate's feasibility predicate for the caps of the path the lateral component chose |
| `plan_le_fpass`, `plan_start_le_bpass` | hence the plan's speeds are pointwise ≤ the certificate's greatest profile for those caps, and the longitudinal decision on them is not min-violation |
| `pathCap_le_capAt`, `pathCap_lateral_sound`, `pathCap_physical_sound` | the joint cap never exceeds the longitudinal certificate's cap; `b ≤ pathCap ⇒ b·pathCurv ≤ a_lat·10⁹`; if the true curvature `0 ≤ κ_t ≤ pathCurv` (assumption) then `b·κ_t ≤ a_lat·10⁹` |
| `toy_*` | concrete instance (obstacle, curvature cap and braking all binding): value 82491 = independent enumeration; extraction `[(250, 10.5·10⁶), (1000, 10.5·10⁶)]`; too-fast start ⇒ `none` and no candidate plan |

Two objectives are distinguished: (1) **declared travel-time optimality** — `joint_value_spec`
for the declared cost table over the joint discrete action space (a calibration of the table to
seconds is a trust item); (2) **speed-cap surrogate optimality** — the pre-existing
`Dp.dp_optimal` (lateral-only, lower bound, old `Bellman.value`). The joint model is an offline
reference optimizer: the submitted controller does not run it (see C, D).

### B′. Planning levers (`Corpus.Levers`)

The joint model is restated for an arbitrary step relation `P` and terminal cap `T`
(`actionsP`, `terminalP`, `PlanP`); `Joint` is the instance `P = stepOK`, `T = terminalCap`
(`joint_value_eq`, `joint_plan_iff`). For every such model: lower bound (`valueP_lower`, needs the
joint `Shaped` form), attainment (`valueP_attained`), exact infeasibility (`valueP_none_iff`).

| theorem | hypotheses | statement |
|---|---|---|
| `lever_value_le` | every step of `P` reachable under an invariant `I` is a step of `P'`; `T ≤ T'`; `I s` | `valueP P … s = some v → ∃ v' ≤ v, valueP P' … s = some v'` |
| `budget_lever` (+ `cornering_lever`, `braking_lever`, `accel_lever`, `horizon_lever`) | `raise c A Acc Brk Rh Vm K` with each value `≥` the original; `0 ≤ a_lat, r_h, v_max`; `GapsNonneg q` | the joint optimum with raised limits is `≤` the original, and feasible whenever the original is |
| `extra_caps_cost`, `extra_caps_mono` | extra departure caps `E ≤ E'` | adding a cap never lowers the optimum; a looser cap never raises it |
| `box_step_circle`, `circle_value_le_box` | `a_lat² + max(a_acc,a_brk)² ≤ A²`, `0 ≤ a_lat`, `0 ≤ b` at the start, `GapsNonneg q` | every certified box step is a friction-circle step (`circleOK`: same offsets/grids, `b ≤ v_max²`, `(b·κ·2d)² + (Δb·10⁹)² ≤ (A·10⁹·2d)²`), so the circle optimum `≤` the certified optimum |
| `toy_cornering_binds`, `toy_braking_slack`, `toy_circle_gain`, `toy_extra_cap_infeasible` | `Joint`'s toy problem | `a_lat` 5 → none, 6/7 → 82491, 8 → 51417; `a_brk` 9 → 82491; circle radius 7 → 74074; cap 12·10⁶ → none (Python enumeration agrees: `tests/test_joint_dp.py`) |

These order **declared** optima. A larger budget is never worse *in the model*; whether the car
can track the resulting plan (rear slip, cone hits) is outside the theorems, and the measurements
in `docs/proof_closure_report.md` show that above `lat_frac ≈ 0.75` it cannot on the bicycle
backend. The circle model is a relaxation and is not certified by `Cert`.

### B″. Certificate interface for a preview-only lattice expert (`Corpus.ExactDp`)

| theorem | hypotheses | statement |
|---|---|---|
| `certificate_no_path_beats`, `certificate_has_optimal_path` | `valueO terminal stages start = some v` | no admissible lattice path costs less than `v`, and an admissible path of cost `v` exists (restatements of `valueO_le_cost` / `valueO_attained`) |
| `terminal_brake_safe` | `0 ≤ 2·a·d`, `1 ≤ n`, `b ≤ n·(2·a·d)` | `n` full-braking intervals bring speed² `b` to 0 (restatement of `full_brake_reaches_zero`) |

Used by an offline experiment, not by `teams/certline.py`. The experiment searched a lattice of
exact bicycle-model rollouts using only the controller's own 40 m preview. It lost to the
shipped controller: on 30 sampled states it found a plan for 14, with mean −0.56 m of 2 s
progress, and needed 44 s per solve. It was not shipped. Optimality here is over the lattice only.

## C. Lateral plan `steer-2` (`Corpus.SteerProofs`, `SteerClear`, `SteerShape`, `SteerRamp`)

| theorem | hypotheses | statement |
|---|---|---|
| `rawPlanAt_hold` | obstacles `= pre ++ [d,lmin,lmax] :: post`, `pairwiseDisjoint hold`, `0 < ramp_in`, `d − hold ≤ s ≤ d + hold` | the selected supplier at `s` is exactly that obstacle |
| `plan_ok_hold_clearance` | `clearanceDomain c q`, `holdTargetsOK c q`, `decide = plan_ok`, `[d,lmin,lmax] ∈ obstacles`, station `j` in its hold window | offset beyond the obstacle edge on the chosen side **and** `clearAt ≥ 0` |
| `plan_ok_clearance`, `plan_ok_clears_nonneg` | same domain predicates, `plan_ok` | every recorded clearance entry is `≥ 0` |
| `ramp_bounds`, `ramp_mono`, `ramp_1000_lt`, `ramp_full`, `ramp_zero` | `0 ≤ A`, `0 ≤ x ≤ L`, `0 < L` | endpoints, amplitude, monotonicity, strict `≤ 999` before the end |
| `ramp_slope_bound` | `0 ≤ A`, `0 < L`, `0 ≤ h` | `2L·(ramp(x+h) − ramp(x)) < 3Ah + 2L` (exact floor) |
| `ramp_second_bound` | `0 ≤ A`, `0 < L`, `0 ≤ x ≤ L` | `L²·|ramp(x+h) − 2ramp(x) + ramp(x−h)| < 6Ah² + 2L²` |
| `sideAmp_edge` | gap fits, obstacle inside the road, even width | target beyond the obstacle edge on the chosen side |
| `offAt_mirror`, `offs_mirror`, `clearAt_mirror`, `curvs_mirror`, `sides_amps_mirror` | `noTies q`, nonempty corridors | mirrored observation ⇒ negated offsets, flipped sides, equal amplitudes, clearances, curvature demand and steering |
| `steerAt_range`, `steerAt_unsaturated_of_condition` | `0 < κ_max`, `0 ≤ κ`, `1000κ ≤ 999κ_max` | `steerAt ∈ [−1000, 1000]`; exact `⌈1000κ/κ_max⌉` |
| counterexamples | — | `old_selector_counterexample`, `narrowing_counterexample`, `supplier_switch_curvature`, `mirror_tie_counterexample` |

## D. Submitted controller (`Corpus.Controller`, `ControllerLong`)

| theorem | statement |
|---|---|
| `spareBox_sound`, `spareBox_complete`, `spareBox_empty_iff` | for ANY obstacle memory: `x ∈ spareBox` ⇔ inside the corridor and beyond every covering remembered obstacle's edge on its remembered side; empty ⇔ no clear offset exists |
| `projectPlan_boxed` | for ≥ 1 iteration, ≥ 2 stations, all intervals nonempty: every projected entry is inside its interval, whatever the input |
| `publishOffset_inside` | cap first, interval clamp last ⇒ published offset in any nonempty interval |
| `decideCaps_gs_feasible`, `decideCaps_gs_greatest`, `decideCaps_accel`, `decideCaps_mv_infeasible`, `decideLong_gs`, `decideLong_mv` | for the controller's OWN caps (curvature of the published path, gap rule, sidestep caps, terminal cap): greatest-safe profile feasible and pointwise greatest; integer acceleration in `[−a_brk, a_acc]` and greatest; min-violation ⇔ infeasible |
| `decideLong_eq_cert` | when the caps coincide with `Cert`'s, `decideLong = Cert.decide` |
| `decideLateral_*_eq` | the controller's lateral decision is `Steer.offs/curvs/steers` on a `Steer.Q` with the SIGNED lateral offset |
| `steerAct_saturated`, `steerAct_rate`, `steerRun_invariant`, `pedalAct_nonoverlap`, `invalid_full_brake`, `gripBudget_not_above` | integer model of the actuation pipeline (clip, bias, clip, rate limit last): every command of every run is in `[−1, 1]` and within the rate limit; pedals never overlap |

Refinement evidence (category R): `tests/test_controller_refinement.py` runs the Python
functions on random in-domain inputs and has the Lean kernel check the identical outputs of
`decideLong`, `Steer.offs/curvs/steers`, `spareBox` (every station) and `projectPlan`; a
perturbed output is rejected.

## E. Closed loop (`Corpus.Closed`)

| theorem | hypotheses (named predicates) | statement |
|---|---|---|
| `longitudinal_step`, `longitudinal_forever` | `RealisedWithin` (realised next speed² in `[0, F₁]`), `PreviewConsistent` (next preview = shifted preview + continuation with caps `≥ a_lat·r_h`), `NextValid` | if the first decision is greatest-safe, every decision along the run is greatest-safe; speed² stays under the backward envelope |
| `realised_under_cap` | same | the realised speed² respects the cap of the station reached |
| `delayed_small_gain` | `DelayedEquation D P d e f` (`D(e_{k+1}−e_k) = −P·e_{k−d} + D·f_k`), `|f_k| ≤ F`, `SmallGain` (`0 ≤ P ≤ D`, `P·d·(P·M + D·F) + D²·F ≤ D·P·M`), `|e_0| ≤ M` | `|e_k| ≤ M` for all `k` (delay `d`, disturbance, sensor error and bias residual inside `F`) |
| `closed_loop_clearance` | the above + published offset keeps margin `M` inside `spareBox` | the ACTUAL offset is inside the corridor and beyond every covering remembered obstacle's edge, at every step |
| `curvature_demand_safe`, `lateral_accel_safe` | `CurvatureSafe`, `LateralAccelSafe` | steering demand ≤ `κ_max`; lateral acceleration ≤ `a_lat` under the error bound |
| `full_brake_reaches_zero`, `full_brake_pointwise_least` | `0 ≤ 2ad`, `b ≤ n·2ad`, `n ≥ 1` | full braking stops within `n` stations and is pointwise least |
| `bias_estimator_bounded` | `0 ≤ lim` | the clamped bias estimate stays in `[−lim, lim]` |
| `concrete_small_gain`, `concrete_run_bounded`, `concrete_curvature_instance` | — | non-vacuity: `v = 10 m/s`, `dt = 0.05 s`, `l_d = 8 m`, delay 2 steps, `F = 6 mm/step` ⇒ `M = 124 mm`; a 200-step worst-case run stays within it |

## F. Longitudinal certificate `cert-2` (`Corpus.Cert`, `CertProofs`, `Quant`)

Pre-existing (Part 1/2 of `Cert.lean`) plus: `decide_classification` (the four kinds are exactly
the four disjoint input cases), `feasible_exists_iff` and `decide_gs_iff_feasible` (exact
infeasible-start detection), `feasible_nonneg_caps` (every profile entry nonnegative and under
its cap), `accel_floor_rounding`, `decide_robust_under` (`InDomainContinuation`), and the
quantiser facts `qfix_floor_bounds`, `qfix_error_bound`, `qfix_monotone_mantissa`,
`qfix_int_exact`. The arithmetic is separated from physics: `a_lat`, `a_acc`, `a_brk`, `r_h`
and the domain radius are declared inputs, never claimed physically correct.
