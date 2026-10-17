/-
Axiom audit: `lake env lean Corpus/Audit.lean` lists the axioms each theorem depends on.
Release gate (`autocross.datagen.lean_bridge.axiom_audit`): every theorem below must be listed and
depend on nothing beyond Lean's standard axioms (propext, Classical.choice, Quot.sound); in
particular no `sorryAx` and no `Lean.ofReduceBool` (native_decide).
-/
import Corpus

#print axioms Corpus.Cert.le_bpass
#print axioms Corpus.Cert.fpass_greatest
#print axioms Corpus.Cert.bpass_head_nonneg
#print axioms Corpus.Cert.hold_feasible
#print axioms Corpus.Cert.robust_extend
#print axioms Corpus.Cert.robust_necessary
#print axioms Corpus.Cert.decide_robust_extend
#print axioms Corpus.Cert.decide_robust_greatest
#print axioms Corpus.Cert.decide_min_violation_robust
#print axioms Corpus.Cert.fpass_feasible
#print axioms Corpus.Cert.accel_greatest
#print axioms Corpus.Cert.brake_least
#print axioms Corpus.Cert.greatest_minimises
#print axioms Corpus.Cert.capExcess_mono
#print axioms Corpus.Cert.brake_minimises_excess
#print axioms Corpus.Cert.inDomain_of_valid
#print axioms Corpus.Cert.decide_gs_facts
#print axioms Corpus.Cert.decide_greatest_safe
#print axioms Corpus.Cert.decide_greatest_safe_feasible
#print axioms Corpus.Cert.decide_greatest_safe_minimises
#print axioms Corpus.Cert.decide_accel_greatest
#print axioms Corpus.Cert.decide_min_violation_infeasible
#print axioms Corpus.Cert.decide_min_violation
#print axioms Corpus.pairs_within_split
#print axioms Corpus.full_coverage_of_exact
#print axioms Corpus.adjCheck_sound
#print axioms Corpus.noCrossSplit_of_check
#print axioms Corpus.mirrorObs_involutive
#print axioms Corpus.mirrorAct_involutive
#print axioms Corpus.mirror_gap_swap
#print axioms Corpus.chooseSide_mirror
#print axioms Corpus.Steer.wellFormed_off_length
#print axioms Corpus.Steer.sides_length
#print axioms Corpus.Steer.amps_length
#print axioms Corpus.Steer.decide_kind_invalid
#print axioms Corpus.Steer.decide_no_gap_iff
#print axioms Corpus.Steer.decide_plan_ok_iff
#print axioms Corpus.Steer.clamp_abs_le
#print axioms Corpus.Steer.clamp_noop_in_hold
#print axioms Corpus.Steer.ramp_zero
#print axioms Corpus.Steer.wOb_zero_before
#print axioms Corpus.Steer.steer_within_range
#print axioms Corpus.Steer.steer_unsaturated
#print axioms Corpus.SteerFeas.clamp_in_corridor
#print axioms Corpus.SteerFeas.clamp_abs_le_gen
#print axioms Corpus.SteerFeas.target_in_corridor_left
#print axioms Corpus.SteerFeas.target_in_corridor_right
#print axioms Corpus.SteerFeas.off_clearance_hold
#print axioms Corpus.SteerFeas.off_clearance_ramps_outside
#print axioms Corpus.SteerFeas.clearance_hold_nonneg
#print axioms Corpus.SteerFeas.clearance_outside_nonneg
#print axioms Corpus.SteerFeas.clearAt_hold_nonneg
#print axioms Corpus.SteerFeas.clearAt_outside_nonneg
#print axioms Corpus.SteerFeas.clearAt_nonneg_of_source
#print axioms Corpus.SteerFeas.clearAt_plan_ok_nonneg
#print axioms Corpus.SteerRamp.n_shift
#print axioms Corpus.SteerRamp.n_full
#print axioms Corpus.SteerRamp.ramp_full
#print axioms Corpus.Track.error_fixed_point
#print axioms Corpus.Track.error_bound_steady
#print axioms Corpus.Track.error_bound_steady_ramp
#print axioms Corpus.Track.corridor_invariant
#print axioms Corpus.Track.steer_unsaturated_of_kappa_le_kmax
#print axioms Corpus.Track.concrete_margin_example
#print axioms Corpus.Track.concrete_margin_bound
#print axioms Corpus.Bellman.greedy_cost
#print axioms Corpus.Bellman.value_attains
#print axioms Corpus.Bellman.no_policy_beats
#print axioms Corpus.Bellman.value_unique
#print axioms Corpus.Dp.dp_optimal
#print axioms Corpus.Dp.transition_certificate
