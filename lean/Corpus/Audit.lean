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
