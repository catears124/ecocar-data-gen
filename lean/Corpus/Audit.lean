/-
Axiom audit: `lake env lean Corpus/Audit.lean` lists the axioms each theorem depends on.
Expected: only Lean's standard axioms (propext, Classical.choice, Quot.sound); no `sorryAx`.
-/
import Corpus

#print axioms Corpus.Cert.le_bpass
#print axioms Corpus.Cert.fpass_greatest
#print axioms Corpus.Cert.fpass_feasible
#print axioms Corpus.Cert.brake_least
#print axioms Corpus.Cert.greatest_minimises
#print axioms Corpus.Cert.decide_greatest_safe
#print axioms Corpus.Cert.decide_min_violation
#print axioms Corpus.pairs_within_split
#print axioms Corpus.full_coverage_of_exact
#print axioms Corpus.adjCheck_sound
#print axioms Corpus.mirrorObs_involutive
#print axioms Corpus.mirrorAct_involutive
#print axioms Corpus.mirror_gap_swap
#print axioms Corpus.chooseSide_mirror
