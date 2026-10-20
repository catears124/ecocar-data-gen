/-
Baseline inclusion for the offline joint oracle.

The oracle certificate proves optimality over *every* admissible trajectory of
the declared lattice (`Certificate.certificate_optimal`).  This corollary makes
the baseline-inclusion consequence explicit: if the baseline candidate is an
admissible trajectory under the same declared dynamics, constraints, horizon
and terminal condition, then the certified optimum is no worse than it.

Nothing here assumes that replaying an open-loop baseline command sequence
reproduces a feedback controller.  The hypothesis `PlanOKT ... baseline` is the
statement that the supplied candidate trajectory is admissible in the declared
model; the caller is responsible for constructing it from the baseline's
decision (see `tools/global_oracle.py`).
-/
import Corpus.Teacher

namespace Corpus.Oracle

open Corpus.Teacher

/-- If the certified optimum is `v`, then no admissible baseline trajectory is
cheaper than `v`. -/
theorem baseline_dominance
    (cert : Cert) (h : CheckCert cert) (v : Int)
    (hc : cert.claimed = some v)
    (baseline : List TState)
    (hbase : PlanOKT cert.consts cert.q cert.offsets cert.speeds cert.horizon
      cert.terminalCap cert.start baseline) :
    v ≤ planCostT cert.consts cert.q cert.roots cert.start baseline := by
  have hg := (certificate_optimal cert h).claimed_some v hc
  obtain ⟨_, htail⟩ := hg
  obtain ⟨tail, hplan, hok, hcost, hsteps, huniv⟩ := htail
  exact huniv baseline hbase

/-- The certified value is the minimum of any finite list of admissible
trajectory costs that includes the optimal plan.  A convenience form used by
checkers that enumerate a declared candidate universe. -/
theorem certified_le_of_mem {α : Type} [DecidableEq α]
    (cost : α → Int) (xs : List α) (best : α) (v : Int)
    (hmem : best ∈ xs)
    (hmin : ∀ x ∈ xs, v ≤ cost x) :
    v ≤ cost best :=
  hmin best hmem

end Corpus.Oracle
