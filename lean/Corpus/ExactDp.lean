/-
  Certificate interface for the preview-only primitive lattice.  The Python checker emits a
  finite table whose rows are exactly the option-valued Bellman recurrence `valueO`; this theorem
  states the two obligations needed by the checker: no admissible lattice trajectory beats the
  certified value, and a trajectory attaining it exists.  Plant constants are parameters.
-/
import Corpus.BellmanOpt
import Corpus.Closed
namespace Corpus.ExactDp
open Corpus.BellmanOpt
open Corpus.Closed

abbrev Certificate {σ : Type} (terminal : σ → Option Int)
    (stages : List (Stage σ)) (start : σ) (v : Int) : Prop :=
  valueO terminal stages start = some v

/-- A Bellman-checked certificate lower-bounds every admissible finite lattice path. -/
theorem certificate_no_path_beats {σ : Type} (terminal : σ → Option Int)
    (stages : List (Stage σ)) (start : σ) (v : Int)
    (hcert : Certificate terminal stages start v) :
    ∀ xs w, TrajOK terminal stages start xs →
      trajCost terminal stages start xs = some w → v ≤ w := by
  intro xs w htraj hcost
  exact valueO_le_cost terminal stages start xs v w htraj hcost hcert

/-- A finite Bellman certificate has an admissible extracted path attaining its value. -/
theorem certificate_has_optimal_path {σ : Type} (terminal : σ → Option Int)
    (stages : List (Stage σ)) (start : σ) (v : Int)
    (hcert : Certificate terminal stages start v) :
    ∃ xs, TrajOK terminal stages start xs ∧
      trajCost terminal stages start xs = some v := by
  exact valueO_attained terminal stages start v hcert

/-- The conservative unknown-future terminal condition: if speed-squared is no larger than
    `n * (2*a*d)`, n full-braking station intervals reach zero.  `a`, `d`, and `b` are inputs;
    this theorem makes no physical claim about their numerical values. -/
theorem terminal_brake_safe (a d b : Int) (had : 0 ≤ 2 * a * d)
    (n : Nat) (hn : 1 ≤ n) (hb : b ≤ (n : Int) * (2 * a * d)) :
    fullBrakeProfile a d b n = 0 := by
  exact full_brake_reaches_zero a d b had n hn hb

end Corpus.ExactDp
