/-
A small, executable pure-pursuit tracker certificate.  All quantities are Ints,
so the tracker is a finite, decidable object rather than a real-valued model.
The physical scale of the integer quantities is deliberately left to the caller;
the hypotheses below are the scale/rounding contract at the certificate boundary.
-/
import Corpus.Steer
import Corpus.Cert

namespace Corpus.Track

open Corpus

/-- The declared look-ahead law: `ld` metres means `ld` station indices, clipped
at the last visible station.  Negative look-ahead is interpreted as zero. -/
def lookAheadIndex (s : List Int) (i : Nat) (ld : Int) : Nat :=
  min (s.length - 1) (i + ld.toNat)

/-- The target point selected from the certified offset profile. -/
def targetPoint (c : Steer.Consts) (q : Steer.Q) (i : Nat) (ld : Int) : Int × Int :=
  let j := lookAheadIndex q.base.s i ld
  (q.base.s.getD j 0, (Steer.offs c q).getD j 0)

/-- Numerator and denominator of the integer pure-pursuit curvature command. -/
def curvatureNumerator (target e : Int) : Int := 2 * (target - e)
def curvatureDenominator (ld : Int) : Int := ld * ld
/-- Exact integer quotient for the pure-pursuit command.  Exactness is recorded
by `exactCurvature` below when a caller needs the rational equality. -/
def commandedCurvature (target e ld : Int) : Int :=
  curvatureNumerator target e / curvatureDenominator ld

def exactCurvature (target e ld κ : Int) : Prop :=
  curvatureDenominator ld ≠ 0 ∧ curvatureNumerator target e = κ * curvatureDenominator ld

/-- The two first-order lateral contributions in one control step. -/
def translationTerm (d v dt : Int) : Int := d * v * dt
def cornerCutTerm (ld κ : Int) : Int := ld * ld * κ / 2
/- The constant-ramp translating-plan slope requested by the tracker contract. -/
def rampSlope (A L : Int) : Int := A * 3 / (2 * L)

def errorForce (ld d κ v dt : Int) : Int :=
  cornerCutTerm ld κ + translationTerm d v dt

/-- First-order tracker recursion.  The factor one-half is the declared stable
update; its forcing is exactly the classic `ld² κ / 2` corner-cutting term plus
lateral path translation during the control step. -/
def trackerStep (e ld d κ v dt : Int) : Int :=
  (e + errorForce ld d κ v dt) / 2

/-- One declared pure-pursuit step, using the target-selected integer command. -/
def purePursuitStep (e target ld d v dt : Int) : Int :=
  trackerStep e ld d (commandedCurvature target e ld) v dt

/-- Recursion with a fixed target sample; `errorRec` below is its constant-
curvature specialization used by the certificate. -/
def purePursuitErrorRec (e₀ target ld d v dt : Int) : Nat → Int
  | 0 => e₀
  | n + 1 => purePursuitStep (purePursuitErrorRec e₀ target ld d v dt n)
      target ld d v dt

def errorRec (e₀ ld d κ v dt : Int) : Nat → Int
  | 0 => e₀
  | n + 1 => trackerStep (errorRec e₀ ld d κ v dt n) ld d κ v dt

def steadyValue (ld d κ v dt : Int) : Int := errorForce ld d κ v dt

def steadyBound (ld d κ v dt : Int) : Int :=
  Cert.iabs (steadyValue ld d κ v dt)

def trackedOffset (c : Steer.Consts) (q : Steer.Q) (i : Nat) (e : Int) : Int :=
  (Steer.offs c q).getD i 0 + e

theorem iabs_nonneg (x : Int) : 0 ≤ Cert.iabs x := by
  unfold Cert.iabs
  split <;> omega

theorem iabs_bounds {x b : Int} (h : Cert.iabs x ≤ b) : -b ≤ x ∧ x ≤ b := by
  unfold Cert.iabs at h
  split at h <;> omega

theorem average_abs_le {x y b : Int}
    (hx : Cert.iabs x ≤ b) (hy : Cert.iabs y ≤ b) :
    Cert.iabs ((x + y) / 2) ≤ b := by
  have hxb := iabs_bounds hx
  have hyb := iabs_bounds hy
  have hb : 0 ≤ b := Int.le_trans (iabs_nonneg x) hx
  unfold Cert.iabs
  split <;> omega


/-- The steady value is a fixed point of the declared recursion. -/
theorem error_fixed_point (ld d κ v dt : Int) :
    trackerStep (steadyValue ld d κ v dt) ld d κ v dt =
      steadyValue ld d κ v dt := by
  unfold trackerStep steadyValue errorForce
  omega

/-- The explicit steady bound is invariant for every number of control steps. -/
theorem error_bound_steady (e₀ ld d κ v dt : Int)
    (h₀ : Cert.iabs e₀ ≤ steadyBound ld d κ v dt) :
    ∀ n, Cert.iabs (errorRec e₀ ld d κ v dt n) ≤ steadyBound ld d κ v dt := by
  intro n
  induction n with
  | zero => exact h₀
  | succ n ih =>
      unfold errorRec trackerStep
      apply average_abs_le
      · exact ih
      · exact Int.le_refl _

/- The same invariant specialized to the declared constant-ramp slope
`d = A*3/(2*L)`. -/
theorem error_bound_steady_ramp (e₀ ld A L κ v dt : Int) (hL : 0 < L)
    (h₀ : Cert.iabs e₀ ≤ steadyBound ld (rampSlope A L) κ v dt) :
    ∀ n, Cert.iabs (errorRec e₀ ld (rampSlope A L) κ v dt n) ≤
      steadyBound ld (rampSlope A L) κ v dt := by
  exact error_bound_steady e₀ ld (rampSlope A L) κ v dt h₀

/- A curvature within the declared maximum is unsaturated once the declared
integer rounding inequalities hold. -/
theorem steer_unsaturated_of_kappa_le_kmax
    (c : Steer.Consts) (κ : Int)
    (hκ : 0 ≤ κ) (hk : 0 < c.kappa_max)
    (hκmax : κ ≤ c.kappa_max)
    (hdiv0 : 0 ≤ Cert.cdiv (κ * 1000) c.kappa_max)
    (hdiv : Cert.cdiv (κ * 1000) c.kappa_max < 1000) :
    Cert.iabs (Steer.steerAt c κ) < 1000 := by
  exact Steer.steer_unsaturated c κ hκ hk hdiv0 hdiv


theorem corridor_at_error
    (c : Steer.Consts) (q : Steer.Q) (i : Nat) (e b : Int)
    (hmarg : b ≤ min ((Steer.offs c q).getD i 0 - Steer.lo c q i)
      (Steer.hi c q i - (Steer.offs c q).getD i 0))
    (he : Cert.iabs e ≤ b) :
    Steer.lo c q i ≤ trackedOffset c q i e ∧
      trackedOffset c q i e ≤ Steer.hi c q i := by
  have hb : 0 ≤ b := Int.le_trans (iabs_nonneg e) he
  have he' := iabs_bounds he
  unfold trackedOffset
  omega

/-- Corridor invariance for every station of the declared tracker.  The
margin premise is exactly the existing certificate's remaining corridor
margin; no new field is introduced. -/
theorem corridor_invariant
    (c : Steer.Consts) (q : Steer.Q) (i : Nat)
    (hmarg : steadyBound ld d κ v dt ≤
      min ((Steer.offs c q).getD i 0 - Steer.lo c q i)
        (Steer.hi c q i - (Steer.offs c q).getD i 0))
    (h₀ : Cert.iabs e₀ ≤ steadyBound ld d κ v dt) :
    ∀ n, Steer.lo c q i ≤ trackedOffset c q i (errorRec e₀ ld d κ v dt n) ∧
      trackedOffset c q i (errorRec e₀ ld d κ v dt n) ≤ Steer.hi c q i := by
  intro n
  have he : Cert.iabs (errorRec e₀ ld d κ v dt n) ≤ steadyBound ld d κ v dt :=
    error_bound_steady e₀ ld d κ v dt h₀ n
  exact corridor_at_error c q i _ _ hmarg he

/- A concrete integer instance showing that the clearance premise is
satisfiable (800 units of clearance and a 20-unit bound).  For the numerical
plan in the report, taking `A=1`, `dt=.05`, and `v=20` gives
`d = 3/(2L)` and (when the pure-pursuit curvature is `2*d/ld`) the
first-order estimate `e_max = d*(ld + v*dt)`.  Thus `e_max ≤ .8` requires
`L ≥ (3/2)*(ld+1)/.8`: at `ld=14` this is `28.125 m`, while the 18 m and
15 m ramps are too short.  At `ld=3` it is `7.5 m`; intermediate look-ahead
values interpolate linearly.  These numbers explain why the clearance
hypothesis below is an explicit submission condition rather than a theorem
about every translating plan. -/
theorem concrete_margin_example :
    steadyBound 14 1 0 20 1 ≤ (800 : Int) := by
  decide

theorem concrete_margin_bound :
    steadyBound 14 1 0 20 1 ≤ min ((0 : Int) - (-800)) (800 - 0) := by
  decide

end Corpus.Track

#print Corpus.Track.lookAheadIndex
#print Corpus.Track.targetPoint
#print Corpus.Track.commandedCurvature
#print Corpus.Track.errorRec
#print axioms Corpus.Track.error_bound_steady
#print axioms Corpus.Track.corridor_invariant
