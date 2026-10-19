/-
Closed-loop conditional guarantees.

What is modelled (integers; every physical claim is conditional on a named predicate):

1. **Longitudinal receding horizon** (`Cert.decide` re-run every step).  Assumptions:
   `RealisedWithin` — the realised speed² at the next station is in `[0, F₁]` where `F₁`
   is the certified next-station value (the actuator does not overshoot the commanded
   end point); `PreviewConsistent` — the next preview is the current one shifted by one
   station, continued by segments whose caps are `≥ a_lat·r_h` (the declared road
   domain), with terminal cap `≥ a_lat·r_h`; the next observation passes the input guard
   and the gap check.  Proved: the next decision is again `greatest_safe`
   (`longitudinal_step`), hence for ever (`longitudinal_forever`): the car never enters
   the min-violation regime and its speed² stays under the backward envelope.
2. **Lateral tracking with actuation delay** — the first-order pure-pursuit error model
   `D·(e_{k+1} - e_k) = -P·e_{k-d} + D·f_k` (`P/D = v·dt/l_d`, delay `d` control steps,
   pre-history `e_{k-d} = e_0` for `k < d`), with `|f_k| ≤ F` where `F` collects path
   forcing, disturbance, sensor error (`P·η/D`), residual steering bias and rounding.
   Proved (`delayed_small_gain`): under the small-gain inequality
   `P·d·(P·M + D·F) + D²·F ≤ D·P·M` (with `0 ≤ P ≤ D`) and `|e_0| ≤ M`, `|e_k| ≤ M`
   for all `k`.  The model itself (linearised kinematics, unsaturated steering) is an
   assumption; `curvature_demand_safe` / `lateral_accel_safe` give the explicit
   inequalities under which the bound keeps steering unsaturated and lateral
   acceleration within budget.
3. **Safe corridor / obstacle clearance** (`closed_loop_clearance`): if the published
   offset keeps margin `≥ M` inside the controller's certified interval `spareBox`
   and `|e| ≤ M`, the actual offset is inside the corridor and beyond every
   remembered obstacle's edge.
4. **Fallback**: full braking reaches zero speed² in `⌈b₀ / (2·a·d)⌉` stations and
   is pointwise least (`full_brake_reaches_zero`, `full_brake_pointwise_least`).
5. **Bias estimator**: the clamped estimate stays in `[-lim, lim]`.

Not covered: tyre saturation, load transfer, slip, the bicycle/CARLA plant, sensor
outliers beyond `η`, delay variation; none is claimed.
-/
import Corpus.Cert
import Corpus.Controller

namespace Corpus.Closed
open Corpus
open Corpus.Cert

/-! ## 1. Longitudinal receding horizon -/

/-- Certified next-station speed² of an observation. -/
def F1 (c : Cert.Consts) (q : Cert.Q) : Int :=
  (fpass c.a_acc c.a_brk (segs c q) (lastCap c q) (q.v0 * q.v0)).getD 1 0

/-- The realised next speed² lies in `[0, F₁]` (no overshoot of the commanded end point). -/
def RealisedWithin (c : Cert.Consts) (q q' : Cert.Q) : Prop :=
  0 ≤ q'.v0 ∧ q'.v0 * q'.v0 ≤ F1 c q

/-- The next preview is the current one shifted by one station and continued inside the
declared road domain (caps `≥ a_lat·r_h`, nonnegative gaps). -/
def PreviewConsistent (c : Cert.Consts) (q q' : Cert.Q) : Prop :=
  ∃ d cont, 0 ≤ d ∧ (∀ p ∈ cont, c.a_lat * c.r_h ≤ p.1 ∧ 0 ≤ p.2) ∧
    c.a_lat * c.r_h ≤ lastCap c q' ∧
    segs c q' = (segs c q).tail ++ (capAt c q (q.s.length - 1), d) :: cont

/-- The next observation passes the guard and the gap check. -/
def NextValid (c : Cert.Consts) (q' : Cert.Q) : Prop :=
  (wellFormed q' && constsOK c && capsNonneg c q') = true ∧ gapsFeasible c q' = true

theorem decide_gs_of (c : Cert.Consts) (q : Cert.Q)
    (hv : (wellFormed q && constsOK c && capsNonneg c q) = true) (hg : gapsFeasible c q = true)
    (hb : q.v0 * q.v0 ≤ (bpass c.a_brk (segs c q) (lastCap c q)).headD 0) :
    (Cert.decide c q).kind = kindGreatestSafe := by
  unfold Cert.decide
  rw [if_neg (by simp [hv]), if_neg (by simp [hg]), if_neg (by omega)]

/-- **One receding-horizon step.** -/
theorem longitudinal_step (c : Cert.Consts) (q q' : Cert.Q)
    (hd : (Cert.decide c q).kind = kindGreatestSafe)
    (hr : RealisedWithin c q q') (hpc : PreviewConsistent c q q') (hv : NextValid c q') :
    (Cert.decide c q').kind = kindGreatestSafe ∧
    q'.v0 * q'.v0 ≤ (bpass c.a_brk (segs c q') (lastCap c q')).headD 0 := by
  obtain ⟨d, cont, hd0, hc, hE, hsegs⟩ := hpc
  obtain ⟨hD, _, _⟩ := decide_gs_facts c q hd
  have hF := decide_greatest_safe_feasible c q hd
  -- the current certified profile, held at its last value, is feasible on the continuation
  have hext := robust_extend c.a_acc c.a_brk (capAt c q (q.s.length - 1)) (c.a_lat * c.r_h)
    hD.aA hD.aB (segs c q) _ hF d cont (lastCap c q') hd0 hc hE
  obtain ⟨rest, hs⟩ := segs_cons c q hD.two
  -- split the profile: x :: F₁ :: tail
  have hfp : fpass c.a_acc c.a_brk (segs c q) (lastCap c q) (q.v0 * q.v0) =
      q.v0 * q.v0 :: fpass c.a_acc c.a_brk rest (lastCap c q)
        (min ((bpass c.a_brk rest (lastCap c q)).headD 0) (q.v0 * q.v0 + 2 * c.a_acc * (q.s.getD 1 0 - q.s.getD 0 0))) := by
    rw [hs]; rfl
  obtain ⟨t, ht⟩ := fpass_head c.a_acc c.a_brk rest (lastCap c q)
    (min ((bpass c.a_brk rest (lastCap c q)).headD 0) (q.v0 * q.v0 + 2 * c.a_acc * (q.s.getD 1 0 - q.s.getD 0 0)))
  have hF1 : F1 c q = min ((bpass c.a_brk rest (lastCap c q)).headD 0)
      (q.v0 * q.v0 + 2 * c.a_acc * (q.s.getD 1 0 - q.s.getD 0 0)) := by
    unfold F1; rw [hfp, ht]; rfl
  rw [hfp, ht, hs] at hext
  have htail : Feasible c.a_acc c.a_brk (rest ++ (capAt c q (q.s.length - 1), d) :: cont) (lastCap c q')
      (F1 c q :: (t ++ List.replicate (cont.length + 1)
        (lastD (q.v0 * q.v0 :: min ((bpass c.a_brk rest (lastCap c q)).headD 0)
          (q.v0 * q.v0 + 2 * c.a_acc * (q.s.getD 1 0 - q.s.getD 0 0)) :: t)))) := by
    rw [hF1]
    exact hext.2.2.2.2
  have hseg' : segs c q' = rest ++ (capAt c q (q.s.length - 1), d) :: cont := by
    rw [hsegs, hs]; rfl
  have hle := LeAll_head (le_bpass _ _ _ _ _ htail)
  rw [← hseg'] at hle
  have hb : q'.v0 * q'.v0 ≤ (bpass c.a_brk (segs c q') (lastCap c q')).headD 0 :=
    Int.le_trans hr.2 hle
  exact ⟨decide_gs_of c q' hv.1 hv.2 hb, hb⟩

/-- **Persistent longitudinal safety**: along any run whose observations are pairwise
consistent and valid and whose realised speeds never overshoot, every decision is
`greatest_safe` — the car never enters the min-violation (cannot-stop-in-time) regime. -/
theorem longitudinal_forever (c : Cert.Consts) (qs : Nat → Cert.Q)
    (h0 : (Cert.decide c (qs 0)).kind = kindGreatestSafe)
    (hr : ∀ k, RealisedWithin c (qs k) (qs (k + 1)))
    (hpc : ∀ k, PreviewConsistent c (qs k) (qs (k + 1)))
    (hv : ∀ k, NextValid c (qs (k + 1))) :
    ∀ k, (Cert.decide c (qs k)).kind = kindGreatestSafe := by
  intro k
  induction k with
  | zero => exact h0
  | succ k ih => exact (longitudinal_step c (qs k) (qs (k + 1)) ih (hr k) (hpc k) (hv k)).1

/-- The realised speed² respects the cap of the station it reaches. -/
theorem realised_under_cap (c : Cert.Consts) (q q' : Cert.Q)
    (hd : (Cert.decide c q).kind = kindGreatestSafe) (hr : RealisedWithin c q q') (c1 d1 : Int)
    (rest : List (Int × Int)) (hs : segs c q = (capAt c q 0, q.s.getD 1 0 - q.s.getD 0 0) :: (c1, d1) :: rest) :
    q'.v0 * q'.v0 ≤ c1 := by
  have hF := decide_greatest_safe_feasible c q hd
  rw [hs] at hF
  obtain ⟨t, ht⟩ := fpass_head c.a_acc c.a_brk ((c1, d1) :: rest) (lastCap c q)
    (min ((bpass c.a_brk ((c1, d1) :: rest) (lastCap c q)).headD 0)
      (q.v0 * q.v0 + 2 * c.a_acc * (q.s.getD 1 0 - q.s.getD 0 0)))
  have hF1 : F1 c q = min ((bpass c.a_brk ((c1, d1) :: rest) (lastCap c q)).headD 0)
      (q.v0 * q.v0 + 2 * c.a_acc * (q.s.getD 1 0 - q.s.getD 0 0)) := by
    unfold F1; rw [hs]; show (q.v0 * q.v0 :: fpass _ _ _ _ _).getD 1 0 = _; rw [ht]; rfl
  have hfp : fpass c.a_acc c.a_brk ((capAt c q 0, q.s.getD 1 0 - q.s.getD 0 0) :: (c1, d1) :: rest)
      (lastCap c q) (q.v0 * q.v0) = q.v0 * q.v0 :: F1 c q :: (t.drop 0) := by
    show q.v0 * q.v0 :: fpass _ _ _ _ _ = _
    rw [ht, hF1]; rfl
  rw [hfp] at hF
  obtain ⟨_, _, _, _, hF'⟩ := hF
  have hcap : F1 c q ≤ c1 := by
    cases t with
    | nil => exact hF'.elim
    | cons y ys => exact hF'.2.1
  exact Int.le_trans hr.2 hcap

/-! ## 2. Lateral tracking with actuation delay -/

/-- The delayed first-order tracking model (pre-history `e_{k-d} = e_0` for `k < d`). -/
def DelayedEquation (D P : Int) (d : Nat) (e f : Nat → Int) : Prop :=
  ∀ k, D * (e (k + 1) - e k) = -(P * e (k - d)) + D * f k

/-- Named forcing budget: path, disturbance, sensor (already scaled by the gain), bias
residual, rounding.  All nonnegative. -/
def forcingTotal (path disturbance sensor bias rounding : Int) : Int :=
  path + disturbance + sensor + bias + rounding

/-- The small-gain condition (integer form of `M·β·(1 - dβ) ≥ F·(1 + dβ)`, `β = P/D`). -/
def SmallGain (D P F M : Int) (d : Nat) : Prop :=
  0 < D ∧ 0 ≤ P ∧ P ≤ D ∧ 0 ≤ F ∧ 0 ≤ M ∧
  P * (d : Int) * (P * M + D * F) + D * D * F ≤ D * P * M

theorem bound_mul {a x M : Int} (ha : 0 ≤ a) (h1 : -M ≤ x) (h2 : x ≤ M) :
    -(a * M) ≤ a * x ∧ a * x ≤ a * M := by
  have l := Int.mul_le_mul_of_nonneg_left h1 ha
  have r := Int.mul_le_mul_of_nonneg_left h2 ha
  rw [Int.mul_neg] at l
  exact ⟨l, r⟩

/-- Increment bound: one step changes the error by at most `(P·M + D·F)/D`. -/
theorem delayed_increment {D P F M : Int} {d : Nat} {e f : Nat → Int}
    (hg : SmallGain D P F M d) (heq : DelayedEquation D P d e f) (hf : ∀ k, -F ≤ f k ∧ f k ≤ F)
    (j : Nat) (hj : -M ≤ e (j - d) ∧ e (j - d) ≤ M) :
    -(P * M + D * F) ≤ D * (e (j + 1) - e j) ∧ D * (e (j + 1) - e j) ≤ P * M + D * F := by
  obtain ⟨hD, hP, _, _, _, _⟩ := hg
  have h1 := bound_mul hP hj.1 hj.2
  have h2 := bound_mul (Int.le_of_lt hD) (hf j).1 (hf j).2
  have := heq j
  generalize D * (e (j + 1) - e j) = L at *
  generalize P * e (j - d) = X at *
  generalize D * f j = Y at *
  omega

/-- Window bound: over `m ≤ k` steps the error moves by at most `m·(P·M + D·F)/D`. -/
theorem delayed_window {D P F M : Int} {d : Nat} {e f : Nat → Int}
    (hg : SmallGain D P F M d) (heq : DelayedEquation D P d e f) (hf : ∀ k, -F ≤ f k ∧ f k ≤ F)
    (k : Nat) (hprev : ∀ j, j ≤ k → -M ≤ e j ∧ e j ≤ M) :
    ∀ m, m ≤ k → -((m : Int) * (P * M + D * F)) ≤ D * (e k - e (k - m)) ∧
      D * (e k - e (k - m)) ≤ (m : Int) * (P * M + D * F) := by
  intro m
  induction m with
  | zero => intro _; simp
  | succ m ih =>
      intro hm
      obtain ⟨l1, r1⟩ := ih (by omega)
      have hinc := delayed_increment hg heq hf (k - (m + 1))
        (hprev _ (by omega))
      have hk : k - (m + 1) + 1 = k - m := by omega
      rw [hk] at hinc
      have hsplit : D * (e k - e (k - (m + 1))) =
          D * (e k - e (k - m)) + D * (e (k - m) - e (k - (m + 1))) := by
        rw [← Int.mul_add]; congr 1; omega
      have hcast : ((m + 1 : Nat) : Int) * (P * M + D * F) =
          (m : Int) * (P * M + D * F) + (P * M + D * F) := by
        push_cast; rw [Int.add_mul, Int.one_mul]
      rw [hsplit, hcast]
      generalize D * (e k - e (k - m)) = A at *
      generalize D * (e (k - m) - e (k - (m + 1))) = B at *
      generalize (m : Int) * (P * M + D * F) = W at *
      omega

/-- **Delayed small-gain forward invariance**: `|e_0| ≤ M` and the small-gain
inequality imply `|e_k| ≤ M` for every `k`. -/
theorem delayed_small_gain {D P F M : Int} {d : Nat} {e f : Nat → Int}
    (hg : SmallGain D P F M d) (heq : DelayedEquation D P d e f) (hf : ∀ k, -F ≤ f k ∧ f k ≤ F)
    (h0 : -M ≤ e 0 ∧ e 0 ≤ M) :
    ∀ k, -M ≤ e k ∧ e k ≤ M := by
  have strong : ∀ n, ∀ k, k ≤ n → -M ≤ e k ∧ e k ≤ M := by
    intro n
    induction n with
    | zero =>
        intro k hk
        obtain rfl : k = 0 := by omega
        exact h0
    | succ n ih =>
        intro k hk
        by_cases hkn : k ≤ n
        · exact ih k hkn
        · have hk' : k = n + 1 := by omega
          subst hk'
          obtain ⟨hD, hP, hPD, hF, hM, hgain⟩ := hg
          -- window over min(d, n) steps
          have hw : -((d : Int) * (P * M + D * F)) ≤ D * (e n - e (n - d)) ∧
              D * (e n - e (n - d)) ≤ (d : Int) * (P * M + D * F) := by
            by_cases hdn : d ≤ n
            · exact delayed_window ⟨hD, hP, hPD, hF, hM, hgain⟩ heq hf n ih d hdn
            · have hw' := delayed_window ⟨hD, hP, hPD, hF, hM, hgain⟩ heq hf n ih n (Nat.le_refl _)
              have hnd : n - d = n - n := by omega
              rw [hnd]
              have hA : 0 ≤ P * M + D * F :=
                Int.add_nonneg (Int.mul_nonneg hP hM) (Int.mul_nonneg (Int.le_of_lt hD) hF)
              have hmono : (n : Int) * (P * M + D * F) ≤ (d : Int) * (P * M + D * F) :=
                Int.mul_le_mul_of_nonneg_right (by omega) hA
              omega
          have hen := ih n (Nat.le_refl _)
          have hfn := hf n
          have heqn := heq n
          -- D·(D·e_{n+1}) = D(D-P)·e_n + P·(D·(e_n - e_{n-d})) + D·D·f_n
          have hid : D * (D * e (n + 1)) =
              D * (D - P) * e n + P * (D * (e n - e (n - d))) + D * D * f n := by
            have : D * e (n + 1) = D * e n - P * e (n - d) + D * f n := by
              have := heqn; rw [Int.mul_sub] at this; omega
            rw [this]; grind
          have hDP : 0 ≤ D * (D - P) := Int.mul_nonneg (Int.le_of_lt hD) (by omega)
          have b1 := bound_mul hDP hen.1 hen.2
          have b2 := bound_mul hP hw.1 hw.2
          have hDD : 0 ≤ D * D := Int.mul_nonneg (Int.le_of_lt hD) (Int.le_of_lt hD)
          have b3 := bound_mul hDD hfn.1 hfn.2
          -- normalise P·(d·A) to the gain form
          have hpd : P * ((d : Int) * (P * M + D * F)) = P * (d : Int) * (P * M + D * F) := by grind
          have hrhs : D * (D * M) = D * (D - P) * M + D * P * M := by grind
          rw [hpd] at b2
          have up : D * (D * e (n + 1)) ≤ D * (D * M) := by
            rw [hid, hrhs]
            generalize D * (D - P) * e n = X1 at *
            generalize D * (D - P) * M = Y1 at *
            generalize P * (D * (e n - e (n - d))) = X2 at *
            generalize P * (d : Int) * (P * M + D * F) = Y2 at *
            generalize D * D * f n = X3 at *
            generalize D * D * F = Y3 at *
            generalize D * P * M = Z at *
            omega
          have lo : -(D * (D * M)) ≤ D * (D * e (n + 1)) := by
            rw [hid, hrhs]
            generalize D * (D - P) * e n = X1 at *
            generalize D * (D - P) * M = Y1 at *
            generalize P * (D * (e n - e (n - d))) = X2 at *
            generalize P * (d : Int) * (P * M + D * F) = Y2 at *
            generalize D * D * f n = X3 at *
            generalize D * D * F = Y3 at *
            generalize D * P * M = Z at *
            omega
          have u1 := Int.le_of_mul_le_mul_left up hD
          have u2 := Int.le_of_mul_le_mul_left u1 hD
          have l0 : D * (D * (-M)) ≤ D * (D * e (n + 1)) := by
            rw [Int.mul_neg, Int.mul_neg]; exact lo
          have l1 := Int.le_of_mul_le_mul_left l0 hD
          have l2 := Int.le_of_mul_le_mul_left l1 hD
          exact ⟨l2, u2⟩
  intro k
  exact strong k k (Nat.le_refl _)

/-- Non-vacuity: `v = 10 m/s`, `dt = 0.05 s`, `l_d = 8 m` (`β = 1/16`), a two-step
actuator delay, and forcing `6 mm/step` admit the invariant bound `M = 124 mm`. -/
theorem concrete_small_gain : SmallGain 160 10 6 124 2 := by
  unfold SmallGain; decide

/-- ... and a run of the model stays within it (kernel-evaluated for 200 steps with
worst-case alternating forcing, from the bound itself). -/
def simErr (D P F : Int) (d : Nat) (e0 : Int) : Nat → List Int
  | 0 => [e0]
  | n + 1 =>
      let h := simErr D P F d e0 n
      let ek := h.headD 0
      let ed := h.getD d (h.getLast?.getD e0)
      let fk := if ek ≥ 0 then F else -F
      ((D * ek - P * ed + D * fk) / D) :: h

set_option maxRecDepth 100000 in
theorem concrete_run_bounded :
    (simErr 160 10 6 2 124 200).all (fun x => decide (-124 ≤ x ∧ x ≤ 124)) = true := by decide

/-! ## 3. Curvature demand and lateral acceleration under the error bound -/

def CurvatureSafe (kpath M ld kmax : Int) : Prop :=
  0 ≤ kpath ∧ 0 ≤ M ∧ 0 < ld ∧ 0 ≤ kmax ∧
  kpath * ld * ld + 2 * M ≤ kmax * ld * ld

def LateralAccelSafe (v kpath M ld alat : Int) : Prop :=
  0 ≤ v ∧ 0 ≤ kpath ∧ 0 ≤ M ∧ 0 < ld ∧ 0 ≤ alat ∧
  v * v * (kpath * ld * ld + 2 * M) ≤ alat * ld * ld

/-- With `|e| ≤ M`, the pure-pursuit curvature demand `κ_path + 2|e|/l_d²` stays
under `κ_max` (steering unsaturated) when `CurvatureSafe` holds. -/
theorem curvature_demand_safe {e kpath M ld kmax : Int}
    (he : -M ≤ e ∧ e ≤ M) (hs : CurvatureSafe kpath M ld kmax) :
    kpath * ld * ld + 2 * Cert.iabs e ≤ kmax * ld * ld := by
  rcases hs with ⟨_, _, _, _, h⟩
  unfold Cert.iabs
  split <;> omega

/-- ... and the lateral acceleration `v²(κ_path + 2|e|/l_d²)` stays under `a_lat`. -/
theorem lateral_accel_safe {e v kpath M ld alat : Int}
    (he : -M ≤ e ∧ e ≤ M) (hs : LateralAccelSafe v kpath M ld alat) :
    v * v * (kpath * ld * ld + 2 * Cert.iabs e) ≤ alat * ld * ld := by
  rcases hs with ⟨hv, _, _, _, _, h⟩
  have hv2 : 0 ≤ v * v := Int.mul_nonneg hv hv
  have hi : kpath * ld * ld + 2 * Cert.iabs e ≤ kpath * ld * ld + 2 * M := by
    unfold Cert.iabs; split <;> omega
  exact Int.le_trans (Int.mul_le_mul_of_nonneg_left hi hv2) h

theorem concrete_curvature_instance : CurvatureSafe 0 10 8 1 ∧ LateralAccelSafe 10 0 10 8 32 := by
  unfold CurvatureSafe LateralAccelSafe; decide

/-! ## 4. Safe corridor and obstacle clearance in closed loop -/

def corridorMargin (lo hi x M : Int) : Prop := lo + M ≤ x ∧ x ≤ hi - M

theorem corridor_forward_invariant {lo hi x e M : Int}
    (hm : corridorMargin lo hi x M) (he : -M ≤ e ∧ e ≤ M) :
    lo ≤ x + e ∧ x + e ≤ hi := by
  rcases hm with ⟨hl, hh⟩; omega

/-- **Closed-loop clearance**: if at every step the published offset `x k` keeps
margin `M` inside the controller's certified interval, and the tracking error obeys
the delayed model with the small-gain condition, then the ACTUAL offset `x k + e k`
is inside the corridor and beyond the edge of every remembered obstacle covering
the station, at every step. -/
theorem closed_loop_clearance (c : Steer.Consts) (q : Nat → Steer.Q)
    (mem : Nat → List Controller.MemoryOb) (i : Nat → Nat) (x : Nat → Int)
    {D P F M : Int} {d : Nat} {e f : Nat → Int}
    (hg : SmallGain D P F M d) (heq : DelayedEquation D P d e f)
    (hf : ∀ k, -F ≤ f k ∧ f k ≤ F) (h0 : -M ≤ e 0 ∧ e 0 ≤ M)
    (hplan : ∀ k, corridorMargin (Controller.spareBox c (q k) (mem k) (i k)).1
      (Controller.spareBox c (q k) (mem k) (i k)).2 (x k) M) :
    ∀ k, (Steer.lo c (q k) (i k) ≤ x k + e k ∧ x k + e k ≤ Steer.hi c (q k) (i k)) ∧
      ∀ o ∈ mem k, Controller.covers c.lon.hold ((q k).base.s.getD (i k) 0) o →
        (o.side > 0 → o.lmax + Steer.halfw c + c.margin ≤ x k + e k) ∧
        (o.side < 0 → x k + e k ≤ o.lmin - Steer.halfw c - c.margin) := by
  intro k
  have he := delayed_small_gain hg heq hf h0 k
  have hin := corridor_forward_invariant (hplan k) he
  exact Controller.spareBox_sound c (q k) (mem k) (i k) (x k + e k) hin.1 hin.2

/-! ## 5. Fallback: full braking -/

def fullBrakeStep (a d b : Int) : Int := max 0 (b - 2 * a * d)
def fullBrakeProfile (a d b : Int) : Nat → Int
  | 0 => b
  | n + 1 => fullBrakeStep a d (fullBrakeProfile a d b n)

theorem full_brake_le (a d b : Int) (had : 0 ≤ 2 * a * d) :
    ∀ n : Nat, fullBrakeProfile a d b n ≤ max 0 (b - (n : Int) * (2 * a * d)) := by
  intro n
  induction n with
  | zero => simp [fullBrakeProfile]; omega
  | succ n ih =>
      have hcast : ((n + 1 : Nat) : Int) * (2 * a * d) = (n : Int) * (2 * a * d) + 2 * a * d := by
        push_cast; rw [Int.add_mul, Int.one_mul]
      simp only [fullBrakeProfile, fullBrakeStep]
      rw [hcast]
      generalize fullBrakeProfile a d b n = p at *
      generalize (n : Int) * (2 * a * d) = w at *
      generalize 2 * a * d = s at *
      omega

/-- **Full braking stops**: after `n ≥ 1` stations with `b ≤ n·2·a·d` the speed² is 0. -/
theorem full_brake_reaches_zero (a d b : Int) (had : 0 ≤ 2 * a * d) (n : Nat) (hn : 1 ≤ n)
    (hb : b ≤ (n : Int) * (2 * a * d)) : fullBrakeProfile a d b n = 0 := by
  have h := full_brake_le a d b had n
  have hpos : 0 ≤ fullBrakeProfile a d b n := by
    obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
    simp only [fullBrakeProfile, fullBrakeStep]; omega
  omega

theorem full_brake_pointwise_least (a : Int) (ds : List Int) (x x' : Int) (t : List Int)
    (hp : DynPossible a ds (x' :: t)) (hx : x ≤ x') :
    LeAll (brakeProfile a ds x) (x' :: t) :=
  brake_least a ds x x' t hp hx

/-! ## 6. Steering-bias estimator -/

def clampB (lim x : Int) : Int := max (-lim) (min x lim)
def biasEstimateStep (lim gNum gDen beta betaHat : Int) : Int :=
  clampB lim (betaHat + gNum * (beta - betaHat) / gDen)

theorem bias_estimator_bounded {lim gNum gDen beta betaHat : Int} (hlim : 0 ≤ lim) :
    -lim ≤ biasEstimateStep lim gNum gDen beta betaHat ∧
      biasEstimateStep lim gNum gDen beta betaHat ≤ lim := by
  unfold biasEstimateStep clampB; omega

end Corpus.Closed
