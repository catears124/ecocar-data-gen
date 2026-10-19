import Corpus.Cert
import Corpus.Quant

/-!
Kernel-checked closure facts for the longitudinal certificate.  The module separates the
boolean input classification from the profile proofs in `Corpus.Cert`; all physics constants
remain explicit integer inputs, while arithmetic side conditions are stated as hypotheses.
-/

namespace Corpus.CertProofs

open Corpus.Cert

/-- The valid branch predicate used by the four-way decision classification. -/
def Valid (c : Consts) (q : Q) : Prop :=
  (wellFormed q && constsOK c && capsNonneg c q) = true

def Gaps (c : Consts) (q : Q) : Prop := gapsFeasible c q = true

/-- The backward-pass budget at station 0. -/
def budget (c : Consts) (q : Q) : Int := (bpass c.a_brk (segs c q) (lastCap c q)).headD 0

/-- **Input-domain validation and failure classification**: the four kinds are
exactly the four disjoint cases of (valid input, gaps fit, start within budget). -/
theorem decide_classification (c : Consts) (q : Q) :
    ((Cert.decide c q).kind = kindInvalid ↔ ¬ Valid c q) ∧
    ((Cert.decide c q).kind = kindNoGap ↔ Valid c q ∧ ¬ Gaps c q) ∧
    ((Cert.decide c q).kind = kindMinViolation ↔ Valid c q ∧ Gaps c q ∧ budget c q < q.v0 * q.v0) ∧
    ((Cert.decide c q).kind = kindGreatestSafe ↔ Valid c q ∧ Gaps c q ∧ q.v0 * q.v0 ≤ budget c q) := by
  unfold Valid Gaps budget
  cases hv : (wellFormed q && constsOK c && capsNonneg c q) with
  | false =>
      have e : (Cert.decide c q).kind = kindInvalid := by unfold Cert.decide; rw [if_pos (by simp [hv])]
      rw [e]; simp [kindInvalid, kindNoGap, kindMinViolation, kindGreatestSafe]
  | true =>
    cases hg : gapsFeasible c q with
    | false =>
        have e : (Cert.decide c q).kind = kindNoGap := by
          unfold Cert.decide; rw [if_neg (by simp [hv]), if_pos (by simp [hg])]
        rw [e]; simp [kindInvalid, kindNoGap, kindMinViolation, kindGreatestSafe]
    | true =>
        by_cases hb : q.v0 * q.v0 > (bpass c.a_brk (segs c q) (lastCap c q)).headD 0
        · have e : (Cert.decide c q).kind = kindMinViolation := by
            unfold Cert.decide; rw [if_neg (by simp [hv]), if_neg (by simp [hg]), if_pos hb]
          have hh : (bpass c.a_brk (segs c q) (lastCap c q)).headD 0 =
              (bpass c.a_brk (segs c q) (lastCap c q)).head?.getD 0 := by
            cases bpass c.a_brk (segs c q) (lastCap c q) <;> rfl
          rw [e]; simp [kindInvalid, kindNoGap, kindMinViolation, kindGreatestSafe]; omega
        · have e : (Cert.decide c q).kind = kindGreatestSafe := by
            unfold Cert.decide; rw [if_neg (by simp [hv]), if_neg (by simp [hg]), if_neg hb]
          have hh : (bpass c.a_brk (segs c q) (lastCap c q)).headD 0 =
              (bpass c.a_brk (segs c q) (lastCap c q)).head?.getD 0 := by
            cases bpass c.a_brk (segs c q) (lastCap c q) <;> rfl
          rw [e]; simp [kindInvalid, kindNoGap, kindMinViolation, kindGreatestSafe]; omega

/-- **Exact infeasible-start detection**: on a valid input, a feasible profile from
the current speed exists iff the start is within the backward-pass budget. -/
theorem feasible_exists_iff (c : Consts) (q : Q) (hv : Valid c q) :
    (∃ t, Feasible c.a_acc c.a_brk (segs c q) (lastCap c q) (q.v0 * q.v0 :: t)) ↔
      q.v0 * q.v0 ≤ budget c q := by
  have hD := inDomain_of_valid c q hv
  constructor
  · intro ⟨t, ht⟩
    exact LeAll_head (le_bpass _ _ _ _ _ ht)
  · intro hb
    have hf := fpass_feasible _ _ hD.aA hD.aB _ _ _ hD.segsOK (Int.mul_nonneg hD.v0 hD.v0) hb
    obtain ⟨t, ht⟩ := fpass_head c.a_acc c.a_brk (segs c q) (lastCap c q) (q.v0 * q.v0)
    rw [ht] at hf
    exact ⟨t, hf⟩

/-- Hence, on valid inputs with fitting gaps: `greatest_safe` iff a feasible
profile exists, `min_violation` iff none does. -/
theorem decide_gs_iff_feasible (c : Consts) (q : Q) (hv : Valid c q) (hg : Gaps c q) :
    (Cert.decide c q).kind = kindGreatestSafe ↔
      ∃ t, Feasible c.a_acc c.a_brk (segs c q) (lastCap c q) (q.v0 * q.v0 :: t) := by
  rw [feasible_exists_iff c q hv, (decide_classification c q).2.2.2]
  exact ⟨fun h => h.2.2, fun h => ⟨hv, hg, h⟩⟩

/-- Every entry of the certified profile is nonnegative and under its cap (and the
last under the terminal cap): restatement of `Feasible` as an indexed fact. -/
theorem feasible_nonneg_caps (aA aB : Int) : ∀ (sg : List (Int × Int)) (cN : Int) (b : List Int),
    Feasible aA aB sg cN b → ∀ i, i < sg.length → 0 ≤ b.getD i 0 ∧ b.getD i 0 ≤ (sg.getD i (0, 0)).1
  | [], _, _, _, i, hi => by simp at hi
  | (cc, d) :: rest, cN, x :: y :: t, ⟨h0, hc, _, _, hf⟩, i, hi => by
      cases i with
      | zero => exact ⟨h0, hc⟩
      | succ i =>
          have := feasible_nonneg_caps aA aB rest cN (y :: t) hf i (by simpa using hi)
          simpa using this
  | (_, _) :: _, _, [], h, _, _ => h.elim
  | (_, _) :: _, _, [_], h, _, _ => h.elim

/-- Every feasible profile is bounded by the backward pass, including its first entry. -/
theorem bpass_greatest (aA aB : Int) (segs : List (Int × Int)) (cN : Int)
    (b : List Int) (hf : Feasible aA aB segs cN b) :
    LeAll b (bpass aB segs cN) := le_bpass aA aB segs cN b hf

/-- The forward pass is feasible and pointwise greatest among profiles with the given start. -/
theorem fpass_profile_greatest (aA aB : Int) (haA : 0 ≤ aA) (haB : 0 ≤ aB)
    (segs : List (Int × Int)) (cN x : Int) (hs : SegsOK segs cN) (hx : 0 ≤ x)
    (hB : x ≤ (bpass aB segs cN).headD 0) :
    Feasible aA aB segs cN (fpass aA aB segs cN x) :=
  fpass_feasible aA aB haA haB segs cN x hs hx hB

theorem fpass_profile_pointwise_greatest (aA aB : Int) (segs : List (Int × Int))
    (cN x x' : Int) (t : List Int) (hf : Feasible aA aB segs cN (x' :: t))
    (hx : x' ≤ x) : LeAll (x' :: t) (fpass aA aB segs cN x) :=
  fpass_greatest aA aB segs cN x x' t hf hx

/-- Floor rounding for an integer speed-squared target and positive station spacing. -/
theorem accel_floor_rounding (F b d : Int) (hd : 0 < d) :
    b + 2 * ((F - b) / (2 * d)) * d ≤ F ∧
      F < b + 2 * ((F - b) / (2 * d) + 1) * d := by
  have h2 : 0 < 2 * d := by omega
  have hlo := Int.ediv_mul_le (F - b) (by omega : 2 * d ≠ 0)
  have hup := Int.lt_ediv_add_one_mul_self (F - b) h2
  generalize (F - b) / (2 * d) = A at hlo hup ⊢
  have he : A * (2 * d) = 2 * A * d := by grind
  have he2 : (A + 1) * (2 * d) = 2 * A * d + 2 * d := by grind
  have hlow : b + 2 * A * d ≤ F := by omega
  have hu : F - b < 2 * A * d + 2 * d := by
    rw [← he2]
    exact hup
  have he3 : 2 * (A + 1) * d = 2 * A * d + 2 * d := by grind
  constructor
  · exact hlow
  · rw [he3]
    omega

/-- A greatest-safe decision enforces every visible cap and its terminal (horizon) cap. -/
theorem decide_greatest_safe_caps (c : Consts) (q : Q)
    (hd : (Cert.decide c q).kind = kindGreatestSafe) :
    Feasible c.a_acc c.a_brk (segs c q) (lastCap c q)
      (fpass c.a_acc c.a_brk (segs c q) (lastCap c q) (q.v0 * q.v0)) :=
  decide_greatest_safe_feasible c q hd


/-- Robust continuation assumption, stated independently of the planner implementation. -/
def InDomainContinuation (c : Consts) (cont : List (Int × Int)) (cE d : Int) : Prop :=
  0 ≤ d ∧ (∀ p ∈ cont, c.a_lat * c.r_h ≤ p.1 ∧ 0 ≤ p.2) ∧ c.a_lat * c.r_h ≤ cE

theorem decide_robust_under (c : Consts) (q : Q)
    (hd : (Cert.decide c q).kind = kindGreatestSafe) (cont : List (Int × Int)) (cE d : Int)
    (hcont : InDomainContinuation c cont cE d) :
    let F := fpass c.a_acc c.a_brk (segs c q) (lastCap c q) (q.v0 * q.v0)
    Feasible c.a_acc c.a_brk (segs c q ++ (capAt c q (q.s.length - 1), d) :: cont) cE
      (F ++ List.replicate (cont.length + 1) (lastD F)) := by
  dsimp [InDomainContinuation] at hcont
  exact decide_robust_extend c q hd d cont cE hcont.1 hcont.2.1 hcont.2.2

end Corpus.CertProofs
