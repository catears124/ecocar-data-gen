/-
Kernel-checked composition facts for steer-2.  The main result below links the
implemented decision's plan_ok branch to non-negative recorded clearance under
an explicit finite, decidable corridor/source domain.  Selection and geometry
facts are stated with their exact assumptions; no simulator or tracking
assumption is hidden in the definitions.
-/
import Corpus.SteerFeas
import Corpus.SteerRamp

namespace Corpus.SteerProofs
open Corpus Corpus.Steer
open Corpus.SteerRamp

/-- Every clamped station is in its nonempty integer corridor. -/
theorem offAt_in_corridor (c : Consts) (q : Q) (i : Nat) (si : Int)
    (hlo : lo c q i ≤ hi c q i) :
    lo c q i ≤ offAt c q si i ∧ offAt c q si i ≤ hi c q i := by
  exact SteerFeas.clamp_in_corridor c q i (rawOffAt c q si) hlo

/-- The list-level corridor fact, expressed at each generated station. -/
theorem offs_in_corridor (c : Consts) (q : Q)
    (hcorr : ∀ i, i < q.base.s.length → lo c q i ≤ hi c q i) :
    ∀ i, i < q.base.s.length →
      lo c q i ≤ offAt c q (q.base.s.getD i 0) i ∧
      offAt c q (q.base.s.getD i 0) i ≤ hi c q i := by
  intro i hi
  exact offAt_in_corridor c q i (q.base.s.getD i 0) (hcorr i hi)

/-- The finite and decidable assumptions consumed by the clearance theorem.
The source-side condition is supplied separately: it is the explicit bridge
from selector/geometry reasoning to the clearance arithmetic. -/
def clearanceDomain (c : Consts) (q : Q) : Bool :=
  decide (wellFormed c q = true ∧
    2 * halfw c = c.lon.veh_width ∧
    0 < c.lon.ramp_in ∧
    ∀ i, i < q.base.s.length → lo c q i ≤ hi c q i)

theorem clearanceDomain_sound (c : Consts) (q : Q)
    (h : clearanceDomain c q = true) :
    wellFormed c q = true ∧
    2 * halfw c = c.lon.veh_width ∧
    0 < c.lon.ramp_in ∧
    ∀ i, i < q.base.s.length → lo c q i ≤ hi c q i := by
  simpa [clearanceDomain] using of_decide_eq_true h

/-- Outside every selected hold, the recorded clearance is nonnegative solely
from corridor containment. -/
theorem outside_hold_clearance
    (c : Consts) (q : Q) (i : Nat) (hlo : lo c q i ≤ hi c q i)
    (si : Int) (x : Int) (hout : ∀ p, rawPlanAt c q si = some p →
      si < p.d - c.lon.hold ∨ p.d + c.lon.hold < si) :
    lo c q i ≤ x ∧ x ≤ hi c q i → 0 ≤ clearAt c q si i x := by
  intro hc
  exact SteerFeas.clearAt_outside_nonneg c q i si x hc hout

/-- A selector proof can be discharged locally when the designated obstacle is
strictly heavier than every other well-formed obstacle.  This is the exact
non-tie condition required by the first-strictly-largest fold. -/
def strictSupplier (c : Consts) (q : Q) (si : Int) (o : List Int) : Prop :=
  Cert.obsTriple o ≠ none ∧ o ∈ q.base.obstacles ∧
    0 < (match Cert.obsTriple o with | some (d, _, _) => wOb c d si | none => 0) ∧
    ∀ z ∈ q.base.obstacles, z ≠ o →
      (match Cert.obsTriple z with | some (d, _, _) => wOb c d si | none => 0) <
      (match Cert.obsTriple o with | some (d, _, _) => wOb c d si | none => 0)

/-- Selection correctness for a singleton obstacle list (a kernel-checkable
base case used by list induction in downstream selector proofs). -/
theorem selection_singleton
    (c : Consts) (q : Q) (q' : Q) (o : List Int) (si d lmin lmax : Int)
    (hobs : q'.base.obstacles = [o])
    (ho : o = [d, lmin, lmax])
    (hw : 0 < wOb c d si) :
    rawPlanAt c q' si =
      some (Plan.mk d lmin lmax
        (sideAmp q' d lmin lmax).1 (sideAmp q' d lmin lmax).2) := by
  rw [ho] at hobs
  simp only [rawPlanAt, rawPlanAt.go, hobs, Cert.obsTriple]
  rw [if_pos hw]

/-- The fixed ramp-out cap is strict before the end of a ramp, preventing a
ramp-out supplier from tying a hold supplier at weight 1000. -/
theorem wOb_ramp_out_lt (c : Consts) (d si : Int)
    (hpre : d - c.lon.hold - c.lon.ramp_in ≤ si)
    (hentry : ¬ si < d - c.lon.hold)
    (h0 : d + c.lon.hold < si)
    (h1 : si ≤ d + c.lon.hold + c.lon.ramp_out) :
    wOb c d si ≤ 999 := by
  have hnpre : ¬ si < d - c.lon.hold - c.lon.ramp_in := by omega
  have nhold : ¬ si ≤ d + c.lon.hold := by omega
  simp [wOb, hnpre, hentry, nhold, h1]
  exact Int.min_le_left _ _

/-- Old steer-1 ramp-out weight ties at the concrete regression station. -/
def wObOld (hold rout d si : Int) : Int :=
  if si ≤ d + hold + rout then 1000 - ramp 1000 (si - d - hold) rout else 0

theorem old_weight_regression :
    wObOld 3000 15000 20900 24000 = 1000 ∧
    min 999 (wObOld 3000 15000 20900 24000) = 999 := by decide

theorem new_weight_regression :
    min 999 (1000 - ramp 1000 (24000 - 20900 - 3000) 15000) = 999 := by decide

/-- Exact steering range, independent of the curvature value. -/
theorem steerAt_range (c : Consts) (κ : Int) :
    -1000 ≤ steerAt c κ ∧ steerAt c κ ≤ 1000 := by
  simp [steerAt]
  omega

/-- Exact unsaturated condition for nonnegative curvature and positive limit. -/
theorem steerAt_unsaturated_of_condition (c : Consts) (κ : Int)
    (hk : 0 < c.kappa_max) (hκ : 0 ≤ κ)
    (hbound : 1000 * κ ≤ 999 * c.kappa_max) :
    steerAt c κ = Cert.cdiv (κ * 1000) c.kappa_max := by
  have hnonneg : 0 ≤ Cert.cdiv (κ * 1000) c.kappa_max := by
    unfold Cert.cdiv
    have hq : (-(κ * 1000)) / c.kappa_max ≤ 0 := by
      apply Int.ediv_le_of_le_mul hk
      omega
    omega
  have hcap : Cert.cdiv (κ * 1000) c.kappa_max ≤ 999 := by
    unfold Cert.cdiv
    have hneg : -999 ≤ (-(κ * 1000)) / c.kappa_max := by
      apply Int.le_ediv_of_mul_le hk
      omega
    omega
  simp only [steerAt]
  rw [show max c.kappa_max 1 = c.kappa_max by omega]
  have hmin : min 1000 (Cert.cdiv (κ * 1000) c.kappa_max) =
      Cert.cdiv (κ * 1000) c.kappa_max := by omega
  have hmax : max (-1000) (Cert.cdiv (κ * 1000) c.kappa_max) =
      Cert.cdiv (κ * 1000) c.kappa_max := by omega
  rw [hmin, hmax]

/-- A hold supplier has exactly the maximum selector weight. -/
theorem wOb_hold (c : Consts) (d si : Int)
    (hri : 0 < c.lon.ramp_in)
    (h0 : d - c.lon.hold ≤ si) (h1 : si ≤ d + c.lon.hold) :
    wOb c d si = 1000 := by
  have hn : ¬ si < d - c.lon.hold - c.lon.ramp_in := by omega
  have he : ¬ si < d - c.lon.hold := by omega
  simp [wOb, hn, he, h1]

/-- Outside a hold window, every active ramp supplier is capped below 1000. -/
theorem wOb_le_999_outside_hold (c : Consts) (d si : Int)
    (hri : 0 < c.lon.ramp_in)
    (hout : si < d - c.lon.hold ∨ d + c.lon.hold < si) :
    wOb c d si ≤ 999 := by
  by_cases hpre : si < d - c.lon.hold - c.lon.ramp_in
  · simp [wOb, hpre]
  · by_cases hin : si < d - c.lon.hold
    · have hx0 : 0 ≤ si - d + c.lon.hold + c.lon.ramp_in := by omega
      have hxL : si - d + c.lon.hold + c.lon.ramp_in < c.lon.ramp_in := by omega
      have hr := ramp_1000_lt
        (si - d + c.lon.hold + c.lon.ramp_in) c.lon.ramp_in hx0 hxL hri
      simp [wOb, hpre, hin, hr]
    · have hhold : ¬ si ≤ d + c.lon.hold := by omega
      by_cases hout' : si ≤ d + c.lon.hold + c.lon.ramp_out
      · simp [wOb, hpre, hin, hhold, hout']
        exact Int.min_le_left _ _
      · have hafter : d + c.lon.hold + c.lon.ramp_out < si := by omega
        simp [wOb, hpre, hin, hhold, hout', hafter]
/-- A true finite feasibility equivalence for the implemented branch. -/
theorem plan_ok_iff_wellFormed_noGap (c : Consts) (q : Q)
    (hw : wellFormed c q = true) :
    (decide c q).kind = kindPlanOk ↔ hasNoGap c q = false := by
  exact decide_plan_ok_iff c q hw

/-- A fold whose current winner dominates every remaining obstacle is stable. -/
theorem go_stable (c : Consts) (q : Q) (si : Int) :
    ∀ (l : List (List Int)) (bestW : Int) (best : Option Plan),
      (∀ ob ∈ l, ∀ d lmin lmax, Cert.obsTriple ob = some (d,lmin,lmax) →
        wOb c d si ≤ bestW) →
      rawPlanAt.go c q si bestW best l = best := by
  intro l
  induction l with
  | nil => intro bestW best h; rfl
  | cons ob tl ih =>
      intro bestW best h
      simp only [rawPlanAt.go]
      cases ho : Cert.obsTriple ob with
      | none =>
          apply ih bestW best
          intro z hz d lmin lmax hztr
          exact h z (by simp [hz]) d lmin lmax hztr
      | some triple =>
          obtain ⟨d,lmin,lmax⟩ := triple
          have hw := h ob (by simp) d lmin lmax ho
          have hnot : ¬ bestW < wOb c d si := by omega
          simp [hnot]
          apply ih bestW best
          intro z hz d' lmin' lmax' hztr
          exact h z (by simp [hz]) d' lmin' lmax' hztr

/-- A 1000-weight obstacle wins after any prefix bounded by 999 and remains
the winner through a post-list bounded by 1000. -/
theorem go_prefix (c : Consts) (q : Q) (si : Int)
    (pre post : List (List Int)) (bestW : Int) (best : Option Plan)
    (o : List Int) (d lmin lmax : Int)
    (ho : o = [d,lmin,lmax]) (hbest : bestW ≤ 999)
    (hpre : ∀ ob ∈ pre, ∀ d' lmin' lmax',
      Cert.obsTriple ob = some (d',lmin',lmax') → wOb c d' si ≤ 999)
    (hwo : wOb c d si = 1000)
    (hpost : ∀ ob ∈ post, ∀ d' lmin' lmax',
      Cert.obsTriple ob = some (d',lmin',lmax') → wOb c d' si ≤ 1000) :
    rawPlanAt.go c q si bestW best (pre ++ o :: post) =
      some (Plan.mk d lmin lmax (sideAmp q d lmin lmax).1
        (sideAmp q d lmin lmax).2) := by
  induction pre generalizing bestW best with
  | nil =>
      rw [ho]
      simp only [List.nil_append, rawPlanAt.go, Cert.obsTriple]
      rw [hwo]
      have hb : bestW < 1000 := by omega
      rw [if_pos hb]
      apply go_stable c q si post 1000
        (some (Plan.mk d lmin lmax (sideAmp q d lmin lmax).1
          (sideAmp q d lmin lmax).2))
      intro z hz d' lmin' lmax' htr
      exact hpost z hz d' lmin' lmax' htr
  | cons ob pre ih =>
      simp only [List.cons_append, rawPlanAt.go]
      cases hx : Cert.obsTriple ob with
      | none =>
          refine ih bestW best hbest ?_
          intro z hz
          exact hpre z (by simp [hz])
      | some triple =>
          obtain ⟨d',lmin',lmax'⟩ := triple
          simp only [hx]
          have hw := hpre ob (by simp) d' lmin' lmax' hx
          by_cases hb : bestW < wOb c d' si
          · rw [if_pos hb]
            refine ih (wOb c d' si)
              (some (Plan.mk d' lmin' lmax'
                (sideAmp q d' lmin' lmax').1
                (sideAmp q d' lmin' lmax').2)) (by omega) ?_
            intro z hz d'' lmin'' lmax'' htr
            exact hpre z (by simp [hz]) d'' lmin'' lmax'' htr
          · rw [if_neg hb]
            refine ih bestW best hbest ?_
            intro z hz d'' lmin'' lmax'' htr
            exact hpre z (by simp [hz]) d'' lmin'' lmax'' htr

/-- Any entry of a disjointness list is separated from a designated distance. -/
theorem disjointAt_mem (hold d : Int) : ∀ xs,
    disjointAt hold d xs = true →
    ∀ z ∈ xs, ∀ d' lmin' lmax', Cert.obsTriple z = some (d',lmin',lmax') →
      d' - hold > d + hold ∨ d - hold > d' + hold := by
  intro xs
  induction xs with
  | nil => intro h z hz; simp at hz
  | cons ob tl ih =>
      intro h z hz d' lmin' lmax' hztr
      rcases List.mem_cons.mp hz with hzo | hz
      · subst z
        cases ht : Cert.obsTriple ob with
        | none => simp [disjointAt, ht] at h
        | some triple =>
            obtain ⟨d0,l0,m0⟩ := triple
            have hh : (decide (d0 - hold > d + hold ∨ d - hold > d0 + hold) &&
                disjointAt hold d tl) = true := by
              simpa [disjointAt, ht] using h
            have hc : decide (d0 - hold > d + hold ∨ d - hold > d0 + hold) = true :=
              ((Bool.and_eq_true _ _).mp hh).1
            have hd : d0 - hold > d + hold ∨ d - hold > d0 + hold := of_decide_eq_true hc
            simp [ht] at hztr
            rcases hztr with ⟨rfl, rfl, rfl⟩
            exact hd
      · cases ht : Cert.obsTriple ob with
        | none => simp [disjointAt, ht] at h
        | some triple =>
            obtain ⟨d0,l0,m0⟩ := triple
            have hh : (decide (d0 - hold > d + hold ∨ d - hold > d0 + hold) &&
                disjointAt hold d tl) = true := by
              simpa [disjointAt, ht] using h
            have ht' : disjointAt hold d tl = true :=
              ((Bool.and_eq_true _ _).mp hh).2
            exact ih ht' z hz d' lmin' lmax' hztr
/-- Pairwise disjoint hold windows separate every other triple from a
designated obstacle throughout that obstacle's hold. -/
theorem hsep_of_pairwise
    (hold : Int) (pre post : List (List Int)) (o : List Int)
    (d lmin lmax : Int) (ho : Cert.obsTriple o = some (d,lmin,lmax))
    (si : Int) (h0 : d - hold ≤ si) (h1 : si ≤ d + hold)
    (hp : pairwiseDisjoint hold (pre ++ o :: post) = true) :
    ∀ z ∈ pre ++ post, ∀ d' lmin' lmax',
      Cert.obsTriple z = some (d',lmin',lmax') →
        (si < d' - hold ∨ d' + hold < si) := by
  induction pre with
  | nil =>
      intro z hz d' lmin' lmax' hztr
      have hh : disjointAt hold d post = true ∧
          pairwiseDisjoint hold (o :: post) = true := by
        simpa [pairwiseDisjoint, ho] using hp
      have hd := disjointAt_mem hold d post hh.1 z hz d' lmin' lmax' hztr
      rcases hd with h | h
      · exact Or.inl (by omega)
      · exact Or.inr (by omega)
  | cons x pre ih =>
      intro z hz d' lmin' lmax' hztr
      cases hx : Cert.obsTriple x with
      | none => simp [pairwiseDisjoint, hx] at hp
      | some triple =>
          obtain ⟨dx,lx,mx⟩ := triple
          have hh : disjointAt hold dx (pre ++ o :: post) = true ∧
              pairwiseDisjoint hold (pre ++ o :: post) = true := by
            simpa [pairwiseDisjoint, hx] using hp
          rcases List.mem_cons.mp hz with hzx | hz
          · subst z
            have hztr' := hztr
            simp [hx] at hztr'
            rcases hztr' with ⟨rfl, rfl, rfl⟩
            have hxo := disjointAt_mem hold dx (pre ++ o :: post) hh.1
              o (by simp) d lmin lmax ho
            rcases hxo with h | h
            · exact Or.inr (by omega)
            · exact Or.inl (by omega)
          · exact ih hh.2 z hz d' lmin' lmax' hztr

/-- Hold-window selection from pairwise-disjoint obstacles. -/
theorem rawPlanAt_hold
    (c : Consts) (q : Q) (si : Int)
    (pre post : List (List Int)) (o : List Int)
    (d lmin lmax : Int)
    (hobs : q.base.obstacles = pre ++ o :: post)
    (ho : Cert.obsTriple o = some (d,lmin,lmax))
    (hol : o = [d,lmin,lmax])
    (hri : 0 < c.lon.ramp_in)
    (hp : pairwiseDisjoint c.lon.hold (pre ++ o :: post) = true)
    (h0 : d - c.lon.hold ≤ si) (h1 : si ≤ d + c.lon.hold) :
    rawPlanAt c q si =
      some (Plan.mk d lmin lmax (sideAmp q d lmin lmax).1
        (sideAmp q d lmin lmax).2) := by
  have hsep := hsep_of_pairwise c.lon.hold pre post o d lmin lmax ho si h0 h1 hp
  unfold rawPlanAt
  rw [hobs]
  apply go_prefix c q si pre post 0 none o d lmin lmax hol (by omega)
    (fun z hz d' lmin' lmax' hztr =>
      wOb_le_999_outside_hold c d' si hri
        (hsep z (by simp [hz]) d' lmin' lmax' hztr))
    (wOb_hold c d si hri h0 h1)
    (fun z hz d' lmin' lmax' hztr => by
      have hh := wOb_le_999_outside_hold c d' si hri
        (hsep z (by simp [hz]) d' lmin' lmax' hztr)
      omega)
end Corpus.SteerProofs
