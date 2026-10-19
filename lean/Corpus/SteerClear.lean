/-
  Obstacle clearance of the `steer-2` lateral plan, derived from the decision
  algorithm (no clearance or plan-selection fact is assumed).

  Main result `plan_ok_clearance`: if `Steer.decide c q` returns `plan_ok` and the
  decidable domain predicates `SteerProofs.clearanceDomain` (well-formed input,
  pairwise-disjoint hold windows, even vehicle width, `0 < ramp_in`, nonempty
  corridors) and `holdTargetsOK` (each obstacle lies inside the road at its
  nearest preview station, and each obstacle's chosen target offset lies inside the
  corridor at every preview station of its hold window) hold, then at EVERY
  preview station the published offset has nonnegative certified clearance
  (`clearAt ≥ 0`, the recorded `clearance_mm` entry), and at every station inside
  an obstacle's hold window the offset is beyond that obstacle's edge
  (`≥ lmax + halfw + margin` when passing left, `≤ lmin - halfw - margin` when
  passing right).

  Necessity of the hypotheses is witnessed by kernel counterexamples:
  `old_selector_counterexample` (the pre-fix ramp-out weight selects the wrong
  supplier and the plan overlaps obstacle B while `clearAt ≥ 0` everywhere) and
  `narrowing_counterexample` (target outside a narrowed corridor ⇒ the clamped
  offset overlaps the obstacle).

  Scope: the obstacle's longitudinal footprint (inflated by the car's
  half-length) lying inside `[d - hold, d + hold]`, and the car tracking the plan,
  are assumptions outside this file (see `Corpus.Closed`).
-/
import Corpus.SteerProofs

namespace Corpus.SteerClear
open Corpus Corpus.Steer

/-- Station `j` of the preview. -/
abbrev st (q : Q) (j : Nat) : Int := q.base.s.getD j 0

/-- Decidable geometric hypotheses: obstacle inside the road at its nearest
station, and its target offset inside the corridor at every hold station. -/
def holdTargetsOK (c : Consts) (q : Q) : Bool :=
  q.base.obstacles.all fun ob => match Cert.obsTriple ob with
    | some (d, lmin, lmax) =>
        let i := Cert.nearestIndex q.base.s d
        let sa := sideAmp q d lmin lmax
        decide (0 ≤ lmax + q.base.wl.getD i 0) && decide (lmin - q.base.wr.getD i 0 ≤ 0) &&
        (List.range q.base.s.length).all fun j =>
          decide (d - c.lon.hold ≤ st q j ∧ st q j ≤ d + c.lon.hold →
            lo c q j ≤ sa.1 * sa.2 ∧ sa.1 * sa.2 ≤ hi c q j)
    | none => true

theorem holdTargetsOK_sound (c : Consts) (q : Q) (h : holdTargetsOK c q = true)
    (d lmin lmax : Int) (hm : [d, lmin, lmax] ∈ q.base.obstacles) :
    0 ≤ lmax + q.base.wl.getD (Cert.nearestIndex q.base.s d) 0 ∧
    lmin - q.base.wr.getD (Cert.nearestIndex q.base.s d) 0 ≤ 0 ∧
    ∀ j, j < q.base.s.length → d - c.lon.hold ≤ st q j → st q j ≤ d + c.lon.hold →
      lo c q j ≤ (sideAmp q d lmin lmax).1 * (sideAmp q d lmin lmax).2 ∧
      (sideAmp q d lmin lmax).1 * (sideAmp q d lmin lmax).2 ≤ hi c q j := by
  unfold holdTargetsOK at h
  have h1 := List.all_eq_true.mp h _ hm
  simp only [Cert.obsTriple, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true,
    List.mem_range] at h1
  obtain ⟨⟨hl, hr⟩, hj⟩ := h1
  exact ⟨hl, hr, fun j hjl ha hb => hj j hjl ⟨ha, hb⟩⟩

/-- The chosen side's target is beyond the obstacle edge whenever the chosen gap
fits the car (and the obstacle lies inside the road, width even). -/
theorem sideAmp_edge (c : Consts) (q : Q) (d lmin lmax : Int)
    (hgap : c.lon.veh_width + 2 * c.margin ≤ max (gapLeft q d lmax) (gapRight q d lmin))
    (hl : 0 ≤ lmax + q.base.wl.getD (Cert.nearestIndex q.base.s d) 0)
    (hr : lmin - q.base.wr.getD (Cert.nearestIndex q.base.s d) 0 ≤ 0)
    (heven : 2 * halfw c = c.lon.veh_width) :
    ((sideAmp q d lmin lmax).1 = 1 ∧
        lmax + halfw c + c.margin ≤ (sideAmp q d lmin lmax).1 * (sideAmp q d lmin lmax).2) ∨
    ((sideAmp q d lmin lmax).1 = -1 ∧
        (sideAmp q d lmin lmax).1 * (sideAmp q d lmin lmax).2 ≤ lmin - halfw c - c.margin) := by
  unfold gapLeft gapRight at hgap
  generalize hwl : q.base.wl.getD (Cert.nearestIndex q.base.s d) 0 = wl at *
  generalize hwr : q.base.wr.getD (Cert.nearestIndex q.base.s d) 0 = wr at *
  have hL : c.lon.veh_width + 2 * c.margin ≤ wl - lmax →
      lmax + halfw c + c.margin ≤ 1 * Cert.cdiv (Cert.iabs (lmax + wl)) 2 := by
    intro hg
    have := (SteerFeas.target_in_corridor_left c wl lmax _ hg hl heven rfl).1
    omega
  have hR : c.lon.veh_width + 2 * c.margin ≤ lmin + wr →
      -1 * Cert.cdiv (Cert.iabs (lmin - wr)) 2 ≤ lmin - halfw c - c.margin := by
    intro hg
    have := (SteerFeas.target_in_corridor_right c wr lmin _ hg hr heven rfl).2.2
    omega
  unfold sideAmp
  simp only [hwl, hwr]
  by_cases h1 : wl - lmax > lmin + wr
  · rw [if_pos h1]; exact Or.inl ⟨rfl, hL (by omega)⟩
  · rw [if_neg h1]
    by_cases h2 : lmin + wr > wl - lmax
    · rw [if_pos h2]; exact Or.inr ⟨rfl, hR (by omega)⟩
    · rw [if_neg h2]
      by_cases h3 : 0 ≤ q.lat
      · rw [if_pos h3]; exact Or.inl ⟨rfl, hL (by omega)⟩
      · rw [if_neg h3]; exact Or.inr ⟨rfl, hR (by omega)⟩

/-- Any plan selected by `rawPlanAt` comes from a listed obstacle triple. -/
theorem go_mem (c : Consts) (q : Q) (si : Int) (P : Plan → Prop) :
    ∀ (l : List (List Int)) (bestW : Int) (best : Option Plan),
      (∀ p, best = some p → P p) →
      (∀ ob ∈ l, ∀ d lmin lmax, Cert.obsTriple ob = some (d, lmin, lmax) →
        P { d := d, lmin := lmin, lmax := lmax, side := (sideAmp q d lmin lmax).1,
            amp := (sideAmp q d lmin lmax).2 }) →
      ∀ p, rawPlanAt.go c q si bestW best l = some p → P p := by
  intro l
  induction l with
  | nil => intro bestW best hb _ p hp; exact hb p hp
  | cons ob tl ih =>
      intro bestW best hb hl p hp
      simp only [rawPlanAt.go] at hp
      have htl : ∀ ob' ∈ tl, ∀ d lmin lmax, Cert.obsTriple ob' = some (d, lmin, lmax) →
          P { d := d, lmin := lmin, lmax := lmax, side := (sideAmp q d lmin lmax).1,
              amp := (sideAmp q d lmin lmax).2 } :=
        fun ob' h' => hl ob' (List.mem_cons_of_mem ob h')
      split at hp
      · rename_i d lmin lmax htr
        split at hp
        · exact ih _ _ (fun p' hp' => by
            simp only [Option.some.injEq] at hp'
            subst hp'
            exact hl ob (by simp) d lmin lmax htr) htl p hp
        · exact ih _ _ hb htl p hp
      · exact ih _ _ hb htl p hp

theorem rawPlanAt_mem (c : Consts) (q : Q) (si : Int) (p : Plan)
    (hp : rawPlanAt c q si = some p) :
    ∃ ob ∈ q.base.obstacles, Cert.obsTriple ob = some (p.d, p.lmin, p.lmax) := by
  exact go_mem c q si (fun p => ∃ ob ∈ q.base.obstacles, Cert.obsTriple ob = some (p.d, p.lmin, p.lmax))
    q.base.obstacles 0 none (fun _ h => by cases h)
    (fun ob hob d lmin lmax htr => ⟨ob, hob, htr⟩) p hp

theorem obsTriple_eq {ob : List Int} {d lmin lmax : Int}
    (h : Cert.obsTriple ob = some (d, lmin, lmax)) : ob = [d, lmin, lmax] := by
  match ob, h with
  | [a, b, e], h => simp only [Cert.obsTriple, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨rfl, rfl, rfl⟩ := h; rfl

theorem pairwise_of_wellFormed (c : Consts) (q : Q) (hw : wellFormed c q = true) :
    pairwiseDisjoint c.lon.hold q.base.obstacles = true := by
  simp only [wellFormed, Bool.and_eq_true] at hw
  exact hw.1.1.1.2

theorem noGap_of_plan_ok (c : Consts) (q : Q) (hw : wellFormed c q = true)
    (hk : (decide c q).kind = kindPlanOk) (d lmin lmax : Int)
    (hm : [d, lmin, lmax] ∈ q.base.obstacles) :
    c.lon.veh_width + 2 * c.margin ≤ max (gapLeft q d lmax) (gapRight q d lmin) := by
  have hng : hasNoGap c q = false := (decide_plan_ok_iff c q hw).mp hk
  unfold hasNoGap at hng
  have := List.any_eq_false.mp hng _ hm
  simp only [Cert.obsTriple, decide_eq_true_eq] at this
  omega

/-- **Hold-window clearance from the algorithm.** -/
theorem plan_ok_hold_clearance (c : Consts) (q : Q)
    (hdom : SteerProofs.clearanceDomain c q = true) (htg : holdTargetsOK c q = true)
    (hk : (decide c q).kind = kindPlanOk)
    (d lmin lmax : Int) (hm : [d, lmin, lmax] ∈ q.base.obstacles)
    (j : Nat) (hj : j < q.base.s.length)
    (h0 : d - c.lon.hold ≤ st q j) (h1 : st q j ≤ d + c.lon.hold) :
    (((sideAmp q d lmin lmax).1 = 1 ∧ lmax + halfw c + c.margin ≤ offAt c q (st q j) j) ∨
      ((sideAmp q d lmin lmax).1 = -1 ∧ offAt c q (st q j) j ≤ lmin - halfw c - c.margin)) ∧
    0 ≤ clearAt c q (st q j) j (offAt c q (st q j) j) := by
  obtain ⟨hw, heven, hri, hcorr⟩ := SteerProofs.clearanceDomain_sound c q hdom
  obtain ⟨hl, hr, htgt⟩ := holdTargetsOK_sound c q htg d lmin lmax hm
  obtain ⟨pre, post, hsplit⟩ := List.append_of_mem hm
  have hsel := SteerProofs.rawPlanAt_hold c q (st q j) pre post [d, lmin, lmax] d lmin lmax hsplit
    rfl rfl hri (by rw [← hsplit]; exact pairwise_of_wellFormed c q hw) h0 h1
  have hoff : offAt c q (st q j) j = clampAt c q j ((sideAmp q d lmin lmax).1 * (sideAmp q d lmin lmax).2) := by
    unfold offAt rawOffAt
    rw [hsel]
    simp only
    rw [off_eq_amp_in_hold c d _ _ (st q j) (by omega) h0 h1]
  have ht := htgt j hj h0 h1
  have hclamp : clampAt c q j ((sideAmp q d lmin lmax).1 * (sideAmp q d lmin lmax).2) =
      (sideAmp q d lmin lmax).1 * (sideAmp q d lmin lmax).2 := by
    unfold clampAt
    simp only
    rw [if_pos (hcorr j hj)]
    omega
  rw [hclamp] at hoff
  have hedge := sideAmp_edge c q d lmin lmax (noGap_of_plan_ok c q hw hk d lmin lmax hm) hl hr heven
  rw [hoff]
  refine ⟨hedge, ?_⟩
  apply SteerFeas.clearAt_hold_nonneg c q j (st q j) _ _ hsel ⟨h0, h1⟩ ht
  rcases hedge with ⟨hs, he⟩ | ⟨hs, he⟩
  · exact Or.inl ⟨hs, he⟩
  · exact Or.inr ⟨hs, he⟩

/-- **Clearance at every preview station.**  Under the same hypotheses every
recorded clearance entry is nonnegative: inside a hold window by the theorem
above, elsewhere because the offset is clamped into the corridor. -/
theorem plan_ok_clearance (c : Consts) (q : Q)
    (hdom : SteerProofs.clearanceDomain c q = true) (htg : holdTargetsOK c q = true)
    (hk : (decide c q).kind = kindPlanOk) (j : Nat) (hj : j < q.base.s.length) :
    0 ≤ clearAt c q (st q j) j (offAt c q (st q j) j) := by
  obtain ⟨hw, _, _, hcorr⟩ := SteerProofs.clearanceDomain_sound c q hdom
  cases hp : rawPlanAt c q (st q j) with
  | none =>
      exact SteerFeas.clearAt_outside_nonneg c q j (st q j) _
        (SteerFeas.clamp_in_corridor c q j _ (hcorr j hj)) (fun p h => by rw [hp] at h; cases h)
  | some p =>
      by_cases hin : p.d - c.lon.hold ≤ st q j ∧ st q j ≤ p.d + c.lon.hold
      · obtain ⟨ob, hob, htr⟩ := rawPlanAt_mem c q (st q j) p hp
        have hob' : [p.d, p.lmin, p.lmax] ∈ q.base.obstacles := by
          rw [← obsTriple_eq htr]; exact hob
        exact (plan_ok_hold_clearance c q hdom htg hk p.d p.lmin p.lmax hob' j hj hin.1 hin.2).2
      · exact SteerFeas.clearAt_outside_nonneg c q j (st q j) _
          (SteerFeas.clamp_in_corridor c q j _ (hcorr j hj))
          (fun p' h => by rw [hp] at h; cases h; omega)

/-- The recorded clearance list is entrywise nonnegative. -/
theorem plan_ok_clears_nonneg (c : Consts) (q : Q)
    (hdom : SteerProofs.clearanceDomain c q = true) (htg : holdTargetsOK c q = true)
    (hk : (decide c q).kind = kindPlanOk) :
    ∀ x ∈ clears c q (offs c q), 0 ≤ x := by
  intro x hx
  unfold clears at hx
  simp only [List.mem_map, List.mem_range] at hx
  obtain ⟨j, hj, rfl⟩ := hx
  have hoffs : (offs c q).getD j 0 = offAt c q (st q j) j := by
    unfold offs
    simp [List.getD_eq_getElem?_getD, hj]
  rw [hoffs]
  exact plan_ok_clearance c q hdom htg hk j hj

/-! ## Kernel-checked instances -/

set_option maxRecDepth 200000

def abLon : Cert.Consts :=
  { a_lat := 6000, a_acc := 2000, a_brk := 3000, a_max := 9000, v_max := 20000,
    r_h := 2000, hold := 3000, ramp_in := 18000, ramp_out := 15000, r_sw_m := 15, veh_width := 2000 }
def abC : Consts := { lon := abLon, margin := 200, step := 1000, kappa_max := 400000 }
/-- Opposing obstacles: A (passed on its right) is listed first, B (passed on its
left) starts its hold window 100 mm after A's ends. -/
def abQ (wl25 : Int) : Q :=
  { base := { v0 := 5000, s := (List.range 41).map (fun (i : Nat) => 1000 * (i : Int)),
              k := List.replicate 41 0,
              wl := (List.range 41).map (fun (i : Nat) => if i = 25 then wl25 else 3000),
              wr := List.replicate 41 3000,
              obstacles := [[20900, 500, 1500], [27000, -1500, -500]] },
    lat := 0 }

/-- **Non-vacuity**: every hypothesis of `plan_ok_clearance` holds on the
opposing-obstacle layout, and at station 24 m (inside B's hold) the plan is on
B's left at `+1250 ≥ -500 + 1000 + 200`. -/
theorem ab_hypotheses_hold :
    SteerProofs.clearanceDomain abC (abQ 3000) = true ∧ holdTargetsOK abC (abQ 3000) = true ∧
    (decide abC (abQ 3000)).kind = kindPlanOk ∧ offAt abC (abQ 3000) 24000 24 = 1250 := by decide

/-- The pre-fix weight (`1000 - ramp`, uncapped) and the plan it selects. -/
def wObOld (c : Consts) (d si : Int) : Int :=
  if si < d - c.lon.hold - c.lon.ramp_in then 0
  else if si < d - c.lon.hold then ramp 1000 (si - d + c.lon.hold + c.lon.ramp_in) c.lon.ramp_in
  else if si ≤ d + c.lon.hold then 1000
  else if si ≤ d + c.lon.hold + c.lon.ramp_out then 1000 - ramp 1000 (si - d - c.lon.hold) c.lon.ramp_out
  else 0

def goOld (c : Consts) (q : Q) (si : Int) : Int → Option Plan → List (List Int) → Option Plan
  | _, best, [] => best
  | bestW, best, ob :: tl => match Cert.obsTriple ob with
    | some (d, lmin, lmax) =>
        if bestW < wObOld c d si then
          goOld c q si (wObOld c d si)
            (some { d := d, lmin := lmin, lmax := lmax, side := (sideAmp q d lmin lmax).1,
                    amp := (sideAmp q d lmin lmax).2 }) tl
        else goOld c q si bestW best tl
    | none => goOld c q si bestW best tl

def offAtOld (c : Consts) (q : Q) (si : Int) (i : Nat) : Int :=
  clampAt c q i (match goOld c q si 0 none q.base.obstacles with
    | some p => offOb c p.d p.side p.amp si
    | none => 0)

def clearAtOld (c : Consts) (q : Q) (si : Int) (i : Nat) (x : Int) : Int :=
  match goOld c q si 0 none q.base.obstacles with
  | some p =>
      if p.d - c.lon.hold ≤ si ∧ si ≤ p.d + c.lon.hold then
        if p.side = 1 then min (x - (p.lmax + halfw c + c.margin)) (hi c q i - x)
        else if p.side = -1 then min ((p.lmin - halfw c - c.margin) - x) (x - lo c q i)
        else min (x - lo c q i) (hi c q i - x)
      else min (x - lo c q i) (hi c q i - x)
  | none => min (x - lo c q i) (hi c q i - x)

/-- **Counterexample to the pre-fix model**: at station 24 m (inside B's hold
window) the old selector keeps A, the planned offset is `-1250` (car body
`[-2250, -250]` overlaps B's extent `[-1500, -500]`), yet the old certified
clearance at every station is nonnegative. -/
theorem old_selector_counterexample :
    offAtOld abC (abQ 3000) 24000 24 = -1250 ∧
    (-1250 - halfw abC < -500 ∧ -1500 < -1250 + halfw abC) ∧
    ((List.range 41).all fun j =>
      decide (0 ≤ clearAtOld abC (abQ 3000) (1000 * (j : Int)) j (offAtOld abC (abQ 3000) (1000 * (j : Int)) j))) = true := by
  decide

/-- **`holdTargetsOK` is needed**: narrowing the road to 1.5 m on the left at
station 25 (inside B's hold) keeps `plan_ok`, but the target `+1250` no longer
fits the corridor; the clamped offset `300` overlaps B and the certificate's own
clearance entry is negative. -/
theorem narrowing_counterexample :
    SteerProofs.clearanceDomain abC (abQ 1500) = true ∧ holdTargetsOK abC (abQ 1500) = false ∧
    (decide abC (abQ 1500)).kind = kindPlanOk ∧ offAt abC (abQ 1500) 25000 25 = 300 ∧
    300 - halfw abC < -500 + 0 + 1000 ∧ clearAt abC (abQ 1500) 25000 25 300 < 0 := by decide

/-- **Supplier switching breaks any smoothness bound**: on the same (fixed,
certified-clear) opposing layout the plan jumps from `-1250` (A's hold) to `+1250`
(B's hold) between consecutive 1 m stations; the second difference is `2500 mm`,
whereas a single smoothstep ramp of amplitude 1250 over 15 m obeys
`L²·|Δ²| < 6·A·h² + 2·L²` (`SteerShape.ramp_second_bound`), i.e. `|Δ²| ≤ 35`.  The
lateral plan is clear but its curvature demand (`2.5·10⁶ µ/m`) saturates steering;
this is why `Corpus.Joint` enforces `pathCurv ≤ kappa_max` and the controller
rate-limits inside the certified interval. -/
theorem supplier_switch_curvature :
    offAt abC (abQ 3000) 23000 23 = -1250 ∧ offAt abC (abQ 3000) 24000 24 = 1250 ∧
    secondAt (offs abC (abQ 3000)) 23 = 2500 ∧
    curvAt (abQ 3000) (offs abC (abQ 3000)) 23 > abC.kappa_max := by decide

end Corpus.SteerClear
