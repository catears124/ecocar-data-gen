/-
  Planning levers of the joint offset/speed model, inside the option-valued
  Bellman suite (`Corpus.BellmanOpt`, `Corpus.Joint`).

  ## What is modelled
  The runtime controller exposes declared budgets ("levers"): the cornering budget
  `a_lat`, the acceleration and braking budgets `a_acc`/`a_brk`, the terminal
  horizon radius `r_h` (dropping the terminal bound near the finish is the limit
  `r_h → ∞`), the speed cap `v_max`, the steering-curvature limit `kappa_max`,
  extra per-station speed² caps (the gap-speed rule and the sidestep gate), and
  the shape of the acceleration budget: an equal-axis box (what `Cert`/`Joint`
  certify) versus a friction circle.

  ## Results
  * Section 1 re-states the joint model for an ARBITRARY step relation `P` and
    terminal cap `T` (`actionsP`, `terminalP`, `PlanP`) and proves the Bellman
    facts for it: lower bound (`valueP_lower`), attainment (`valueP_attained`),
    exact infeasibility (`valueP_none_iff`).  `Joint` is the instance
    `P = stepOK`, `T = terminalCap` (`joint_value_eq`, `joint_plan_iff`).
  * `lever_value_le` (the lever theorem): if every step of model `P` that is
    reachable under an invariant `I` is a step of model `P'`, and `T ≤ T'`, then
    the optimal declared cost of `P'` is `≤` that of `P` (and `P'` is feasible
    whenever `P` is).  Relaxing a constraint can never make the optimum worse.
  * `budget_lever`: raising any of `a_lat`, `a_acc`, `a_brk`, `r_h`, `v_max`,
    `kappa_max` (jointly or singly; corollaries `cornering_lever`,
    `braking_lever`, `accel_lever`, `horizon_lever`) can only lower the optimal
    declared cost of the joint plan.
  * `extra_caps_cost`, `extra_caps_mono`: an extra speed² cap (gap rule, sidestep
    gate) can only raise the optimum, and monotonically in the cap.
  * `box_step_circle`, `circle_value_le_box`: when `a_lat² + max(a_acc,a_brk)² ≤
    A²`, every certified (box) step from a nonnegative speed² is a step of the
    friction-circle model of radius `A`, so the circle model's optimum is `≤` the
    certified optimum.  The circle model is NOT certified by `Cert`: its plans can
    combine braking and cornering beyond the box.
  * Kernel-checked instances on `Joint`'s toy problem: which lever binds there.

  Nothing here says a declared budget is physically achievable, and the
  objective is the declared per-transition cost of `Joint`.
-/
import Corpus.Joint

namespace Corpus.Levers
open Corpus Corpus.Joint

/-! ## 1. The joint model for an arbitrary step relation -/

def actionsP (P : JState → JState → Prop) [∀ s t, Decidable (P s t)] (len : Nat) (grid : List Int)
    (time : Nat → Int → Int → Int) (s : JState) : List (Act JState) :=
  if s.station + 1 < len then
    Dp.bucketGrid.flatMap fun n => grid.filterMap fun w =>
      let t : JState := { station := s.station + 1, previous := s.current, current := n, b := w }
      if P s t then some (t, time s.station s.b w) else none
  else []

def terminalP (T : Int) : JState → Option Int := fun s => if s.b ≤ T then some 0 else none

/-- Declarative plans of the model `(P, len, T)`. -/
def PlanP (P : JState → JState → Prop) (len : Nat) (T : Int) : JState → List JState → Prop
  | s, [] => s.b ≤ T
  | s, t :: ts => P s t ∧ s.station + 1 < len ∧ PlanP P len T t ts

/-- A step relation has the joint shape: one station forward, offset history
carried, offset on the bucket grid, speed² on the declared grid. -/
def Shaped (P : JState → JState → Prop) (grid : List Int) : Prop :=
  ∀ s t, P s t → t.station = s.station + 1 ∧ t.previous = s.current ∧
    t.current ∈ Dp.bucketGrid ∧ t.b ∈ grid

/-- Optimal declared cost over `m` stages. -/
abbrev valueP (P : JState → JState → Prop) [∀ s t, Decidable (P s t)] (len : Nat) (T : Int)
    (grid : List Int) (time : Nat → Int → Int → Int) (m : Nat) (s : JState) : Option Int :=
  BellmanOpt.valueO (terminalP T) (List.replicate m (actionsP P len grid time)) s

theorem mem_actionsP {P : JState → JState → Prop} [∀ s t, Decidable (P s t)] {len : Nat}
    {grid : List Int} {time : Nat → Int → Int → Int} {s : JState} {a : Act JState}
    (h : a ∈ actionsP P len grid time s) :
    P s a.1 ∧ s.station + 1 < len ∧ a.2 = time s.station s.b a.1.b := by
  unfold actionsP at h
  split at h
  · rename_i hlen
    simp only [List.mem_flatMap, List.mem_filterMap] at h
    obtain ⟨n, _, w, _, hw⟩ := h
    split at hw
    · rename_i hok
      injection hw with hw
      subst hw
      exact ⟨hok, hlen, rfl⟩
    · cases hw
  · simp at h

theorem actionsP_complete {P : JState → JState → Prop} [∀ s t, Decidable (P s t)] {len : Nat}
    {grid : List Int} {time : Nat → Int → Int → Int} {s t : JState} (hsh : Shaped P grid)
    (hok : P s t) (hlen : s.station + 1 < len) :
    (t, time s.station s.b t.b) ∈ actionsP P len grid time s := by
  obtain ⟨hst, hpr, hgr, hbg⟩ := hsh s t hok
  cases t with
  | mk ts tp tc tb =>
    simp only at hst hpr hgr hbg
    subst hst hpr
    unfold actionsP
    rw [if_pos hlen]
    exact List.mem_flatMap.mpr ⟨tc, hgr, List.mem_filterMap.mpr ⟨tb, hbg, by rw [if_pos hok]; rfl⟩⟩

def costP (time : Nat → Int → Int → Int) : JState → List JState → Int
  | _, [] => 0
  | s, t :: ts => time s.station s.b t.b + costP time t ts

theorem costP_eq (time : Nat → Int → Int → Int) :
    ∀ (ts : List JState) (s : JState), costP time s ts = planCost time s ts := by
  intro ts
  induction ts with
  | nil => intro _; rfl
  | cons t ts ih => intro s; simp only [costP, planCost, ih]

theorem trajP_plan (P : JState → JState → Prop) [∀ s t, Decidable (P s t)] (len : Nat) (T : Int)
    (grid : List Int) (time : Nat → Int → Int → Int) :
    ∀ (m : Nat) (s : JState) (xs : List (Act JState)),
      BellmanOpt.TrajOK (terminalP T) (List.replicate m (actionsP P len grid time)) s xs →
      PlanP P len T s (xs.map Prod.fst) ∧
      BellmanOpt.trajCost (terminalP T) (List.replicate m (actionsP P len grid time)) s xs =
        some (costP time s (xs.map Prod.fst)) ∧ xs.length = m := by
  intro m
  induction m with
  | zero =>
      intro s xs h
      cases xs with
      | nil =>
          obtain ⟨v, hv⟩ := h
          unfold terminalP at hv
          split at hv
          · rename_i hb
            refine ⟨hb, ?_, rfl⟩
            show terminalP T s = some 0
            simp [terminalP, hb]
          · cases hv
      | cons _ _ => exact h.elim
  | succ m ih =>
      intro s xs h
      cases xs with
      | nil => exact h.elim
      | cons a ys =>
          obtain ⟨ha, ht⟩ := h
          obtain ⟨hok, hlen, hcost⟩ := mem_actionsP ha
          obtain ⟨hp, hc, hl⟩ := ih a.1 ys ht
          refine ⟨⟨hok, hlen, hp⟩, ?_, by rw [List.length_cons, hl]⟩
          show (BellmanOpt.trajCost (terminalP T) (List.replicate m (actionsP P len grid time)) a.1 ys).map
              (fun v => a.2 + v) = some (time s.station s.b a.1.b + costP time a.1 (ys.map Prod.fst))
          rw [hc, hcost]
          rfl

theorem planP_traj (P : JState → JState → Prop) [∀ s t, Decidable (P s t)] (len : Nat) (T : Int)
    (grid : List Int) (time : Nat → Int → Int → Int) (hsh : Shaped P grid) :
    ∀ (ts : List JState) (s : JState), PlanP P len T s ts →
      ∃ xs, xs.map Prod.fst = ts ∧
        BellmanOpt.TrajOK (terminalP T) (List.replicate ts.length (actionsP P len grid time)) s xs ∧
        BellmanOpt.trajCost (terminalP T) (List.replicate ts.length (actionsP P len grid time)) s xs =
          some (costP time s ts) := by
  intro ts
  induction ts with
  | nil =>
      intro s h
      have hb : s.b ≤ T := h
      refine ⟨[], rfl, ⟨0, by simp [terminalP, hb]⟩, ?_⟩
      show terminalP T s = some 0
      simp [terminalP, hb]
  | cons t ts ih =>
      intro s h
      obtain ⟨hok, hlen, hp⟩ := h
      obtain ⟨ys, hys, hok', hc⟩ := ih t hp
      refine ⟨((t, time s.station s.b t.b) : Act JState) :: ys, ?_,
        ⟨actionsP_complete hsh hok hlen, hok'⟩, ?_⟩
      · show t :: ys.map Prod.fst = t :: ts
        rw [hys]
      · show (BellmanOpt.trajCost (terminalP T) (List.replicate ts.length (actionsP P len grid time)) t ys).map
            (fun v => time s.station s.b t.b + v) = some (time s.station s.b t.b + costP time t ts)
        rw [hc]
        rfl

/-- Lower bound: the value is `≤` the declared cost of every plan of length `m`. -/
theorem valueP_lower (P : JState → JState → Prop) [∀ s t, Decidable (P s t)] (len : Nat) (T : Int)
    (grid : List Int) (time : Nat → Int → Int → Int) (hsh : Shaped P grid) (m : Nat) (s : JState)
    (v : Int) (hv : valueP P len T grid time m s = some v) :
    ∀ ts, PlanP P len T s ts → ts.length = m → v ≤ costP time s ts := by
  intro ts hp hl
  obtain ⟨xs, _, hok, hc⟩ := planP_traj P len T grid time hsh ts s hp
  rw [hl] at hok hc
  exact BellmanOpt.valueO_le_cost _ _ s xs v _ hok hc hv

/-- Attainment: some plan of length `m` has cost exactly the value. -/
theorem valueP_attained (P : JState → JState → Prop) [∀ s t, Decidable (P s t)] (len : Nat) (T : Int)
    (grid : List Int) (time : Nat → Int → Int → Int) (m : Nat) (s : JState) (v : Int)
    (hv : valueP P len T grid time m s = some v) :
    ∃ ts, PlanP P len T s ts ∧ ts.length = m ∧ costP time s ts = v := by
  obtain ⟨xs, hok, hc⟩ := BellmanOpt.valueO_attained _ _ s v hv
  obtain ⟨hp, hc', hl⟩ := trajP_plan P len T grid time m s xs hok
  rw [hc'] at hc
  simp only [Option.some.injEq] at hc
  exact ⟨xs.map Prod.fst, hp, (map_fst_length xs).trans hl, hc⟩

/-- Exact infeasibility. -/
theorem valueP_none_iff (P : JState → JState → Prop) [∀ s t, Decidable (P s t)] (len : Nat) (T : Int)
    (grid : List Int) (time : Nat → Int → Int → Int) (hsh : Shaped P grid) (m : Nat) (s : JState) :
    valueP P len T grid time m s = none ↔ ¬ ∃ ts, PlanP P len T s ts ∧ ts.length = m := by
  rw [valueP, BellmanOpt.valueO_none_iff]
  constructor
  · intro hn ⟨ts, hp, hl⟩
    obtain ⟨xs, _, hok, _⟩ := planP_traj P len T grid time hsh ts s hp
    rw [hl] at hok
    exact hn ⟨xs, hok⟩
  · intro hn ⟨xs, hok⟩
    obtain ⟨hp, _, hl⟩ := trajP_plan P len T grid time m s xs hok
    exact hn ⟨xs.map Prod.fst, hp, (map_fst_length xs).trans hl⟩

/-! ### `Joint` is the instance `P = stepOK`, `T = terminalCap` -/

theorem stepOK_shaped (c : Steer.Consts) (q : Steer.Q) (grid : List Int) :
    Shaped (stepOK c q grid) grid :=
  fun _ _ h => ⟨h.1.1, h.1.2.1, h.1.2.2.1, h.2.1⟩

theorem joint_value_eq (c : Steer.Consts) (q : Steer.Q) (grid : List Int)
    (time : Nat → Int → Int → Int) (m : Nat) (s : JState) :
    BellmanOpt.valueO (terminal c q) (List.replicate m (actions c q grid time)) s =
      valueP (stepOK c q grid) q.base.s.length (terminalCap c q) grid time m s := rfl

theorem joint_plan_iff (c : Steer.Consts) (q : Steer.Q) (grid : List Int) :
    ∀ (ts : List JState) (s : JState),
      PlanOK c q grid s ts ↔ PlanP (stepOK c q grid) q.base.s.length (terminalCap c q) s ts := by
  intro ts
  induction ts with
  | nil => intro _; exact Iff.rfl
  | cons t ts ih => intro s; exact and_congr Iff.rfl (and_congr Iff.rfl (ih t))

/-! ## 2. The lever theorem -/

theorem planP_mono {P P' : JState → JState → Prop} {len : Nat} {T T' : Int} (I : JState → Prop)
    (hP : ∀ s t, I s → s.station + 1 < len → P s t → P' s t ∧ I t) (hT : T ≤ T') :
    ∀ (ts : List JState) (s : JState), I s → PlanP P len T s ts → PlanP P' len T' s ts := by
  intro ts
  induction ts with
  | nil => intro s _ h; exact Int.le_trans (show s.b ≤ T from h) hT
  | cons t ts ih =>
      intro s hI h
      obtain ⟨hok, hlen, hp⟩ := h
      obtain ⟨hok', hIt⟩ := hP s t hI hlen hok
      exact ⟨hok', hlen, ih t hIt hp⟩

/-- **Lever theorem.**  If every reachable step of `P` (under the invariant `I`)
is a step of `P'` and the terminal cap does not shrink, then whenever `P` has an
optimal plan of cost `v`, `P'` has one of cost `v' ≤ v`. -/
theorem lever_value_le {P P' : JState → JState → Prop} [∀ s t, Decidable (P s t)]
    [∀ s t, Decidable (P' s t)] {len : Nat} {T T' : Int} {grid : List Int}
    {time : Nat → Int → Int → Int} (I : JState → Prop) (hsh' : Shaped P' grid)
    (hP : ∀ s t, I s → s.station + 1 < len → P s t → P' s t ∧ I t) (hT : T ≤ T')
    (m : Nat) (s : JState) (v : Int) (hI : I s) (hv : valueP P len T grid time m s = some v) :
    ∃ v', valueP P' len T' grid time m s = some v' ∧ v' ≤ v := by
  obtain ⟨ts, hp, hl, hc⟩ := valueP_attained P len T grid time m s v hv
  have hp' := planP_mono I hP hT ts s hI hp
  cases hv' : valueP P' len T' grid time m s with
  | none => exact absurd ⟨ts, hp', hl⟩ ((valueP_none_iff P' len T' grid time hsh' m s).mp hv')
  | some v' =>
      refine ⟨v', rfl, ?_⟩
      rw [← hc]
      exact valueP_lower P' len T' grid time hsh' m s v' hv' ts hp' hl

/-! ## 3. Budget levers: `a_lat`, `a_acc`, `a_brk`, `r_h`, `v_max`, `kappa_max` -/

/-- The joint model with raised declared limits; every other constant unchanged. -/
def raise (c : Steer.Consts) (A Acc Brk Rh Vm K : Int) : Steer.Consts :=
  { c with kappa_max := K,
           lon := { c.lon with a_lat := A, a_acc := Acc, a_brk := Brk, r_h := Rh, v_max := Vm } }

/-- Raised limits, plus the sign facts the monotonicity needs. -/
structure Raised (c : Steer.Consts) (A Acc Brk Rh Vm K : Int) : Prop where
  lat : c.lon.a_lat ≤ A
  acc : c.lon.a_acc ≤ Acc
  brk : c.lon.a_brk ≤ Brk
  rh : c.lon.r_h ≤ Rh
  vm : c.lon.v_max ≤ Vm
  kap : c.kappa_max ≤ K
  lat0 : 0 ≤ c.lon.a_lat
  rh0 : 0 ≤ c.lon.r_h
  vm0 : 0 ≤ c.lon.v_max

theorem cdiv_nonneg {a b : Int} (ha : 0 ≤ a) (hb : 0 ≤ b) : 0 ≤ Cert.cdiv a b := by
  unfold Cert.cdiv
  rcases Int.lt_or_eq_of_le hb with hb | hb
  · have h := Int.ediv_le_ediv hb (show -a ≤ 0 by omega)
    rw [Int.zero_ediv] at h
    omega
  · subst hb
    rw [Int.ediv_zero]
    omega

theorem self_mul_nonneg (x : Int) : 0 ≤ x * x := by
  rcases Int.le_total 0 x with h | h
  · exact Int.mul_nonneg h h
  · have := Int.mul_nonneg (Int.neg_nonneg_of_nonpos h) (Int.neg_nonneg_of_nonpos h)
    rwa [Int.neg_mul_neg] at this

theorem foldl_ge (f : Int → List Int → Int) (hf : ∀ a ob, a ≤ f a ob) :
    ∀ (l : List (List Int)) (a : Int), a ≤ l.foldl f a := by
  intro l
  induction l with
  | nil => intro a; exact Int.le_refl a
  | cons ob l ih => intro a; exact Int.le_trans (hf a ob) (ih (f a ob))

theorem stationY_nonneg (c : Cert.Consts) (q : Cert.Q) (si : Int) : 0 ≤ Cert.stationY c q si := by
  unfold Cert.stationY
  apply foldl_ge
  intro a ob
  split
  · split
    · exact Int.le_max_left _ _
    · exact Int.le_refl _
  · exact Int.le_refl _

theorem kbarOf_nonneg (c : Cert.Consts) (ka y : Int) (hka : 0 ≤ ka) (hy : 0 ≤ y) :
    0 ≤ Cert.kbarOf c ka y := by
  unfold Cert.kbarOf
  have hsw : 0 ≤ Cert.cdiv (6000 * y) (c.r_sw_m * c.r_sw_m) :=
    cdiv_nonneg (by omega) (self_mul_nonneg _)
  exact cdiv_nonneg (Int.mul_nonneg (by omega) (by decide)) (Int.le_trans (by decide) (Int.le_max_right _ _))

theorem iabs_nonneg (x : Int) : 0 ≤ Cert.iabs x := by
  unfold Cert.iabs; split <;> omega

theorem capAt_le_vmax (c : Cert.Consts) (q : Cert.Q) (i : Nat) :
    Cert.capAt c q i ≤ c.v_max * c.v_max := by
  unfold Cert.capAt
  dsimp only
  split
  · exact Int.le_refl _
  · exact Int.min_le_left _ _

theorem capAt_raise (c : Steer.Consts) (A Acc Brk Rh Vm K : Int) (h : Raised c A Acc Brk Rh Vm K)
    (q : Cert.Q) (i : Nat) : Cert.capAt c.lon q i ≤ Cert.capAt (raise c A Acc Brk Rh Vm K).lon q i := by
  have hv : c.lon.v_max * c.lon.v_max ≤ Vm * Vm :=
    Int.mul_le_mul h.vm h.vm h.vm0 (Int.le_trans h.vm0 h.vm)
  have hk0 : 0 ≤ Cert.kbarOf c.lon (Cert.iabs (q.k.getD i 0)) (Cert.stationY c.lon q (q.s.getD i 0)) :=
    kbarOf_nonneg _ _ _ (iabs_nonneg _) (stationY_nonneg _ _ _)
  have ha : c.lon.a_lat * 1000000000 ≤ A * 1000000000 :=
    Int.mul_le_mul_of_nonneg_right h.lat (by decide)
  show (if Cert.kbarOf c.lon (Cert.iabs (q.k.getD i 0)) (Cert.stationY c.lon q (q.s.getD i 0)) = 0
        then c.lon.v_max * c.lon.v_max
        else min (c.lon.v_max * c.lon.v_max) (c.lon.a_lat * 1000000000 /
          Cert.kbarOf c.lon (Cert.iabs (q.k.getD i 0)) (Cert.stationY c.lon q (q.s.getD i 0)))) ≤
      (if Cert.kbarOf c.lon (Cert.iabs (q.k.getD i 0)) (Cert.stationY c.lon q (q.s.getD i 0)) = 0
        then Vm * Vm
        else min (Vm * Vm) (A * 1000000000 /
          Cert.kbarOf c.lon (Cert.iabs (q.k.getD i 0)) (Cert.stationY c.lon q (q.s.getD i 0))))
  generalize Cert.kbarOf c.lon (Cert.iabs (q.k.getD i 0)) (Cert.stationY c.lon q (q.s.getD i 0)) = k at hk0 ⊢
  by_cases hk : k = 0
  · rw [if_pos hk, if_pos hk]; exact hv
  · rw [if_neg hk, if_neg hk]
    have hkp : 0 < k := by omega
    exact Int.le_min.mpr ⟨Int.le_trans (Int.min_le_left _ _) hv,
      Int.le_trans (Int.min_le_right _ _) (Int.ediv_le_ediv hkp ha)⟩

theorem pathCurv_raise (c : Steer.Consts) (A Acc Brk Rh Vm K : Int) (q : Steer.Q) (s : JState) (n : Int) :
    pathCurv (raise c A Acc Brk Rh Vm K) q s n = pathCurv c q s n := rfl

theorem pathCap_raise (c : Steer.Consts) (A Acc Brk Rh Vm K : Int) (h : Raised c A Acc Brk Rh Vm K)
    (q : Steer.Q) (s : JState) (n : Int) :
    pathCap c q s n ≤ pathCap (raise c A Acc Brk Rh Vm K) q s n := by
  have hcap := capAt_raise c A Acc Brk Rh Vm K h q.base s.station
  have ha : c.lon.a_lat * 1000000000 ≤ A * 1000000000 :=
    Int.mul_le_mul_of_nonneg_right h.lat (by decide)
  unfold pathCap
  rw [pathCurv_raise]
  show _ ≤ (if pathCurv c q s n ≤ 0 then Cert.capAt (raise c A Acc Brk Rh Vm K).lon q.base s.station
    else min (Cert.capAt (raise c A Acc Brk Rh Vm K).lon q.base s.station) (A * 1000000000 / pathCurv c q s n))
  by_cases hk : pathCurv c q s n ≤ 0
  · rw [if_pos hk, if_pos hk]; exact hcap
  · rw [if_neg hk, if_neg hk]
    exact Int.le_min.mpr ⟨Int.le_trans (Int.min_le_left _ _) hcap,
      Int.le_trans (Int.min_le_right _ _) (Int.ediv_le_ediv (by omega) ha)⟩

theorem terminalCap_raise (c : Steer.Consts) (A Acc Brk Rh Vm K : Int) (h : Raised c A Acc Brk Rh Vm K)
    (q : Steer.Q) : terminalCap c q ≤ terminalCap (raise c A Acc Brk Rh Vm K) q := by
  have hcap := capAt_raise c A Acc Brk Rh Vm K h q.base (q.base.s.length - 1)
  have hm : c.lon.a_lat * c.lon.r_h ≤ A * Rh :=
    Int.mul_le_mul h.lat h.rh h.rh0 (Int.le_trans h.lat0 h.lat)
  show min (Cert.capAt c.lon q.base (q.base.s.length - 1)) (c.lon.a_lat * c.lon.r_h) ≤
    min (Cert.capAt (raise c A Acc Brk Rh Vm K).lon q.base (q.base.s.length - 1)) (A * Rh)
  exact Int.le_min.mpr ⟨Int.le_trans (Int.min_le_left _ _) hcap,
    Int.le_trans (Int.min_le_right _ _) hm⟩

theorem go_raise (c : Steer.Consts) (A Acc Brk Rh Vm K : Int) (q : Steer.Q) (si : Int) :
    ∀ (l : List (List Int)) (bw : Int) (best : Option Steer.Plan),
      Steer.rawPlanAt.go (raise c A Acc Brk Rh Vm K) q si bw best l = Steer.rawPlanAt.go c q si bw best l := by
  intro l
  induction l with
  | nil => intro _ _; rfl
  | cons ob tl ih =>
      intro bw best
      simp only [Steer.rawPlanAt.go]
      split
      · have hw : Steer.wOb (raise c A Acc Brk Rh Vm K) = Steer.wOb c := rfl
        rw [hw]
        split <;> exact ih _ _
      · exact ih _ _

theorem obstacleClear_raise (c : Steer.Consts) (A Acc Brk Rh Vm K : Int) (q : Steer.Q) (i : Nat) (x : Int) :
    Dp.obstacleClear (raise c A Acc Brk Rh Vm K) q i x ↔ Dp.obstacleClear c q i x := by
  have hp : Steer.rawPlanAt (raise c A Acc Brk Rh Vm K) q = Steer.rawPlanAt c q := by
    funext si
    exact go_raise c A Acc Brk Rh Vm K q si q.base.obstacles 0 none
  unfold Dp.obstacleClear Steer.clearAt
  rw [hp]
  exact Iff.rfl

theorem stepOK_raise (c : Steer.Consts) (A Acc Brk Rh Vm K : Int) (h : Raised c A Acc Brk Rh Vm K)
    (q : Steer.Q) (grid : List Int) (s t : JState) (hgap : 0 ≤ gapAt q s.station)
    (hok : stepOK c q grid s t) : stepOK (raise c A Acc Brk Rh Vm K) q grid s t := by
  obtain ⟨⟨h1, h2, h3, h4, h5, h6⟩, ⟨g1, g2, g3, g4, g5⟩⟩ := hok
  have hacc : 2 * c.lon.a_acc * gapAt q s.station ≤ 2 * Acc * gapAt q s.station :=
    Int.mul_le_mul_of_nonneg_right (by have := h.acc; omega) hgap
  have hbrk : 2 * c.lon.a_brk * gapAt q s.station ≤ 2 * Brk * gapAt q s.station :=
    Int.mul_le_mul_of_nonneg_right (by have := h.brk; omega) hgap
  refine ⟨⟨h1, h2, h3, h4, (obstacleClear_raise c A Acc Brk Rh Vm K q _ _).mpr h5, ?_⟩, ⟨g1, g2, ?_, ?_, ?_⟩⟩
  · rw [pathCurv_raise]; exact Int.le_trans h6 h.kap
  · exact Int.le_trans g3 (pathCap_raise c A Acc Brk Rh Vm K h q s t.current)
  · show t.b ≤ s.b + 2 * Acc * gapAt q s.station
    omega
  · show s.b ≤ t.b + 2 * Brk * gapAt q s.station
    omega

/-- Nonnegative segment lengths inside the preview (true for every well-formed
observation; stated as a decidable hypothesis). -/
def GapsNonneg (q : Steer.Q) : Prop := ∀ i, i + 1 < q.base.s.length → 0 ≤ gapAt q i

/-- **Budget lever.**  Raising any declared limit (`a_lat`, `a_acc`, `a_brk`,
`r_h`, `v_max`, `kappa_max`) can only lower the optimal declared cost of the joint
plan, and keeps every feasible start feasible. -/
theorem budget_lever (c : Steer.Consts) (A Acc Brk Rh Vm K : Int) (h : Raised c A Acc Brk Rh Vm K)
    (q : Steer.Q) (hq : GapsNonneg q) (grid : List Int) (time : Nat → Int → Int → Int)
    (m : Nat) (s : JState) (v : Int)
    (hv : BellmanOpt.valueO (terminal c q) (List.replicate m (actions c q grid time)) s = some v) :
    ∃ v', BellmanOpt.valueO (terminal (raise c A Acc Brk Rh Vm K) q)
        (List.replicate m (actions (raise c A Acc Brk Rh Vm K) q grid time)) s = some v' ∧ v' ≤ v := by
  rw [joint_value_eq] at hv ⊢
  exact lever_value_le (fun _ => True) (stepOK_shaped _ q grid)
    (fun s t _ hlen hok => ⟨stepOK_raise c A Acc Brk Rh Vm K h q grid s t (hq s.station hlen) hok, trivial⟩)
    (terminalCap_raise c A Acc Brk Rh Vm K h q) m s v trivial hv

/-- The cornering budget alone. -/
theorem cornering_lever (c : Steer.Consts) (A : Int) (hA : c.lon.a_lat ≤ A) (hlat0 : 0 ≤ c.lon.a_lat)
    (hrh0 : 0 ≤ c.lon.r_h) (hvm0 : 0 ≤ c.lon.v_max) (q : Steer.Q) (hq : GapsNonneg q) (grid : List Int)
    (time : Nat → Int → Int → Int) (m : Nat) (s : JState) (v : Int)
    (hv : BellmanOpt.valueO (terminal c q) (List.replicate m (actions c q grid time)) s = some v) :
    ∃ v', BellmanOpt.valueO
        (terminal (raise c A c.lon.a_acc c.lon.a_brk c.lon.r_h c.lon.v_max c.kappa_max) q)
        (List.replicate m (actions (raise c A c.lon.a_acc c.lon.a_brk c.lon.r_h c.lon.v_max c.kappa_max)
          q grid time)) s = some v' ∧ v' ≤ v :=
  budget_lever c _ _ _ _ _ _
    ⟨hA, Int.le_refl _, Int.le_refl _, Int.le_refl _, Int.le_refl _, Int.le_refl _, hlat0, hrh0, hvm0⟩
    q hq grid time m s v hv

/-- The braking budget alone. -/
theorem braking_lever (c : Steer.Consts) (B : Int) (hB : c.lon.a_brk ≤ B) (hlat0 : 0 ≤ c.lon.a_lat)
    (hrh0 : 0 ≤ c.lon.r_h) (hvm0 : 0 ≤ c.lon.v_max) (q : Steer.Q) (hq : GapsNonneg q) (grid : List Int)
    (time : Nat → Int → Int → Int) (m : Nat) (s : JState) (v : Int)
    (hv : BellmanOpt.valueO (terminal c q) (List.replicate m (actions c q grid time)) s = some v) :
    ∃ v', BellmanOpt.valueO
        (terminal (raise c c.lon.a_lat c.lon.a_acc B c.lon.r_h c.lon.v_max c.kappa_max) q)
        (List.replicate m (actions (raise c c.lon.a_lat c.lon.a_acc B c.lon.r_h c.lon.v_max c.kappa_max)
          q grid time)) s = some v' ∧ v' ≤ v :=
  budget_lever c _ _ _ _ _ _
    ⟨Int.le_refl _, Int.le_refl _, hB, Int.le_refl _, Int.le_refl _, Int.le_refl _, hlat0, hrh0, hvm0⟩
    q hq grid time m s v hv

/-- The acceleration budget alone. -/
theorem accel_lever (c : Steer.Consts) (Acc : Int) (hAcc : c.lon.a_acc ≤ Acc) (hlat0 : 0 ≤ c.lon.a_lat)
    (hrh0 : 0 ≤ c.lon.r_h) (hvm0 : 0 ≤ c.lon.v_max) (q : Steer.Q) (hq : GapsNonneg q) (grid : List Int)
    (time : Nat → Int → Int → Int) (m : Nat) (s : JState) (v : Int)
    (hv : BellmanOpt.valueO (terminal c q) (List.replicate m (actions c q grid time)) s = some v) :
    ∃ v', BellmanOpt.valueO
        (terminal (raise c c.lon.a_lat Acc c.lon.a_brk c.lon.r_h c.lon.v_max c.kappa_max) q)
        (List.replicate m (actions (raise c c.lon.a_lat Acc c.lon.a_brk c.lon.r_h c.lon.v_max c.kappa_max)
          q grid time)) s = some v' ∧ v' ≤ v :=
  budget_lever c _ _ _ _ _ _
    ⟨Int.le_refl _, hAcc, Int.le_refl _, Int.le_refl _, Int.le_refl _, Int.le_refl _, hlat0, hrh0, hvm0⟩
    q hq grid time m s v hv

/-- The terminal horizon radius alone (dropping the terminal bound near the finish
is the case of a large `Rh`). -/
theorem horizon_lever (c : Steer.Consts) (Rh : Int) (hRh : c.lon.r_h ≤ Rh) (hlat0 : 0 ≤ c.lon.a_lat)
    (hrh0 : 0 ≤ c.lon.r_h) (hvm0 : 0 ≤ c.lon.v_max) (q : Steer.Q) (hq : GapsNonneg q) (grid : List Int)
    (time : Nat → Int → Int → Int) (m : Nat) (s : JState) (v : Int)
    (hv : BellmanOpt.valueO (terminal c q) (List.replicate m (actions c q grid time)) s = some v) :
    ∃ v', BellmanOpt.valueO
        (terminal (raise c c.lon.a_lat c.lon.a_acc c.lon.a_brk Rh c.lon.v_max c.kappa_max) q)
        (List.replicate m (actions (raise c c.lon.a_lat c.lon.a_acc c.lon.a_brk Rh c.lon.v_max c.kappa_max)
          q grid time)) s = some v' ∧ v' ≤ v :=
  budget_lever c _ _ _ _ _ _
    ⟨Int.le_refl _, Int.le_refl _, Int.le_refl _, hRh, Int.le_refl _, Int.le_refl _, hlat0, hrh0, hvm0⟩
    q hq grid time m s v hv

/-! ## 4. Extra speed² caps (gap-speed rule, sidestep gate) -/

/-- The joint step with an extra per-station speed² cap `E` on the departure speed. -/
def capped (c : Steer.Consts) (q : Steer.Q) (grid : List Int) (E : Nat → Int) (s t : JState) : Prop :=
  stepOK c q grid s t ∧ s.b ≤ E s.station

instance cappedDecidable (c : Steer.Consts) (q : Steer.Q) (grid : List Int) (E : Nat → Int)
    (s t : JState) : Decidable (capped c q grid E s t) := by
  unfold capped; infer_instance

theorem capped_shaped (c : Steer.Consts) (q : Steer.Q) (grid : List Int) (E : Nat → Int) :
    Shaped (capped c q grid E) grid :=
  fun s t h => stepOK_shaped c q grid s t h.1

/-- An extra cap can only raise the optimum: dropping it never costs. -/
theorem extra_caps_cost (c : Steer.Consts) (q : Steer.Q) (grid : List Int) (E : Nat → Int)
    (time : Nat → Int → Int → Int) (m : Nat) (s : JState) (v : Int)
    (hv : valueP (capped c q grid E) q.base.s.length (terminalCap c q) grid time m s = some v) :
    ∃ v', BellmanOpt.valueO (terminal c q) (List.replicate m (actions c q grid time)) s = some v' ∧
      v' ≤ v := by
  rw [joint_value_eq]
  exact lever_value_le (fun _ => True) (stepOK_shaped c q grid)
    (fun _ _ _ _ h => ⟨h.1, trivial⟩) (Int.le_refl _) m s v trivial hv

/-- Monotone in the cap: a looser cap never makes the optimum worse. -/
theorem extra_caps_mono (c : Steer.Consts) (q : Steer.Q) (grid : List Int) (E E' : Nat → Int)
    (hE : ∀ i, E i ≤ E' i) (time : Nat → Int → Int → Int) (m : Nat) (s : JState) (v : Int)
    (hv : valueP (capped c q grid E) q.base.s.length (terminalCap c q) grid time m s = some v) :
    ∃ v', valueP (capped c q grid E') q.base.s.length (terminalCap c q) grid time m s = some v' ∧
      v' ≤ v :=
  lever_value_le (fun _ => True) (capped_shaped c q grid E')
    (fun s _ _ _ h => ⟨⟨h.1, Int.le_trans h.2 (hE s.station)⟩, trivial⟩) (Int.le_refl _) m s v trivial hv

/-! ## 5. Box versus friction circle -/

/-- Lateral acceleration at departure, scaled by `10⁹·2d` (mm/s² · 10⁹ · 2d). -/
def latTerm (c : Steer.Consts) (q : Steer.Q) (s t : JState) : Int :=
  s.b * max (pathCurv c q s t.current) 0 * (2 * gapAt q s.station)

/-- Longitudinal acceleration over the segment, scaled the same way. -/
def lonTerm (s t : JState) : Int := (t.b - s.b) * 1000000000

/-- Radius `A` (mm/s²), scaled the same way. -/
def radTerm (q : Steer.Q) (A : Int) (s : JState) : Int := A * 1000000000 * (2 * gapAt q s.station)

/-- Friction-circle step of radius `A`: the same offsets and grids as the joint
model, speed² under `v_max²`, and combined lateral (at the departure station,
for the declared path curvature) and longitudinal acceleration inside the circle. -/
def circleOK (c : Steer.Consts) (q : Steer.Q) (grid : List Int) (A : Int) (s t : JState) : Prop :=
  offsetOK c q s t ∧ t.b ∈ grid ∧ 0 ≤ t.b ∧ 0 ≤ s.b ∧ s.b ≤ c.lon.v_max * c.lon.v_max ∧
    latTerm c q s t * latTerm c q s t + lonTerm s t * lonTerm s t ≤ radTerm q A s * radTerm q A s

instance circleOKDecidable (c : Steer.Consts) (q : Steer.Q) (grid : List Int) (A : Int)
    (s t : JState) : Decidable (circleOK c q grid A s t) := by
  unfold circleOK offsetOK Dp.corridorClear Dp.obstacleClear
  infer_instance

theorem circleOK_shaped (c : Steer.Consts) (q : Steer.Q) (grid : List Int) (A : Int) :
    Shaped (circleOK c q grid A) grid :=
  fun _ _ h => ⟨h.1.1, h.1.2.1, h.1.2.2.1, h.2.1⟩

theorem sq_le_sq_of_abs {u r : Int} (h1 : u ≤ r) (h2 : -r ≤ u) : u * u ≤ r * r := by
  have hr : 0 ≤ r := by omega
  rcases Int.le_total 0 u with hu | hu
  · exact Int.mul_le_mul h1 h1 hu hr
  · have h := Int.mul_le_mul (show -u ≤ r by omega) (show -u ≤ r by omega) (by omega) hr
    rwa [Int.neg_mul_neg] at h

/-- **Box ⊆ circle.**  If the box `(a_lat, max(a_acc, a_brk))` fits in the circle
of radius `A`, every certified joint step from a nonnegative speed² is a circle
step. -/
theorem box_step_circle (c : Steer.Consts) (q : Steer.Q) (grid : List Int) (A : Int) (s t : JState)
    (hlat0 : 0 ≤ c.lon.a_lat)
    (hA : c.lon.a_lat * c.lon.a_lat + max c.lon.a_acc c.lon.a_brk * max c.lon.a_acc c.lon.a_brk ≤ A * A)
    (hgap : 0 ≤ gapAt q s.station) (hs0 : 0 ≤ s.b) (hok : stepOK c q grid s t) :
    circleOK c q grid A s t := by
  obtain ⟨hoff, ⟨g1, g2, g3, g4, g5⟩⟩ := hok
  refine ⟨hoff, g1, g2, hs0,
    Int.le_trans g3 (Int.le_trans (pathCap_le_capAt c q s t.current) (capAt_le_vmax _ _ _)), ?_⟩
  -- the lateral term
  generalize hd : gapAt q s.station = d at hgap g4 g5
  have hdd : 0 ≤ 2 * d := by omega
  have hlat : s.b * max (pathCurv c q s t.current) 0 ≤ c.lon.a_lat * 1000000000 := by
    by_cases hk : 0 < pathCurv c q s t.current
    · rw [show max (pathCurv c q s t.current) 0 = pathCurv c q s t.current from
        Int.max_eq_left (Int.le_of_lt hk)]
      exact pathCap_lateral_sound c q s t.current s.b hk g3
    · rw [show max (pathCurv c q s t.current) 0 = 0 from Int.max_eq_right (by omega), Int.mul_zero]
      exact Int.mul_nonneg hlat0 (by decide)
  have hX0 : 0 ≤ s.b * max (pathCurv c q s t.current) 0 * (2 * d) :=
    Int.mul_nonneg (Int.mul_nonneg hs0 (Int.le_max_right _ _)) hdd
  have hX : s.b * max (pathCurv c q s t.current) 0 * (2 * d) ≤ c.lon.a_lat * 1000000000 * (2 * d) :=
    Int.mul_le_mul_of_nonneg_right hlat hdd
  have hXX := Int.mul_le_mul hX hX hX0 (Int.le_trans hX0 hX)
  -- the longitudinal term
  have ha1 : c.lon.a_acc ≤ max c.lon.a_acc c.lon.a_brk := Int.le_max_left _ _
  have ha2 : c.lon.a_brk ≤ max c.lon.a_acc c.lon.a_brk := Int.le_max_right _ _
  generalize hamax : max c.lon.a_acc c.lon.a_brk = a at hA ha1 ha2
  have hu1 : t.b - s.b ≤ 2 * a * d := by
    have := Int.mul_le_mul_of_nonneg_right (show 2 * c.lon.a_acc ≤ 2 * a by omega) hgap
    omega
  have hu2 : -(2 * a * d) ≤ t.b - s.b := by
    have := Int.mul_le_mul_of_nonneg_right (show 2 * c.lon.a_brk ≤ 2 * a by omega) hgap
    omega
  have hUU := sq_le_sq_of_abs hu1 hu2
  have hYY : lonTerm s t * lonTerm s t ≤ (2 * a * d) * (2 * a * d) * (1000000000 * 1000000000) := by
    unfold lonTerm
    have := Int.mul_le_mul_of_nonneg_right hUU (show (0 : Int) ≤ 1000000000 * 1000000000 by decide)
    have e : (t.b - s.b) * 1000000000 * ((t.b - s.b) * 1000000000) =
        (t.b - s.b) * (t.b - s.b) * (1000000000 * 1000000000) := by grind
    rw [e]; exact this
  -- the radius
  have hR : (1000000000 * (2 * d)) * (1000000000 * (2 * d)) * (c.lon.a_lat * c.lon.a_lat + a * a) ≤
      (1000000000 * (2 * d)) * (1000000000 * (2 * d)) * (A * A) :=
    Int.mul_le_mul_of_nonneg_left hA (self_mul_nonneg _)
  have e1 : c.lon.a_lat * 1000000000 * (2 * d) * (c.lon.a_lat * 1000000000 * (2 * d)) +
      (2 * a * d) * (2 * a * d) * (1000000000 * 1000000000) =
      (1000000000 * (2 * d)) * (1000000000 * (2 * d)) * (c.lon.a_lat * c.lon.a_lat + a * a) := by grind
  have e2 : radTerm q A s * radTerm q A s = (1000000000 * (2 * d)) * (1000000000 * (2 * d)) * (A * A) := by
    unfold radTerm; rw [hd]; grind
  show latTerm c q s t * latTerm c q s t + lonTerm s t * lonTerm s t ≤ radTerm q A s * radTerm q A s
  unfold latTerm
  rw [hd, e2]
  omega

/-- **Circle relaxation.**  Under the same fit condition, from a nonnegative
start speed², the friction-circle model is feasible whenever the certified joint
model is, and its optimal declared cost is no larger. -/
theorem circle_value_le_box (c : Steer.Consts) (q : Steer.Q) (hq : GapsNonneg q) (grid : List Int)
    (A : Int) (hlat0 : 0 ≤ c.lon.a_lat)
    (hA : c.lon.a_lat * c.lon.a_lat + max c.lon.a_acc c.lon.a_brk * max c.lon.a_acc c.lon.a_brk ≤ A * A)
    (time : Nat → Int → Int → Int) (m : Nat) (s : JState) (hs0 : 0 ≤ s.b) (v : Int)
    (hv : BellmanOpt.valueO (terminal c q) (List.replicate m (actions c q grid time)) s = some v) :
    ∃ v', valueP (circleOK c q grid A) q.base.s.length (terminalCap c q) grid time m s = some v' ∧
      v' ≤ v := by
  rw [joint_value_eq] at hv
  exact lever_value_le (fun s => 0 ≤ s.b) (circleOK_shaped c q grid A)
    (fun s t hI hlen hok =>
      ⟨box_step_circle c q grid A s t hlat0 hA (hq s.station hlen) hI hok, hok.2.2.1⟩)
    (Int.le_refl _) m s v hs0 hv

/-! ## 6. Kernel-checked instances on `Joint`'s toy problem

`Joint.toyC`: three stations 1 m apart, a forced left sidestep around an obstacle
at 2 m, start speed 4 m/s, `a_lat = 6`, `a_acc = 2`, `a_brk = 3` m/s².  Base
optimum 82491 (`Joint.toy_value`).  These instances show which levers bind for
THIS problem; they are not claims about other courses. -/

set_option maxRecDepth 200000

/-- `toyC` with only the cornering budget changed. -/
def toyLat (A : Int) : Steer.Consts := raise toyC A 2000 3000 4000 20000 1000000
/-- `toyC` with only the braking budget changed. -/
def toyBrk (B : Int) : Steer.Consts := raise toyC 6000 2000 B 4000 20000 1000000

/-- Cornering binds: 5 m/s² makes the start infeasible, 7 m/s² changes nothing
(the grid cannot use it), 8 m/s² lowers the optimum from 82491 to 51417. -/
theorem toy_cornering_binds :
    valueP (stepOK (toyLat 5000) (toyQ 4000) toyGrid) 3 (terminalCap (toyLat 5000) (toyQ 4000))
      toyGrid toyTime 2 (start (toyQ 4000)) = none ∧
    valueP (stepOK (toyLat 7000) (toyQ 4000) toyGrid) 3 (terminalCap (toyLat 7000) (toyQ 4000))
      toyGrid toyTime 2 (start (toyQ 4000)) = some 82491 ∧
    valueP (stepOK (toyLat 8000) (toyQ 4000) toyGrid) 3 (terminalCap (toyLat 8000) (toyQ 4000))
      toyGrid toyTime 2 (start (toyQ 4000)) = some 51417 := by decide

/-- Braking does not bind here: tripling `a_brk` leaves the optimum at 82491. -/
theorem toy_braking_slack :
    valueP (stepOK (toyBrk 9000) (toyQ 4000) toyGrid) 3 (terminalCap (toyBrk 9000) (toyQ 4000))
      toyGrid toyTime 2 (start (toyQ 4000)) = some 82491 := by decide

/-- The friction circle of radius 7 m/s² contains the certified box
(`6² + 3² ≤ 7²`) and strictly improves the optimum: 74074 < 82491. -/
theorem toy_circle_gain :
    toyLon.a_lat * toyLon.a_lat + max toyLon.a_acc toyLon.a_brk * max toyLon.a_acc toyLon.a_brk ≤
      7000 * 7000 ∧
    valueP (circleOK toyC (toyQ 4000) toyGrid 7000) 3 (terminalCap toyC (toyQ 4000))
      toyGrid toyTime 2 (start (toyQ 4000)) = some 74074 := by decide

/-- An extra speed² cap below the start speed² makes the start infeasible. -/
theorem toy_extra_cap_infeasible :
    valueP (capped toyC (toyQ 4000) toyGrid (fun _ => 12000000)) 3 (terminalCap toyC (toyQ 4000))
      toyGrid toyTime 2 (start (toyQ 4000)) = none := by decide

end Corpus.Levers
