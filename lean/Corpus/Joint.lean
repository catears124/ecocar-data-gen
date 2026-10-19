/-
  Joint lateral-offset / speed² planning over the real `Steer.Q`, solved by the
  option-valued Bellman recursion of `Corpus.BellmanOpt`.

  ## Model (all integers: mm, mm²/s², mm/s², 1e-6 / m)
  A state is `(station i, previous offset, current offset, b)` where `b` is the
  speed² AT station `i`.  A step `s → t` is allowed (`stepOK`) iff
  * offsets: `t.station = s.station + 1`, `t.previous = s.current` (history is
    carried, so the curvature stencil cannot be rewritten), `t.current ∈
    Dp.bucketGrid`, corridor (`Dp.corridorClear`) and obstacle clearance
    (`Dp.obstacleClear`, i.e. `Steer.clearAt ≥ 0`) at the arrival station, and
    the declared path curvature at the departure station
    `pathCurv = |kbar_road(i)| + 10⁹·|n - 2x + p| / step²` is `≤ kappa_max`;
  * speed: the DEPARTURE speed is under the cap of the departure station's actual
    stencil (`s.b ≤ pathCap`), `t.b` is on the declared speed² grid (mm²/s²),
    `0 ≤ t.b`, and `t.b ≤ s.b + 2·a_acc·d`, `s.b ≤ t.b + 2·a_brk·d`
    (`d = s_{i+1} - s_i`): exactly the per-segment constraints of `Cert.Feasible`.
  The terminal state is feasible iff its speed² is `≤ Cert.lastCap` (the
  longitudinal certificate's declared terminal constraint).  The objective is an
  arbitrary declared per-transition cost `time i b_i b_{i+1}`; nothing here
  claims that a particular table equals physical seconds.

  ## Results
  * `joint_value_spec` / `joint_infeasible_iff`: the DP value is the minimum of the
    declared cost over the DECLARATIVE set of joint plans (`PlanOK`), is attained,
    and is `none` exactly when no joint plan exists.
  * `joint_extract_spec`: the executable extraction returns a plan in that set
    with optimal cost; every step satisfies `stepOK`.
  * `plan_cert_feasible` (composition): the speed² sequence of every joint plan is
    `Cert.Feasible` for the caps of the path the lateral component chose, hence
    pointwise below the longitudinal certificate's greatest profile for those caps
    (`plan_le_fpass`) and the longitudinal decision on them is not min-violation
    (`plan_start_le_bpass`).
  * `pathCap_le_capAt`, `pathCap_lateral_sound`, `pathCap_physical_sound`.
  * kernel-checked counterexamples to `Dp.vModel` (unit mismatch, curvature
    under-count) and a kernel-checked concrete instance (`toy_*`).
-/
import Corpus.BellmanOpt
import Corpus.Dp

namespace Corpus.Joint
open Corpus

abbrev Act := Bellman.Action

structure JState where
  station : Nat
  previous : Int
  current : Int
  b : Int
deriving DecidableEq, Repr

def JState.toJ (s : JState) : Dp.JState :=
  { station := s.station, previous := s.previous, current := s.current }

def gapAt (q : Steer.Q) (i : Nat) : Int := q.base.s.getD (i + 1) 0 - q.base.s.getD i 0

/-- Lateral curvature (1e-6/m) of the stencil `p, x, n` (mm) at uniform spacing
`h` (mm): `|n - 2x + p| / h²`, rounded up. -/
def latCurv (h p x n : Int) : Int := Cert.cdiv (1000000000 * Cert.iabs (n - 2 * x + p)) (h * h)

/-- Certified road term of the longitudinal model at station `i`. -/
def roadCurv (c : Steer.Consts) (q : Steer.Q) (i : Nat) : Int :=
  Cert.iabs (Cert.kbarOf c.lon (Cert.iabs (q.base.k.getD i 0))
    (Cert.stationY c.lon q.base (q.base.s.getD i 0)))

/-- Declared curvature bound of the chosen path at the departure station. -/
def pathCurv (c : Steer.Consts) (q : Steer.Q) (s : JState) (n : Int) : Int :=
  roadCurv c q s.station + latCurv c.step s.previous s.current n

/-- Speed² cap of the chosen path at the departure station: the longitudinal
certificate's own cap, tightened by the lateral-acceleration bound of the path. -/
def pathCap (c : Steer.Consts) (q : Steer.Q) (s : JState) (n : Int) : Int :=
  if pathCurv c q s n ≤ 0 then Cert.capAt c.lon q.base s.station
  else min (Cert.capAt c.lon q.base s.station) (c.lon.a_lat * 1000000000 / pathCurv c q s n)

def offsetOK (c : Steer.Consts) (q : Steer.Q) (s t : JState) : Prop :=
  t.station = s.station + 1 ∧ t.previous = s.current ∧ t.current ∈ Dp.bucketGrid ∧
  Dp.corridorClear c q t.station t.current ∧ Dp.obstacleClear c q t.station t.current ∧
  pathCurv c q s t.current ≤ c.kappa_max

def speedOK (c : Steer.Consts) (q : Steer.Q) (grid : List Int) (s t : JState) : Prop :=
  t.b ∈ grid ∧ 0 ≤ t.b ∧ s.b ≤ pathCap c q s t.current ∧
  t.b ≤ s.b + 2 * c.lon.a_acc * gapAt q s.station ∧
  s.b ≤ t.b + 2 * c.lon.a_brk * gapAt q s.station

def stepOK (c : Steer.Consts) (q : Steer.Q) (grid : List Int) (s t : JState) : Prop :=
  offsetOK c q s t ∧ speedOK c q grid s t

instance stepOKDecidable (c : Steer.Consts) (q : Steer.Q) (grid : List Int) (s t : JState) :
    Decidable (stepOK c q grid s t) := by
  unfold stepOK offsetOK speedOK Dp.corridorClear Dp.obstacleClear
  infer_instance

def terminalCap (c : Steer.Consts) (q : Steer.Q) : Int := Cert.lastCap c.lon q.base

/-- Terminal feasibility: the declared terminal speed constraint. -/
def terminal (c : Steer.Consts) (q : Steer.Q) : JState → Option Int :=
  fun s => if s.b ≤ terminalCap c q then some 0 else none

/-- Offered actions: every certified offset bucket and every speed² grid value. -/
def actions (c : Steer.Consts) (q : Steer.Q) (grid : List Int) (time : Nat → Int → Int → Int)
    (s : JState) : List (Act JState) :=
  if s.station + 1 < q.base.s.length then
    Dp.bucketGrid.flatMap fun n => grid.filterMap fun w =>
      let t : JState := { station := s.station + 1, previous := s.current, current := n, b := w }
      if stepOK c q grid s t then some (t, time s.station s.b w) else none
  else []

/-- `s.length - 1` identical stages: one transition per preview segment. -/
def stages (c : Steer.Consts) (q : Steer.Q) (grid : List Int) (time : Nat → Int → Int → Int) :
    List (BellmanOpt.Stage JState) :=
  List.replicate (q.base.s.length - 1) (actions c q grid time)

/-- The real start: origin offsets, observed speed². -/
def start (q : Steer.Q) : JState := { station := 0, previous := 0, current := 0, b := q.base.v0 * q.base.v0 }

/-! ## Declarative joint plans -/

/-- A joint plan from `s`: a list of successive states, each step allowed and
inside the preview, ending at a terminal-feasible state. -/
def PlanOK (c : Steer.Consts) (q : Steer.Q) (grid : List Int) : JState → List JState → Prop
  | s, [] => s.b ≤ terminalCap c q
  | s, t :: ts => stepOK c q grid s t ∧ s.station + 1 < q.base.s.length ∧ PlanOK c q grid t ts

def planCost (time : Nat → Int → Int → Int) : JState → List JState → Int
  | _, [] => 0
  | s, t :: ts => time s.station s.b t.b + planCost time t ts

theorem map_fst_length (xs : List (Act JState)) : (xs.map Prod.fst).length = xs.length := by
  induction xs with
  | nil => rfl
  | cons a t ih => exact congrArg (· + 1) ih

theorem mem_actions {c : Steer.Consts} {q : Steer.Q} {grid : List Int}
    {time : Nat → Int → Int → Int} {s : JState} {a : Act JState}
    (h : a ∈ actions c q grid time s) :
    stepOK c q grid s a.1 ∧ s.station + 1 < q.base.s.length ∧ a.2 = time s.station s.b a.1.b := by
  unfold actions at h
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

theorem actions_complete {c : Steer.Consts} {q : Steer.Q} {grid : List Int}
    {time : Nat → Int → Int → Int} {s t : JState}
    (hok : stepOK c q grid s t) (hlen : s.station + 1 < q.base.s.length) :
    (t, time s.station s.b t.b) ∈ actions c q grid time s := by
  have hst := hok.1.1
  have hpr := hok.1.2.1
  have hgr := hok.1.2.2.1
  have hbg := hok.2.1
  cases t with
  | mk ts tp tc tb =>
    simp only at hst hpr hgr hbg
    subst hst hpr
    unfold actions
    rw [if_pos hlen]
    exact List.mem_flatMap.mpr ⟨tc, hgr, List.mem_filterMap.mpr ⟨tb, hbg, by rw [if_pos hok]; rfl⟩⟩

/-- Soundness of the DP encoding: admissible trajectories are joint plans with
the same cost. -/
theorem traj_plan (c : Steer.Consts) (q : Steer.Q) (grid : List Int) (time : Nat → Int → Int → Int) :
    ∀ (m : Nat) (s : JState) (xs : List (Act JState)),
      BellmanOpt.TrajOK (terminal c q) (List.replicate m (actions c q grid time)) s xs →
      PlanOK c q grid s (xs.map Prod.fst) ∧
      BellmanOpt.trajCost (terminal c q) (List.replicate m (actions c q grid time)) s xs =
        some (planCost time s (xs.map Prod.fst)) ∧ xs.length = m := by
  intro m
  induction m with
  | zero =>
      intro s xs h
      cases xs with
      | nil =>
          obtain ⟨v, hv⟩ := h
          unfold terminal at hv
          split at hv
          · rename_i hb
            refine ⟨hb, ?_, rfl⟩
            show terminal c q s = some 0
            simp [terminal, hb]
          · cases hv
      | cons _ _ => exact h.elim
  | succ m ih =>
      intro s xs h
      cases xs with
      | nil => exact h.elim
      | cons a ys =>
          obtain ⟨ha, ht⟩ := h
          obtain ⟨hok, hlen, hcost⟩ := mem_actions ha
          obtain ⟨hp, hc, hl⟩ := ih a.1 ys ht
          refine ⟨⟨hok, hlen, hp⟩, ?_, by rw [List.length_cons, hl]⟩
          show (BellmanOpt.trajCost (terminal c q) (List.replicate m (actions c q grid time)) a.1 ys).map
              (fun v => a.2 + v) = some (time s.station s.b a.1.b + planCost time a.1 (ys.map Prod.fst))
          rw [hc, hcost]
          rfl

/-- Completeness of the DP encoding: every joint plan of the right length is an
admissible trajectory with the same cost. -/
theorem plan_traj (c : Steer.Consts) (q : Steer.Q) (grid : List Int) (time : Nat → Int → Int → Int) :
    ∀ (ts : List JState) (s : JState), PlanOK c q grid s ts →
      ∃ xs, xs.map Prod.fst = ts ∧
        BellmanOpt.TrajOK (terminal c q) (List.replicate ts.length (actions c q grid time)) s xs ∧
        BellmanOpt.trajCost (terminal c q) (List.replicate ts.length (actions c q grid time)) s xs =
          some (planCost time s ts) := by
  intro ts
  induction ts with
  | nil =>
      intro s h
      have hb : s.b ≤ terminalCap c q := h
      refine ⟨[], rfl, ⟨0, by simp [terminal, hb]⟩, ?_⟩
      show terminal c q s = some 0
      simp [terminal, hb]
  | cons t ts ih =>
      intro s h
      obtain ⟨hok, hlen, hp⟩ := h
      obtain ⟨ys, hys, hok', hc⟩ := ih t hp
      refine ⟨((t, time s.station s.b t.b) : Act JState) :: ys, ?_, ⟨actions_complete hok hlen, hok'⟩, ?_⟩
      · show t :: ys.map Prod.fst = t :: ts
        rw [hys]
      · show (BellmanOpt.trajCost (terminal c q) (List.replicate ts.length (actions c q grid time)) t ys).map
            (fun v => time s.station s.b t.b + v) = some (time s.station s.b t.b + planCost time t ts)
        rw [hc]
        rfl

/-! ## Joint optimality over the declarative plan set -/

/-- **Joint optimality.**  If the DP value from `s` over `m` stages is `some v`,
then `v` is a lower bound on the declared cost of every joint plan of length `m`
from `s`, and some joint plan of length `m` attains it. -/
theorem joint_value_spec (c : Steer.Consts) (q : Steer.Q) (grid : List Int)
    (time : Nat → Int → Int → Int) (m : Nat) (s : JState) (v : Int)
    (hv : BellmanOpt.valueO (terminal c q) (List.replicate m (actions c q grid time)) s = some v) :
    (∀ ts, PlanOK c q grid s ts → ts.length = m → v ≤ planCost time s ts) ∧
    (∃ ts, PlanOK c q grid s ts ∧ ts.length = m ∧ planCost time s ts = v) := by
  constructor
  · intro ts hp hl
    obtain ⟨xs, _, hok, hc⟩ := plan_traj c q grid time ts s hp
    rw [hl] at hok hc
    exact BellmanOpt.valueO_le_cost _ _ s xs v _ hok hc hv
  · obtain ⟨xs, hok, hc⟩ := BellmanOpt.valueO_attained _ _ s v hv
    obtain ⟨hp, hc', hl⟩ := traj_plan c q grid time m s xs hok
    rw [hc'] at hc
    simp only [Option.some.injEq] at hc
    exact ⟨xs.map Prod.fst, hp, (map_fst_length xs).trans hl, hc⟩

/-- **Infeasibility is exact**: the DP value is `none` iff no joint plan of length
`m` exists (dead ends and infeasible starts are never reported as feasible). -/
theorem joint_infeasible_iff (c : Steer.Consts) (q : Steer.Q) (grid : List Int)
    (time : Nat → Int → Int → Int) (m : Nat) (s : JState) :
    BellmanOpt.valueO (terminal c q) (List.replicate m (actions c q grid time)) s = none ↔
      ¬ ∃ ts, PlanOK c q grid s ts ∧ ts.length = m := by
  rw [BellmanOpt.valueO_none_iff]
  constructor
  · intro hn ⟨ts, hp, hl⟩
    obtain ⟨xs, _, hok, _⟩ := plan_traj c q grid time ts s hp
    rw [hl] at hok
    exact hn ⟨xs, hok⟩
  · intro hn ⟨xs, hok⟩
    obtain ⟨hp, _, hl⟩ := traj_plan c q grid time m s xs hok
    exact hn ⟨xs.map Prod.fst, hp, (map_fst_length xs).trans hl⟩

/-- **Extraction.**  The executable extraction returns an optimal joint plan. -/
theorem joint_extract_spec (c : Steer.Consts) (q : Steer.Q) (grid : List Int)
    (time : Nat → Int → Int → Int) (m : Nat) (s : JState) (v : Int)
    (hv : BellmanOpt.valueO (terminal c q) (List.replicate m (actions c q grid time)) s = some v) :
    ∃ xs, BellmanOpt.extract (terminal c q) (List.replicate m (actions c q grid time)) s = some xs ∧
      PlanOK c q grid s (xs.map Prod.fst) ∧ (xs.map Prod.fst).length = m ∧
      planCost time s (xs.map Prod.fst) = v ∧
      (∀ ts, PlanOK c q grid s ts → ts.length = m → planCost time s (xs.map Prod.fst) ≤ planCost time s ts) := by
  obtain ⟨xs, hx, hok, hc, _⟩ := BellmanOpt.extract_spec _ _ s v hv
  obtain ⟨hp, hc', hl⟩ := traj_plan c q grid time m s xs hok
  rw [hc'] at hc
  simp only [Option.some.injEq] at hc
  refine ⟨xs, hx, hp, (map_fst_length xs).trans hl, hc, ?_⟩
  intro ts hts hlen
  rw [hc]
  exact (joint_value_spec c q grid time m s v hv).1 ts hts hlen

/-- The same statements for the real model from the real start. -/
theorem stages_eq (c : Steer.Consts) (q : Steer.Q) (grid : List Int) (time : Nat → Int → Int → Int) :
    stages c q grid time = List.replicate (q.base.s.length - 1) (actions c q grid time) := rfl

/-! ## Every step of every joint plan -/

/-- Station progression and the full step certificate along a plan. -/
def StepsOK (c : Steer.Consts) (q : Steer.Q) (grid : List Int) : JState → List JState → Prop
  | _, [] => True
  | s, t :: ts => stepOK c q grid s t ∧ StepsOK c q grid t ts

theorem plan_steps (c : Steer.Consts) (q : Steer.Q) (grid : List Int) :
    ∀ (ts : List JState) (s : JState), PlanOK c q grid s ts → StepsOK c q grid s ts := by
  intro ts
  induction ts with
  | nil => intro _ _; trivial
  | cons t ts ih => intro s h; exact ⟨h.1, ih t h.2.2⟩

/-- Last state of a plan. -/
def lastState : JState → List JState → JState
  | s, [] => s
  | _, t :: ts => lastState t ts

theorem plan_progress (c : Steer.Consts) (q : Steer.Q) (grid : List Int) :
    ∀ (ts : List JState) (s : JState), PlanOK c q grid s ts →
      (lastState s ts).station = s.station + ts.length ∧
      (lastState s ts).b ≤ terminalCap c q := by
  intro ts
  induction ts with
  | nil => intro s h; exact ⟨by simp [lastState], h⟩
  | cons t ts ih =>
      intro s h
      obtain ⟨hst, hb⟩ := ih t h.2.2
      refine ⟨?_, hb⟩
      simp only [lastState, List.length_cons]
      rw [hst, h.1.1.1]
      omega

/-! ## Composition with the longitudinal certificate -/

def segsOf (c : Steer.Consts) (q : Steer.Q) : JState → List JState → List (Int × Int)
  | _, [] => []
  | s, t :: ts => (pathCap c q s t.current, gapAt q s.station) :: segsOf c q t ts

def speedsOf : JState → List JState → List Int
  | s, [] => [s.b]
  | s, t :: ts => s.b :: speedsOf t ts

theorem speedsOf_head (s : JState) (ts : List JState) : ∃ r, speedsOf s ts = s.b :: r := by
  cases ts with
  | nil => exact ⟨[], rfl⟩
  | cons t ts => exact ⟨speedsOf t ts, rfl⟩

/-- **Composition.**  The speed² sequence of every joint plan (from a state with
`0 ≤ b`) is a `Cert.Feasible` profile for the caps of the path the lateral
component chose and the declared terminal cap. -/
theorem plan_cert_feasible (c : Steer.Consts) (q : Steer.Q) (grid : List Int) :
    ∀ (ts : List JState) (s : JState), PlanOK c q grid s ts → 0 ≤ s.b →
      Cert.Feasible c.lon.a_acc c.lon.a_brk (segsOf c q s ts) (terminalCap c q) (speedsOf s ts) := by
  intro ts
  induction ts with
  | nil => intro s h h0; exact ⟨h0, h⟩
  | cons t ts ih =>
      intro s h h0
      obtain ⟨hok, _, hp⟩ := h
      obtain ⟨r, hr⟩ := speedsOf_head t ts
      have hrec := ih t hp hok.2.2.1
      rw [hr] at hrec
      show Cert.Feasible _ _ ((pathCap c q s t.current, gapAt q s.station) :: segsOf c q t ts) _
        (s.b :: speedsOf t ts)
      rw [hr]
      exact ⟨h0, hok.2.2.2.1, hok.2.2.2.2.1, hok.2.2.2.2.2, hrec⟩

/-- The plan's speeds are pointwise below the longitudinal certificate's greatest
feasible profile on the chosen path's caps. -/
theorem plan_le_fpass (c : Steer.Consts) (q : Steer.Q) (grid : List Int)
    (ts : List JState) (s : JState) (h : PlanOK c q grid s ts) (h0 : 0 ≤ s.b) :
    Cert.LeAll (speedsOf s ts)
      (Cert.fpass c.lon.a_acc c.lon.a_brk (segsOf c q s ts) (terminalCap c q) s.b) := by
  have hf := plan_cert_feasible c q grid ts s h h0
  obtain ⟨r, hr⟩ := speedsOf_head s ts
  rw [hr] at hf ⊢
  exact Cert.fpass_greatest _ _ _ _ _ _ r hf (Int.le_refl _)

/-- A feasible joint plan implies the longitudinal certificate on the same path
is in its greatest-safe regime (the start is within the braking envelope). -/
theorem plan_start_le_bpass (c : Steer.Consts) (q : Steer.Q) (grid : List Int)
    (ts : List JState) (s : JState) (h : PlanOK c q grid s ts) (h0 : 0 ≤ s.b) :
    s.b ≤ (Cert.bpass c.lon.a_brk (segsOf c q s ts) (terminalCap c q)).headD 0 := by
  have hf := plan_cert_feasible c q grid ts s h h0
  obtain ⟨r, hr⟩ := speedsOf_head s ts
  rw [hr] at hf
  exact Cert.LeAll_head (Cert.le_bpass _ _ _ _ _ hf)

theorem start_nonneg (q : Steer.Q) : 0 ≤ (start q).b := by
  show 0 ≤ q.base.v0 * q.base.v0
  rcases Int.le_total 0 q.base.v0 with h | h
  · exact Int.mul_nonneg h h
  · have := Int.mul_nonneg (Int.neg_nonneg_of_nonpos h) (Int.neg_nonneg_of_nonpos h)
    rwa [Int.neg_mul_neg] at this

/-! ## Soundness of the curvature-based speed restriction -/

theorem pathCap_le_capAt (c : Steer.Consts) (q : Steer.Q) (s : JState) (n : Int) :
    pathCap c q s n ≤ Cert.capAt c.lon q.base s.station := by
  unfold pathCap
  split
  · exact Int.le_refl _
  · exact Int.min_le_left _ _

/-- Arithmetic soundness: a speed² under the cap satisfies `b·κ ≤ a_lat·10⁹`
(i.e. `v²κ ≤ a_lat` in SI units) for the declared curvature `κ`. -/
theorem pathCap_lateral_sound (c : Steer.Consts) (q : Steer.Q) (s : JState) (n b : Int)
    (hk : 0 < pathCurv c q s n) (hb : b ≤ pathCap c q s n) :
    b * pathCurv c q s n ≤ c.lon.a_lat * 1000000000 := by
  unfold pathCap at hb
  rw [if_neg (by omega)] at hb
  have h1 : b ≤ c.lon.a_lat * 1000000000 / pathCurv c q s n := Int.le_trans hb (Int.min_le_right _ _)
  have h2 := Int.mul_le_mul_of_nonneg_right h1 (Int.le_of_lt hk)
  exact Int.le_trans h2 (Int.ediv_mul_le _ (by omega))

/-- Conditional physical form: IF the true curvature `kt` of the driven path is
covered by the declared bound (`0 ≤ kt ≤ pathCurv`, an assumption about geometry
and tracking, not proved here), then lateral acceleration is within `a_lat`. -/
theorem pathCap_physical_sound (c : Steer.Consts) (q : Steer.Q) (s : JState) (n b kt : Int)
    (ha : 0 ≤ c.lon.a_lat) (hb0 : 0 ≤ b) (hb : b ≤ pathCap c q s n) (hkt0 : 0 ≤ kt)
    (hkt : kt ≤ pathCurv c q s n) :
    b * kt ≤ c.lon.a_lat * 1000000000 := by
  by_cases hk : 0 < pathCurv c q s n
  · exact Int.le_trans (Int.mul_le_mul_of_nonneg_left hkt hb0) (pathCap_lateral_sound c q s n b hk hb)
  · have : kt = 0 := by omega
    subst this
    rw [Int.mul_zero]
    omega

/-! ## Kernel-checked concrete instance

Three stations 1 m apart, a 4 m half-width road, one obstacle at 2 m on the
right (`[-1500, -300]` mm), start speed 4 m/s (`b₀ = 16·10⁶`), lateral budget
6 m/s², acceleration 2 m/s², braking 3 m/s², a declared time table decreasing in
speed.  The car must sidestep left to offset 1000 at station 2; the curvature of
that sidestep caps the speed at station 1 (`pathCap < 16·10⁶`), so the optimum
must brake to `10.5·10⁶` first: curvature, braking and the obstacle all bind.
-/

set_option maxRecDepth 200000

def toyLon : Cert.Consts :=
  { a_lat := 6000, a_acc := 2000, a_brk := 3000, a_max := 9000, v_max := 20000,
    r_h := 4000, hold := 500, ramp_in := 1000, ramp_out := 1000, r_sw_m := 15, veh_width := 2000 }
def toyC : Steer.Consts := { lon := toyLon, margin := 200, step := 1000, kappa_max := 1000000 }
def toyBase (v0 : Int) : Cert.Q :=
  { v0 := v0, s := [0, 1000, 2000], k := [0, 0, 0], wl := [4000, 4000, 4000],
    wr := [4000, 4000, 4000], obstacles := [[2000, -1500, -300]] }
def toyQ (v0 : Int) : Steer.Q := { base := toyBase v0, lat := 0 }
def toyGrid : List Int := [0, 9000000, 10500000, 16000000, 20000000]
/-- Declared per-transition cost (decreasing in the mean speed²). -/
def toyTime (_ : Nat) (b w : Int) : Int := 1000000 / (1 + (b + w) / 1000000)

/-- Bool form of `PlanOK` for the independent enumeration. -/
def planOKb (c : Steer.Consts) (q : Steer.Q) (grid : List Int) : JState → List JState → Bool
  | s, [] => decide (s.b ≤ terminalCap c q)
  | s, t :: ts => decide (stepOK c q grid s t) && decide (s.station + 1 < q.base.s.length) &&
      planOKb c q grid t ts

theorem planOKb_sound (c : Steer.Consts) (q : Steer.Q) (grid : List Int) :
    ∀ (ts : List JState) (s : JState), planOKb c q grid s ts = true → PlanOK c q grid s ts := by
  intro ts
  induction ts with
  | nil => intro s h; exact (of_decide_eq_true h : s.b ≤ terminalCap c q)
  | cons t ts ih =>
      intro s h
      simp only [planOKb, Bool.and_eq_true, decide_eq_true_eq] at h
      exact ⟨h.1.1, h.1.2, ih t h.2⟩

theorem planOKb_complete (c : Steer.Consts) (q : Steer.Q) (grid : List Int) :
    ∀ (ts : List JState) (s : JState), PlanOK c q grid s ts → planOKb c q grid s ts = true := by
  intro ts
  induction ts with
  | nil => intro s h; exact decide_eq_true (show s.b ≤ terminalCap c q from h)
  | cons t ts ih =>
      intro s h
      simp only [planOKb, Bool.and_eq_true, decide_eq_true_eq]
      exact ⟨⟨h.1, h.2.1⟩, ih t h.2.2⟩

/-- Every two-step candidate on the declared grids (independent of the DP). -/
def toyCandidates : List (List JState) :=
  Dp.bucketGrid.flatMap fun n1 => toyGrid.flatMap fun b1 => Dp.bucketGrid.flatMap fun n2 =>
    toyGrid.map fun b2 =>
      [{ station := 1, previous := 0, current := n1, b := b1 },
       { station := 2, previous := n1, current := n2, b := b2 }]

def toyEnumMin : Option Int :=
  (toyCandidates.filter (planOKb toyC (toyQ 4000) toyGrid (start (toyQ 4000)))).foldr
    (fun ts acc => match acc with
      | none => some (planCost toyTime (start (toyQ 4000)) ts)
      | some a => some (min a (planCost toyTime (start (toyQ 4000)) ts))) none

theorem toy_value : BellmanOpt.valueO (terminal toyC (toyQ 4000))
    (stages toyC (toyQ 4000) toyGrid toyTime) (start (toyQ 4000)) = some 82491 := by decide

theorem toy_value_eq_enumeration : toyEnumMin = some 82491 := by decide

theorem toy_extract :
    (BellmanOpt.extract (terminal toyC (toyQ 4000)) (stages toyC (toyQ 4000) toyGrid toyTime)
      (start (toyQ 4000))).map (List.map fun a => ((a.1.current, a.1.b) : Int × Int)) =
      some [(250, 10500000), (1000, 10500000)] := by decide

/-- The curvature of the forced sidestep binds: the cap at station 1 is below the
start speed², so the car must brake before the sidestep. -/
theorem toy_curvature_binds :
    pathCap toyC (toyQ 4000) { station := 1, previous := 0, current := 250, b := 0 } 1000 < 16000000 ∧
    planOKb toyC (toyQ 4000) toyGrid (start (toyQ 4000))
      [{ station := 1, previous := 0, current := 250, b := 16000000 },
       { station := 2, previous := 250, current := 1000, b := 16000000 }] = false := by decide

/-- Too fast to brake for the sidestep: the DP reports infeasibility (no dead-end
masquerade), and indeed no candidate plan exists. -/
theorem toy_infeasible :
    BellmanOpt.valueO (terminal toyC (toyQ 6000)) (stages toyC (toyQ 6000) toyGrid toyTime)
      (start (toyQ 6000)) = none ∧
    (toyCandidates.filter (planOKb toyC (toyQ 6000) toyGrid (start (toyQ 6000)))).length = 0 := by decide

/-! ## Counterexamples to the old `Dp` speed model (kernel-checked) -/

/-- `Dp.speedFeasible` compares bucket indices (≤ 1000) with caps in mm²/s²: a cap
of 1000 mm²/s² (0.03 m/s) and a cap of 10⁹ mm²/s² (31.6 m/s) are
indistinguishable on the whole grid, i.e. the cap is inert. -/
theorem old_speed_cap_inert :
    (Dp.speedGrid.all fun v => Dp.speedGrid.all fun w =>
      Dp.speedFeasible 1000 v w == Dp.speedFeasible 1000000000 v w) = true := by decide

/-- `Dp.lateralStencil` is in mm (no 10⁹/step² scaling) while the road term of
`Dp.jointCurvature` is in 1e-6/m: a 1000 mm second difference over 1 m stations
(curvature 1 /m = 10⁶ µ/m, as `Steer.curvAt` and `latCurv` compute) contributes
only 1000. -/
theorem old_stencil_undercount :
    Dp.lateralStencil 0 0 1000 = 1000 ∧ latCurv 1000 0 0 1000 = 1000000 ∧
    Steer.curvAt (toyQ 0) [0, 0, 1000] 1 = 1000000 := by decide

end Corpus.Joint
