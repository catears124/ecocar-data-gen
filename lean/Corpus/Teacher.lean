/-
  Executable, proof-carrying teacher model.

  This module is the Lean counterpart of ``autocross/optimal/model.py`` and
  ``autocross/optimal/certificate.py``.  The Python teacher emits a finite table
  of Bellman values and a plan; ``checkCert`` is a decidable checker for that
  table.  ``checkCert_sound`` (below) turns a passing check into the mathematical
  statements:

    * ``valueO`` from the observed start equals the claimed value;
    * every declarative plan costs at least that value;
    * the exported plan is admissible, reaches the horizon, and costs exactly
      the claimed value;
    * a claimed ``none`` means no declarative plan exists.

  The checker does not trust the solver.  It recomputes the successor relation
  from the encoded instance (grids, corridor, inflated obstacles, curvature,
  acceleration/braking, terminal cap), requires the declared state universe to
  be closed under that relation, and verifies the Bellman equation in every row.
  Roots of the speed² grid are declared too; ``rootsOK`` proves each is the
  unique integer square root, so the integer cost is exactly the ``isqrt`` cost
  of the Python model without making the Lean kernel evaluate ``Nat.sqrt``.

  Information structure: the model reads only the quantised policy observation
  (preview stations, curvatures, widths, obstacle extents, current speed) and the
  declared terminal cap.  Nothing beyond the preview is used.  Optimality is
  therefore optimality over the finite preview model, not over a whole course.
-/
import Corpus.Joint
import Lean.Data.Json
import Corpus.BellmanOpt
import Corpus.Cert
import Corpus.Steer
import Std.Data.HashMap

open Lean

namespace Corpus.Teacher

open Corpus
open Corpus.BellmanOpt

/-- Decidability of bounded universal quantification over a list, even when the
element type is infinite.  This is the form used by the certificate checker. -/
local instance listForallDecidable {α : Type} (l : List α) (p : α → Prop)
    [DecidablePred p] : Decidable (∀ x ∈ l, p x) :=
  decidable_of_iff (l.all p) (by simp [List.all_eq_true])

structure TState where
  station : Nat
  previous : Int
  current : Int
  b : Int
deriving Repr, DecidableEq, FromJson, ToJson

abbrev Act := Bellman.Action TState

def gapAt (q : Steer.Q) (i : Nat) : Int :=
  q.base.s.getD (i + 1) 0 - q.base.s.getD i 0

/-- Integer square-root certificate: `r = floor(sqrt b)` without invoking an
opaque square-root primitive. -/
def rootOK (b r : Int) : Prop :=
  0 ≤ r ∧ r * r ≤ b ∧ b < (r + 1) * (r + 1)

instance rootOKDecidable (b r : Int) : Decidable (rootOK b r) := by
  unfold rootOK
  infer_instance

def rootOf : List (List Int) → Int → Int
  | [], _ => 0
  | p :: ps, b => if p.getD 0 0 = b then p.getD 1 0 else rootOf ps b

def rootsOK (roots : List (List Int)) (bs : List Int) : Prop :=
  ∀ b ∈ bs, rootOK b (rootOf roots b)

instance rootsOKDecidable (roots : List (List Int)) (bs : List Int) :
    Decidable (rootsOK roots bs) := by
  unfold rootsOK
  infer_instance

/-- Declared traversal-time charge (microseconds) of one segment. -/
def timeCost (d rb rw : Int) : Int :=
  Cert.cdiv (2 * d * 1000000) (rb + rw)

def terminalCap (c : Steer.Consts) : Int :=
  min (c.lon.v_max * c.lon.v_max) (c.lon.a_lat * c.lon.r_h)

def pathCurvT (c : Steer.Consts) (q : Steer.Q) (i : Nat) (p x n : Int) : Int :=
  Cert.iabs (q.base.k.getD i 0) + Joint.latCurv c.step p x n

def pathCapT (c : Steer.Consts) (curv : Int) : Int :=
  let vmax2 := c.lon.v_max * c.lon.v_max
  if curv ≤ 0 then vmax2 else min vmax2 (c.lon.a_lat * 1000000000 / curv)

def halfWidth (c : Steer.Consts) : Int :=
  c.lon.veh_width / 2

def obstacleOKT (c : Steer.Consts) (q : Steer.Q) (i : Nat) (x : Int) : Bool :=
  q.base.obstacles.all fun ob =>
    match Cert.obsTriple ob with
    | some (d, lmin, lmax) =>
        if d - c.lon.hold ≤ q.base.s.getD i 0 ∧ q.base.s.getD i 0 ≤ d + c.lon.hold then
          decide (x ≤ lmin - (halfWidth c + c.margin) ∨ lmax + (halfWidth c + c.margin) ≤ x)
        else true
    | none => true

def lateralOKT (c : Steer.Consts) (q : Steer.Q) (offsets : List Int) (i : Nat)
    (p x : Int) : Prop :=
  p ∈ offsets ∧ x ∈ offsets ∧ Steer.lo c q i ≤ x ∧ x ≤ Steer.hi c q i ∧
  obstacleOKT c q i x = true

instance lateralOKTDecidable (c : Steer.Consts) (q : Steer.Q) (offsets : List Int)
    (i : Nat) (p x : Int) : Decidable (lateralOKT c q offsets i p x) := by
  unfold lateralOKT
  infer_instance

def stepOKT (c : Steer.Consts) (q : Steer.Q) (offsets speeds : List Int)
    (s t : TState) : Prop :=
  t.station = s.station + 1 ∧ t.previous = s.current ∧
  lateralOKT c q offsets t.station t.previous t.current ∧
  pathCurvT c q s.station s.previous s.current t.current ≤ c.kappa_max ∧
  0 ≤ s.b ∧ s.b ≤ pathCapT c (pathCurvT c q s.station s.previous s.current t.current) ∧
  t.b ∈ speeds ∧ 0 ≤ t.b ∧ (0 < s.b ∨ 0 < t.b) ∧
  t.b ≤ s.b + 2 * c.lon.a_acc * gapAt q s.station ∧
  s.b ≤ t.b + 2 * c.lon.a_brk * gapAt q s.station

instance stepOKTDecidable (c : Steer.Consts) (q : Steer.Q) (offsets speeds : List Int)
    (s t : TState) : Decidable (stepOKT c q offsets speeds s t) := by
  unfold stepOKT
  infer_instance

/-- Offered actions: every offset and every speed grid value, filtered by the
full step relation.  The list order is the deterministic tie-break order. -/
def actionsT (c : Steer.Consts) (q : Steer.Q) (offsets speeds : List Int)
    (roots : List (List Int)) (s : TState) : List Act :=
  if s.station + 1 < q.base.s.length then
    offsets.flatMap fun n =>
      speeds.filterMap fun w =>
        let t : TState := { station := s.station + 1, previous := s.current, current := n, b := w }
        if stepOKT c q offsets speeds s t then
          some (t, timeCost (gapAt q s.station) (rootOf roots s.b) (rootOf roots w))
        else none
  else []

def terminalT (cap : Int) : TState → Option Int :=
  fun s => if s.b ≤ cap then some 0 else none

def PlanOKT (c : Steer.Consts) (q : Steer.Q) (offsets speeds : List Int) (H : Nat)
    (cap : Int) : TState → List TState → Prop
  | s, [] => s.station = H ∧ terminalT cap s = some 0
  | s, t :: ts => stepOKT c q offsets speeds s t ∧ s.station + 1 < q.base.s.length ∧
      PlanOKT c q offsets speeds H cap t ts

def planCostT (c : Steer.Consts) (q : Steer.Q) (roots : List (List Int)) :
    TState → List TState → Int
  | _, [] => 0
  | s, t :: ts =>
      timeCost (gapAt q s.station) (rootOf roots s.b) (rootOf roots t.b) +
      planCostT c q roots t ts

/-! ## Certificate representation -/

structure Cert where
  consts : Steer.Consts
  q : Steer.Q
  offsets : List Int
  speeds : List Int
  roots : List (List Int)
  horizon : Nat
  start : TState
  terminalCap : Int
  states : List (List TState)
  values : List (List (Option Int))
  plan : List TState
  claimed : Option Int
  planCost : Option Int
deriving Repr, FromJson, ToJson

def stagesT (cert : Cert) : List (TState → List Act) :=
  List.replicate cert.horizon (actionsT cert.consts cert.q cert.offsets cert.speeds cert.roots)

def lookupVal (xs : List TState) (vs : List (Option Int)) (s : TState) : Option Int :=
  match xs, vs with
  | [], _ => none
  | _, [] => none
  | x :: xs, v :: vs => if x = s then v else lookupVal xs vs s

def statesAt (cert : Cert) (i : Nat) : List TState := cert.states.getD i []
def valuesAt (cert : Cert) (i : Nat) : List (Option Int) := cert.values.getD i []

def valAt (cert : Cert) (i : Nat) (s : TState) : Option Int :=
  lookupVal (statesAt cert i) (valuesAt cert i) s

def hasRow (cert : Cert) (i : Nat) (s : TState) : Prop :=
  ∃ v, lookupVal (statesAt cert i) (valuesAt cert i) s = some v

def rowBellmanOK (cert : Cert) (i : Nat) (s : TState) : Prop :=
  valAt cert i s =
    bestO (fun a => (valAt cert (i + 1) a.1).map (fun v => a.2 + v))
      (actionsT cert.consts cert.q cert.offsets cert.speeds cert.roots s)

instance rowBellmanOKDecidable (cert : Cert) (i : Nat) (s : TState) :
    Decidable (rowBellmanOK cert i s) := by
  unfold rowBellmanOK
  infer_instance

def closedUnder (cert : Cert) (i : Nat) : Prop :=
  ∀ s ∈ statesAt cert i, ∀ a ∈ actionsT cert.consts cert.q cert.offsets cert.speeds cert.roots s,
    a.1 ∈ statesAt cert (i + 1)

instance closedUnderDecidable (cert : Cert) (i : Nat) : Decidable (closedUnder cert i) := by
  unfold closedUnder
  infer_instance

def planFromOK (cert : Cert) : TState → List TState → Prop :=
  PlanOKT cert.consts cert.q cert.offsets cert.speeds cert.horizon cert.terminalCap

instance planOKTDecidable (c : Steer.Consts) (q : Steer.Q) (offsets speeds : List Int)
    (H : Nat) (cap : Int) : ∀ (s : TState) (ts : List TState),
    Decidable (PlanOKT c q offsets speeds H cap s ts)
  | s, [] => inferInstanceAs (Decidable (s.station = H ∧ terminalT cap s = some 0))
  | s, t :: ts => by
      letI : Decidable (PlanOKT c q offsets speeds H cap t ts) :=
        planOKTDecidable c q offsets speeds H cap t ts
      unfold PlanOKT
      infer_instance

instance planFromOKDecidable (cert : Cert) (s : TState) (ts : List TState) :
    Decidable (planFromOK cert s ts) :=
  planOKTDecidable cert.consts cert.q cert.offsets cert.speeds cert.horizon cert.terminalCap s ts

def firstOptimal (cert : Cert) (i : Nat) (s : TState) : Option TState :=
  let target := valAt cert i s
  ((actionsT cert.consts cert.q cert.offsets cert.speeds cert.roots s).find? fun a =>
    (valAt cert (i + 1) a.1).map (fun v => a.2 + v) = target).map (fun a => a.1)

def planTieOK (cert : Cert) : TState → List TState → Prop
  | _, [] => True
  | s, t :: ts => firstOptimal cert s.station s = some t ∧ planTieOK cert t ts

instance planTieOKDecidable (cert : Cert) : ∀ (s : TState) (ts : List TState),
    Decidable (planTieOK cert s ts)
  | s, [] => inferInstanceAs (Decidable True)
  | s, t :: ts => by
      letI : Decidable (planTieOK cert t ts) := planTieOKDecidable cert t ts
      unfold planTieOK
      infer_instance

def claimOK (cert : Cert) : Prop :=
  match cert.claimed, cert.plan with
  | none, [] => True
  | none, _ => False
  | some _, [] => False
  | some v, s :: tail =>
      s = cert.start ∧ cert.plan.length = cert.horizon + 1 ∧
      planFromOK cert s tail ∧
      planCostT cert.consts cert.q cert.roots s tail = v ∧
      planTieOK cert s tail ∧ cert.planCost = some v

instance claimOKDecidable (cert : Cert) : Decidable (claimOK cert) := by
  unfold claimOK
  split <;> infer_instance

/-- The full decidable certificate-checking predicate. -/
def CheckCert (cert : Cert) : Prop :=
  -- domain and grid shape
  Steer.wellFormed cert.consts cert.q = true ∧
  0 ≤ cert.terminalCap ∧
  cert.horizon + 1 ≤ cert.q.base.s.length ∧
  1 ≤ cert.horizon ∧
  cert.start.station = 0 ∧
  cert.start.previous ∈ cert.offsets ∧ cert.start.current ∈ cert.offsets ∧
  0 ≤ cert.start.b ∧
  cert.offsets.Nodup ∧ cert.speeds.Nodup ∧
  (∀ b ∈ cert.speeds, 0 ≤ b) ∧
  rootsOK cert.roots (cert.start.b :: cert.speeds) ∧
  -- table shape and universe
  cert.states.length = cert.horizon + 1 ∧
  cert.values.length = cert.horizon + 1 ∧
  (∀ i ∈ List.range (cert.horizon + 1), (statesAt cert i).length = (valuesAt cert i).length) ∧
  (∀ i ∈ List.range (cert.horizon + 1), (statesAt cert i).Nodup) ∧
  statesAt cert 0 = [cert.start] ∧
  (∀ i ∈ List.range (cert.horizon + 1), ∀ s ∈ statesAt cert i,
      s.station = i ∧ s.previous ∈ cert.offsets ∧ s.current ∈ cert.offsets ∧
      (s.b ∈ cert.start.b :: cert.speeds)) ∧
  (∀ i ∈ List.range cert.horizon, closedUnder cert i) ∧
  -- terminal rows and Bellman recurrence
  (∀ s ∈ statesAt cert cert.horizon,
      valAt cert cert.horizon s = terminalT cert.terminalCap s) ∧
  (∀ i ∈ List.range cert.horizon, ∀ s ∈ statesAt cert i, rowBellmanOK cert i s) ∧
  -- claimed start value
  valAt cert 0 cert.start = cert.claimed ∧
  -- exported plan
  claimOK cert


/-! ## Successor-set completeness -/

theorem mem_actionsT {c : Steer.Consts} {q : Steer.Q} {offsets speeds : List Int}
    {roots : List (List Int)} {s : TState} {a : Act}
    (h : a ∈ actionsT c q offsets speeds roots s) :
    stepOKT c q offsets speeds s a.1 ∧
    a.2 = timeCost (gapAt q s.station) (rootOf roots s.b) (rootOf roots a.1.b) := by
  unfold actionsT at h
  split at h
  · rename_i hlen
    simp only [List.mem_flatMap, List.mem_filterMap] at h
    obtain ⟨n, _, w, _, hw⟩ := h
    split at hw
    · rename_i hok
      injection hw with hw
      subst hw
      exact ⟨hok, rfl⟩
    · cases hw
  · simp at h

theorem actionsT_complete {c : Steer.Consts} {q : Steer.Q} {offsets speeds : List Int}
    {roots : List (List Int)} {s t : TState}
    (hok : stepOKT c q offsets speeds s t) (hlen : s.station + 1 < q.base.s.length) :
    (t, timeCost (gapAt q s.station) (rootOf roots s.b) (rootOf roots t.b)) ∈
      actionsT c q offsets speeds roots s := by
  obtain ⟨hst, hprev, hlat, hcurv, hb0, hcap, hb, hb0', hpos, hacc, hbrk⟩ := hok
  obtain ⟨hpo, hco, hlo, hhi, hob⟩ := hlat
  cases t with
  | mk ts tp tc tb =>
    simp only at hst hprev hb hb0' hacc hbrk hcurv hcap hpos hlo hhi hob hco hpo
    subst hst hprev
    unfold actionsT
    rw [if_pos hlen]
    apply List.mem_flatMap.mpr
    refine ⟨tc, hco, ?_⟩
    apply List.mem_filterMap.mpr
    refine ⟨tb, hb, ?_⟩
    have hstep : stepOKT c q offsets speeds s
        { station := s.station + 1, previous := s.current, current := tc, b := tb } :=
      ⟨rfl, rfl, ⟨hpo, hco, hlo, hhi, hob⟩, hcurv, hb0, hcap, hb, hb0', hpos, hacc, hbrk⟩
    rw [if_pos hstep]
    rfl

/-! ## Every declarative plan is an option-valued trajectory -/

theorem plan_trajT (c : Steer.Consts) (q : Steer.Q) (offsets speeds : List Int)
    (roots : List (List Int)) (H : Nat) (cap : Int) :
    ∀ (ts : List TState) (s : TState), PlanOKT c q offsets speeds H cap s ts →
      ∃ xs, xs.map Prod.fst = ts ∧
        TrajOK (terminalT cap) (List.replicate ts.length (actionsT c q offsets speeds roots)) s xs ∧
        trajCost (terminalT cap) (List.replicate ts.length (actionsT c q offsets speeds roots)) s xs =
          some (planCostT c q roots s ts) := by
  intro ts
  induction ts with
  | nil =>
      intro s h
      obtain ⟨_, ht⟩ := h
      refine ⟨[], rfl, ⟨0, ht⟩, ?_⟩
      show terminalT cap s = some (planCostT c q roots s [])
      simpa [terminalT, planCostT] using ht
  | cons t ts ih =>
      intro s h
      obtain ⟨hstep, hlen, hp⟩ := h
      obtain ⟨ys, hys, hok, hc⟩ := ih t hp
      refine ⟨((t, timeCost (gapAt q s.station) (rootOf roots s.b) (rootOf roots t.b)) : Act) :: ys,
        ?_, ⟨actionsT_complete hstep hlen, hok⟩, ?_⟩
      · show t :: ys.map Prod.fst = t :: ts
        rw [hys]
      · show (trajCost (terminalT cap)
            (List.replicate ts.length (actionsT c q offsets speeds roots)) t ys).map
              (fun v => timeCost (gapAt q s.station) (rootOf roots s.b) (rootOf roots t.b) + v) =
          some (timeCost (gapAt q s.station) (rootOf roots s.b) (rootOf roots t.b) +
            planCostT c q roots t ts)
        rw [hc]
        rfl


/-! ## Sparse Bellman tables -/

theorem bestO_congr_on {f g : Act → Option Int} :
    ∀ (l : List Act), (∀ a ∈ l, f a = g a) → bestO f l = bestO g l := by
  intro l
  induction l with
  | nil => intro _; rfl
  | cons a as ih =>
      intro h
      simp only [bestO]
      rw [h a (by simp), ih (fun b hb => h b (by simp [hb]))]

theorem sparse_value
    (terminal : TState → Option Int) (actions : TState → List Act)
    (states : Nat → List TState) (val : Nat → TState → Option Int)
    (H : Nat)
    (hterm : ∀ s ∈ states H, val H s = terminal s)
    (hrec : ∀ i < H, ∀ s ∈ states i,
      val i s = bestO (fun a => (val (i + 1) a.1).map (fun v => a.2 + v)) (actions s))
    (hclosed : ∀ i < H, ∀ s ∈ states i, ∀ a ∈ actions s, a.1 ∈ states (i + 1)) :
    ∀ (k i : Nat), H - i = k → i ≤ H → ∀ s ∈ states i,
      val i s = valueO terminal (List.replicate (H - i) actions) s := by
  intro k
  induction k with
  | zero =>
      intro i hk hi s hs
      have hieq : i = H := by omega
      subst hieq
      simpa [valueO] using hterm s hs
  | succ k ih =>
      intro i hk hi s hs
      have hiH : i < H := by omega
      have hk' : H - (i + 1) = k := by omega
      have hik : H - i = k + 1 := by omega
      rw [hik, List.replicate_succ]
      rw [valueO]
      rw [hrec i hiH s hs]
      apply bestO_congr_on
      intro a ha
      rw [ih (i + 1) hk' (by omega) a.1 (hclosed i hiH s hs a ha)]
      rw [hk']


/-! ## Checker boolean and its soundness -/

instance checkCertDecidable (cert : Cert) : Decidable (CheckCert cert) := by
  unfold CheckCert
  infer_instance

def checkCert (cert : Cert) : Bool := decide (CheckCert cert)

theorem checkCert_sound (cert : Cert) (h : checkCert cert = true) : CheckCert cert :=
  of_decide_eq_true h

def lastStateT : TState → List TState → TState
  | s, [] => s
  | _, t :: ts => lastStateT t ts

theorem plan_length (c : Steer.Consts) (q : Steer.Q) (offsets speeds : List Int)
    (H : Nat) (cap : Int) :
    ∀ (ts : List TState) (s : TState), PlanOKT c q offsets speeds H cap s ts →
      (lastStateT s ts).station = s.station + ts.length := by
  intro ts
  induction ts with
  | nil => intro s h; simp [lastStateT]
  | cons t ts ih =>
      intro s h
      obtain ⟨hstep, _, hp⟩ := h
      have hst : t.station = s.station + 1 := hstep.1
      have := ih t hp
      simp only [lastStateT, List.length_cons]
      rw [this, hst]
      omega

theorem plan_terminal_station (c : Steer.Consts) (q : Steer.Q) (offsets speeds : List Int)
    (H : Nat) (cap : Int) :
    ∀ (ts : List TState) (s : TState), PlanOKT c q offsets speeds H cap s ts →
      (lastStateT s ts).station = H := by
  intro ts
  induction ts with
  | nil => intro s h; exact h.1
  | cons t ts ih => intro s h; exact ih t h.2.2

theorem table_value_cert (cert : Cert) (h : CheckCert cert)
    (i : Nat) (s : TState) (hs : s ∈ statesAt cert i) :
    valAt cert i s =
      valueO (terminalT cert.terminalCap)
        (List.replicate (cert.horizon - i)
          (actionsT cert.consts cert.q cert.offsets cert.speeds cert.roots)) s := by
  rcases h with ⟨hdom, hcap, hlen, hge, hst0, hstp, hstc, hb0, hoff, hsp, hspn, hroots,
    hsl, hvl, hlen2, hnodup, huniv0, hstate, hclosed, hterm, hrec, hval, hplan⟩
  have hi : i ≤ cert.horizon := by
    by_cases hle : i ≤ cert.horizon
    · exact hle
    · have hnil : statesAt cert i = [] := by
        unfold statesAt
        have hstates_le : cert.states.length ≤ i := by
          rw [hsl]
          omega
        rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none hstates_le]
        rfl
      rw [hnil] at hs
      simp at hs
  exact sparse_value (terminalT cert.terminalCap)
    (actionsT cert.consts cert.q cert.offsets cert.speeds cert.roots)
    (statesAt cert) (valAt cert) cert.horizon
    (fun s hs => hterm s hs)
    (fun i hi s hs => hrec i (List.mem_range.mpr hi) s hs)
    (fun i hi s hs a ha => hclosed i (List.mem_range.mpr hi) s hs a ha)
    (cert.horizon - i) i rfl hi s hs

/-- The checker's claimed value is the true Bellman optimum of the encoded
finite model. -/
theorem certificate_value (cert : Cert) (h : CheckCert cert) :
    valueO (terminalT cert.terminalCap) (stagesT cert) cert.start = cert.claimed := by
  rcases h with ⟨hdom, hcap, hlen, hge, hst0, hstp, hstc, hb0, hoff, hsp, hspn, hroots,
    hsl, hvl, hlen2, hnodup, huniv0, hstate, hclosed, hterm, hrec, hval, hplan⟩
  have hstart : cert.start ∈ statesAt cert 0 := by
    rw [huniv0]; simp
  have ht := table_value_cert cert ⟨hdom, hcap, hlen, hge, hst0, hstp, hstc, hb0, hoff, hsp,
    hspn, hroots, hsl, hvl, hlen2, hnodup, huniv0, hstate, hclosed, hterm, hrec, hval, hplan⟩
    0 cert.start hstart
  rw [Nat.sub_zero] at ht
  rw [hval] at ht
  exact ht.symm

/-- Every declarative plan costs at least the claimed value. -/
theorem certificate_lower_bound (cert : Cert) (h : CheckCert cert)
    (ts : List TState) (hp : PlanOKT cert.consts cert.q cert.offsets cert.speeds
      cert.horizon cert.terminalCap cert.start ts) :
    ∀ v, cert.claimed = some v → v ≤ planCostT cert.consts cert.q cert.roots cert.start ts := by
  intro v hv
  rcases h with ⟨hdom, hcap, hlen, hge, hst0, hstp, hstc, hb0, hoff, hsp, hspn, hroots,
    hsl, hvl, hlen2, hnodup, huniv0, hstate, hclosed, hterm, hrec, hval, hplan⟩
  have hplen := plan_length cert.consts cert.q cert.offsets cert.speeds cert.horizon cert.terminalCap ts cert.start hp
  have hstart0 : cert.start.station = 0 := hst0
  have hlen_ts : ts.length = cert.horizon := by
    have htermstation : (lastStateT cert.start ts).station = cert.horizon :=
      plan_terminal_station cert.consts cert.q cert.offsets cert.speeds cert.horizon cert.terminalCap ts cert.start hp
    rw [hplen] at htermstation
    omega
  obtain ⟨xs, hxs, hok, hc⟩ := plan_trajT cert.consts cert.q cert.offsets cert.speeds cert.roots
    cert.horizon cert.terminalCap ts cert.start hp
  rw [hlen_ts] at hok hc
  have hv' : valueO (terminalT cert.terminalCap) (stagesT cert) cert.start = some v := by
    rw [← hv]
    exact certificate_value cert ⟨hdom, hcap, hlen, hge, hst0, hstp, hstc, hb0, hoff, hsp,
      hspn, hroots, hsl, hvl, hlen2, hnodup, huniv0, hstate, hclosed, hterm, hrec, hval, hplan⟩
  have hbound := BellmanOpt.valueO_le_cost (terminalT cert.terminalCap)
    (stagesT cert) cert.start xs v (planCostT cert.consts cert.q cert.roots cert.start ts)
    (by simpa [stagesT] using hok) (by simpa [stagesT] using hc) hv'
  exact hbound

/-- The exported plan is admissible, reaches the horizon, and attains the
claimed value. -/
theorem certificate_plan (cert : Cert) (h : CheckCert cert) :
    match cert.claimed with
    | none => True
    | some v =>
        ∃ tail, cert.plan = cert.start :: tail ∧
          PlanOKT cert.consts cert.q cert.offsets cert.speeds cert.horizon cert.terminalCap cert.start tail ∧
          planCostT cert.consts cert.q cert.roots cert.start tail = v ∧
          cert.plan.length = cert.horizon + 1
    := by
  rcases h with ⟨hdom, hcap, hlen, hge, hst0, hstp, hstc, hb0, hoff, hsp, hspn, hroots,
    hsl, hvl, hlen2, hnodup, huniv0, hstate, hclosed, hterm, hrec, hval, hplan⟩
  cases hc : cert.claimed with
  | none => trivial
  | some v =>
      unfold claimOK at hplan
      rw [hc] at hplan
      cases hp : cert.plan with
      | nil => simp [hp] at hplan
      | cons s tail =>
          simp only [hp] at hplan
          rcases hplan with ⟨hs, hplen, hok, hcost, htie, hpcost⟩
          subst hs
          exact ⟨tail, rfl, hok, hcost, hplen⟩

/-- **Infeasibility is exact**: a claimed `none` rules out every declarative
plan of the horizon model. -/
theorem certificate_infeasible (cert : Cert) (h : CheckCert cert)
    (hclaim : cert.claimed = none) :
    ¬ ∃ ts, PlanOKT cert.consts cert.q cert.offsets cert.speeds cert.horizon cert.terminalCap cert.start ts := by
  intro ⟨ts, hp⟩
  have hv := certificate_value cert h
  rw [hclaim] at hv
  obtain ⟨xs, _, hok, _⟩ := plan_trajT cert.consts cert.q cert.offsets cert.speeds cert.roots
    cert.horizon cert.terminalCap ts cert.start hp
  have hlen_ts := plan_length cert.consts cert.q cert.offsets cert.speeds cert.horizon cert.terminalCap ts cert.start hp
  have htermstation : (lastStateT cert.start ts).station = cert.horizon :=
    plan_terminal_station cert.consts cert.q cert.offsets cert.speeds cert.horizon cert.terminalCap ts cert.start hp
  have hstart0 : cert.start.station = 0 := by
    rcases h with ⟨_, _, _, _, hst0, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _⟩
    exact hst0
  have hlen_eq : ts.length = cert.horizon := by
    rw [hlen_ts] at htermstation; omega
  rw [hlen_eq] at hok
  exact (BellmanOpt.valueO_none_iff (terminalT cert.terminalCap) (stagesT cert) cert.start).mp
    (by simpa [stagesT] using hv) ⟨xs, by simpa [stagesT] using hok⟩

def allStepsOKT (c : Steer.Consts) (q : Steer.Q) (offsets speeds : List Int) :
    TState → List TState → Prop
  | _, [] => True
  | s, t :: ts =>
      stepOKT c q offsets speeds s t ∧ allStepsOKT c q offsets speeds t ts

structure TeacherGuarantee (cert : Cert) : Prop where
  value :
    valueO (terminalT cert.terminalCap) (stagesT cert) cert.start = cert.claimed
  claimed_some :
    ∀ v, cert.claimed = some v →
      valueO (terminalT cert.terminalCap) (stagesT cert) cert.start = some v ∧
      (∃ tail, cert.plan = cert.start :: tail ∧
        PlanOKT cert.consts cert.q cert.offsets cert.speeds cert.horizon cert.terminalCap cert.start tail ∧
        planCostT cert.consts cert.q cert.roots cert.start tail = v ∧
        allStepsOKT cert.consts cert.q cert.offsets cert.speeds cert.start tail ∧
        (∀ ts, PlanOKT cert.consts cert.q cert.offsets cert.speeds cert.horizon cert.terminalCap cert.start ts →
          v ≤ planCostT cert.consts cert.q cert.roots cert.start ts))
  claimed_none :
    cert.claimed = none →
      valueO (terminalT cert.terminalCap) (stagesT cert) cert.start = none ∧
      ¬ ∃ ts, PlanOKT cert.consts cert.q cert.offsets cert.speeds cert.horizon cert.terminalCap cert.start ts

theorem certificate_optimal (cert : Cert) (h : CheckCert cert) :
    TeacherGuarantee cert := by
  have hvalue := certificate_value cert h
  have hplan := certificate_plan cert h
  have hnone : cert.claimed = none →
      ¬ ∃ ts, PlanOKT cert.consts cert.q cert.offsets cert.speeds cert.horizon cert.terminalCap cert.start ts :=
    fun hc => certificate_infeasible cert h hc
  have hsteps :
      ∀ (s : TState) (ts : List TState),
        PlanOKT cert.consts cert.q cert.offsets cert.speeds cert.horizon cert.terminalCap s ts →
          allStepsOKT cert.consts cert.q cert.offsets cert.speeds s ts := by
    intro s ts hp
    induction ts generalizing s with
    | nil => trivial
    | cons t ts ih =>
        exact ⟨hp.1, ih t hp.2.2⟩
  refine
    { value := hvalue
      claimed_some := ?_
      claimed_none := ?_ }
  · intro v hc
    have hv : valueO (terminalT cert.terminalCap) (stagesT cert) cert.start = some v := by
      simpa [hc] using hvalue
    obtain ⟨tail, htail, hp, hcost, _hlen⟩ := by
      simpa [hc] using hplan
    refine ⟨hv, ?_⟩
    refine ⟨tail, htail, hp, hcost, hsteps cert.start tail hp, ?_⟩
    intro ts hts
    exact certificate_lower_bound cert h ts hts v hc
  · intro hc
    exact ⟨by simpa [hc] using hvalue, hnone hc⟩

theorem timeCost_eq_isqrt {d b w rb rw : Int}
    (hb : rootOK b rb) (hw : rootOK w rw) :
    timeCost d rb rw =
      Cert.cdiv (2 * d * 1000000)
        (Int.ofNat (Nat.sqrt b.toNat) + Int.ofNat (Nat.sqrt w.toNat)) := by
  have root_unique : ∀ (n r : Nat), r * r ≤ n → n < (r + 1) * (r + 1) →
      r = Nat.sqrt n := by
    intro n r h1 h2
    have hs1 := Nat.sqrt_le n
    have hs2 := Nat.lt_succ_sqrt n
    have hle : r ≤ Nat.sqrt n := by
      by_cases hr : r ≤ Nat.sqrt n
      · exact hr
      · have hr' : Nat.sqrt n + 1 ≤ r := by omega
        have hh : (Nat.sqrt n + 1) * (Nat.sqrt n + 1) ≤ r * r :=
          Nat.mul_le_mul hr' hr'
        have hh2 : (Nat.sqrt n + 1) * (Nat.sqrt n + 1) ≤ n :=
          Nat.le_trans hh h1
        exact False.elim ((Nat.not_le_of_lt
          (by simpa [Nat.succ_eq_add_one] using hs2)) hh2)
    have hge : Nat.sqrt n ≤ r := by
      by_cases hr : Nat.sqrt n ≤ r
      · exact hr
      · have hr' : r + 1 ≤ Nat.sqrt n := by omega
        have hh : (r + 1) * (r + 1) ≤ Nat.sqrt n * Nat.sqrt n :=
          Nat.mul_le_mul hr' hr'
        have hh2 : (r + 1) * (r + 1) ≤ n :=
          Nat.le_trans hh hs1
        exact False.elim ((Nat.not_le_of_lt h2) hh2)
    omega
  have root_int : ∀ (x r : Int), rootOK x r →
      r = Int.ofNat (Nat.sqrt x.toNat) := by
    intro x r h
    rcases h with ⟨hr, hsq, hlt⟩
    have hx : 0 ≤ x := by
      exact Int.le_trans (Int.mul_nonneg hr hr) hsq
    have hsqNat := Int.toNat_le_toNat hsq
    rw [Int.toNat_mul (by omega) (by omega)] at hsqNat
    have hrnat : r.toNat * r.toNat ≤ x.toNat := hsqNat
    have hden : 0 < (r + 1) * (r + 1) := Int.mul_pos (by omega) (by omega)
    have hltNat := (Int.toNat_lt_toNat hden).2 hlt
    rw [Int.toNat_mul (by omega) (by omega)] at hltNat
    have hplus : (r + 1).toNat = r.toNat + 1 := by
      simpa using (Int.toNat_add (a := r) (b := 1) (by omega) (by omega))
    rw [hplus] at hltNat
    have hlnat : x.toNat < (r.toNat + 1) * (r.toNat + 1) := hltNat
    have heq := root_unique x.toNat r.toNat hrnat hlnat
    rw [← Int.toNat_of_nonneg hr]
    exact congrArg Int.ofNat heq
  rw [timeCost, root_int b rb hb, root_int w rw hw]

end Corpus.Teacher
