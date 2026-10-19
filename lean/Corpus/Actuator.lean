/-
  Executable, proof-carrying checker for the finite actuator model.

  This module is the Lean counterpart of ``autocross/actuator/model.py`` and
  ``autocross/actuator/certificate.py``.  ``checkCert`` is a decidable checker
  for the sparse certificate emitted there; ``checkCert_sound`` turns a passing
  check into a kernel-checked statement about the encoded finite model:

    * the Bellman value from the declared start equals the claimed value;
    * every declarative action plan costs at least the claimed value;
    * an exported plan is admissible, reaches the horizon, and attains exactly
      the claimed value;
    * a claimed ``none`` rules out every declarative plan.

  The checker never trusts the solver.  It re-encodes the instance, recomputes
  the transition, every admissible (steer, accel) pair, the terminal rows, the
  closure of the declared state universe, and the exported plan.  Integer
  division here is Lean's ``Int`` Euclidean division, which is floor division
  for positive divisors; every divisor in the model is positive on accepted
  certificates.
-/
import Lean.Data.Json
import Corpus.Cert
import Corpus.BellmanOpt

open Lean

namespace Corpus.Actuator

open Corpus
open Corpus.BellmanOpt

/-- Decidability of bounded universal quantification over a list, even when the
element type is infinite. -/
local instance listForallDecidable {α : Type} (l : List α) (p : α → Prop)
    [DecidablePred p] : Decidable (∀ x ∈ l, p x) :=
  decidable_of_iff (l.all p) (by simp [List.all_eq_true])

/-! ## State and encoded instance -/

structure AState where
  station : Nat
  d : Int
  psi : Int
  b : Int
  delta : Int
deriving Repr, DecidableEq, FromJson, ToJson

structure ActLon where
  a_lat : Int
  a_acc : Int
  a_brk : Int
  v_max : Int
  r_h : Int
  hold : Int
  veh_width : Int
deriving Repr, DecidableEq, FromJson, ToJson

structure ActConsts where
  lon : ActLon
  margin : Int
  step : Int
  max_steer : Int
  steer_rate_urad_s : Int
  steer_tau_us : Int
  wheelbase_mm : Int
deriving Repr, DecidableEq, FromJson, ToJson

structure ActBase where
  s : List Int
  k : List Int
  wl : List Int
  wr : List Int
  obstacles : List (List Int)
deriving Repr, DecidableEq, FromJson, ToJson

/-- The instance's `q` object: the quantised observation plus `lat`. -/
structure ActQ where
  base : ActBase
  lat : Int
deriving Repr, DecidableEq, FromJson, ToJson

/-- The declared finite model, re-encoded from the certificate instance. -/
structure ActModel where
  consts : ActConsts
  q : ActQ
  offsets : List Int
  psis : List Int
  speeds : List Int
  deltas : List Int
  steer_inputs : List Int
  acc_inputs : List Int
  horizon : Nat
  start : AState
  terminal_distance : Int
  understeer_num : Int
  understeer_den : Int
  terminal_min_b : Int
  terminal_max_psi : Int
  terminal_max_delta : Int
  ref_d : List Int
  ref_psi : List Int
deriving Repr, ToJson

/-- Optional JSON list: absent means the zero reference (as in Python). -/
def getListD (j : Json) (key : String) : Except String (List Int) :=
  match j.getObjVal? key with
  | .ok v => fromJson? v
  | .error _ => .ok []

instance : FromJson ActModel where
  fromJson? j := do
    let consts : ActConsts ← fromJson? (← j.getObjVal? "consts")
    let q : ActQ ← fromJson? (← j.getObjVal? "q")
    let offsets : List Int ← fromJson? (← j.getObjVal? "offsets")
    let psis : List Int ← fromJson? (← j.getObjVal? "psis")
    let speeds : List Int ← fromJson? (← j.getObjVal? "speeds")
    let deltas : List Int ← fromJson? (← j.getObjVal? "deltas")
    let steer_inputs : List Int ← fromJson? (← j.getObjVal? "steer_inputs")
    let acc_inputs : List Int ← fromJson? (← j.getObjVal? "acc_inputs")
    let horizon : Nat ← fromJson? (← j.getObjVal? "horizon")
    let start : AState ← fromJson? (← j.getObjVal? "start")
    let terminal_distance : Int ← fromJson? (← j.getObjVal? "terminal_distance")
    let understeer_num : Int ← fromJson? (← j.getObjVal? "understeer_num")
    let understeer_den : Int ← fromJson? (← j.getObjVal? "understeer_den")
    let terminal_min_b : Int ← fromJson? (← j.getObjVal? "terminal_min_b")
    let terminal_max_psi : Int ← fromJson? (← j.getObjVal? "terminal_max_psi")
    let terminal_max_delta : Int ← fromJson? (← j.getObjVal? "terminal_max_delta")
    let ref_d ← getListD j "ref_d"
    let ref_psi ← getListD j "ref_psi"
    let m : ActModel :=
      { consts := consts, q := q, offsets := offsets, psis := psis, speeds := speeds,
        deltas := deltas, steer_inputs := steer_inputs, acc_inputs := acc_inputs,
        horizon := horizon, start := start, terminal_distance := terminal_distance,
        understeer_num := understeer_num, understeer_den := understeer_den,
        terminal_min_b := terminal_min_b, terminal_max_psi := terminal_max_psi,
        terminal_max_delta := terminal_max_delta, ref_d := ref_d, ref_psi := ref_psi }
    return m

abbrev Act := Corpus.Bellman.Action AState

def V_MIN_MM_S : Int := 1000

def gapAt (q : ActQ) (i : Nat) : Int :=
  q.base.s.getD (i + 1) 0 - q.base.s.getD i 0

/-! ## Integer square roots, declared in the certificate -/

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

/-- Nearest declared grid value (ties to the larger value), then clamp.  This
is the scalar reference's `snap` for strictly increasing grids. -/
def snap (x : Int) : List Int → Int
  | [] => x
  | [v] => v
  | v :: w :: rest =>
      if x ≤ v then v
      else if x ≤ w then (if x - v ≥ w - x then w else v)
      else snap x (w :: rest)

/-! ## Scalar transition, exactly as in `model.py` -/

def halfWidth (c : ActConsts) : Int := c.lon.veh_width / 2

def loA (m : ActModel) (i : Nat) : Int :=
  -(m.q.base.wr.getD i 0 - halfWidth m.consts - m.consts.margin)

def hiA (m : ActModel) (i : Nat) : Int :=
  m.q.base.wl.getD i 0 - halfWidth m.consts - m.consts.margin

def obstacleOKA (m : ActModel) (i : Nat) (x : Int) : Bool :=
  m.q.base.obstacles.all fun ob =>
    match Cert.obsTriple ob with
    | some (dist, lmin, lmax) =>
        if dist - m.consts.lon.hold ≤ m.q.base.s.getD i 0 ∧
            m.q.base.s.getD i 0 ≤ dist + m.consts.lon.hold then
          decide (x ≤ lmin - (halfWidth m.consts + m.consts.margin) ∨
            lmax + (halfWidth m.consts + m.consts.margin) ≤ x)
        else true
    | none => true

/-- Corridor and inflated-obstacle test at one station (the Python
`lateral_ok`; grid membership is checked separately for table rows). -/
def midOKB (m : ActModel) (i : Nat) (x : Int) : Bool :=
  decide (loA m i ≤ x ∧ x ≤ hiA m i) && obstacleOKA m i x

def lateralOKB (m : ActModel) (i : Nat) (x : Int) : Bool :=
  decide (x ∈ m.offsets) && midOKB m i x

def capAtA (m : ActModel) (vmax2 : Int) (j : Nat) : Int :=
  let kj := Cert.iabs (m.q.base.k.getD j 0)
  if kj = 0 then vmax2
  else min vmax2 (m.consts.lon.a_lat * 1000000000 / kj)

/-- Largest terminal speed² that can still brake for every visible station
after the horizon (`model.py::terminal_cap`). -/
def terminalCapA (m : ActModel) : Int :=
  let vmax2 := m.consts.lon.v_max * m.consts.lon.v_max
  let h := min m.horizon (m.q.base.k.length - 1)
  let cap0 := capAtA m vmax2 h
  let cap := (List.range m.q.base.k.length).foldl (fun cap j =>
    if h < j then
      min cap (capAtA m vmax2 j +
        2 * m.consts.lon.a_brk * (m.q.base.s.getD j 0 - m.q.base.s.getD h 0))
    else cap) cap0
  max cap 0

def terminalOKB (m : ActModel) (s : AState) : Bool :=
  decide (s.station = m.horizon) &&
  midOKB m m.horizon s.d &&
  decide (Cert.iabs s.psi ≤ m.terminal_max_psi) &&
  decide (Cert.iabs s.delta ≤ m.terminal_max_delta) &&
  decide (m.terminal_min_b ≤ s.b ∧ s.b ≤ terminalCapA m)

/-- The candidate successor state, computed exactly as Python does.  Only the
domain test decides whether it is offered. -/
def nextStateCore (m : ActModel) (roots : List (List Int)) (s : AState) (u a : Int) : AState :=
  let ds := gapAt m.q s.station
  let v := rootOf roots s.b
  let dt := ds * 1000000 / max v V_MIN_MM_S
  let target := u * m.consts.max_steer / 1000
  let lag := (target - s.delta) * min dt m.consts.steer_tau_us / m.consts.steer_tau_us
  let lim := m.consts.steer_rate_urad_s * dt / 1000000
  let change := min (max lag (-lim)) lim
  let delta := snap (s.delta + change) m.deltas
  let den := m.consts.wheelbase_mm + m.understeer_num * s.b / m.understeer_den
  let kappa := delta * 1000 / den
  let kroad := m.q.base.k.getD s.station 0
  let psi := snap (s.psi + (kappa - kroad) * ds / 1000) m.psis
  let d := snap (s.d + ((s.psi + psi) / 2) * ds / 1000000) m.offsets
  let b := snap (s.b + 2 * a * ds) m.speeds
  ⟨s.station + 1, d, psi, b, delta⟩

/-- All `none` branches of Python `next_state`. -/
def stepDomainOK (m : ActModel) (roots : List (List Int)) (s : AState) (u a : Int) : Prop :=
  s.station < m.horizon ∧
  u ∈ m.steer_inputs ∧ a ∈ m.acc_inputs ∧
  0 < m.consts.wheelbase_mm + m.understeer_num * s.b / m.understeer_den ∧
  0 ≤ s.b + 2 * a * gapAt m.q s.station ∧
  midOKB m (s.station + 1) (nextStateCore m roots s u a).d = true ∧
  midOKB m (s.station + 1) (((nextStateCore m roots s u a).d + s.d) / 2) = true

instance stepDomainOKDecidable (m : ActModel) (roots : List (List Int)) (s : AState)
    (u a : Int) : Decidable (stepDomainOK m roots s u a) := by
  unfold stepDomainOK
  infer_instance

/-- Python `next_state`, including the declared-input and domain guards. -/
def nextStateA (m : ActModel) (roots : List (List Int)) (s : AState) (u a : Int) :
    Option AState :=
  if stepDomainOK m roots s u a then some (nextStateCore m roots s u a) else none

def kappaPathA (m : ActModel) (s t : AState) : Int :=
  m.q.base.k.getD s.station 0 + (t.psi - s.psi) * 1000 / gapAt m.q s.station

def aLongA (m : ActModel) (a : Int) : Int :=
  if 0 ≤ a then m.consts.lon.a_acc else m.consts.lon.a_brk

/-- The three admissibility tests of Python `action_ok`. -/
def actionValidA (m : ActModel) (s t : AState) (a : Int) : Bool :=
  let kp := kappaPathA m s t
  let alat := m.consts.lon.a_lat
  let along := aLongA m a
  let lat := t.b * Cert.iabs kp / 1000000000
  decide (-(m.consts.lon.a_brk) ≤ a ∧ a ≤ m.consts.lon.a_acc) &&
  decide (t.b * Cert.iabs kp ≤ alat * 1000000000) &&
  decide ((lat * along) * (lat * along) + (a * alat) * (a * alat) ≤
    (alat * along) * (alat * along))

def actionOKA (m : ActModel) (roots : List (List Int)) (s : AState) (u a : Int) :
    Option AState :=
  match nextStateA m roots s u a with
  | none => none
  | some t => if actionValidA m s t a then some t else none

def timeCost (d rb rw : Int) : Int :=
  Cert.cdiv (2 * d * 1000000) (rb + rw)

def refAtA (xs : List Int) (i : Nat) : Int :=
  if xs.length > i then xs.getD i 0 else 0

def actionCostA (m : ActModel) (roots : List (List Int)) (s t : AState) : Int :=
  1000 * timeCost (gapAt m.q s.station) (rootOf roots s.b) (rootOf roots t.b) +
  Cert.iabs (t.delta - s.delta) / 1000 +
  5000 * Cert.iabs (t.d - refAtA m.ref_d (s.station + 1)) +
  5000 * Cert.iabs (t.psi - refAtA m.ref_psi (s.station + 1)) / 100

/-- Offered actions in the declared order: steering input outer, acceleration
input inner.  This is the deterministic tie-break order of the solver. -/
def actionsA (m : ActModel) (roots : List (List Int)) (s : AState) : List Act :=
  m.steer_inputs.flatMap fun u =>
    m.acc_inputs.filterMap fun a =>
      match actionOKA m roots s u a with
      | some t => some ((t, actionCostA m roots s t) : Act)
      | none => none

def terminalCostA (m : ActModel) (roots : List (List Int)) (s : AState) : Option Int :=
  if terminalOKB m s then
    if m.terminal_distance ≤ 0 then some 0
    else
      let v := rootOf roots s.b
      if v ≤ 0 then none
      else some (Cert.cdiv (m.terminal_distance * 1000000) v)
  else none

def terminalValueA (m : ActModel) (roots : List (List Int)) (s : AState) : Option Int :=
  (terminalCostA m roots s).map (fun c => 1000 * c)

def stagesA (m : ActModel) (roots : List (List Int)) : List (Stage AState) :=
  List.replicate m.horizon (actionsA m roots)

/-- Split a JSON `[u,a]` list into a pair, retaining only well-formed entries. -/
def pairList (xs : List (List Int)) : List (Int × Int) :=
  xs.filterMap fun p =>
    match p with
    | [u, a] => some (u, a)
    | _ => none

def pairListOK (xs : List (List Int)) : Bool :=
  xs.all fun p => decide (p.length = 2)

def firstPair : List Int → Int × Int
  | [u, a] => (u, a)
  | _ => (0, 0)

structure Cert where
  inst : ActModel
  roots : List (List Int)
  states : List (List AState)
  values : List (List (Option Int))
  plan : List AState
  planActions : List (List Int)
  firstAction : Option (List Int)
  claimed : Option Int
  planCost : Option Int := none
deriving Repr

/-- The Python certificate nests the model under `instance`; `planCost` is an
optional redundant field (the checker recomputes the plan cost). -/
instance : FromJson Cert where
  fromJson? j := do
    let inst : ActModel ← fromJson? (← j.getObjVal? "instance")
    let roots : List (List Int) ← fromJson? (← j.getObjVal? "roots")
    let states : List (List AState) ← fromJson? (← j.getObjVal? "states")
    let values : List (List (Option Int)) ← fromJson? (← j.getObjVal? "values")
    let plan : List AState ← fromJson? (← j.getObjVal? "plan")
    let planActions : List (List Int) ← fromJson? (← j.getObjVal? "planActions")
    let firstAction : Option (List Int) ← fromJson? (← j.getObjVal? "firstAction")
    let claimed : Option Int ← fromJson? (← j.getObjVal? "claimed")
    let planCost : Option Int ←
      match j.getObjVal? "planCost" with
      | .ok v => fromJson? v
      | .error _ => pure none
    let c : Cert :=
      { inst := inst, roots := roots, states := states, values := values, plan := plan,
        planActions := planActions, firstAction := firstAction, claimed := claimed,
        planCost := planCost }
    return c

def lookupVal (xs : List AState) (vs : List (Option Int)) (s : AState) : Option Int :=
  match xs, vs with
  | [], _ => none
  | _, [] => none
  | x :: xs, v :: vs => if x = s then v else lookupVal xs vs s

def statesAt (cert : Cert) (i : Nat) : List AState := cert.states.getD i []
def valuesAt (cert : Cert) (i : Nat) : List (Option Int) := cert.values.getD i []

def valAt (cert : Cert) (i : Nat) (s : AState) : Option Int :=
  lookupVal (statesAt cert i) (valuesAt cert i) s

def hasRow (cert : Cert) (i : Nat) (s : AState) : Prop :=
  ∃ v, valAt cert i s = some v

/-- Check the first action against the plan's second state and the stage-1
value, exactly as the Python checker does. -/
def firstActionOKB (cert : Cert) (s : AState) (v : Int) (tail : List AState) : Bool :=
  match cert.firstAction with
  | none => false
  | some fa =>
      if fa.length = 2 then
        let u := (firstPair fa).1
        let a := (firstPair fa).2
        match actionOKA cert.inst cert.roots s u a with
        | none => false
        | some t =>
            match tail with
            | t1 :: _ =>
                decide (t = t1) &&
                (match valAt cert 1 t with
                 | some vt => decide (actionCostA cert.inst cert.roots s t + vt = v)
                 | none => false)
            | [] => false
      else false

def planCostAgrees (cert : Cert) (v : Int) : Prop :=
  match cert.planCost with
  | some pc => pc = v
  | none => True

instance planCostAgreesDecidable (cert : Cert) (v : Int) : Decidable (planCostAgrees cert v) := by
  unfold planCostAgrees
  split <;> infer_instance

/-! ## Declarative plans -/

/-- Deterministic replay of an action sequence. -/
def runPlanA (m : ActModel) (roots : List (List Int)) :
    AState → List (Int × Int) → Option AState
  | s, [] => some s
  | s, (u, a) :: rest =>
      match actionOKA m roots s u a with
      | some t => runPlanA m roots t rest
      | none => none

/-- Trajectory cost; includes the terminal value. -/
def planCostOpt (m : ActModel) (roots : List (List Int)) :
    AState → List (Int × Int) → Option Int
  | s, [] => terminalValueA m roots s
  | s, (u, a) :: rest =>
      match actionOKA m roots s u a with
      | none => none
      | some t => (planCostOpt m roots t rest).map (fun v => actionCostA m roots s t + v)

def PlanOKA (m : ActModel) (roots : List (List Int)) (s : AState)
    (acts : List (Int × Int)) : Prop :=
  ∃ t v, runPlanA m roots s acts = some t ∧ terminalValueA m roots t = some v

/-- Replay the exported state sequence and check that every action produces the
next declared state. -/
def planStepsOK (cert : Cert) : AState → List AState → List (Int × Int) → Prop
  | _, [], [] => True
  | s, t :: ts, (u, a) :: rest =>
      actionOKA cert.inst cert.roots s u a = some t ∧
      planStepsOK cert t ts rest
  | _, [], _ :: _ => False
  | _, _ :: _, [] => False

instance planStepsOKDecidable (cert : Cert) : ∀ s tail acts,
    Decidable (planStepsOK cert s tail acts)
  | _, [], [] => isTrue trivial
  | _, [], _ :: _ => isFalse (by simp [planStepsOK])
  | _, _ :: _, [] => isFalse (by simp [planStepsOK])
  | s, t :: ts, (u, a) :: rest => by
      haveI : Decidable (planStepsOK cert t ts rest) :=
        planStepsOKDecidable cert t ts rest
      unfold planStepsOK
      infer_instance

/-- Certificate claim: no plan for `none`, and an admissible, correctly-costed
plan for `some v`.  Both `planActions` and `firstAction` are recomputed. -/
def claimOK (cert : Cert) : Prop :=
  pairListOK cert.planActions = true ∧
  match cert.claimed, cert.plan with
  | none, [] => cert.firstAction = none ∧ cert.planActions = []
  | some v, s :: tail =>
      s = cert.inst.start ∧
      cert.plan.length = cert.inst.horizon + 1 ∧
      (pairList cert.planActions).length = cert.inst.horizon ∧
      planStepsOK cert s tail (pairList cert.planActions) ∧
      planCostOpt cert.inst cert.roots s (pairList cert.planActions) = some v ∧
      firstActionOKB cert s v tail = true ∧
      planCostAgrees cert v
  | _, _ => False

instance claimOKDecidable (cert : Cert) : Decidable (claimOK cert) := by
  unfold claimOK
  repeat' split
  all_goals infer_instance

/-! ## Decidable checker -/

def lastStateA : AState → List AState → AState
  | s, [] => s
  | _, t :: ts => lastStateA t ts

def sortedLt : List Int → Bool
  | [] => true
  | [_] => true
  | x :: y :: t => decide (x < y) && sortedLt (y :: t)

abbrev domainOK (m : ActModel) : Prop :=
  m.start.station = 0 ∧
  (∀ i ∈ List.range m.horizon, 0 < gapAt m.q i) ∧
  1 ≤ m.horizon ∧
  m.horizon + 1 ≤ m.q.base.s.length ∧
  m.horizon + 1 ≤ m.q.base.k.length ∧
  m.q.base.k.length ≤ m.q.base.s.length ∧
  m.horizon + 1 ≤ m.q.base.wl.length ∧
  m.horizon + 1 ≤ m.q.base.wr.length ∧
  m.start.d ∈ m.offsets ∧
  m.start.psi ∈ m.psis ∧
  m.start.delta ∈ m.deltas ∧
  0 ≤ m.start.b ∧
  sortedLt m.offsets = true ∧
  sortedLt m.psis = true ∧
  sortedLt m.speeds = true ∧
  sortedLt m.deltas = true ∧
  sortedLt m.steer_inputs = true ∧
  sortedLt m.acc_inputs = true ∧
  (∀ b ∈ m.speeds, 0 < b) ∧
  0 < m.consts.steer_tau_us ∧
  0 < m.understeer_den

abbrev shapeOK (cert : Cert) : Prop :=
  cert.states.length = cert.inst.horizon + 1 ∧
  cert.values.length = cert.inst.horizon + 1 ∧
  (∀ i ∈ List.range (cert.inst.horizon + 1),
      (statesAt cert i).length = (valuesAt cert i).length) ∧
  (∀ i ∈ List.range (cert.inst.horizon + 1), (statesAt cert i).Nodup) ∧
  statesAt cert 0 = [cert.inst.start] ∧
  (∀ i ∈ List.range (cert.inst.horizon + 1), ∀ s ∈ statesAt cert i,
      s.station = i ∧ s.d ∈ cert.inst.offsets ∧ s.psi ∈ cert.inst.psis ∧
      s.b ∈ cert.inst.start.b :: cert.inst.speeds ∧ s.delta ∈ cert.inst.deltas)

instance domainOKDecidable (m : ActModel) : Decidable (domainOK m) := by
  unfold domainOK
  infer_instance

instance shapeOKDecidable (cert : Cert) : Decidable (shapeOK cert) := by
  unfold shapeOK
  infer_instance

def closedUnder (cert : Cert) (i : Nat) : Prop :=
  ∀ s ∈ statesAt cert i, ∀ a ∈ actionsA cert.inst cert.roots s,
    a.1 ∈ statesAt cert (i + 1)

def rowBellmanOK (cert : Cert) (i : Nat) (s : AState) : Prop :=
  valAt cert i s =
    bestO (fun a => (valAt cert (i + 1) a.1).map (fun v => a.2 + v))
      (actionsA cert.inst cert.roots s)

instance rowBellmanOKDecidable (cert : Cert) (i : Nat) (s : AState) :
    Decidable (rowBellmanOK cert i s) := by
  unfold rowBellmanOK
  infer_instance

instance closedUnderDecidable (cert : Cert) (i : Nat) : Decidable (closedUnder cert i) := by
  unfold closedUnder
  infer_instance

abbrev closedOK (cert : Cert) : Prop :=
  ∀ i ∈ List.range cert.inst.horizon, closedUnder cert i

instance closedOKDecidable (cert : Cert) : Decidable (closedOK cert) := by
  unfold closedOK
  infer_instance

abbrev terminalRowsOK (cert : Cert) : Prop :=
  ∀ s ∈ statesAt cert cert.inst.horizon,
    valAt cert cert.inst.horizon s = terminalValueA cert.inst cert.roots s

instance terminalRowsOKDecidable (cert : Cert) : Decidable (terminalRowsOK cert) := by
  unfold terminalRowsOK
  infer_instance

abbrev bellmanOK (cert : Cert) : Prop :=
  ∀ i ∈ List.range cert.inst.horizon, ∀ s ∈ statesAt cert i, rowBellmanOK cert i s

instance bellmanOKDecidable (cert : Cert) : Decidable (bellmanOK cert) := by
  unfold bellmanOK
  infer_instance

/-- The full decidable certificate-checking predicate. -/
def CheckCert (cert : Cert) : Prop :=
  domainOK cert.inst ∧
  rootsOK cert.roots (cert.inst.start.b :: cert.inst.speeds) ∧
  shapeOK cert ∧
  closedOK cert ∧
  terminalRowsOK cert ∧
  bellmanOK cert ∧
  valAt cert 0 cert.inst.start = cert.claimed ∧
  claimOK cert


instance checkCertDecidable (cert : Cert) : Decidable (CheckCert cert) := by
  unfold CheckCert
  infer_instance

def checkCert (cert : Cert) : Bool := decide (CheckCert cert)

theorem checkCert_sound (cert : Cert) (h : checkCert cert = true) : CheckCert cert :=
  of_decide_eq_true h

/-! ## Successor-set completeness -/

theorem nextStateA_station {m : ActModel} {roots : List (List Int)} {s : AState}
    {u a : Int} {t : AState} (h : nextStateA m roots s u a = some t) :
    t.station = s.station + 1 := by
  unfold nextStateA at h
  have hdom : stepDomainOK m roots s u a := by
    by_cases hd : stepDomainOK m roots s u a
    · exact hd
    · rw [if_neg hd] at h
      cases h
  rw [if_pos hdom] at h
  cases h
  dsimp [nextStateCore]

theorem nextStateA_steer_mem {m : ActModel} {roots : List (List Int)} {s : AState}
    {u a : Int} {t : AState} (h : nextStateA m roots s u a = some t) :
    u ∈ m.steer_inputs := by
  unfold nextStateA at h
  have hdom : stepDomainOK m roots s u a := by
    by_cases hd : stepDomainOK m roots s u a
    · exact hd
    · rw [if_neg hd] at h
      cases h
  exact hdom.2.1

theorem nextStateA_acc_mem {m : ActModel} {roots : List (List Int)} {s : AState}
    {u a : Int} {t : AState} (h : nextStateA m roots s u a = some t) :
    a ∈ m.acc_inputs := by
  unfold nextStateA at h
  have hdom : stepDomainOK m roots s u a := by
    by_cases hd : stepDomainOK m roots s u a
    · exact hd
    · rw [if_neg hd] at h
      cases h
  exact hdom.2.2.1

theorem actionOKA_some_next {m : ActModel} {roots : List (List Int)} {s : AState}
    {u a : Int} {t : AState} (h : actionOKA m roots s u a = some t) :
    nextStateA m roots s u a = some t := by
  unfold actionOKA at h
  cases hnext : nextStateA m roots s u a with
  | none => rw [hnext] at h; cases h
  | some t0 =>
      rw [hnext] at h
      cases hv : actionValidA m s t0 a with
      | false => simp [hv] at h
      | true =>
          have hsome : some t0 = some t := by simpa [hv] using h
          have ht : t0 = t := by simpa using hsome
          rw [← ht]

theorem actionOKA_station {m : ActModel} {roots : List (List Int)} {s : AState}
    {u a : Int} {t : AState} (h : actionOKA m roots s u a = some t) :
    t.station = s.station + 1 :=
  nextStateA_station (actionOKA_some_next h)

theorem actionOKA_steer_mem {m : ActModel} {roots : List (List Int)} {s : AState}
    {u a : Int} {t : AState} (h : actionOKA m roots s u a = some t) :
    u ∈ m.steer_inputs :=
  nextStateA_steer_mem (actionOKA_some_next h)

theorem actionOKA_acc_mem {m : ActModel} {roots : List (List Int)} {s : AState}
    {u a : Int} {t : AState} (h : actionOKA m roots s u a = some t) :
    a ∈ m.acc_inputs :=
  nextStateA_acc_mem (actionOKA_some_next h)

theorem mem_actionsA {m : ActModel} {roots : List (List Int)} {s : AState} {act : Act}
    (h : act ∈ actionsA m roots s) :
    ∃ u a t, u ∈ m.steer_inputs ∧ a ∈ m.acc_inputs ∧
      actionOKA m roots s u a = some t ∧ act = ((t, actionCostA m roots s t) : Act) := by
  unfold actionsA at h
  rcases List.mem_flatMap.mp h with ⟨u, hu, h⟩
  rcases List.mem_filterMap.mp h with ⟨a, ha, hres⟩
  split at hres
  · rename_i t ht
    injection hres with hres
    exact ⟨u, a, t, hu, ha, ht, hres.symm⟩
  · cases hres

theorem actionsA_complete {m : ActModel} {roots : List (List Int)} {s : AState}
    {u a : Int} {t : AState} (hu : u ∈ m.steer_inputs) (ha : a ∈ m.acc_inputs)
    (h : actionOKA m roots s u a = some t) :
    ((t, actionCostA m roots s t) : Act) ∈ actionsA m roots s := by
  unfold actionsA
  refine List.mem_flatMap.mpr ⟨u, hu, ?_⟩
  refine List.mem_filterMap.mpr ⟨a, ha, ?_⟩
  simp [h]
  rfl

/-! ## Plans are trajectories -/

theorem planSteps_runPlan (cert : Cert) :
    ∀ (tail : List AState) (acts : List (Int × Int)) (s : AState),
      planStepsOK cert s tail acts →
      runPlanA cert.inst cert.roots s acts = some (lastStateA s tail) := by
  intro tail
  induction tail with
  | nil =>
      intro acts s h
      cases acts with
      | nil => simp [planStepsOK, runPlanA, lastStateA]
      | cons p rest =>
          obtain ⟨u, a⟩ := p
          simp [planStepsOK] at h
  | cons t ts ih =>
      intro acts s h
      cases acts with
      | nil => simp [planStepsOK] at h
      | cons p rest =>
          obtain ⟨u, a⟩ := p
          simp only [planStepsOK] at h
          obtain ⟨hact, hrest⟩ := h
          simp only [runPlanA, hact, lastStateA]
          exact ih rest t hrest

theorem planSteps_length (cert : Cert) :
    ∀ (tail : List AState) (acts : List (Int × Int)) (s : AState),
      planStepsOK cert s tail acts → tail.length = acts.length := by
  intro tail
  induction tail with
  | nil =>
      intro acts s h
      cases acts with
      | nil => rfl
      | cons p rest =>
          obtain ⟨u, a⟩ := p
          simp [planStepsOK] at h
  | cons t ts ih =>
      intro acts s h
      cases acts with
      | nil => simp [planStepsOK] at h
      | cons p rest =>
          obtain ⟨u, a⟩ := p
          simp only [planStepsOK] at h
          obtain ⟨_, hrest⟩ := h
          have := ih rest t hrest
          simp only [List.length_cons]
          omega

theorem planCostOpt_terminal_some (cert : Cert) :
    ∀ (tail : List AState) (acts : List (Int × Int)) (s : AState) (v : Int),
      planStepsOK cert s tail acts →
      planCostOpt cert.inst cert.roots s acts = some v →
      ∃ tv, terminalValueA cert.inst cert.roots (lastStateA s tail) = some tv := by
  intro tail
  induction tail with
  | nil =>
      intro acts s v hsteps hcost
      cases acts with
      | nil =>
          simp [planStepsOK, planCostOpt, lastStateA] at hsteps hcost
          exact ⟨v, hcost⟩
      | cons p rest =>
          obtain ⟨u, a⟩ := p
          simp [planStepsOK] at hsteps
  | cons t ts ih =>
      intro acts s v hsteps hcost
      cases acts with
      | nil => simp [planStepsOK] at hsteps
      | cons p rest =>
          obtain ⟨u, a⟩ := p
          simp only [planStepsOK] at hsteps
          obtain ⟨hact, hrest⟩ := hsteps
          simp only [planCostOpt, hact] at hcost
          cases hc : planCostOpt cert.inst cert.roots t rest with
          | none => rw [hc] at hcost; cases hcost
          | some w =>
              rw [hc] at hcost
              simp only [Option.map_some, Option.some.injEq] at hcost
              simpa [lastStateA] using ih rest t w hrest hc

theorem plan_trajA (m : ActModel) (roots : List (List Int)) :
    ∀ (acts : List (Int × Int)) (s : AState), PlanOKA m roots s acts →
      ∃ xs, TrajOK (terminalValueA m roots)
          (List.replicate acts.length (actionsA m roots)) s xs ∧
        trajCost (terminalValueA m roots)
          (List.replicate acts.length (actionsA m roots)) s xs =
          planCostOpt m roots s acts := by
  intro acts
  induction acts with
  | nil =>
      intro s h
      rcases h with ⟨t, v, hrun, hterm⟩
      simp [runPlanA] at hrun
      have hts : t = s := by simpa using hrun.symm
      subst hts
      refine ⟨[], ⟨v, hterm⟩, ?_⟩
      simpa [trajCost, planCostOpt] using hterm
  | cons p rest ih =>
      intro s h
      obtain ⟨u, a⟩ := p
      rcases h with ⟨t0, v, hrun, hterm⟩
      simp only [runPlanA] at hrun
      cases hact : actionOKA m roots s u a with
      | none => rw [hact] at hrun; cases hrun
      | some t =>
          rw [hact] at hrun
          have hplan : PlanOKA m roots t rest := ⟨t0, v, hrun, hterm⟩
          obtain ⟨ys, hok, hc⟩ := ih t hplan
          have hmem := actionsA_complete (actionOKA_steer_mem hact)
            (actionOKA_acc_mem hact) hact
          refine ⟨((t, actionCostA m roots s t) : Act) :: ys, ?_, ?_⟩
          · simp only [List.length_cons, List.replicate_succ]
            exact ⟨hmem, hok⟩
          · simp only [List.length_cons, List.replicate_succ, trajCost]
            rw [hc]
            simp [planCostOpt, hact]

theorem runPlan_station (m : ActModel) (roots : List (List Int)) :
    ∀ (acts : List (Int × Int)) (s t : AState),
      runPlanA m roots s acts = some t → t.station = s.station + acts.length := by
  intro acts
  induction acts with
  | nil =>
      intro s t h
      simp [runPlanA] at h
      subst h
      simp
  | cons p rest ih =>
      intro s t h
      obtain ⟨u, a⟩ := p
      simp only [runPlanA] at h
      cases hact : actionOKA m roots s u a with
      | none => rw [hact] at h; cases h
      | some t' =>
          rw [hact] at h
          have hst := actionOKA_station hact
          have hrest := ih t' t h
          simp only [List.length_cons]
          omega

theorem terminalOKB_station {m : ActModel} {s : AState} (h : terminalOKB m s = true) :
    s.station = m.horizon := by
  unfold terminalOKB at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  exact h.1.1.1.1

theorem planOK_length (m : ActModel) (roots : List (List Int))
    (acts : List (Int × Int)) (s : AState) (h : PlanOKA m roots s acts)
    (hs : s.station = 0) : acts.length = m.horizon := by
  rcases h with ⟨t, v, hrun, hterm⟩
  have hstat : t.station = m.horizon := by
    cases htc : terminalCostA m roots t with
    | none =>
        have hz : terminalValueA m roots t = none := by simp [terminalValueA, htc]
        rw [hz] at hterm
        cases hterm
    | some c =>
        have hs : terminalValueA m roots t = some (1000 * c) := by
          simp [terminalValueA, htc]
        rw [hs] at hterm
        unfold terminalCostA at htc
        split at htc
        · rename_i hok
          exact terminalOKB_station hok
        · cases htc
  have hrunstat := runPlan_station m roots acts s t hrun
  omega

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
    (terminal : AState → Option Int) (actions : AState → List Act)
    (states : Nat → List AState) (val : Nat → AState → Option Int)
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
      rw [hik]
      simp only [List.replicate_succ]
      rw [valueO]
      rw [hrec i hiH s hs]
      apply bestO_congr_on
      intro a ha
      rw [ih (i + 1) hk' (by omega) a.1 (hclosed i hiH s hs a ha)]
      rw [hk']

theorem table_value_cert (cert : Cert) (h : CheckCert cert)
    (i : Nat) (s : AState) (hs : s ∈ statesAt cert i) :
    valAt cert i s =
      valueO (terminalValueA cert.inst cert.roots)
        (List.replicate (cert.inst.horizon - i) (actionsA cert.inst cert.roots)) s := by
  rcases h with ⟨hdom, hroots, hshape, hclosed, hterm, hrec, hval, hclaim⟩
  rcases hshape with ⟨hsl, hvl, hlen2, hnodup, hzero, hstate⟩
  have hi : i ≤ cert.inst.horizon := by
    by_cases hle : i ≤ cert.inst.horizon
    · exact hle
    · exfalso
      have hnil : statesAt cert i = [] := by
        unfold statesAt
        have hstates_le : cert.states.length ≤ i := by
          rw [hsl]
          omega
        rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none hstates_le]
        rfl
      rw [hnil] at hs
      simp at hs
  exact sparse_value (terminalValueA cert.inst cert.roots)
    (actionsA cert.inst cert.roots) (statesAt cert) (valAt cert) cert.inst.horizon
    (fun s hs => hterm s hs)
    (fun i hi s hs => hrec i (List.mem_range.mpr hi) s hs)
    (fun i hi s hs a ha => hclosed i (List.mem_range.mpr hi) s hs a ha)
    (cert.inst.horizon - i) i rfl hi s hs

/-- The checker's claimed value is the true Bellman optimum of the encoded
finite model. -/
theorem certificate_value (cert : Cert) (h : CheckCert cert) :
    valueO (terminalValueA cert.inst cert.roots) (stagesA cert.inst cert.roots)
      cert.inst.start = cert.claimed := by
  rcases h with ⟨hdom, hroots, hshape, hclosed, hterm, hrec, hval, hclaim⟩
  rcases hshape with ⟨hsl, hvl, hlen2, hnodup, hzero, hstate⟩
  have hstart : cert.inst.start ∈ statesAt cert 0 := by
    rw [hzero]; simp
  have ht := table_value_cert cert ⟨hdom, hroots, ⟨hsl, hvl, hlen2, hnodup, hzero, hstate⟩,
    hclosed, hterm, hrec, hval, hclaim⟩ 0 cert.inst.start hstart
  rw [Nat.sub_zero] at ht
  rw [hval] at ht
  simpa [stagesA] using ht.symm

/-- Every declarative plan costs at least the claimed value. -/
theorem certificate_lower_bound (cert : Cert) (h : CheckCert cert)
    (acts : List (Int × Int)) (hp : PlanOKA cert.inst cert.roots cert.inst.start acts) :
    ∀ v w, cert.claimed = some v → planCostOpt cert.inst cert.roots cert.inst.start acts = some w →
      v ≤ w := by
  intro v w hv hw
  rcases h with ⟨hdom, hroots, hshape, hclosed, hterm, hrec, hval, hclaim⟩
  have hlen := planOK_length cert.inst cert.roots acts cert.inst.start hp hdom.1
  obtain ⟨xs, hok, hc⟩ := plan_trajA cert.inst cert.roots acts cert.inst.start hp
  rw [hlen] at hok hc
  have hv' : valueO (terminalValueA cert.inst cert.roots) (stagesA cert.inst cert.roots)
      cert.inst.start = some v := by
    rw [← hv]
    exact certificate_value cert ⟨hdom, hroots, hshape, hclosed, hterm, hrec, hval, hclaim⟩
  have hcost : trajCost (terminalValueA cert.inst cert.roots)
      (stagesA cert.inst cert.roots) cert.inst.start xs = some w := by
    dsimp [stagesA]
    rw [hc, hw]
  exact BellmanOpt.valueO_le_cost (terminalValueA cert.inst cert.roots)
    (stagesA cert.inst cert.roots) cert.inst.start xs v w
    (by simpa [stagesA] using hok) hcost hv'

/-- The claimed value is attained by the exported plan. -/
theorem certificate_plan_some (cert : Cert) (h : CheckCert cert) (v : Int)
    (hc : cert.claimed = some v) :
    ∃ tail acts, cert.plan = cert.inst.start :: tail ∧
      pairList cert.planActions = acts ∧
      PlanOKA cert.inst cert.roots cert.inst.start acts ∧
      planCostOpt cert.inst cert.roots cert.inst.start acts = some v ∧
      cert.plan.length = cert.inst.horizon + 1 := by
  rcases h with ⟨hdom, hroots, hshape, hclosed, hterm, hrec, hval, hclaim⟩
  unfold claimOK at hclaim
  rw [hc] at hclaim
  rcases hclaim with ⟨hpair, hclaim⟩
  cases hp : cert.plan with
  | nil => simp [hp] at hclaim
  | cons s tail =>
      simp only [hp] at hclaim
      rcases hclaim with ⟨hs, hlen, halen, hsteps, hcost, hfa, hpc⟩
      rw [hs] at hsteps hcost
      have hrun := planSteps_runPlan cert tail (pairList cert.planActions)
        cert.inst.start hsteps
      obtain ⟨tv, htv⟩ := planCostOpt_terminal_some cert tail
        (pairList cert.planActions) cert.inst.start v hsteps hcost
      refine ⟨tail, pairList cert.planActions, ?_, rfl, ?_, hcost, hlen⟩
      · simpa [hp, hs]
      · exact ⟨lastStateA cert.inst.start tail, tv, hrun, htv⟩

/-- The exported plan is admissible, reaches the horizon, and attains the
claimed value. -/
theorem certificate_plan (cert : Cert) (h : CheckCert cert) :
    match cert.claimed with
    | none => True
    | some v =>
        ∃ tail acts, cert.plan = cert.inst.start :: tail ∧
          pairList cert.planActions = acts ∧
          PlanOKA cert.inst cert.roots cert.inst.start acts ∧
          planCostOpt cert.inst cert.roots cert.inst.start acts = some v ∧
          cert.plan.length = cert.inst.horizon + 1
    := by
  cases hc : cert.claimed with
  | none => trivial
  | some v => exact certificate_plan_some cert h v hc

/-- **Infeasibility is exact**: a claimed `none` rules out every declarative
plan of the horizon model. -/
theorem certificate_infeasible (cert : Cert) (h : CheckCert cert)
    (hclaim : cert.claimed = none) :
    ¬ ∃ acts, PlanOKA cert.inst cert.roots cert.inst.start acts := by
  intro ⟨acts, hp⟩
  rcases h with ⟨hdom, hroots, hshape, hclosed, hterm, hrec, hval, hck⟩
  have hv := certificate_value cert ⟨hdom, hroots, hshape, hclosed, hterm, hrec, hval, hck⟩
  rw [hclaim] at hv
  have hlen := planOK_length cert.inst cert.roots acts cert.inst.start hp hdom.1
  obtain ⟨xs, hok, hc⟩ := plan_trajA cert.inst cert.roots acts cert.inst.start hp
  rw [hlen] at hok
  have hnot := (BellmanOpt.valueO_none_iff (terminalValueA cert.inst cert.roots)
    (stagesA cert.inst cert.roots) cert.inst.start).mp (by simpa [stagesA] using hv)
  exact hnot ⟨xs, by simpa [stagesA] using hok⟩

structure ActuatorGuarantee (cert : Cert) : Prop where
  value :
    valueO (terminalValueA cert.inst cert.roots) (stagesA cert.inst cert.roots)
      cert.inst.start = cert.claimed
  claimed_some :
    ∀ v, cert.claimed = some v →
      valueO (terminalValueA cert.inst cert.roots) (stagesA cert.inst cert.roots)
        cert.inst.start = some v ∧
      (∃ tail acts, cert.plan = cert.inst.start :: tail ∧
        pairList cert.planActions = acts ∧
        PlanOKA cert.inst cert.roots cert.inst.start acts ∧
        planCostOpt cert.inst cert.roots cert.inst.start acts = some v ∧
        cert.plan.length = cert.inst.horizon + 1 ∧
        (∀ other, PlanOKA cert.inst cert.roots cert.inst.start other →
          ∀ w, planCostOpt cert.inst cert.roots cert.inst.start other = some w → v ≤ w))
  claimed_none :
    cert.claimed = none →
      valueO (terminalValueA cert.inst cert.roots) (stagesA cert.inst cert.roots)
        cert.inst.start = none ∧
      ¬ ∃ acts, PlanOKA cert.inst cert.roots cert.inst.start acts

theorem certificate_optimal (cert : Cert) (h : CheckCert cert) :
    ActuatorGuarantee cert := by
  have hvalue := certificate_value cert h
  have hplan := certificate_plan cert h
  have hnone : cert.claimed = none →
      ¬ ∃ acts, PlanOKA cert.inst cert.roots cert.inst.start acts :=
    fun hc => certificate_infeasible cert h hc
  refine { value := hvalue, claimed_some := ?_, claimed_none := ?_ }
  · intro v hc
    have hv : valueO (terminalValueA cert.inst cert.roots) (stagesA cert.inst cert.roots)
        cert.inst.start = some v := by
      simpa [hc] using hvalue
    obtain ⟨tail, acts, htail, hpairs, hp, hcost, hlen⟩ :=
      certificate_plan_some cert h v hc
    refine ⟨hv, tail, acts, htail, hpairs, hp, hcost, hlen, ?_⟩
    intro other hother w hw
    exact certificate_lower_bound cert h other hother v w hc hw
  · intro hc
    exact ⟨by simpa [hc] using hvalue, hnone hc⟩

end Corpus.Actuator
