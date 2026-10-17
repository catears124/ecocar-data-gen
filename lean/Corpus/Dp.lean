/-
  A declared finite-grid lateral-offset dynamic program (station, previous
  offset, current offset), plus the abstract propagation theorems it relies on.

  Objective (explicitly a surrogate, see `jointCap`/`toyCap`): for every
  transition the model charges the negative of a declared integer **speed-cap
  surrogate**; minimising the accumulated cost therefore maximises the sum of
  the caps over the horizon.  It is a *lateral-trajectory* DP with a speed-cap
  surrogate, not a travel-time optimum and not a joint path-and-speed optimum.

  Admissibility (`AdmissibleFrom`) is translation-relative: a policy must offer
  a legal action at every state it *visits*.  Availability is a hard
  requirement: an empty action list admits no continuation, and the DP value on
  such a state is only a bound that no trajectory can attain.
-/
import Corpus.Bellman
import Corpus.Cert
import Corpus.SteerFeas

namespace Corpus.Dp
open Corpus
open Corpus.Bellman

/-! ## Abstract propagation along a single admissible path -/

/-- A policy is admissible *from a start state* when it is legal at every state
it actually visits: the head action is offered at `s`, and the tail is
admissible from the successor.  Unlike `Bellman.Admissible` this never demands
an action at an unvisited (e.g. terminal) state, which is what made the earlier
joint statement vacuous. -/
inductive AdmissibleFrom {σ : Type} (m : Model σ) :
    List (Stage σ) → Policy σ → σ → Prop
  | nil (s : σ) : AdmissibleFrom m [] [] s
  | cons {st : Stage σ} {rest : List (Stage σ)} {p : σ → Action σ} {ps : Policy σ} {s : σ}
      (head : p s ∈ st s)
      (tail : AdmissibleFrom m rest ps (m.successor s (p s))) :
      AdmissibleFrom m (st :: rest) (p :: ps) s

/-- **No admissible path beats the Bellman value.**  The hypothesis is exactly
the existence of a legal action at each visited state along `policy`. -/
theorem no_policy_beats_from {σ : Type} (m : Model σ) :
    ∀ (stages : List (Stage σ)) (policy : Policy σ) (s : σ),
      AdmissibleFrom m stages policy s →
      policyCost m stages policy s ≥ value m stages s := by
  intro stages
  induction stages with
  | nil =>
    intro policy s h
    cases h
    simp [policyCost, value]
  | cons st rest ih =>
    intro policy s h
    cases policy with
    | nil => cases h
    | cons p ps =>
      cases h with
      | cons hhead htail =>
        have hle := bestCost_le (fun q => value m rest (m.successor s q)) (st s) (p s) hhead
        have hrec := ih ps (m.successor s (p s)) htail
        change (p s).2 + policyCost m rest ps (m.successor s (p s)) ≥
          bestCost (fun q => value m rest (m.successor s q)) (st s)
        omega

/-- A lower bound on every offered action's cost bounds `bestCost` from below,
provided the action list is nonempty. -/
theorem bestCost_ge_of_le {σ : Type} (f : Action σ → Int) :
    ∀ (x : Action σ) (xs : List (Action σ)) (c : Int),
      (∀ b, b ∈ x :: xs → c ≤ b.2 + f b) → c ≤ bestCost f (x :: xs)
  | x, [], c, h => by simpa [bestCost] using h x (by simp)
  | x, y :: ys, c, h => by
    have hx : c ≤ x.2 + f x := h x (by simp)
    have hy : c ≤ bestCost f (y :: ys) :=
      bestCost_ge_of_le f y ys c (fun b hb => h b (List.mem_cons_of_mem x hb))
    exact Int.le_min.mpr ⟨hx, hy⟩

/-- **`bestCost` is attained by any offered action of minimal cost.** -/
theorem bestCost_eq_of_min {σ : Type} (f : Action σ → Int) :
    ∀ (as : List (Action σ)) (a : Action σ), a ∈ as →
      (∀ b, b ∈ as → a.2 + f a ≤ b.2 + f b) → bestCost f as = a.2 + f a := by
  intro as
  cases as with
  | nil => intro a h; cases h
  | cons x xs =>
    intro a hmem hmin
    exact Int.le_antisymm
      (bestCost_le f (x :: xs) a hmem)
      (bestCost_ge_of_le f x xs (a.2 + f a) hmin)

/-- A policy that follows a `bestCost`-attaining action at every visited state
(the greedy choice of `Bellman`) attains the Bellman value. -/
inductive GreedyTraj {σ : Type} (m : Model σ) :
    List (Stage σ) → Policy σ → σ → Prop
  | nil (s : σ) : GreedyTraj m [] [] s
  | cons {st : Stage σ} {rest : List (Stage σ)} {p : σ → Action σ} {ps : Policy σ} {s : σ}
      (hp : p s ∈ st s)
      (hopt : bestCost (fun a => value m rest (m.successor s a)) (st s) =
        (p s).2 + value m rest (m.successor s (p s)))
      (htail : GreedyTraj m rest ps (m.successor s (p s))) :
      GreedyTraj m (st :: rest) (p :: ps) s

/-- **The extracted greedy trajectory both is admissible and attains the value.** -/
theorem greedyTraj_attains {σ : Type} (m : Model σ) :
    ∀ (stages : List (Stage σ)) (policy : Policy σ) (s : σ),
      GreedyTraj m stages policy s →
      AdmissibleFrom m stages policy s ∧ policyCost m stages policy s = value m stages s := by
  intro stages
  induction stages with
  | nil =>
    intro policy s h
    cases h
    exact ⟨AdmissibleFrom.nil s, rfl⟩
  | cons st rest ih =>
    intro policy s h
    cases policy with
    | nil => cases h
    | cons p ps =>
      cases h with
      | cons hp hopt htail =>
        obtain ⟨hadm, hcost⟩ := ih ps (m.successor s (p s)) htail
        refine ⟨AdmissibleFrom.cons hp hadm, ?_⟩
        show (p s).2 + policyCost m rest ps (m.successor s (p s)) =
          bestCost (fun a => value m rest (m.successor s a)) (st s)
        rw [hcost]
        exact hopt.symm

/-! ## The declared lateral grid and step relation -/

/-- Integer lateral-offset bucket grid, in millimetres. -/
def bucketGrid : List Int := [-1000, -750, -500, -250, 0, 250, 500, 750, 1000]

/-- A joint state stores the station and the previous/current offset buckets. -/
structure JState where
  station : Nat
  previous : Int
  current : Int
deriving DecidableEq, Repr

/-- The declared finite state set for a preview of `n` stations. -/
def stateGrid (n : Nat) : List JState :=
  (List.range n).flatMap fun i =>
    bucketGrid.flatMap fun p => bucketGrid.map fun x =>
      { station := i, previous := p, current := x }

/-- Integer curvature contribution of a three-bucket lateral stencil. -/
def lateralStencil (p x n : Int) : Int :=
  Cert.iabs (Cert.cdiv (n - 2 * x + p) 1)

/-- `i = 0` is the *origin*, not an admissible first state: the model starts at
`initialState` and the first transition advances the station by one. -/
def initialState : JState :=
  { station := 0, previous := 0, current := 0 }

/-- Curvature bound used by the joint model.  The road term is the certified
`kbarOf` term, while the second term is the integer lateral stencil. -/
def jointCurvature (c : Cert.Consts) (q : Steer.Q) (i : Nat)
    (p x n : Int) : Int :=
  Cert.iabs (Cert.kbarOf c (Cert.iabs (q.base.k.getD i 0))
    (Cert.stationY c q.base (q.base.s.getD i 0))) + lateralStencil p x n

def corridorClear (c : Steer.Consts) (q : Steer.Q) (i : Nat) (x : Int) : Prop :=
  Steer.lo c q i ≤ x ∧ x ≤ Steer.hi c q i

def obstacleClear (c : Steer.Consts) (q : Steer.Q) (i : Nat) (x : Int) : Prop :=
  0 ≤ Steer.clearAt c q (q.base.s.getD i 0) i x

def bucketOK (x : Int) : Prop := x ∈ bucketGrid

/-- **State continuity.**  A transition must advance exactly one station, carry
the current offset into the previous slot, land on the declared grid, satisfy the
corridor and obstacle constraints at the *arrival* station, and obey the
curvature bound of the actual three-offset stencil.  Without the first two
conjuncts the DP could rewrite its history and bypass the curvature limit. -/
def transitionOK (c : Steer.Consts) (q : Steer.Q) (s t : JState) : Prop :=
  t.station = s.station + 1 ∧ t.previous = s.current ∧ bucketOK t.current ∧
  corridorClear c q t.station t.current ∧ obstacleClear c q t.station t.current ∧
  jointCurvature c.lon q s.station s.previous s.current t.current ≤ c.kappa_max

instance transitionOKDecidable (c : Steer.Consts) (q : Steer.Q) (s t : JState) :
    Decidable (transitionOK c q s t) := by
  unfold transitionOK corridorClear obstacleClear bucketOK
  infer_instance

/-- A speed² cap attached to a joint state.  It combines the certified
longitudinal cap with the lateral curvature cap of the stencil that *entered*
the state (`s.previous → s.current → n`), so the charge depends on the actual
transition, not on a state in isolation. -/
def jointCap (c : Steer.Consts) (q : Steer.Q) (s : JState) (n : Int) : Int :=
  let k := jointCurvature c.lon q s.station s.previous s.current n
  if k = 0 then Cert.capAt c.lon q.base s.station
  else min (Cert.capAt c.lon q.base s.station) (c.lon.a_lat * 1000000000 / k)

/-- The action list at a state is the declared next-bucket grid filtered by the
corridor, obstacle, and curvature conditions.  The pair stores the next state and
the negative speed² surrogate, so minimising cost maximises the certified cap.
An empty list is a *hard* failure: no continuation exists there. -/
def jointActions (c : Steer.Consts) (q : Steer.Q) (s : JState) : List (Action JState) :=
  if s.station + 1 < q.base.s.length then
    bucketGrid.filterMap fun n =>
      let t : JState := { station := s.station + 1, previous := s.current, current := n }
      if decide (transitionOK c q s t) then some (t, -(jointCap c q s n)) else none
  else []

/-- The successor is the explicitly selected next grid state. -/
def jointSuccessor (_ : JState) (a : Action JState) : JState := a.1

/-- The finite-horizon joint model.  The horizon is `s.length - 1` transitions
between `s.length` stations. -/
def jointModel (c : Steer.Consts) (q : Steer.Q) : Model JState where
  states := stateGrid q.base.s.length
  stages := (List.range (q.base.s.length - 1)).map fun _ => jointActions c q
  successor := jointSuccessor
  terminal := fun _ => 0

/-- **Joint optimality (nonvacuous form).**  For every policy that is legal at
each state it visits starting from `initialState`, the accumulated negative
speed-cap cost is at least the backward-induction value: no admissible lateral
trajectory has a lower declared cost.  The premise is a genuine trajectory
existence statement, not a global action-list requirement. -/
theorem dp_optimal (c : Steer.Consts) (q : Steer.Q) (p : Policy JState)
    (hfeasible : AdmissibleFrom (jointModel c q) (jointModel c q).stages p initialState) :
    policyCost (jointModel c q) (jointModel c q).stages p initialState
      ≥ value (jointModel c q) (jointModel c q).stages initialState :=
  no_policy_beats_from (jointModel c q) _ p _ hfeasible

/-- The certified per-transition curvature and clearance obligations used by the
joint action filter, exposed for clients that need to audit a selected edge. -/
theorem transition_certificate (c : Steer.Consts) (q : Steer.Q) (s t : JState)
    (h : transitionOK c q s t) :
    t.station = s.station + 1 ∧ t.previous = s.current ∧ bucketOK t.current ∧
      corridorClear c q t.station t.current ∧ obstacleClear c q t.station t.current ∧
      jointCurvature c.lon q s.station s.previous s.current t.current ≤ c.kappa_max :=
  h

/-! ## A concrete, kernel-computed instance

Three stations, the three-point grid `[-250, 0, 250]`, and a curvature limit of
500 on the integer stencil `|n - 2x + p|`.  The executable DP (`toyModel`) is
the same definition the theorems are about; the independent enumeration
`toyCandidates`/`toyChain` checks the optimum exhaustively, and the extraction
`toyExtract` is the greedy choice of the model itself.  Everything below is
decided by the kernel (`decide`), with no `native_decide`.
-/

def toyGrid : List Int := [-250, 0, 250]

def toyStencilAbs (p x n : Int) : Nat := (n - 2 * x + p).natAbs

/-- Speed-cap surrogate, in the declared integer units. -/
def toyCap (p x n : Int) : Int := 1000 - (toyStencilAbs p x n : Int)

def toyStepOK (s t : JState) : Bool :=
  (t.station == s.station + 1) && (t.previous == s.current) &&
    decide (t.current ∈ toyGrid) && decide (toyStencilAbs s.previous s.current t.current ≤ 500)

def toyActions (s : JState) : List (Action JState) :=
  if s.station + 1 < 3 then
    toyGrid.filterMap fun n =>
      let t : JState := { station := s.station + 1, previous := s.current, current := n }
      if toyStepOK s t then some (t, -(toyCap s.previous s.current n)) else none
  else []

def toyModel : Model JState where
  states := stateGrid 3
  stages := [toyActions, toyActions]
  successor := jointSuccessor
  terminal := fun _ => 0

/-- The last station offers nothing: this is the explicit infeasible case that
`AdmissibleFrom` never demands an action from. -/
theorem toy_terminal_no_actions : toyActions { station := 2, previous := 0, current := 0 } = [] := by
  decide

/-- Independent enumeration of every length-2 offset chain on the toy grid. -/
def toyCandidates : List (List Int) :=
  toyGrid.flatMap fun a => toyGrid.map fun b => [a, b]

def toyChain (l : List Int) : Bool :=
  match l with
  | [a, b] => (decide (a ∈ toyGrid)) && (decide (b ∈ toyGrid)) && decide (toyStencilAbs 0 a b ≤ 500)
  | _ => false

def toyChainCost (l : List Int) : Int :=
  match l with
  | [a, b] => -(toyCap 0 0 a) - toyCap 0 a b
  | _ => 0

/-- Exhaustive optimum over the enumerated chains. -/
def toyEnumMin : Int :=
  (toyCandidates.filterMap fun l => if toyChain l then some (toyChainCost l) else none).foldr min 1000000

/-- **Nonempty admissible set**, checked exhaustively by the kernel. -/
theorem toy_nonempty : toyCandidates.any (fun l => toyChain l) = true := by
  decide

/-- **The DP value equals the exhaustive enumeration minimum.** -/
theorem toy_dp_eq_enumeration : value toyModel toyModel.stages initialState = toyEnumMin := by
  decide

/-- **No admissible chain is cheaper than the enumerated optimum**: the kernel
checks the inequality on every enumerated candidate. -/
theorem toy_optimal_exhaustive :
    (toyCandidates.filterMap
      fun l => if toyChain l then some (decide (toyEnumMin ≤ toyChainCost l)) else none).all id
      = true := by
  decide


/-- The greedy extraction from the model itself walks the optimum. -/
def toyExtract : List Int :=
  match (greedy? (fun a => value toyModel [toyActions] (jointSuccessor initialState a))
      (toyActions initialState)) with
  | some a =>
    match (greedy? (fun b => value toyModel [] (jointSuccessor a.1 b)) (toyActions a.1)) with
    | some b => [a.1.current, b.1.current]
    | none => [a.1.current]
  | none => []

/-- The extracted trajectory is the enumerated optimum. -/
theorem toy_extract_optimal : toyChainCost toyExtract = toyEnumMin ∧ toyChain toyExtract = true := by
  decide

/-! ## Speed dynamics: acceleration/braking reachability and a travel-time objective

The DP above charges a speed-cap *surrogate* per transition.  This section adds
speed as a state component: a declared integer speed² bucket `v`, an
acceleration limit `accelMax`, a braking limit `brakeMax`, the curvature/clearance
cap of the actual transition, and a cost that is a *declared integer travel time*
(`timeTable`, milliseconds to traverse one station at that speed² bucket).

Objective, named precisely: minimise the sum of the declared per-station
traversal times over trajectories whose every transition advances one station,
lands on the offset grid, obeys the corridor/obstacle/curvature certificate, and
keeps the speed² bucket within the acceleration/braking envelope and under the
cap.  The optimality over that discrete set is proved; the *calibration* of
`timeTable` to physical seconds is data, i.e. a trust item, not a theorem.
-/

/-- Declared speed² buckets. -/
def speedGrid : List Int := [0, 250, 500, 750, 1000]

/-- Declared per-station traversal time in integer milliseconds, keyed by the
speed² bucket of the transition.  Monotonically decreasing (faster is quicker),
which the concrete instance checks by kernel reduction. -/
def timeTable : List (Int × Int) :=
  [(0, 120), (250, 60), (500, 40), (750, 30), (1000, 24)]

/-- Declared traversal time of one station at speed² bucket `v`; `0` for an
unlisted bucket (which the reachability test rejects, since `w ∈ speedGrid`). -/
def timeAt (v : Int) : Int :=
  ((timeTable.find? fun p => p.1 == v).map fun p => p.2).getD 0

/-- Acceleration and braking limits on one station's speed² change, in bucket
units. -/
def accelMax : Int := 250
def brakeMax : Int := 500

/-- **Speed reachability.**  The next speed² bucket must be declared, must not
exceed the certified cap of the transition, and must lie inside the
acceleration/braking envelope `v - brakeMax ≤ w ≤ v + accelMax`. -/
def speedFeasible (cap v w : Int) : Bool :=
  decide (w ∈ speedGrid) && decide (w ≤ cap) && decide (v - brakeMax ≤ w) &&
    decide (w ≤ v + accelMax)

/-- The speed component of a state is only ever a declared bucket. -/
structure VState where
  station : Nat
  previous : Int
  current : Int
  v : Int
deriving DecidableEq, Repr

def VState.toJ (s : VState) : JState :=
  { station := s.station, previous := s.previous, current := s.current }

/-- Actions of the joint offset/speed model: for each certified offset successor
`t`, every reachable, cap-respecting speed bucket `w`; the cost is the declared
traversal time at `w`, so the accumulated cost is the declared travel time. -/
def vActions (c : Steer.Consts) (q : Steer.Q) (s : VState) : List (Action VState) :=
  if s.station + 1 < q.base.s.length then
    bucketGrid.flatMap fun n =>
      let t : JState := { station := s.station + 1, previous := s.current, current := n }
      if decide (transitionOK c q s.toJ t) then
        speedGrid.filterMap fun w =>
          if speedFeasible (jointCap c q s.toJ n) s.v w then
            some (({ station := t.station, previous := t.previous, current := t.current, v := w } : VState), timeAt w)
          else none
      else []
  else []

/-- The declared joint offset/speed horizon: `s.length - 1` transitions. -/
def vModel (c : Steer.Consts) (q : Steer.Q) : Model VState where
  states := (List.range q.base.s.length).flatMap fun i =>
    bucketGrid.flatMap fun p => bucketGrid.flatMap fun x =>
      speedGrid.map fun v => { station := i, previous := p, current := x, v := v }
  stages := (List.range (q.base.s.length - 1)).map fun _ => vActions c q
  successor := fun _ a => a.1
  terminal := fun _ => 0

def vInitial : VState := { station := 0, previous := 0, current := 0, v := 0 }

/-- **Declared travel-time optimality with speed state.**  No policy that is
legal at each visited state from `vInitial` accumulates less declared travel time
than the backward-induction value of the declared model. -/
theorem speed_dp_optimal (c : Steer.Consts) (q : Steer.Q) (p : Policy VState)
    (hfeasible : AdmissibleFrom (vModel c q) (vModel c q).stages p vInitial) :
    policyCost (vModel c q) (vModel c q).stages p vInitial
      ≥ value (vModel c q) (vModel c q).stages vInitial := by
  exact no_policy_beats_from (vModel c q) _ p _ hfeasible

/-! ### Concrete speed instance (kernel-checked)

Three stations, straight offset profile (so the offset certificate is
satisfied), speed² buckets `[0, 250, 500]`, acceleration and braking 250 per
station, caps `500, 500, 250` (a straight pair and then a turn).  The optimum
must build speed on the straight and brake for the turn: fixing the target speed
alone is not enough, reachability decides it.
-/

def toySpeedGrid : List Int := [0, 250, 500]
def toyCaps : List Int := [500, 500, 250]
def toyAccel : Int := 250

/-- Step check of the concrete instance: target bucket declared, under the cap of
the *target* station, inside the acceleration envelope. -/
def toySpeedOK (targetStation : Nat) (v w : Int) : Bool :=
  decide (w ∈ toySpeedGrid) && decide (w ≤ toyCaps.getD (targetStation - 1) 0) &&
    decide (w ≤ v + toyAccel)

def toySpeedActions (s : Nat × Int) : List (Action (Nat × Int)) :=
  if s.1 + 1 < 4 then
    toySpeedGrid.filterMap fun w =>
      if toySpeedOK (s.1 + 1) s.2 w then some ((s.1 + 1, w), timeAt w) else none
  else []

def toySpeedModel : Model (Nat × Int) where
  states := (List.range 4).flatMap fun i => toySpeedGrid.map fun v => (i, v)
  stages := [toySpeedActions, toySpeedActions, toySpeedActions]
  successor := fun _ a => a.1
  terminal := fun _ => 0

def toySpeedStart : Nat × Int := (0, 0)

/-- Independent enumeration of every declared speed chain. -/
def toySpeedCandidates : List (List Int) :=
  toySpeedGrid.flatMap fun a => toySpeedGrid.flatMap fun b => toySpeedGrid.map fun c => [a, b, c]

def toySpeedChain (l : List Int) : Bool :=
  match l with
  | [a, b, c] => toySpeedOK 1 0 a && toySpeedOK 2 a b && toySpeedOK 3 b c
  | _ => false

def toySpeedCost (l : List Int) : Int :=
  match l with
  | [a, b, c] => timeAt a + timeAt b + timeAt c
  | _ => 0

def toySpeedEnumMin : Int :=
  (toySpeedCandidates.filterMap fun l => if toySpeedChain l then some (toySpeedCost l) else none).foldr
    min 1000000

/-- The declared time table is strictly decreasing in speed, so minimising time is
not the same as maximising speed blindly. -/
theorem toy_time_table_decreasing : timeAt 0 > timeAt 250 ∧ timeAt 250 > timeAt 500 := by
  decide

/-- **Reachability bites**: from a standstill one station cannot reach the top
bucket, so the cap tests alone do not determine the optimum. -/
theorem toy_accel_reachability_bites : toySpeedOK 1 0 500 = false := by
  decide

/-- The turn caps the arrival speed: the fastest arrival at station 3 is 250. -/
theorem toy_turn_cap_bites : toySpeedOK 3 500 500 = false ∧ toySpeedOK 3 500 250 = true := by
  decide

/-- **Nonempty admissible speed set**, kernel-checked. -/
theorem toy_speed_nonempty : toySpeedCandidates.any (fun l => toySpeedChain l) = true := by
  decide

/-- **The declared-time DP value equals the exhaustive enumeration minimum.** -/
theorem toy_speed_dp_eq_enum :
    value toySpeedModel toySpeedModel.stages toySpeedStart = toySpeedEnumMin := by
  decide

/-- **Nothing admissible is quicker than the enumerated optimum.** -/
theorem toy_speed_optimal_exhaustive :
    (toySpeedCandidates.filterMap
      fun l => if toySpeedChain l then some (decide (toySpeedEnumMin ≤ toySpeedCost l)) else none).all id
      = true := by
  decide

/-- Greedy extraction from the model itself. -/
def toySpeedExtract : List Int :=
  match greedy? (fun a => value toySpeedModel [toySpeedActions, toySpeedActions]
      (toySpeedModel.successor toySpeedStart a)) (toySpeedActions toySpeedStart) with
  | some a =>
    match greedy? (fun b => value toySpeedModel [toySpeedActions] (toySpeedModel.successor a.1 b))
        (toySpeedActions a.1) with
    | some b =>
      match greedy? (fun c => value toySpeedModel [] (toySpeedModel.successor b.1 c))
          (toySpeedActions b.1) with
      | some c => [a.1.2, b.1.2, c.1.2]
      | none => [a.1.2, b.1.2]
    | none => [a.1.2]
  | none => []

/-- The extracted speed chain builds speed on the straight and brakes for the
turn, is admissible, and attains the enumerated travel-time minimum. -/
theorem toy_speed_extract_optimal :
    toySpeedExtract = [250, 500, 250] ∧ toySpeedChain toySpeedExtract = true ∧
      toySpeedCost toySpeedExtract = toySpeedEnumMin := by
  decide

end Corpus.Dp
