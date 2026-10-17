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

end Corpus.Dp
