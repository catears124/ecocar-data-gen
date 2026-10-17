/-
  A declared finite-grid joint (station, lateral offset, speed-cost) model.

  The grid is intentionally data, not an implicit real interval.  Feasibility of
  each transition is therefore a decidable integer predicate and Bellman.lean
  supplies the optimality theorem for every admissible policy on that grid.
-/
import Corpus.Bellman
import Corpus.Cert
import Corpus.SteerFeas

namespace Corpus.Dp
open Corpus
open Corpus.Bellman

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

def transitionOK (c : Steer.Consts) (q : Steer.Q) (s t : JState) : Prop :=
  t.station = s.station + 1 ∧ bucketOK t.current ∧
  corridorClear c q t.station t.current ∧ obstacleClear c q t.station t.current ∧
  jointCurvature c.lon q s.station s.previous s.current t.current ≤ c.kappa_max

instance transitionOKDecidable (c : Steer.Consts) (q : Steer.Q) (s t : JState) :
    Decidable (transitionOK c q s t) := by
  unfold transitionOK corridorClear obstacleClear bucketOK
  infer_instance

/-- A speed² cap attached to a joint state.  It combines the certified
longitudinal cap with the lateral curvature cap. -/
def jointCap (c : Steer.Consts) (q : Steer.Q) (s : JState) : Int :=
  let k := jointCurvature c.lon q s.station s.previous s.current s.current
  if k = 0 then Cert.capAt c.lon q.base s.station
  else min (Cert.capAt c.lon q.base s.station) (c.lon.a_lat * 1000000000 / k)

/-- The action list at a state is the declared next-bucket grid filtered by the
corridor, obstacle, and curvature conditions.  The pair stores next state and
negative speed², so minimising cost maximises the certified speed cap. -/
def jointActions (c : Steer.Consts) (q : Steer.Q) (s : JState) : List (Action JState) :=
  if s.station + 1 < q.base.s.length then
    (stateGrid q.base.s.length).filterMap fun t =>
      if decide (transitionOK c q s t) then some (t, -(jointCap c q t)) else none
  else []

/-- The successor is the explicitly selected next grid state. -/
def jointSuccessor (_ : JState) (a : Action JState) : JState := a.1

/-- The finite-horizon joint model. -/
def jointModel (c : Steer.Consts) (q : Steer.Q) : Model JState where
  states := stateGrid q.base.s.length
  stages := (List.range q.base.s.length).map fun _ => jointActions c q
  successor := jointSuccessor
  terminal := fun _ => 0

/-- A policy is feasible in the declared joint model exactly when every selected
transition is one of the filtered grid actions. -/
def JointFeasible (c : Steer.Consts) (q : Steer.Q) (p : Policy JState) : Prop :=
  Admissible (jointModel c q).stages p

/-- The abstract Bellman theorem instantiated with the declared joint model.
For every policy whose action at each stage is a feasible grid transition, its
integer accumulated negative-speed cost is at least the value returned by
backward induction.  Thus no feasible policy has a smaller cost (equivalently,
no faster policy in this discretised additive speed objective). -/
theorem dp_optimal (c : Steer.Consts) (q : Steer.Q) (p : Policy JState)
    (hfeasible : JointFeasible c q p) :
    JointFeasible c q p ∧
      policyCost (jointModel c q) (jointModel c q).stages p initialState
        ≥ value (jointModel c q) (jointModel c q).stages initialState := by
  refine ⟨hfeasible, ?_⟩
  exact no_policy_beats (jointModel c q) _ p hfeasible _

/-- The certified per-transition curvature and clearance obligations used by the
joint action filter, exposed for clients that need to audit a selected edge. -/
theorem transition_certificate (c : Steer.Consts) (q : Steer.Q) (s t : JState)
    (h : transitionOK c q s t) :
    t.station = s.station + 1 ∧ bucketOK t.current ∧
      corridorClear c q t.station t.current ∧ obstacleClear c q t.station t.current ∧
      jointCurvature c.lon q s.station s.previous s.current t.current ≤ c.kappa_max := by
  exact h

end Corpus.Dp
