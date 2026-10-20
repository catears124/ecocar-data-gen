/-
`simcheck CERT.json`

Checks the arithmetic of a simulator-evaluated finite-candidate certificate:
the claimed cost of the selected candidate is the first lexicographic minimum
of the declared cost list, and the baseline candidate is in the list and no
cheaper than the selection.

This executable does **not** re-run the bicycle ODE.  The simulation is the
trusted harness transition function (the same function the simulator uses);
this checker binds the selection arithmetic to the declared integer costs.
-/
import Lean.Data.Json

open Lean

structure SimCert where
  costs : List (List Int)
  selected : Nat
  selectedCost : List Int
  baselineIndex : Nat
  baselineCost : List Int
  lipschitzMilli : Int
  resolutionMilli : Int
  epsilonMilli : Int
deriving Repr

instance : FromJson SimCert where
  fromJson? j := do
    let costs ← j.getObjValAs? (List (List Int)) "costs"
    let selected ← j.getObjValAs? Nat "selected"
    let selectedCost ← j.getObjValAs? (List Int) "selected_cost"
    let baselineIndex ← j.getObjValAs? Nat "baseline_index"
    let baselineCost ← j.getObjValAs? (List Int) "baseline_cost"
    let lipschitzMilli ← j.getObjValAs? Int "lipschitz_milli"
    let resolutionMilli ← j.getObjValAs? Int "resolution_milli"
    let epsilonMilli ← j.getObjValAs? Int "epsilon_milli"
    pure { costs, selected, selectedCost, baselineIndex, baselineCost,
           lipschitzMilli, resolutionMilli, epsilonMilli }

def lexLE : List Int → List Int → Bool
  | [], _ => true
  | _ :: _, [] => false
  | x :: xs, y :: ys =>
      if x < y then true else if y < x then false else lexLE xs ys

def check (c : SimCert) : Bool :=
  (c.costs[c.selected]? == some c.selectedCost) &&
  (c.costs[c.baselineIndex]? == some c.baselineCost) &&
  c.costs.all (fun x => lexLE c.selectedCost x) &&
  lexLE c.selectedCost c.baselineCost &&
  c.epsilonMilli = c.lipschitzMilli * c.resolutionMilli / 1000

def main (args : List String) : IO UInt32 := do
  match args with
  | [path] =>
      let text ← IO.FS.readFile path
      match Json.parse text with
      | .error e =>
          IO.println (Json.mkObj [("ok", false), ("error", "json: " ++ e)]).compress
          return 1
      | .ok j =>
          match (fromJson? j : Except String SimCert) with
          | .error e =>
              IO.println (Json.mkObj [("ok", false), ("error", "schema: " ++ e)]).compress
              return 1
          | .ok c =>
              let ok := check c
              IO.println (Json.mkObj [
                ("ok", toJson ok),
                ("selected", toJson c.selected),
                ("selected_cost", toJson c.selectedCost),
                ("baseline_cost", toJson c.baselineCost),
                ("candidates", toJson c.costs.length),
                ("epsilon_milli", toJson c.epsilonMilli)]).compress
              return (if ok then 0 else 1)
  | _ =>
      IO.println "usage: simcheck CERT.json"
      return 2
