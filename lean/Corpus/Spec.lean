/-
The formal model of the scenario family `brake_turn_obstacle` v1: bounds, bins, structural
constraints, region-based split rule and required coverage cells.

This file is written independently of the Python generator. The checker requires the
generator's exported spec to be *equal* to `btoSpec`, and recomputes every validity verdict,
bin and split, so a change on either side that is not mirrored on the other fails the check.
-/
import Corpus.Model

namespace Corpus

/-- `brake_turn_obstacle` v1. Units: parameters in 1/1000 (mm, mm/s, millidegrees, 1e-3). -/
def btoSpec : Spec where
  name := "brake_turn_obstacle"
  version := 1
  factors := [
    ⟨"turn_radius_m", [⟨"tight", 9000, 14000, false⟩, ⟨"medium", 14000, 22000, false⟩,
                        ⟨"open", 22000, 35000, true⟩], []⟩,
    ⟨"turn_direction", [], ["left", "right"]⟩,
    ⟨"preview_m", [⟨"short", 15000, 25000, false⟩, ⟨"nominal", 25000, 45000, false⟩,
                    ⟨"long", 45000, 60000, true⟩], []⟩,
    ⟨"approach_speed_mps", [⟨"low", 8000, 12000, false⟩, ⟨"mid", 12000, 16000, false⟩,
                             ⟨"high", 16000, 20000, true⟩], []⟩,
    ⟨"obstacle_side", [], ["inside", "outside"]⟩,
    ⟨"obstacle_clearance_m", [⟨"tight", 2800, 3300, false⟩, ⟨"moderate", 3300, 3900, false⟩,
                               ⟨"wide", 3900, 4600, true⟩], []⟩,
    ⟨"road_width_m", [⟨"narrow", 5000, 5600, false⟩, ⟨"standard", 5600, 6800, false⟩,
                       ⟨"wide", 6800, 8000, true⟩], []⟩]
  nuisance := [⟨"turn_angle_deg", 70000, 110000⟩, ⟨"obstacle_arc_frac", 300, 600⟩]
  constants := [("obstacle_inset_m", 600), ("min_obstacle_width_m", 600), ("radius_margin_m", 1500)]

def knownSpecs : List Spec := [btoSpec]

def Spec.konst (sp : Spec) (k : String) : Int := (sp.constants.lookup k).getD 0

def Factor.labels (f : Factor) : List String :=
  if f.bins = [] then f.levels else f.bins.map Bin.label

def Factor.lo (f : Factor) : Int := (f.bins.head?.map Bin.lo).getD 0
def Factor.hi (f : Factor) : Int := (f.bins.getLast?.map Bin.hi).getD 0

def Spec.paramNames (sp : Spec) : List String :=
  sp.factors.map Factor.name ++ sp.nuisance.map Nuisance.name

/-! ### Validity -/

def NumIn (s : Scenario) (n : String) (lo hi : Int) : Prop :=
  match s.nums.lookup n with
  | some v => lo ≤ v ∧ v ≤ hi
  | none => False

instance (s : Scenario) (n : String) (lo hi : Int) : Decidable (NumIn s n lo hi) := by
  unfold NumIn; split <;> infer_instance

def CatIn (s : Scenario) (n : String) (levels : List String) : Prop :=
  match s.cats.lookup n with
  | some l => l ∈ levels
  | none => False

instance (s : Scenario) (n : String) (levels : List String) : Decidable (CatIn s n levels) := by
  unfold CatIn; split <;> infer_instance

/-- Value of a factor within the factor's overall bounds (numeric) or one of its levels. -/
def FactorOK (s : Scenario) (f : Factor) : Prop :=
  if f.bins = [] then CatIn s f.name f.levels else NumIn s f.name f.lo f.hi

instance (s : Scenario) (f : Factor) : Decidable (FactorOK s f) := by
  unfold FactorOK; split <;> infer_instance

/-- Structural constraints of `brake_turn_obstacle`:
  * the obstacle keeps at least two cones: clearance ≤ width − inset − min obstacle width;
  * the corridor fits the turn (course builder rule): radius ≥ width/2 + margin. -/
def BtoStructural (sp : Spec) (s : Scenario) : Prop :=
  match s.nums.lookup "obstacle_clearance_m", s.nums.lookup "road_width_m", s.nums.lookup "turn_radius_m" with
  | some c, some w, some r =>
      c ≤ w - sp.konst "obstacle_inset_m" - sp.konst "min_obstacle_width_m" ∧
      w + 2 * sp.konst "radius_margin_m" ≤ 2 * r
  | _, _, _ => False

instance (sp : Spec) (s : Scenario) : Decidable (BtoStructural sp s) := by
  unfold BtoStructural; split <;> infer_instance

/-- Family dispatch; unknown families are never structurally valid. -/
def Structural (sp : Spec) (s : Scenario) : Prop :=
  if sp.name = "brake_turn_obstacle" then BtoStructural sp s else False

instance (sp : Spec) (s : Scenario) : Decidable (Structural sp s) := by
  unfold Structural; split <;> infer_instance

/-- A scenario is formally valid for `sp`. -/
def Valid (sp : Spec) (s : Scenario) : Prop :=
  s.family = sp.name ∧ s.familyVersion = sp.version ∧
  (∀ f ∈ sp.factors, FactorOK s f) ∧
  (∀ n ∈ sp.nuisance, NumIn s n.name n.lo n.hi) ∧
  Structural sp s

instance (sp : Spec) (s : Scenario) : Decidable (Valid sp s) := by
  unfold Valid; infer_instance

/-! ### Bins -/

def Bin.contains (b : Bin) (v : Int) : Bool :=
  decide (b.lo ≤ v) && (if b.hiInclusive then decide (v ≤ b.hi) else decide (v < b.hi))

def binOf (f : Factor) (v : Int) : Option String :=
  (f.bins.find? (·.contains v)).map Bin.label

def expectedBin (s : Scenario) (f : Factor) : Option String :=
  if f.bins = [] then
    match s.cats.lookup f.name with
    | some l => if l ∈ f.levels then some l else none
    | none => none
  else (s.nums.lookup f.name).bind (binOf f)

/-- The bins the generator recorded are the bins of the parameter values. -/
def BinsCorrect (sp : Spec) (s : Scenario) : Prop :=
  ∀ f ∈ sp.factors, s.bins.lookup f.name = expectedBin s f

instance (sp : Spec) (s : Scenario) : Decidable (BinsCorrect sp s) := by
  unfold BinsCorrect; infer_instance

/-! ### Splits -/

def cellHolds (bins : List (String × String)) (cell : List (String × String)) : Bool :=
  cell.all (fun kv => bins.lookup kv.1 == some kv.2)

/-- Region split: a listed test region wins over a val region; everything else is train. -/
def regionSplit (p : SplitPolicy) (bins : List (String × String)) : String :=
  if p.regions.any (fun r => r.split == "test" && cellHolds bins r.cell) then "test"
  else if p.regions.any (fun r => r.split == "val" && cellHolds bins r.cell) then "val"
  else "train"

def regionKey (p : SplitPolicy) (bins : List (String × String)) : String :=
  "|".intercalate (p.regionFactors.map fun f => f ++ "=" ++ (bins.lookup f).getD "")

def splitNames : List String := ["train", "val", "test"]

/-! ### Coverage cells -/

def pairsOf {α : Type} : List α → List (α × α)
  | [] => []
  | x :: xs => xs.map (fun y => (x, y)) ++ pairsOf xs

def cells1 (sp : Spec) : List (List (String × String)) :=
  sp.factors.flatMap fun f => f.labels.map fun l => [(f.name, l)]

def cells2 (sp : Spec) : List (List (String × String)) :=
  (pairsOf sp.factors).flatMap fun fg =>
    fg.1.labels.flatMap fun a => fg.2.labels.map fun b => [(fg.1.name, a), (fg.2.name, b)]

/-- Required coverage: every bin of every factor (strength 1) or every pair of bins of every
two factors (strength 2). -/
def requiredCells (sp : Spec) : Nat → List (List (String × String))
  | 1 => cells1 sp
  | 2 => cells2 sp
  | _ => []

def cellId (c : List (String × String)) : String :=
  "|".intercalate (c.map fun kv => kv.1 ++ "=" ++ kv.2)

end Corpus
