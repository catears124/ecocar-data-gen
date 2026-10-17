/-
Interchange data model. Mirrors `lean_interchange.json` written by
`autocross/datagen/lean_bridge.py`. Every number is an integer: scenario parameters in
1/1000 of their unit (the generator's canonical fixed-point form), observations in 1/10^6,
SHA-256 digests as natural numbers.
-/
import Lean.Data.Json

open Lean

namespace Corpus

structure Bin where
  label : String
  lo : Int
  hi : Int
  hiInclusive : Bool
deriving Repr, DecidableEq, FromJson, ToJson

/-- A coverage factor: numeric (non-empty `bins`) or categorical (non-empty `levels`). -/
structure Factor where
  name : String
  bins : List Bin
  levels : List String
deriving Repr, DecidableEq, FromJson, ToJson

structure Nuisance where
  name : String
  lo : Int
  hi : Int
deriving Repr, DecidableEq, FromJson, ToJson

structure Spec where
  name : String
  version : Nat
  factors : List Factor
  nuisance : List Nuisance
  constants : List (String × Int)
deriving Repr, DecidableEq, FromJson, ToJson

structure Scenario where
  id : String
  hash : Nat
  family : String
  familyVersion : Nat
  nums : List (String × Int)
  cats : List (String × String)
  /-- bins claimed by the generator (factors whose value lies in no bin are omitted) -/
  bins : List (String × String)
  accepted : Bool
  /-- the generator rejected this draw for a `formal:*` reason (bounds / structure) -/
  formalRejected : Bool
  split : String
  group : String
  region : String
deriving Repr, FromJson

structure Pair where
  base : String
  variant : String
  factor : String
deriving Repr, FromJson

structure Region where
  split : String
  cell : List (String × String)
deriving Repr, FromJson

structure SplitPolicy where
  regionFactors : List String
  regions : List Region
deriving Repr, FromJson

/-- Label-relevant part of an observation, fixed-point (1e-6). -/
structure ObstacleM where
  dist : Int
  latMin : Int
  latMax : Int
deriving Repr, DecidableEq, FromJson

structure ObsM where
  lateral : Int
  heading : Int
  curvature : Int
  widthLeft : Int
  widthRight : Int
  distLeft : Int
  distRight : Int
  vy : Int
  yawRate : Int
  ay : Int
  steer : Int
  previewCurvature : List Int
  previewWidthLeft : List Int
  previewWidthRight : List Int
  centerline : List (Int × Int)
  leftBoundary : List (Int × Int)
  rightBoundary : List (Int × Int)
  obstacles : List ObstacleM
  cornerDirections : List Int
  cornerTurnAngles : List Int
deriving Repr, DecidableEq, FromJson

structure ActM where
  steer : Int
  throttle : Int
  brake : Int
  path : List (Int × Int)
deriving Repr, DecidableEq, FromJson

/-- A recorded sample and the generator's mirrored version of it. -/
structure MirrorCheck where
  sampleId : String
  obs : ObsM
  act : ActM
  mirroredObs : ObsM
  mirroredAct : ActM
deriving Repr, FromJson

structure Interchange where
  schema : String
  schemaVersion : Nat
  spec : Spec
  strength : Nat
  splitPolicy : SplitPolicy
  scenarios : List Scenario
  pairs : List Pair
  requiredCells : List String
  manifestMissing : List String
  hasCorpus : Bool
  backend : String
  realisedClean : List String
  realisedMissing : List String
  /-- (content hash, split index 0 train / 1 val / 2 test) of every clean sample, sorted by hash -/
  cleanSamples : List (Nat × Nat)
  mirrorChecks : List MirrorCheck
deriving Repr, FromJson

end Corpus
