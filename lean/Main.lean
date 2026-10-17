/-
`corpuscheck INTERCHANGE.json [CERT_DIR]`

Decodes the generator's exported manifest/index interchange and every per-scenario certificate
file in CERT_DIR, and decides every property below with its `Decidable` instance (or a checker
with a proved soundness theorem). Prints a JSON report; exit code 0 iff every property holds.
-/
import Corpus

open Lean Corpus

structure CertSample where
  sampleId : String
  clean : Bool
  q : Cert.Q
  decision : Cert.Decision
deriving FromJson

structure CertFile where
  schema : String
  scenarioId : String
  consts : Cert.Consts
  samples : List CertSample
deriving FromJson

structure Result where
  name : String
  ok : Bool
  basis : String
  detail : String

def Result.toJson (r : Result) : Json :=
  Json.mkObj [("name", r.name), ("ok", r.ok), ("basis", r.basis), ("detail", r.detail)]

def check (name basis : String) (p : Prop) [Decidable p] (detail : Unit → String := fun _ => "") : Result :=
  let ok := decide p
  { name, ok, basis, detail := if ok then "" else detail () }

def firstIds (l : List String) : String := ", ".intercalate (l.take 5) ++ (if l.length > 5 then s!" (+{l.length - 5})" else "")

def manifestChecks (d : Interchange) : List Result :=
  let sp := btoSpec
  let acc := accepted d.scenarios
  let cells := requiredCells sp d.strength
  [ check "interchange_schema" "string equality" (d.schema = "autocross.datagen.lean_interchange" ∧ d.schemaVersion = 1),
    check "spec_equals_lean_model" "DecidableEq Spec: generator spec = Corpus.btoSpec" (d.spec = sp),
    check "formal_verdicts_agree" "Decidable (FormalVerdictsAgree): Python formal rejection iff ¬ Corpus.Valid"
      (FormalVerdictsAgree sp d.scenarios)
      (fun _ => firstIds ((d.scenarios.filter fun s => decide (s.formalRejected = true ↔ ¬ Valid sp s) == false).map (·.id))),
    check "accepted_valid_and_bins_correct" "Decidable (AcceptedValid)" (AcceptedValid sp d.scenarios)
      (fun _ => firstIds ((acc.filter fun s => !decide (Valid sp s ∧ BinsCorrect sp s)).map (·.id))),
    check "scenario_ids_unique" "List.Nodup" (UniqueIds d.scenarios),
    check "scenario_hashes_unique" "List.Nodup" (UniqueHashes d.scenarios),
    check "contrast_pairs_integrity" "Decidable (PairsOK): same family and group, exactly the declared factor differs (value and bin)"
      (PairsOK sp d.scenarios d.pairs)
      (fun _ => firstIds ((d.pairs.filter fun p => !decide (PairOK sp acc p)).map fun p => p.base ++ "~" ++ p.factor)),
    check "split_rule" "Decidable (SplitRule): split and region recomputed from the split policy"
      (SplitRule d.splitPolicy d.scenarios),
    check "group_split_consistent" "Decidable (GroupSplitConsistent); implies pairs_within_split" (GroupSplitConsistent d.scenarios),
    check "hash_split_consistent" "Decidable (HashSplitConsistent)" (HashSplitConsistent d.scenarios),
    check "required_cells_match" "requiredCells btoSpec strength = exported cell list"
      ((cells.map cellId) = d.requiredCells),
    check "manifest_coverage_exact" "Decidable (CoverageExact); with [] missing, full_coverage_of_exact"
      (CoverageExact acc cells d.manifestMissing) ]

def corpusChecks (d : Interchange) : List Result :=
  if !d.hasCorpus then [] else
  let acc := accepted d.scenarios
  let cells := requiredCells btoSpec d.strength
  let clean := acc.filter fun s => s.id ∈ d.realisedClean
  [ check "realised_scenarios_accepted" "membership" (∀ i ∈ d.realisedClean, ∃ s ∈ acc, s.id = i),
    check "realised_coverage_exact" "Decidable (CoverageExact) over scenarios with a clean run"
      (CoverageExact clean cells d.realisedMissing),
    check "clean_samples_no_cross_split_duplicates" "adjCheck on hash-sorted list; adjCheck_sound"
      (adjCheck d.cleanSamples = true),
    check "mirror_transform_matches_model" "mirrorObs / mirrorAct equality on exported samples"
      (∀ m ∈ d.mirrorChecks, mirrorObs m.obs = m.mirroredObs ∧ mirrorAct m.act = m.mirroredAct)
      (fun _ => firstIds ((d.mirrorChecks.filter fun m => !decide (mirrorObs m.obs = m.mirroredObs ∧ mirrorAct m.act = m.mirroredAct)).map (·.sampleId))) ]

def constsSane (c : Cert.Consts) : Prop :=
  0 ≤ c.a_acc ∧ 0 ≤ c.a_brk ∧ 0 < c.a_max ∧ 0 < c.a_lat ∧ 0 < c.r_sw_m

instance (c : Cert.Consts) : Decidable (constsSane c) := by unfold constsSane; infer_instance

structure CertStats where
  files : Nat := 0
  samples : Nat := 0
  clean : Nat := 0
  greatestSafe : Nat := 0
  minViolation : Nat := 0
  mismatches : List String := []
  cleanUncertified : List String := []
  badConsts : List String := []

def certFile (st : CertStats) (f : CertFile) : CertStats :=
  let st := { st with files := st.files + 1,
                      badConsts := if decide (constsSane f.consts) then st.badConsts else f.scenarioId :: st.badConsts }
  f.samples.foldl (fun st s =>
    let r := Cert.decide f.consts s.q
    let st := { st with samples := st.samples + 1,
                        mismatches := if r = s.decision then st.mismatches else s.sampleId :: st.mismatches }
    if s.clean then
      { st with clean := st.clean + 1,
                greatestSafe := st.greatestSafe + (if r.kind = Cert.kindGreatestSafe then 1 else 0),
                minViolation := st.minViolation + (if r.kind = Cert.kindMinViolation then 1 else 0),
                cleanUncertified := if r.kind = Cert.kindGreatestSafe ∨ r.kind = Cert.kindMinViolation then st.cleanUncertified
                                    else s.sampleId :: st.cleanUncertified }
    else st) st

def certChecks (st : CertStats) : List Result :=
  if st.files = 0 then [] else
  [ check "cert_constants_sane" "non-negative budgets (hypotheses of fpass_feasible)" (st.badConsts = []) (fun _ => firstIds st.badConsts),
    check "cert_decisions_recomputed" "Cert.decide on the quantised observation = recorded label (every exported sample)"
      (st.mismatches = []) (fun _ => s!"{st.mismatches.length} mismatches: " ++ firstIds st.mismatches.reverse),
    check "clean_labels_certified" "every clean sample is greatest_safe (decide_greatest_safe) or min_violation (decide_min_violation)"
      (st.cleanUncertified = []) (fun _ => firstIds st.cleanUncertified.reverse) ]

def readJson (α : Type) [FromJson α] (path : String) : IO α := do
  let txt ← IO.FS.readFile path
  match Json.parse txt >>= fromJson? with
  | .ok v => pure v
  | .error e => throw <| IO.userError s!"{path}: {e}"

def main (args : List String) : IO UInt32 := do
  match args with
  | [] =>
    IO.eprintln "usage: corpuscheck INTERCHANGE.json [CERT_DIR]"
    return 2
  | inter :: rest =>
    let d ← readJson Interchange inter
    let mut st : CertStats := {}
    if let some dir := rest.head? then
      let entries ← System.FilePath.readDir dir
      let paths := (entries.map (·.path)).filter (·.extension == some "json")
      for p in paths.qsort (fun a b => a.toString < b.toString) do
        let f ← readJson CertFile p.toString
        st := certFile st f
    let results := manifestChecks d ++ corpusChecks d ++ certChecks st
    let ok := results.all (·.ok)
    let stats := Json.mkObj [
      ("scenarios", (accepted d.scenarios).length), ("draws", d.scenarios.length), ("pairs", d.pairs.length),
      ("required_cells", d.requiredCells.length), ("clean_sample_hashes", d.cleanSamples.length),
      ("mirror_checks", d.mirrorChecks.length), ("cert_files", st.files), ("cert_samples", st.samples),
      ("cert_clean_samples", st.clean), ("cert_greatest_safe", st.greatestSafe), ("cert_min_violation", st.minViolation)]
    let report := Json.mkObj [("checker", "corpuscheck"), ("lean_version", Lean.versionString), ("ok", ok),
      ("backend", d.backend), ("stats", stats), ("results", Json.arr (results.map Result.toJson).toArray)]
    IO.println report.pretty
    return if ok then 0 else 1
