/-
`corpuscheck INTERCHANGE.json [CORPUS_DIR]`

1. Decodes the generator's manifest interchange and decides the manifest properties below with
   their `Decidable` instances (or checkers with proved soundness theorems).
2. With CORPUS_DIR: reads `index.json` and EVERY sample record of every completed run DIRECTLY
   (the JSONL files the corpus ships; no exporter in between), quantises each sample's policy
   observation itself (exact decimal arithmetic on the JSON numbers, `qfix`), evaluates
   `Corpus.Cert.decide` on it, and compares the result with the certificate recorded in the
   sample. The set of clean samples must equal the set of clean samples whose certificate was
   checked and is certified, and every count must equal the index.

Trust boundary: the decisions are recomputed by this COMPILED executable (Lean compiler and
runtime, Lean's JSON parser); the theorems about `Cert.decide` are kernel-checked once when the
library is built. Per-sample kernel proofs are produced separately (`Corpus/Witness*.lean`).
Prints a JSON report; exit code 0 iff every property holds.
-/
import Corpus
import Std.Data.HashMap

open Lean Corpus

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
  [ check "interchange_schema" "string equality" (d.schema = "autocross.datagen.lean_interchange" ∧ d.schemaVersion = 2),
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

/-! ## Reading the corpus -/

/-- `floor(x * scale + 1/2)` for the exact decimal value `x = mantissa / 10^exponent` of a JSON
number (identical to `cert.qfix` on the float's shortest repr, which is the JSON text). -/
def qfix (n : JsonNumber) (scale : Nat) : Int :=
  let den : Int := ((10 : Nat) ^ n.exponent : Nat)
  (2 * n.mantissa * scale + den) / (2 * den)

def jnum (j : Json) (k : String) (scale : Nat) : Option Int :=
  match j.getObjValD k with
  | .num n => some (qfix n scale)
  | _ => none

def jnumArr (j : Json) (k : String) (scale : Nat) : Option (List Int) :=
  match j.getObjValD k with
  | .arr a => a.toList.mapM fun x => match x with | .num n => some (qfix n scale) | _ => none
  | _ => none

/-- Quantised inputs of model cert-2 from a policy observation (`none` = invalid input). -/
def quantise (po : Json) : Option Cert.Q := do
  let road := po.getObjValD "road"
  match road.getObjValD "valid" with
  | .bool true => pure ()
  | _ => none
  let v0 ← jnum po "speed" 1000
  let s ← jnumArr road "preview_s" 1000
  let k ← jnumArr road "preview_curvature" 1000000
  let wl ← jnumArr road "preview_width_left" 1000
  let wr ← jnumArr road "preview_width_right" 1000
  let obstacles ← match road.getObjValD "obstacles" with
    | .arr a => a.toList.mapM fun o => do
        pure [← jnum o "distance" 1000, ← jnum o "lateral_min" 1000, ← jnum o "lateral_max" 1000]
    | _ => none
  if s.length < 2 then none
  pure { v0, s, k, wl, wr, obstacles }

def decideOpt (c : Cert.Consts) : Option Cert.Q → Cert.Decision
  | some q => Cert.decide c q
  | none => ⟨Cert.kindInvalid, -c.a_max, none, none, none⟩

def kindName : Nat → String
  | 0 => "greatest_safe" | 1 => "min_violation" | 2 => "no_feasible_gap" | 3 => "invalid_input" | _ => "unknown"


def kindId : String → Option Nat
  | "greatest_safe" => some 0 | "min_violation" => some 1 | "no_feasible_gap" => some 2 | "invalid_input" => some 3
  | _ => none
def optInt : Json → Except String (Option Int)
  | .null => .ok none
  | .num n => if n.exponent = 0 then .ok (some n.mantissa) else .error "non-integer"
  | _ => .error "not an integer"

structure CertRec where
  model : String
  kind : String
  certified : Bool
  a : Int
  b1 : Option Int
  b0 : Option Int
  B0 : Option Int

def parseCert (j : Json) : Except String CertRec := do
  let o ← j.getObj?
  if o.size != 7 then throw "certificate has unexpected fields"
  let a ← match j.getObjValD "accel_mm_s2" with | .num n => if n.exponent = 0 then pure n.mantissa else throw "a" | _ => throw "a"
  pure { model := ← j.getObjValAs? String "model", kind := ← j.getObjValAs? String "kind",
         certified := ← j.getObjValAs? Bool "certified", a,
         b1 := ← optInt (j.getObjValD "b1_mm2_s2"), b0 := ← optInt (j.getObjValD "b0_mm2_s2"),
         B0 := ← optInt (j.getObjValD "B0_mm2_s2") }

def CertRec.matches (r : CertRec) (d : Cert.Decision) : Bool :=
  r.kind == kindName d.kind && r.a == d.a && r.b1 == d.b1 && r.b0 == d.b0 && r.B0 == d.B0 &&
  r.certified == (d.kind == Cert.kindGreatestSafe || d.kind == Cert.kindMinViolation)

def pad5 (n : Nat) : String := let s := toString n; "".pushn '0' (5 - s.length) ++ s

def hexNat (s : String) : Option Nat :=
  s.foldl (fun acc ch => do
    let a ← acc
    let v ← if '0' ≤ ch ∧ ch ≤ '9' then some (ch.toNat - '0'.toNat)
            else if 'a' ≤ ch ∧ ch ≤ 'f' then some (ch.toNat - 'a'.toNat + 10) else none
    pure (16 * a + v)) (some 0)

def splitIdx : String → Option Nat
  | "train" => some 0 | "val" => some 1 | "test" => some 2 | _ => none

/-- Named failure counters; a check passes iff its counter is 0. -/
structure St where
  fails : Std.HashMap String (Nat × Array String) := {}
  samples : Nat := 0
  clean : Nat := 0
  certChecked : Nat := 0
  certMismatch : Nat := 0
  certifiedClaims : Nat := 0
  cleanVerified : Nat := 0
  greatestSafe : Nat := 0
  minViolation : Nat := 0
  cleanIds : Array String := #[]
  verifiedCleanIds : Array String := #[]
  contentHashes : Array (Nat × Nat) := #[]
  /-- (sample id, run index, quantised observation, recorded decision) of every clean sample -/
  witnesses : Array (String × Nat × Cert.Q × Cert.Decision) := #[]

def St.fail (st : St) (name id : String) : St :=
  let (n, ex) := st.fails.getD name (0, #[])
  { st with fails := st.fails.insert name (n + 1, if ex.size < 5 then ex.push id else ex) }

structure Run where
  id : String
  split : String
  status : String
  cleanRun : Bool
  nSamples : Nat
  nClean : Nat
  nCertClean : Nat
  certModel : String
  consts : Option Cert.Consts
  samplesPath : Option String

def parseRun (j : Json) : Run :=
  let gs (k : String) := (j.getObjValAs? String k).toOption.getD ""
  let gn (k : String) := (j.getObjValAs? Nat k).toOption.getD 0
  { id := gs "scenario_id", split := gs "split", status := gs "status",
    cleanRun := (j.getObjValAs? Bool "clean_run").toOption.getD false,
    nSamples := gn "n_samples", nClean := gn "n_clean", nCertClean := gn "n_certified_clean",
    certModel := gs "cert_model", consts := (j.getObjValAs? Cert.Consts "cert_constants").toOption,
    samplesPath := ((j.getObjValD "files").getObjValD "samples" |>.getObjValAs? String "path").toOption }

/-- Check one sample record (line `i` of run `r`). -/
def checkSample (r : Run) (c : Cert.Consts) (ri i : Nat) (j : Json) (st : St) : St := Id.run do
  let mut st := { st with samples := st.samples + 1 }
  let sidOpt := (j.getObjValAs? String "sample_id").toOption
  let sid := sidOpt.getD s!"{r.id}#{i}"
  -- identity: schema, run, split, sample_id = run:step with step = line index
  let schemaOk := (j.getObjValAs? String "schema").toOption == some "autocross.datagen.sample" &&
    (j.getObjValAs? Nat "schema_version").toOption == some 2
  let step := (j.getObjValAs? Nat "step").toOption
  if !(schemaOk && step == some i && i < 100000 && sidOpt == some (r.id ++ ":" ++ pad5 i) &&
       (j.getObjValAs? String "scenario_id").toOption == some r.id &&
       (j.getObjValAs? String "split").toOption == some r.split) then
    st := st.fail "sample_identity" sid
  let clean := (j.getObjValD "quality").getObjValAs? Bool "clean" |>.toOption
  let reasons := (j.getObjValD "quality").getObjValD "exclusion_reasons"
  let oracle := ((j.getObjValD "privileged").getObjValD "oracle").getObjValAs? String "phase" |>.toOption
  let target := j.getObjValD "target"
  let expertPhase := (j.getObjValD "expert").getObjValAs? String "phase" |>.toOption
  if clean.isNone || oracle.isNone then
    st := st.fail "records_parse" sid
  let isClean := clean == some true
  if isClean then
    st := { st with clean := st.clean + 1, cleanIds := st.cleanIds.push sid }
  -- certificate: present iff the target is an expert-phase target
  let lon := target.getObjValD "longitudinal"
  let hasTarget := !(target matches .null)
  let hasCert := !(lon matches .null)
  if hasTarget && (expertPhase == some "expert") != hasCert then
    st := st.fail "cert_presence_matches_phase" sid
  if !hasTarget && hasCert then
    st := st.fail "cert_presence_matches_phase" sid
  let mut verifiedCertified := false
  let mut recorded : Option Cert.Decision := none
  let q := quantise (j.getObjValD "policy_observation")
  if hasCert then
    match parseCert lon with
    | .error _ => st := st.fail "records_parse" sid
    | .ok rc =>
      let d := decideOpt c q
      recorded := (kindId rc.kind).map fun k => ⟨k, rc.a, rc.b1, rc.b0, rc.B0⟩
      st := { st with certChecked := st.certChecked + 1,
                      certifiedClaims := st.certifiedClaims + (if rc.certified then 1 else 0) }
      if rc.model != "cert-2" then
        st := st.fail "cert_model_and_constants" sid
      if rc.matches d then
        verifiedCertified := d.kind == Cert.kindGreatestSafe || d.kind == Cert.kindMinViolation
      else
        st := { st.fail "cert_decisions_recomputed" sid with certMismatch := st.certMismatch + 1 }
  if isClean then
    let okClean := verifiedCertified && oracle == some "decision" && r.cleanRun &&
      (reasons matches .arr #[])
    if okClean then
      let d := decideOpt c q
      st := { st with cleanVerified := st.cleanVerified + 1, verifiedCleanIds := st.verifiedCleanIds.push sid,
                      greatestSafe := st.greatestSafe + (if d.kind == Cert.kindGreatestSafe then 1 else 0),
                      minViolation := st.minViolation + (if d.kind == Cert.kindMinViolation then 1 else 0) }
    else
      st := st.fail "clean_samples_certified" sid
    -- per-sample kernel witness: `Cert.decide c q = <recorded decision>` (q quantised here, by Lean)
    match q, recorded with
    | some q, some rd => st := { st with witnesses := st.witnesses.push (sid, ri, q, rd) }
    | _, _ => pure ()
    match ((j.getObjValD "hashes").getObjValAs? String "content").toOption.bind hexNat, splitIdx r.split with
    | some h, some sp => st := { st with contentHashes := st.contentHashes.push (h, sp) }
    | _, _ => st := st.fail "records_parse" sid
  return st

def readJson (α : Type) [FromJson α] (path : System.FilePath) : IO α := do
  let txt ← IO.FS.readFile path
  match Json.parse txt >>= fromJson? with
  | .ok v => pure v
  | .error e => throw <| IO.userError s!"{path}: {e}"

/-- All checks on a corpus directory. -/
def corpusChecks (d : Interchange) (dir : System.FilePath) :
    IO (List Result × Json × Array (String × Nat × Cert.Q × Cert.Decision) × Array (Option Cert.Consts)) := do
  let index ← readJson Json (dir / "index.json")
  let runsJ := match index.getObjValD "scenarios" with | .arr a => a.toList | _ => []
  let runs := runsJ.map parseRun
  let completed : List Run := runs.filter fun (r : Run) => r.status == "completed"
  let mut st : St := {}
  -- index identity
  let idxOk := (index.getObjValAs? String "schema").toOption == some "autocross.datagen.index" &&
    (index.getObjValAs? Nat "schema_version").toOption == some 2 &&
    (index.getObjValAs? String "certificate_model").toOption == some "cert-2" &&
    ((index.getObjValD "backend").getObjValAs? String "name").toOption == some d.backend
  -- the samples directory holds exactly the referenced sample files
  let referenced : List String := completed.filterMap fun (r : Run) => r.samplesPath
  let present ← if ← (dir / "samples").isDir then
      (← System.FilePath.readDir (dir / "samples")).toList.map (fun e => "samples/" ++ e.fileName) |>.mergeSort (· ≤ ·) |> pure
    else pure []
  let filesExact := present == referenced.mergeSort (· ≤ ·) && referenced.length == completed.length
  for (r, ri) in completed.zipIdx do
    match r.consts, r.samplesPath with
    | some c, some p =>
      if r.certModel != "cert-2" || !Cert.constsOK c then
        st := st.fail "cert_model_and_constants" r.id
      let before := st
      let h ← IO.FS.Handle.mk (dir / p) .read
      let mut i := 0
      repeat
        let line ← h.getLine
        if line.isEmpty then break
        match Json.parse line with
        | .ok j => st := checkSample r c ri i j st
        | .error _ => st := { st.fail "records_parse" s!"{r.id}#{i}" with samples := st.samples + 1 }
        i := i + 1
      if st.samples - before.samples != r.nSamples || st.clean - before.clean != r.nClean ||
         st.cleanVerified - before.cleanVerified != r.nCertClean then
        st := st.fail "run_counts_match_index" r.id
    | _, _ => st := st.fail "records_parse" r.id
  let totals := index.getObjValD "totals"
  let tn (k : String) := (totals.getObjValAs? Nat k).toOption
  let realised : List String := (completed.filter fun (r : Run) => r.cleanRun && r.nClean > 0).map fun (r : Run) => r.id
  let claimed : Nat := runs.foldl (fun a (r : Run) => a + r.nClean) 0
  let pairs : List (Nat × Nat) := st.contentHashes.toList
  let failN (n : String) := (st.fails.getD n (0, #[])).1
  let failEx (n : String) := firstIds (st.fails.getD n (0, #[])).2.toList
  let counter (name basis : String) : Result :=
    check name basis (failN name = 0) (fun _ => s!"{failN name} failures: {failEx name}")
  let results := [
    check "index_identity" "index schema 2, certificate model cert-2, backend = interchange backend" (idxOk = true),
    check "run_ids_unique" "List.Nodup of completed run ids" (List.Nodup (completed.map fun (r : Run) => r.id : List String)),
    check "sample_files_exact" "samples/ holds exactly one file per completed run, all referenced by index.json"
      (filesExact = true) (fun _ => s!"present {present.length}, referenced {referenced.length}, runs {completed.length}"),
    counter "records_parse" "every record and certificate decodes (fail closed)",
    counter "sample_identity" "schema 2; sample_id = run ++ \":\" ++ pad5 step, step = line index (< 10^5), run and split match; with run_ids_unique the sample ids are distinct",
    counter "cert_model_and_constants" "certificate model cert-2; constants satisfy Cert.constsOK",
    counter "cert_decisions_recomputed" "Cert.decide on the Lean-quantised policy observation = recorded kind, a, b1, b0, B0, certified (every certificate)",
    counter "cert_presence_matches_phase" "certificate present iff expert phase",
    counter "clean_samples_certified" "every clean sample: verified certificate of kind greatest_safe or min_violation, decision window, clean run, no exclusion reasons",
    check "clean_ids_equal_checked_ids" "list of clean sample ids = list of clean ids with a verified certified label; non-empty when the index claims clean samples"
      (st.cleanIds.toList = st.verifiedCleanIds.toList ∧ (claimed > 0 → st.verifiedCleanIds.size > 0))
      (fun _ => s!"clean {st.cleanIds.size}, verified {st.verifiedCleanIds.size}, claimed {claimed}"),
    counter "run_counts_match_index" "per run: samples, clean, certified clean = index counts",
    check "totals_match_index" "index totals = counts of the records read"
      (tn "samples" = some st.samples ∧ tn "clean_samples" = some st.clean ∧ tn "clean_certified_samples" = some st.cleanVerified)
      (fun _ => s!"index {tn "samples"}/{tn "clean_samples"}/{tn "clean_certified_samples"}, read {st.samples}/{st.clean}/{st.cleanVerified}"),
    check "realised_clean_matches_index" "interchange realisedClean = clean runs of index.json" (d.realisedClean = realised),
    check "realised_scenarios_accepted" "membership" (∀ i ∈ d.realisedClean, ∃ s ∈ accepted d.scenarios, s.id = i),
    check "realised_coverage_exact" "Decidable (CoverageExact) over scenarios with a clean run"
      (CoverageExact ((accepted d.scenarios).filter fun s => s.id ∈ d.realisedClean) (requiredCells btoSpec d.strength) d.realisedMissing),
    check "clean_content_hashes_match_interchange" "content hashes read from the samples = interchange cleanSamples"
      (sortPairs pairs = d.cleanSamples),
    check "clean_samples_no_cross_split_duplicates" "adjCheck (sortPairs l); noCrossSplit_of_check"
      (adjCheck (sortPairs pairs) = true),
    check "mirror_transform_matches_model" "mirrorObs / mirrorAct equality on exported samples"
      (∀ m ∈ d.mirrorChecks, mirrorObs m.obs = m.mirroredObs ∧ mirrorAct m.act = m.mirroredAct)
      (fun _ => firstIds ((d.mirrorChecks.filter fun m => !decide (mirrorObs m.obs = m.mirroredObs ∧ mirrorAct m.act = m.mirroredAct)).map (·.sampleId))) ]
  let stats := Json.mkObj [("runs_completed", completed.length), ("samples_read", st.samples), ("clean_samples", st.clean),
    ("certificates_recomputed", st.certChecked), ("certificate_mismatches", st.certMismatch),
    ("certified_claims", st.certifiedClaims), ("clean_certified_verified", st.cleanVerified),
    ("clean_greatest_safe", st.greatestSafe), ("clean_min_violation", st.minViolation),
    ("mirror_checks", d.mirrorChecks.length)]
  return (results, stats, st.witnesses, completed.toArray.map (·.consts))

/-! ## Per-sample kernel witnesses -/

def intLit (x : Int) : String := s!"({x})"
def listLit (l : List Int) : String := "[" ++ ", ".intercalate (l.map intLit) ++ "]"
def optLit : Option Int → String
  | none => "none"
  | some x => s!"(some {intLit x})"
def qLit (q : Cert.Q) : String :=
  s!"\{ v0 := {intLit q.v0}, s := {listLit q.s}, k := {listLit q.k}, wl := {listLit q.wl}, wr := {listLit q.wr}, " ++
  s!"obstacles := [{", ".intercalate (q.obstacles.map listLit)}] }"
def decLit (d : Cert.Decision) : String := s!"⟨{d.kind}, {intLit d.a}, {optLit d.b1}, {optLit d.b0}, {optLit d.B0}⟩"
def constsLit (c : Cert.Consts) : String :=
  s!"\{ a_lat := {intLit c.a_lat}, a_acc := {intLit c.a_acc}, a_brk := {intLit c.a_brk}, a_max := {intLit c.a_max}, " ++
  s!"v_max := {intLit c.v_max}, r_h := {intLit c.r_h}, hold := {intLit c.hold}, ramp_in := {intLit c.ramp_in}, " ++
  s!"ramp_out := {intLit c.ramp_out}, r_sw_m := {intLit c.r_sw_m}, veh_width := {intLit c.veh_width} }"

/-- Write `Witness_<k>.lean` files: one theorem `Cert.decide c q = d` per clean sample, with `q`
quantised by this checker and `d` the decision RECORDED in the sample; `by decide` makes the
kernel evaluate `Cert.decide`. Returns the index (file, theorem, sample id). -/
def writeWitnesses (out : System.FilePath) (ws : Array (String × Nat × Cert.Q × Cert.Decision))
    (consts : Array (Option Cert.Consts)) (chunk : Nat := 60) : IO Json := do
  IO.FS.createDirAll out
  let mut idx : Array Json := #[]
  let mut k := 0
  let mut start := 0
  while start < ws.size do
    let part := ws.extract start (start + chunk)
    let name := s!"Witness_{k}"
    let runs := (part.toList.map fun w => w.2.1).eraseDups
    let mut src := "import Corpus\nset_option maxRecDepth 100000\nopen Corpus.Cert\nnamespace " ++ name ++ "\n"
    for ri in runs do
      match consts[ri]?.join with
      | some c => src := src ++ s!"def c{ri} : Consts := {constsLit c}\n"
      | none => pure ()
    let mut n := start
    for (sid, ri, q, d) in part do
      src := src ++ s!"/-- {sid} -/\ntheorem s{n} : decide c{ri} {qLit q} = {decLit d} := by decide\n#print axioms s{n}\n"
      idx := idx.push (Json.mkObj [("file", name ++ ".lean"), ("theorem", s!"{name}.s{n}"), ("sample_id", sid)])
      n := n + 1
    IO.FS.writeFile (out / (name ++ ".lean")) (src ++ "end " ++ name ++ "\n")
    start := start + chunk
    k := k + 1
  return Json.arr idx

def main (args : List String) : IO UInt32 := do
  let (witnessDir, args) := match args with
    | "--witness" :: w :: rest => (some w, rest)
    | _ => (none, args)
  match args with
  | [] =>
    IO.eprintln "usage: corpuscheck [--witness OUT_DIR] INTERCHANGE.json [CORPUS_DIR]"
    return 2
  | inter :: rest =>
    let d ← readJson Interchange inter
    let (cres, cstats, ws, consts) ← match rest.head? with
      | some dir => corpusChecks d dir
      | none => pure ([], Json.null, #[], #[])
    let mres := manifestChecks d
    let corpusMissing := d.hasCorpus && rest.head?.isNone
    let results := mres ++ cres ++
      (if corpusMissing then [check "corpus_dir_given" "interchange declares a corpus" False] else [])
    let ok := results.all (·.ok)
    let witness ← match witnessDir with
      | some w => do
        let idx ← writeWitnesses w ws consts
        IO.FS.writeFile (System.FilePath.mk w / "witness_index.json") idx.compress
        pure (Json.mkObj [("dir", w), ("theorems", ws.size)])
      | none => pure Json.null
    let stats := Json.mkObj [
      ("scenarios", (accepted d.scenarios).length), ("draws", d.scenarios.length), ("pairs", d.pairs.length),
      ("required_cells", d.requiredCells.length), ("corpus", cstats)]
    let report := Json.mkObj [("checker", "corpuscheck"), ("lean_version", Lean.versionString), ("ok", ok),
      ("backend", d.backend),
      ("verification_method", "compiled Lean executable: Lean JSON parser + exact decimal quantisation (qfix) + Corpus.Cert.decide evaluated by the Lean runtime on every sample; the theorems about Cert.decide are kernel-checked at lake build"),
      ("witness", witness), ("stats", stats), ("results", Json.arr (results.map Result.toJson).toArray)]
    IO.println report.pretty
    return if ok then 0 else 1
