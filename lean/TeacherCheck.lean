/-
`teachercheck CERT.json`

Parses a proof-carrying teacher certificate and runs the compiled Lean checker
`Corpus.Teacher.checkCert`.  Soundness is the kernel theorem
`Corpus.Teacher.checkCert_sound`; this executable only evaluates it.  Prints a
JSON report and exits 0 iff the certificate is accepted.
-/
import Corpus
import Corpus.Teacher
import Lean.Data.Json

open Lean

def main (args : List String) : IO UInt32 := do
  match args with
  | [path] =>
      let text ← IO.FS.readFile path
      match Json.parse text with
      | .error e =>
          IO.println (Json.mkObj [("ok", false), ("error", "json: " ++ e)]).compress
          return 1
      | .ok j =>
          match (fromJson? j : Except String Corpus.Teacher.Cert) with
          | .error e =>
              IO.println (Json.mkObj [("ok", false), ("error", "schema: " ++ e)]).compress
              return 1
          | .ok cert =>
              let ok : Bool := Corpus.Teacher.checkCert cert
              let obj := Json.mkObj [
                ("ok", toJson ok),
                ("claimed", match cert.claimed with | none => Json.null | some v => toJson v),
                ("horizon", toJson cert.horizon),
                ("states", toJson cert.states.length),
                ("model", "Corpus.Teacher")
              ]
              IO.println obj.compress
              return (if ok then 0 else 1)
  | _ =>
      IO.println "usage: teachercheck CERT.json"
      return 2
