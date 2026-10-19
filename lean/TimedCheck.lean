/-
`timedcheck CERT.json`

Parses a proof-carrying ``timed-1`` certificate and runs the compiled Lean
checker `Corpus.Timed.checkCert`.  Soundness is the kernel theorem
`Corpus.Timed.checkCert_sound`; this executable only evaluates it.  Prints a
JSON report and exits 0 iff the certificate is accepted.
-/
import Corpus
import Corpus.Timed
import Lean.Data.Json

open Lean

def tripleJson (t : Int × Int × Int) : Json :=
  Json.arr #[toJson t.1, toJson t.2.1, toJson t.2.2]

def main (args : List String) : IO UInt32 := do
  match args with
  | [path] =>
      let text ← IO.FS.readFile path
      match Json.parse text with
      | .error e =>
          IO.println (Json.mkObj [("ok", false), ("error", "json: " ++ e)]).compress
          return 1
      | .ok j =>
          match (fromJson? j : Except String Corpus.Timed.Cert) with
          | .error e =>
              IO.println (Json.mkObj [("ok", false), ("error", "schema: " ++ e)]).compress
              return 1
          | .ok cert =>
              let ok : Bool := Corpus.Timed.checkCert cert
              let obj := Json.mkObj [
                ("ok", toJson ok),
                ("claimed", tripleJson cert.claimed),
                ("plans", toJson (Corpus.Timed.allPlans cert.inst).length),
                ("model", "Corpus.Timed")
              ]
              IO.println obj.compress
              return (if ok then 0 else 1)
  | _ =>
      IO.println "usage: timedcheck CERT.json"
      return 2
