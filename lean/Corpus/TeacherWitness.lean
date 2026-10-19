import Corpus.Teacher

namespace Corpus.Teacher

open Corpus

private def witnessConsts : Steer.Consts := Joint.toyC

private def witnessQ : Steer.Q :=
  { base :=
      { v0 := 1
        s := [0, 1000]
        k := [0, 0]
        wl := [4000, 4000]
        wr := [4000, 4000]
        obstacles := [] }
    lat := 0 }

private def witnessStart : TState :=
  { station := 0, previous := 0, current := 0, b := 1 }

private def witnessNext : TState :=
  { station := 1, previous := 0, current := 0, b := 9 }

private def witnessRoots : List (List Int) := [[1, 1], [4, 2], [9, 3]]

private def witnessCert : Cert :=
  { consts := witnessConsts
    q := witnessQ
    offsets := [0]
    speeds := [1, 4, 9]
    roots := witnessRoots
    horizon := 1
    start := witnessStart
    terminalCap := 9
    states :=
      [ [witnessStart],
        [ { station := 1, previous := 0, current := 0, b := 1 },
          { station := 1, previous := 0, current := 0, b := 4 },
          { station := 1, previous := 0, current := 0, b := 9 } ] ]
    values :=
      [ [some 500000000],
        [some 0, some 0, some 0] ]
    plan := [witnessStart, witnessNext]
    claimed := some 500000000
    planCost := some 500000000 }

theorem witness_check : Corpus.Teacher.checkCert witnessCert = true := by
  decide

#print axioms witness_check

end Corpus.Teacher
