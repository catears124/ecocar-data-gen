import Corpus.Timed

set_option maxRecDepth 1000000

namespace Corpus.Timed

private def witnessConsts : TimedConsts :=
  {
    A_BT := 8000
    A_LT := 8000
    CD := 330
    CF := 130919
    CR := 122473
    FB := 17500
    FBF := 600
    FBMAX := 16000
    FD := 9900
    FDF := 400
    FRR := 217
    G := 9810
    HCG := 500
    HL := 2395
    HW := 1230
    H_US := 25000
    IZ := 4147
    KSTEER := 233000
    L := 3000
    LAG_DEN := 100000
    LAG_NUM := 65334
    LF := 1450
    LR := 1550
    MASS := 1845
    MAX_STEER := 610865
    MU := 870
    MU_T := 970
    PQ := 1
    PW := 200000
    RMIN := 6500
    SLEW := 50
    STEP := 1000
    TICKS := 2
    T_TAIL := 500
    VMAX := 40000
    V_FLOOR := 1000
    V_KIN := 3000
  }

private def witnessInstance : Instance :=
  { consts := witnessConsts
    k := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    wl := [3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000]
    wr := [3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000]
    obstacles := []
    start := { s := 0, d := 0, psi := 0, v := 9000, vy := 0, r := 0, sigma := 0, delta := 0 }
    p_prev := 0
    blocks := [
    { steps := 1, steer := [-100, 0, 100], pedal := [-1000, 0, 1000] },
    { steps := 2, steer := [-200, 0, 200], pedal := [-1000, 0, 1000] }
    ] }

private def witnessCert : Cert :=
  { schema := "autocross.timed.certificate"
    schemaVersion := 1
    inst := witnessInstance
    instanceSha256 := "dc1bbc4da9e9652c0329bbf6ae797aaac7631383c27279aa89a0572d5140d6b6"
    plan := [5, 5]
    claimed := (0, -6306, 1000)
    first := (0, 1000)
    objLex := ["violation", "neg_progress", "effort"]
    objTieBreak := "first plan in enumeration order" }

theorem witness_check : Corpus.Timed.checkCert witnessCert = true := by
  decide

#print axioms witness_check

end Corpus.Timed
