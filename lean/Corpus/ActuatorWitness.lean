import Corpus.Actuator

namespace Corpus.Actuator

private def witnessConsts : ActConsts :=
  { lon :=
      { a_lat := 1000000, a_acc := 1000, a_brk := 1000, v_max := 100,
        r_h := 100, hold := 1000, veh_width := 0 }
    margin := 0
    step := 1000
    max_steer := 1000
    steer_rate_urad_s := 1000000
    steer_tau_us := 1000000
    wheelbase_mm := 1 }

private def witnessQ : ActQ :=
  { base :=
      { s := [0, 1000]
        k := [0, 0]
        wl := [100000, 100000]
        wr := [100000, 100000]
        obstacles := [] }
    lat := 0 }

private def witnessStart : AState :=
  { station := 0, d := 0, psi := 0, b := 1, delta := 0 }

private def witnessNext : AState :=
  { station := 1, d := 0, psi := 0, b := 1, delta := 0 }

private def witnessModel : ActModel :=
  { consts := witnessConsts
    q := witnessQ
    offsets := [0]
    psis := [0]
    speeds := [1]
    deltas := [0]
    steer_inputs := [0]
    acc_inputs := [0]
    horizon := 1
    start := witnessStart
    terminal_distance := 0
    understeer_num := 0
    understeer_den := 1
    terminal_min_b := 0
    terminal_max_psi := 0
    terminal_max_delta := 0
    ref_d := []
    ref_psi := [] }

private def witnessCert : Cert :=
  { inst := witnessModel
    roots := [[1, 1]]
    states := [[witnessStart], [witnessNext]]
    values := [[some 1000000000000], [some 0]]
    plan := [witnessStart, witnessNext]
    planActions := [[0, 0]]
    firstAction := some [0, 0]
    claimed := some 1000000000000
    planCost := some 1000000000000 }

theorem witness_check : Corpus.Actuator.checkCert witnessCert = true := by
  decide

#print axioms witness_check

end Corpus.Actuator
