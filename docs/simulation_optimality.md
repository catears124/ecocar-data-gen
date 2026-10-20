# Simulation-optimal decisions: exactly what is proven

The request was: *prove that the realtime controller is optimal for the
continuous vehicle given the information and task constraints.*  That statement
is not provable with the artifacts in this repository, and it is not provable
from the available information.  This note states the strongest true statement
that is implemented and machine-checked, and the precise gap to the request.

## 1. The proven statement (simulator, finite candidates, time \(t\))

At a recorded decision time \(t\) the certificate in
`experiments/bench/v3/simcert_160.json` binds:

- the measured simulator state \((x,y,\psi,v_x,v_y,r,\delta)\),
- the vehicle constants and the harness bicycle transition
  (`autocross.backends.bicycle.substep`), two 25 ms ticks per control period,
- the 41-station preview, widths, curvature and obstacle boxes,
- 44 declared command sequences (15 constant steering/pedal plans, 8
  pure-pursuit/curvature-speed closed-loop policies, the baseline command,
  and the previous plan shifted),
- every candidate's integer cost `(violation, -progress, effort)`,
- the selected index, its first command, and the baseline candidate.

The theorem is:

\[
\forall c\in\mathcal C:\quad
J_{\text{sim}}(c^*) \;\le_{\text{lex}}\; J_{\text{sim}}(c),
\qquad c_{\text{baseline}}\in\mathcal C,
\]

where \(J_{\text{sim}}\) is computed by the simulator's own transition
function and the declared cost/constraint code.  It is checked twice:

- `autocross.simcert.certificate.check_certificate` recomputes the rollout and
  the selection in Python;
- the compiled Lean checker `simcheck` verifies the integer cost list, the
  first-minimum property, baseline membership and dominance, and the declared
  epsilon arithmetic.

`simcheck` on the recorded step returns:

```json
{"baseline_cost":[4089,-11761,1093],"candidates":44,"epsilon_milli":1580000,
 "ok":true,"selected":14,"selected_cost":[0,-9862,332]}
```

This is **optimality over the declared finite candidate set in the bicycle
simulator at the recorded time**.  It is not optimality over all continuous
commands, and it is not a claim about CARLA or the physical car.

## 2. The continuous-control gap

The candidate set has a resolution \(\delta=(0.2\) steering, \(0.4\) pedal\()\).
For a continuous command sequence there is a candidate within \(\delta\) in each
coordinate, but the cost is not proved Lipschitz over that box.  The certificate
reports a **measured** first-command sensitivity \(L\) and
\(\varepsilon = L\,\delta\,H\); for the recorded step
\(\varepsilon = 1{,}580{,}000\) cost units (1.58 m of progress-equivalent
cost).  This is a measured, conditional gap, not a proof.  The Lean checker
only verifies the arithmetic \(\varepsilon=L\delta H\); it does not verify
\(L\).

A genuine continuous proof would require all of:

1. a formal continuous-time plant with validated parameters and disturbance
   bounds (the bicycle ODE is available, but tyre/actuator/load-transfer
   parameters are not validated);
2. validated numerical integration or interval arithmetic for the ODE in Lean;
3. a global branch-and-bound/HJB proof over the continuous control set, with
   admissible lower bounds;
4. formal semantics for perception, the 50 ms scheduler and the executed
   command;
5. a proof that the simulator/CARLA implements the formal plant.

None of these exist in the repository.  Without them, any “continuous global
optimum” label would be a false claim.

## 3. Measured cost of using the proof-oriented controller

`teams/sim_opt.py` is the controller that makes the certified
simulator-optimal decision (44 candidates, horizon 10 control periods,
`(violation, -progress, effort)`).  On `practice_basic` it is not competitive:

| controller | time | cones | off-course | technical | step p99 |
|---|---:|---:|---:|---:|---:|
| `teams/line_opt.py` (deployed safe portfolio) | 29.18 s | 0 | 0 | 71.4 | 16.0 ms |
| `teams/sim_opt.py` (simulator-optimal, finite candidates) | 36.71 s | 12 | 0 | 8.0 | 71.2 ms |

The reason is not the certificate: it is the **declared finite horizon and
candidate set**.  A 0.5 s constant/pure-pursuit candidate set is locally
feasible and locally better, while the actual course requires a longer,
feedback-rich plan.  The certificate truthfully proves the best member of that
set; it does not make that set a good driving policy.

The deployed realtime controller therefore remains `teams/line_opt.py`, which
matches `certline` on the five public and four held-out courses.  `sim_opt` is
kept as the proof-oriented entry point and as evidence that a locally exact
simulator optimum is not automatically a competitive controller.

## 4. Reproduce

```bash
python tools/simulation_certificate.py tmp/cert_rec/certline_practice_basic_s0 \
  --step 160 --out tmp/simcert_160.json --check
python - <<'PY'
import subprocess, pathlib
p = pathlib.Path('lean/.lake/build/bin/simcheck.exe').resolve()
print(subprocess.run([str(p), 'tmp/simcert_160.json'],
                     capture_output=True, text=True).stdout)
PY
```

## 5. Honest verdict

- Proven: finite-candidate, simulator-exact optimality at the recorded time,
  with baseline inclusion and an independent Python + Lean check.
- Measured: the proof-oriented controller is slower and less safe than the
  deployed portfolio; the finite candidate set, not the certificate, is the
  limitation.
- Not proven, and not claimed: global optimality over continuous controls, over
  a validated continuous vehicle model, or over the physical/CARLA plant.
- To obtain the requested continuous guarantee, the formal ODE/interval
  verification and global continuous optimization work in §2 must be built
  first.  Until then the strongest true statement is the one above.
