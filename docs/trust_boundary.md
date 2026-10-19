# Trust boundary

What the Lean results do **not** cover, stated as narrowly as possible. Items marked
*predicate* appear as named hypotheses of a theorem (their consequences are proved); items
marked *trust* are outside every theorem.

## Toolchain and execution semantics (trust)

- Lean 4.35.0 kernel and the standard axioms `propext`, `Classical.choice`, `Quot.sound`.
- The **compiled** corpus checker (`corpuscheck`): Lean compiler/runtime and Lean's JSON parser.
  Per-sample kernel witnesses re-establish each recorded decision in the kernel.
- CPython semantics of the Python implementations (`autocross/datagen/*.py`,
  `teams/certline.py`). Python ⇄ Lean agreement is checked on generated inputs (kernel `decide`
  witnesses, corpus recomputation); no theorem quantifies over arbitrary Python executions.
- IEEE-754 binary64 in the controller: the racing-line QP, `np.gradient` curvature, path
  geometry `cl + n·off`, path projection and tracking trigonometry, delay roll-out. The safety
  argument relies only on exact float `min`/`max` (the final interval clamp) and on integer
  data below 2⁵³ for the projection; the curvature estimate itself is not verified.

## Modelling predicates (predicate)

| predicate | where | meaning |
|---|---|---|
| `clearanceDomain` | `SteerProofs` | well-formed input with pairwise-disjoint hold windows, even vehicle width, `0 < ramp_in`, nonempty corridors |
| `holdTargetsOK` | `SteerClear` | obstacle inside the road at its nearest preview station; its target offset fits the corridor at every hold station |
| `noTies` | `SteerShape` | no obstacle has equal left/right gaps (mirror equivariance) |
| `RealisedWithin` | `Closed` | the realised next-station speed² is in `[0, F₁]` (actuator never overshoots the certified end point) |
| `PreviewConsistent` | `Closed` | the next preview is the current one shifted by one station, continued inside the declared road domain (caps `≥ a_lat·r_h`) |
| `NextValid` | `Closed` | the next observation passes the input guard and the gap check |
| `DelayedEquation`, `SmallGain` | `Closed` | first-order pure-pursuit lateral-error model with `d`-step delay, `β = P/D = v·dt/l_d`, bounded forcing `F` (path forcing + disturbance + sensor error·gain + bias residual + rounding) and the small-gain inequality |
| `CurvatureSafe`, `LateralAccelSafe` | `Closed` | explicit inequalities keeping steering unsaturated and lateral acceleration in budget under the error bound |
| `InDomainContinuation` | `CertProofs`/`Cert` | unseen road beyond the preview has caps `≥ a_lat·r_h` |
| true curvature `≤ pathCurv` | `Joint.pathCap_physical_sound` | the driven path's curvature is covered by the declared bound |

## Physical and environmental items (trust)

- Tyre/grip budgets (`lat_frac`, `acc_frac`, `brk_frac`, `mu`), `max_steer`, wheelbase and the
  derived integer constants are **declared**, not identified; no theorem claims they are
  physically achievable.
- The declared road domain (`DOMAIN_R_MIN_M = 6.0`, `DOMAIN_Y_MAX_M = 3.25`, `R_SW = 15`) is a
  measurement over `gen-track` outputs, not a proof.
- The joint DP's time table (`toyTime` or any `time` argument) is a declared cost, not seconds.
- An obstacle's longitudinal footprint, inflated by the car's half-length, lies inside
  `[d − hold, d + hold]`: clearance is certified only there (corridor elsewhere).
- Perception: obstacle extents and road widths in the observation are correct; the obstacle
  memory (`_remember`: matching tolerance, 3 m drop-out, side chosen once) is correct.
- Vehicle dynamics: the bicycle backend and CARLA are **not** shown to satisfy
  `DelayedEquation`, and the runtime tracker (second-order path-error feedback) is not the
  first-order model of `Closed`; slip, saturation, load transfer, delay variation and sensor
  outliers are not modelled. The sidestep speed gate (`v² ≤ a_lat·D²/(4Δy)`), the braking budget
  below the friction box (`brk_frac = 0.50`) and the tracker gains are uncertified engineering
  reserves chosen by bicycle-backend measurement.

## Uncertified runtime modes (explicit)

- Controller mode 4 (centreline) and any station whose clearance interval is empty
  (`debug.cert_clear = 0`, `debug.uncertified_stations > 0`).
- `KIND_NO_GAP` / `KIND_MIN_VIOLATION` braking branches: certified as *braking decisions*
  (min-violation is exact infeasibility; full braking is pointwise least), not as collision-free.
- Recovery guard (`recover_lat`/`recover_head`): a Stanley-style steer toward the centreline,
  not covered by any theorem.

## Unresolved obligations

See `docs/proof_closure_report.md` § "Remaining obligations" for the complete list with the
failed approach or missing ingredient for each.

## Optimal teacher trust boundary

The proof-carrying teacher adds a finite, integer model; it does not remove the
following trust boundaries:

- **Quantisation.** The teacher consumes the quantised preview (`s`, curvature,
  widths, obstacle triples, and current speed), not raw continuous perception.
  The quantisation implementation and the correspondence between a Python
  observation and the intended physical observation remain outside the teacher
  optimality theorem.
- **Integer curvature model.** Curvature demand is the declared integer road
  curvature plus a ceiling of the integer second-offset difference. It is not a
  proof that a continuous interpolant, driven vehicle, or tracked path has that
  curvature bound.
- **Obstacle inflation and hold.** The lateral obstacle extents are inflated by
  the declared vehicle half-width and margin, and the obstacle constraint is
  applied only in the declared longitudinal hold window. Correct obstacle
  extents, hold-window coverage, and the adequacy of this inflation are model
  choices.
- **Declared grids, horizon, and terminal cap.** Optimality is over the
  declared offset grid, speed-squared grid, finite preview horizon, and terminal
  cap `min(v_max², a_lat·r_h)`. The cap is the only information supplied beyond
  the preview; it is not a forecast or a whole-course boundary condition.
- **Objective quantisation.** The cost is the sum of integer ceiling charges
  using floored integer speeds. `Corpus.Teacher` proves this declared objective,
  not a physical stopwatch time or an unconstrained continuous-time minimum.
- **Lean kernel and axioms.** The formal result relies on the Lean 4.35.0
  kernel and the standard axioms audited for the corpus (`propext`,
  `Classical.choice`, and `Quot.sound`). The theorem is about the definitions
  in `Corpus.Teacher`.
- **Compiled `teachercheck`.** The executable parses JSON and evaluates
  `Corpus.Teacher.checkCert`; its exit status and output are trusted as a
  compiled Lean/JSON execution. The kernel theorems establish what an accepted
  `CheckCert` means; they do not prove that an arbitrary executable invocation
  was performed or that a file was not changed afterward.
- **Python semantics.** Model construction, sparse DP, certificate serialization,
  and the Python checker run under CPython semantics. Python/Lean parity is
  checked on generated certificates; no Lean theorem quantifies over arbitrary
  Python executions.
- **Tracking and actuation.** The teacher certificate does not cover controller
  tracking error, steering or pedal actuation, delay, saturation, perception
  updates, or collision-free execution of the exported plan.
- **Finite optimum versus physical traversal time.** A finite-model optimum is
  an optimum for the declared quantised objective and constraints. It is not a
  claim that a physical vehicle attains that cost, that CARLA executes the plan,
  or that the plan is optimal on a continuous vehicle or a full course.

- **Offline versus realtime declaration.** Offline exact mode uses the requested
  full finite declaration. Realtime mode uses a reduced horizon/grid and is exact
  only for that reduced declaration; it is not the full-model optimum.
- **Recovery and student tracking layers.** Centreline recovery is an explicit
  uncertified tracking mode. The student controller's startup speed floor and
  its copied pursuit and pedal formulas are additional uncertified tracking and
  actuation layers; they do not inherit the teacher certificate.
- **Simulator metrics.** Cone-hit, off-course, completion, and timing metrics
  are observations from the bicycle simulator runs. They are not safety proofs,
  CARLA results, or proofs of continuous-model optimality.
