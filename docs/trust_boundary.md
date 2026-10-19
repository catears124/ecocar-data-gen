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
