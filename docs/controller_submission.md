# VeriDrive controller: `teams/certline.py`

## Reviewer’s map

The controller is a single `CertifiedRacingLine` subclass.  It uses the live road preview only; the
certificate-bearing decisions are integer/fixed-point decisions, while path smoothing, tracking, and
actuation are deliberately outside the certificate.

```text
road observation
    │
    ├─ decide_lateral: certified steer-1 pass side, offsets, curvature demand
    │       │
    │       └─ _remember obstacle state
    │              └─ _spare_boxes / _plan_boxes: clearance interval at every station
    │                     └─ _racing_line: box-constrained least-squares minimum-curvature path
    │                            │
    │                            └─ chosen-path curvature bound
    │                               (|k_road| + 1000|Δ²offset|, window maximum, floor)
    │                                  └─ decide_long: certified greatest-safe acceleration
    │                                         │
    │                                         └─ tracker: pure pursuit + curvature feed-forward
    │                                            + yaw-rate damping + grip clamp + steer-rate limit
    │                                               └─ pedal map: non-overlapping pedals,
    │                                                  side-slip easing, friction-circle clamp
    └─ invalid/exception paths brake rather than publish an invalid command
```

`decide_lateral` is the reference steer-1 decision.  It supplies the obstacle pass side and a
corridor-clamped offset profile.  `_spare_boxes` intersects that corridor with each remembered
obstacle’s clearance half-space; `_plan_boxes` optionally shrinks the resulting intervals by
`line_shrink`.  `_racing_line` minimizes the squared second difference of the *path points*
`P_i = centerline_i + normal_i * offset_i`, subject to those per-station boxes and the current
lateral offset at station zero.  Thus this is a small box-constrained quadratic program, not a
second independent obstacle planner.

The speed plan is computed from the path that will actually be published.  The runtime forms the
road curvature plus offset curvature, uses the maximum over the configured `kappa_window`, and
applies `kappa_floor`; the corresponding certified integer curvature input has the model form
`|k_road| + 1000|Δ²offset|`.  `decide_long` then applies the grip speed cap, obstacle gap-speed cap,
acceleration/braking reachability, and the terminal-horizon cap.  The tracker follows the same
published path, and the pedal map leaves lateral grip available rather than commanding throttle
and brake together.

## Why the certificate follows the racing line

The certificate chain is:

1. `decide_lateral` chooses a pass side and an offset profile under the steer-1 corridor.
2. `_spare_boxes` computes `lo <= d <= hi` at each station.  These bounds are the corridor walls
   plus the obstacle clearance half-spaces used by steer-1’s `clearAt` predicate.  Staying in the
   interval is therefore exactly the clearance hypothesis (`clearAt >= 0`), rather than merely
   being visually between the cones.
3. `_plan_boxes` preserves those intervals (and normally makes them stricter by `line_shrink`).
   `_racing_line` constrains every solved offset to them and pins the first offset to the car’s
   current lateral position.  Its output is clipped to the boxes before use.
4. The speed curvature is recomputed from that selected path, not copied from the unsmoothed
   steer-1 profile.  `decide_long` receives the resulting caps and applies its terminal-horizon
   bound (`b_n <= a_lat * r_h`) to the same planned path.

Consequently, smoothing changes the path only inside the interval on which steer-1’s clearance
predicate holds; it does not silently invalidate the hypothesis used by the terminal-horizon
bound.  The certificate covers the integer lateral/longitudinal model decisions, not the physical
tracking error, steering actuator, or CARLA/bicycle dynamics.

### Plan selection and safety ordering

The selection order in `CertifiedRacingLine._step` is intentional:

1. Use the `_racing_line` result when it is finite and `_plausible`.
2. Otherwise rate-limit and re-project the line (or the certified offsets) with `_project_plan`,
   clamping it back into the certified intervals after each pass; use it if plausible.
3. Otherwise use the certified `decide_lateral` offsets as-is (with the controller’s offset cap),
   if that path is plausible.
4. If even that path is not plausible, publish the bare centerline as the final emergency
   drivability fallback.

For the first three choices, the interval is always re-applied: **clearance safety outranks
smoothness, and smoothness outranks the published-path metric**.  The last centerline branch is an
explicit escape hatch for an undrivable preview; it should be read as a recovery behavior, not as a
new proof that an obstacle clearance interval is satisfied.  The selected path is also saved as
`prev_line` when appropriate, so the next QP can favor continuity without overriding its boxes.

## Robustness and actuation layers

These layers are intentionally inactive in nominal conditions (zero delay, zero steering bias,
true grip, and no obstacle dropout):

- **Actuation-delay compensation:** when queued commands exist,
  `_predict_pose` rolls the vehicle forward and transforms the path into the predicted body frame.
  With `actuation_delay_steps == 0`, this branch is a no-op.
- **Steering-bias estimation:** after the actuator settles, the controller compares the measured
  steering with the command sent earlier, estimates `bias`, and subtracts it.  An unbiased nominal
  vehicle leaves `bias` at zero.
- **Online grip adaptation:** `_adapt_grip` detects sustained yaw response below the kinematic
  response while cornering hard, lowers `mu_hat`, and scales the declared budgets.  At the
  nominal vehicle grip there is no saturation, so `mu_hat` remains at the vehicle value.
- **Obstacle memory:** `_remember` keeps a detected obstacle in course coordinates and keeps its
  chosen side while the road sensor’s approximately 3 m rear dropout could otherwise make the
  planner pull back toward the obstacle.  With no obstacle there is nothing to remember.
- **Tracker grip clamp and rate limit:** steering is limited by a lateral-grip bound and by the
  actuator slew rate.  These are protection layers, not certificate proofs.
- **Pedal protection:** `_pedals` never overlaps throttle and brake, reduces either pedal when
  measured side slip is large, and reduces throttle according to measured lateral acceleration
  (a friction-circle reserve).

## Known weaknesses and trust boundaries

- The steer-1 combination rule in `_raw_off` keeps the first strictly largest obstacle weight.
  Two closely spaced opposing obstacles can therefore switch the winning supplier abruptly and
  produce a lateral fold.  The controller code documents this mechanism in `_project_plan`; that
  function rate-limits and re-clamps the result, but a genuinely pinched clearance interval can
  still force a rate change.  The controller invariant test includes the opposing-obstacle case
  `(10.0, 1.1, 2.0)` followed by `(22.0, -2.2, -1.2)` in
  `tests/test_controller_certline.py`.
- The fallback chain above is conservative about certified boxes until the final centerline
  escape hatch.  A preview that makes every certified candidate geometrically implausible can
  therefore trade the clearance assertion for a drivable recovery path.
- `lat_frac`, `acc_frac`, `brk_frac`, and `horizon_r` are declared model budgets.  They are not
  measurements of the tyres.  The grip adaptation layer can react to observed saturation, but it
  does not turn the original certificate into a tyre-identification result.
- The tracker, pedal mapping, actuator delay model, bias estimate, and grip adaptation are not
  Lean-certified.  The certification claim is limited to the vendored `decide_lateral` /
  `decide_long` model decisions and their stated hypotheses.

## Verification recipe

Run from the repository root without changing the controller path:

```powershell
python -m autocross selftest -c teams/certline.py
python -m autocross selftest -c teams/certline.py --carla
python -m pytest -q
python -m pytest -q tests/test_controller_certline.py
```

- The first command runs the controller on every public course with the official settings on the
  bicycle backend, reporting finishes, penalties, step time, technical points, and sandbox-rule
  violations.
- The second command runs every public course on the bicycle backend again and additionally on CARLA.  It requires the CARLA client and a running CARLA server; without those, it cannot establish the CARLA result.
- `python -m pytest -q` runs the repository’s complete Python test suite, including certificate,
  corpus, backend, and controller checks.
- `python -m pytest -q tests/test_controller_certline.py` specifically checks clean public-course
  runs, the obstacle course and closed course, absence of controller errors, the published-path
  clearance invariant, and agreement between the chosen-path curvature and the longitudinal
  decision.  These tests are not a substitute for the CARLA self-test.

## Configuration knobs (`DEFAULTS`)

The controller accepts only keys present in `DEFAULTS`; unknown controller-config keys are ignored.
Defaults below are the declarations in `teams/certline.py`.

### Certified budgets and lateral plan

| Key | Default | Meaning |
|---|---:|---|
| `lat_frac` | `BOX_FRAC` | Fraction of `mu*g` assigned to the certified lateral budget. |
| `acc_frac` | `BOX_FRAC` | Fraction of `mu*g` assigned to certified acceleration, bounded by vehicle maximum acceleration. |
| `brk_frac` | `BOX_FRAC` | Fraction of `mu*g` assigned to certified braking, bounded by vehicle maximum deceleration. |
| `horizon_r` | `DOMAIN_HORIZON_MM / 1000.0` | Terminal-horizon radius used by the longitudinal certificate. |
| `gap_speed` | `1.0` | Enables the declared obstacle gap-speed rule. |
| `gap_base` | `5.0` | Base speed in the gap-speed rule, in m/s. |
| `gap_per_m` | `6.0` | Additional gap-rule speed per metre of free slack. |
| `hold` | `3.0` | Obstacle half-length of the lateral hold window, in metres. |
| `ramp_in` | `18.0` | Lateral sidestep ramp-in length, in metres. |
| `ramp_out` | `15.0` | Lateral sidestep ramp-out length, in metres. |
| `margin` | `0.30` | Certified cone/body clearance margin, in metres. |
| `gap_margin` | `0.25` | Extra free-space margin on each side for a gap to count, in metres. |
| `max_offset` | `1.50` | Controller cap on absolute lateral offset, in metres. |

### Chosen-path speed caps

| Key | Default | Meaning |
|---|---:|---|
| `kappa_floor` | `0.0020` | Curvature floor used for speed caps, in 1/m. |
| `kappa_window` | `1` | Number of stations on each side included in the cap maximum. |
| `kappa_safety` | `1.00` | Multiplier on the chosen-path curvature bound. |
| `cap_margin` | `1.00` | Scale applied to the speed-cap decision; `1.0` is the declared budget. |

### Tracking and recovery

| Key | Default | Meaning |
|---|---:|---|
| `k_ff` | `0.35` | Gain on curvature feed-forward steering. |
| `k_pp` | `1.3` | Pure-pursuit gain. |
| `k_psi` | `0.40` | Look-ahead path-tangent heading-error gain. |
| `k_r` | `0.30` | Yaw-rate damping gain, in seconds. |
| `k_v` | `0.50` | Speed-error gain added to certified acceleration, in 1/s. |
| `ld_base` | `2.0` | Base pure-pursuit look-ahead, in metres. |
| `ld_gain` | `0.30` | Look-ahead increase per m/s. |
| `ld_min` | `3.0` | Minimum look-ahead, in metres. |
| `ld_max` | `8.0` | Maximum look-ahead, in metres. |
| `ld_ff` | `3.0` | Look-ahead for feed-forward curvature and yaw-rate reference, in metres. |
| `grip_clamp` | `1.00` | Fraction of `mu*g` permitted by the steering grip clamp. |
| `recover_lat` | `0.70` | Lateral-offset fraction that engages recovery. |
| `recover_head` | `0.30` | Heading-error threshold that engages recovery, in radians. |
| `k_rec_head` | `1.40` | Recovery steering gain on heading error. |
| `k_rec_lat` | `0.20` | Recovery steering gain on lateral offset. |
| `brake_deadband` | `0.2` | Deceleration left to drag before the brake pedal is applied, in m/s². |
| `recover_brake` | `0.0` | Brake pedal command during recovery. |
| `recover_throttle` | `0.25` | Throttle held during low-speed recovery. |
| `recover_vmin` | `4.0` | Speed below which recovery may hold `recover_throttle`, in m/s. |

### Racing-line solver

| Key | Default | Meaning |
|---|---:|---|
| `line_on` | `1.0` | Enables the minimum-curvature line; `0` disables the QP and starts the fallback from the certified plan. |
| `line_shrink` | `0.15` | Shrinks each usable certified interval inward, in metres. |
| `line_w_prev` | `3.0` | Stickiness weight for the previous published line. |
| `line_w_center` | `0.002` | Weak centerline regularization weight. |
| `line_ramp` | `0.35` | Maximum fallback offset change per preview station, in metres of offset. |
| `line_iters` | `8` | Active-set iterations in `_box_qp`. |
| `line_eps` | `1e-6` | Tikhonov regularization for the free QP block. |

### Obstacle memory and robustness

| Key | Default | Meaning |
|---|---:|---|
| `obs_pre` | `4.5` | Remember an obstacle this far before its station, in metres. |
| `obs_post` | `4.5` | Retain an obstacle this far after its station, in metres. |
| `delay_comp` | `1.0` | Enables actuation-delay pose prediction when delayed commands exist. |
| `bias_est` | `1.0` | Enables steering-bias estimation when settled-history conditions hold. |
| `grip_adapt` | `1.0` | Enables online grip-budget adaptation. |
| `sat_ratio` | `0.68` | Yaw-rate/kinematic-yaw-rate ratio below which saturation is suspected. |
| `sat_ay` | `4.5` | Minimum lateral acceleration for saturation detection, in m/s². |
| `bias_gain` | `0.15` | Steering-bias estimator gain. |
| `bias_lim` | `0.15` | Absolute steering-bias limit. |
| `beta_ease` | `1.0` | Enables side-slip pedal easing. |
| `beta_ease_lim` | `0.05` | Side-slip angle threshold for pedal easing, in radians. |
