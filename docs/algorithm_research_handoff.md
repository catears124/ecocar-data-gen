# Algorithm research handoff: learning from the certified imitation corpus

For whoever selects the learning algorithm. This file covers what the data is, what is proven
about it, what is only measured, and the constraints the final controller must meet. The full
data specification is in [`datagen.md`](datagen.md). All numbers below come from the `full`
bicycle corpus generated on 2026-10-17 (`datasets/full_bicycle`, manifest content
`115f4db2bf99…`, expert 2.0.0, certificate model `cert-1`). **No CARLA corpus has been generated
yet**: CARLA was not available on the development machine.

## 1. Hard constraints on the final policy (from the harness)

| Constraint | Value | Consequence for the algorithm |
|---|---|---|
| Runtime environment | sandboxed process; **numpy only** (no torch, no onnxruntime, no native code via ctypes), cannot write files or open sockets | the model must run as plain numpy (or pure Python) at inference; weights are read from files next to the controller |
| Compute | 1 CPU core, **50 ms per step** (20 Hz), 1 GB memory, no GPU; `on_start` ≤ 60 s | step budget is generous for small MLPs (~0.05–1 ms) |
| Inputs | `Observation` only: road model within `range_m` (40 m in the event), ego state, applied actuators, raw cones. **No map**; course name withheld | the policy must be causal and map-free; the dataset matches this exactly |
| Scoring | both backends weighted 50/50 (CARLA, bicycle); technical course 60 %; penalties for cones (−2), off-course (−5), missed gates (−10) | robustness on tight technical elements matters more than raw speed |
| Path metric | `Command.path` (planned path) is scored against the actual trajectory 0.5 s later | predicting a path, not just steer, earns points; the dataset has the expert path per sample |

## 2. The data

### 2.1 Scenario family (only one so far)

`brake_turn_obstacle` v1: a straight approach at a set speed, then one constant-radius turn with a
cone obstacle against the inside or outside boundary. The turn and the obstacle are out of sight
at the hand-over point and appear while the expert drives.

| Factor | Bins |
|---|---|
| turn radius | tight 9–14 m, medium 14–22, open 22–35 |
| direction | left, right |
| preview (sight distance) | short 15–25 m, nominal 25–45, long 45–60 |
| approach speed | low 8–12 m/s, mid 12–16, high 16–20 |
| obstacle side | inside, outside (of the turn) |
| free gap beside obstacle | tight 2.8–3.3 m, moderate 3.3–3.9, wide 3.9–4.6 (vehicle is 2.16 m wide) |
| road width | narrow 5.0–5.6 m, standard 5.6–6.8, wide 6.8–8.0 |
| nuisance | turn angle 70–110°, obstacle position 30–60 % along the arc |

**Not covered:** slaloms, chicanes, esses, hairpin sequences, decreasing-radius turns, multiple
obstacles, closed courses, noisy perception. The unseen evaluation courses contain these (see
`tracks/practice_technical.yaml`). A policy trained only on this family **will not cover the
evaluation course**. More families are needed (see §7).

### 2.2 Sizes (`full` profile, bicycle backend)

| | train | val | test | total |
|---|---|---|---|---|
| scenarios (accepted) | 176 | 12 | 24 | 212 |
| clean runs | 150 | 12 | 18 | 180 |
| clean samples (20 Hz) | 23,977 | 1,982 | 2,578 | 28,537 |
| required pairwise cells covered by clean runs | 152 / 154 | 57 / 154 | 71 / 154 | 154 / 154 |

- 72,043 samples were recorded in total. 43,506 are not clean: scripted approach, after the
  decision window, or from an excluded run.
- 32 of 212 runs were excluded: 24 boundary cone hits, 13 obstacle cone hits, 6 off-course,
  4 missed gates (one run can have several reasons). All of them are tight turns and/or narrow
  gaps (see §6.1).
- Samples within one run are strongly correlated, so the **effective sample size is roughly
  the 180 clean runs**, not 28.5 k.
- Throughput: 212 runs in 162 s with 6 workers. The `large` profile (8 in-cell draws per design
  row, about 4× `full`) takes about 10 min.
- Measured cost: the expert takes **0.13 ms/step**; `cert.decide` takes 0.04 ms/step; the
  sandbox round trip is 0.7 ms median, 2.5 ms p99.

### 2.3 Splits

Splits hold out whole parameter regions of (turn radius × preview):
- `val` = (open radius, nominal preview);
- `test` = (medium radius, short preview);
- everything else is `train`.

Lean checks that a held-out region never appears in another split, that each group (a base
scenario plus its contrast variants) and each contrast pair sits in one split, and that no
identical clean sample appears in two splits. Val and test are small (12 and 18 clean runs), so
their metrics are noisy. For model selection use **group-wise cross-validation on `group_id`**
inside train, and keep test untouched.

### 2.4 Record schema (one JSON line per control step: `samples/<scenario_id>.jsonl`)

| Field | Use |
|---|---|
| `observation.road.*` | **primary inputs.** Scalars: `lateral_offset, heading_error, curvature, width_left/right, dist_to_left/right_boundary, range_m, finish_distance` (null = NaN). Arrays at 1 m stations: `preview_s, preview_curvature, preview_width_left/right` (length 16–58, it varies with preview), `preview_centerline / left / right_boundary` (K×2, vehicle frame). `corners[]` (direction, entry/apex/exit distance, min radius, turn angle, fully_visible). `obstacles[]` (distance, lateral_min/max, center, points); 0 or 1 visible in this family |
| `observation.{speed, vx, vy, yaw_rate, ax, ay, steer, throttle, brake}` | ego inputs (steer/throttle/brake = actuator state actually applied) |
| `observation.cones` | raw cone detections (noisy, seeded); the expert does not use them |
| `observation.{x, y, yaw, progress, lap, t, step, timing_started}` | **do not use as inputs**: they identify the course layout, not the situation |
| `expert.certificate.accel_mps2` | **certified longitudinal label** (see §3); `kind` = `greatest_safe` or `min_violation` |
| `expert.steer` | steering label in [−1, 1], + left (uncertified) |
| `expert.path` | planned path, vehicle frame, one point per preview station (uncertified) |
| `expert.throttle / brake / target_speed` | pedal mapping of the certified acceleration (tracking layer, uncertified) |
| `applied.*` | what the simulator applied (steer is slew-limited at 2/s) |
| `oracle.*` | privileged ground truth (distances to the corner and obstacle, phase): **analysis only** |
| `quality.clean`, `quality.exclusion_reasons` | train only on `clean == true` unless you are deliberately using negatives |
| `split`, `group_id`, `scenario_id`, `sample_id`, `content_sha256`, `backend` | bookkeeping; `backend` is `bicycle` for this corpus |

Units are SI; frames are vehicle x forward, y left. Loading:

```python
import json, pathlib
root = pathlib.Path("datasets/full_bicycle")
idx = json.loads((root / "index.json").read_text())
rows = [json.loads(l) for r in idx["scenarios"] if r.get("status") == "completed"
        for l in (root / r["files"]["samples"]["path"]).open() if json.loads(l)["quality"]["clean"]]
```

The preview length varies, so resample onto fixed stations (e.g. 0, 1, …, 60 m) with a validity mask,
or summarise it (curvature at fixed look-aheads, distance to the next corner, its min radius).

### 2.5 Label distribution (clean samples)

- Certificate kinds: 26,216 `greatest_safe` and 2,321 `min_violation` (8 %, full braking
  because the car is already over its braking budget).
- Certified acceleration is nearly bang-bang:
  - 49.5 % at the acceleration budget (+5.2 m/s²);
  - 23.5 % negative, including the −9.0 m/s² `min_violation` samples;
  - percentiles: 5 % −9.0, 25 % +0.15, median +4.1, 75 % +5.2.

  Regression with MSE will blur the switching surface, so consider classification of the
  regime plus regression of the magnitude, or simply compute it exactly (§4).
- Steering: 5th–95th percentile −0.48 … +0.52, saturating at ±1 at the extremes.
- 57 % of clean samples have no obstacle in view and 43 % have one.

### 2.6 Contrast pairs (184)

Each pair changes exactly one factor between base and variant. Lean checks this, and checks that
the factor moves to a different bin. `contrast_report.json` measures whether the expert's action
actually changed, on clean pairs:

| Factor | Evaluated | Actions differ | Declared hypothesis holds |
|---|---|---|---|
| turn radius | 19 | 19 | 19 (tighter → lower min speed) |
| direction (mirror) | 25 | 25 | 25 (steer negated; residual ~1e-16 on bicycle) |
| preview | 19 | 19 | 18 (shorter → slower before the turn) |
| approach speed | 25 | 25 | 21 (faster → more braking) |
| obstacle side | 19 | 19 | 19 (opposite pass side) |
| clearance | 17 | 17 | 17 (tighter → slower beside obstacle) |
| road width | 25 | 25 | no hypothesis |

These are natural inputs for contrastive or consistency losses: penalise a model whose outputs
for a pair don't move in the declared direction. Mirror augmentation is exact for this data:
observation and label transform are in `records.py`, and Lean checks them against its model on
540 exported samples.

## 3. What is proven, exactly

Lean 4.35.0, core library only, under `lean/`. The axiom audit shows only `propext`,
`Classical.choice` and `Quot.sound`; there are no `sorry`s.

### 3.1 Theorems (hold for all inputs)

The model `cert-1` uses squared speeds b_i at the preview stations (1 m apart, integers: mm,
mm/s, 1e-6/m), subject to:
- **Grip:** b_i · κ̄_i ≤ a_lat. κ̄ bounds the planned path's curvature: road curvature, plus
  the obstacle sidestep 6|y|/15², corrected for the offset path.
- **Gap speed:** beside an obstacle, b_i ≤ (5 + 6·slack)².
- **Speed cap:** b_i ≤ 25².
- **Horizon:** b_last ≤ a_lat · 9 m. A 9 m corner might begin just past the visible range.
- **Budgets:** acceleration ≤ 5.2 m/s² and braking ≤ 6.5 m/s².
- **Start:** b_0 = current speed².

| Theorem | Meaning |
|---|---|
| `Cert.fpass_greatest` (+ `le_bpass`) | every profile satisfying these constraints from the current speed is pointwise ≤ the forward–backward profile F |
| `Cert.fpass_feasible` | F itself satisfies the constraints |
| `Cert.greatest_minimises` | F minimises any objective that does not increase when speeds rise (e.g. traversal time) |
| `Cert.decide_greatest_safe` | the label's next-station speed is ≥ that of every feasible profile: **no command that keeps the plan feasible for every continuation consistent with what is visible can be faster** |
| `Cert.brake_least`, `Cert.decide_min_violation` | when already over budget, the label is full braking at 9 m/s², which yields the pointwise-least physically possible speeds (minimises every constraint violation) |
| `adjCheck_sound` | the linear check on sorted sample hashes implies no clean sample content appears in two splits |
| `pairs_within_split`, `full_coverage_of_exact` | contrast pairs never cross splits; an exact coverage report with no missing cells means every required cell is covered |
| `mirrorObs_involutive`, `mirrorAct_involutive`, `mirror_gap_swap`, `mirror_interval`, `mirror_steer_bound`, `chooseSide_mirror` | mirroring is an involution; left/right gaps swap; obstacle intervals and steer bounds are preserved; the gap-side choice flips (strict cases only; the expert's exact-tie break is not symmetric) |

### 3.2 Checked on the actual corpus files (`lean_report.json`, all pass on `full`)

- The generator's family spec equals the Lean spec.
- Python's formal accept/reject verdicts equal Lean's for all 228 draws.
- Accepted scenarios are valid with correct bins; IDs and hashes are unique.
- Contrast-pair integrity holds for all 184 pairs.
- The split rule, group and hash split consistency hold.
- The required cells, manifest coverage report and realised coverage report are exact.
- No clean sample content appears in two splits (28,537 hashes).
- The mirror transform equals the Lean model on 540 samples.
- **Lean recomputed the certified decision for all 43,758 expert-phase samples and matched
  every recorded label. Every one of the 28,537 clean labels is `greatest_safe` or
  `min_violation`.**

### 3.3 Not proven (do not claim it)

- That model M describes the bicycle model, CARLA, or a real car. Grip, budgets and the 1 m
  station discretisation are modelling choices; nothing between stations is checked.
- Optimality of the **steering / path** labels: those come from a heuristic planner plus pure
  pursuit.
- Optimality over the whole course or the whole lap. The certificate is about one decision
  under limited sight, with a worst-case assumption about the unseen road.
- Anything about a learned policy: its error, its safety, or how it generalises.
- Trusted but unverified: the Python float → integer export, JSON decoding, the Lean
  compiler/runtime. `data-validate` cross-checks the export against the records.

## 4. What this means for the algorithm choice

1. **The certified longitudinal label is a known function of the observation.** `cert.decide`
   (pure integer Python, 0.04 ms) computes it exactly from `observation.road` + speed, and Lean
   proves what it means. A network trained to imitate it can only lose accuracy and lose the
   certificate. **Recommendation: use the certified law directly at inference** (it is
   numpy/pure-Python and sandbox-safe). Spend learning capacity on what is not certified, or
   use the label as an auxiliary target or regulariser.
2. **The whole expert is also a pure function of the observation** (0.13 ms). In this
   scenario family, plain behaviour cloning can at best reproduce something you could run
   directly. Learning pays off where the expert is weak or absent:
   - (a) the lateral tracking layer, which causes the excluded runs (§6.1);
   - (b) situations the expert was never designed or tested for (other families);
   - (c) CARLA dynamics mismatch, e.g. as a residual on top of the expert's command.
3. **DAgger is cheap and recommended.** The expert can label *any* observation offline with
   `expert_action(obs, vp, None)`. The certificate is per observation, so labels on
   student-visited states are just as certifiable: export them and rerun `data-lean`. This
   removes behaviour cloning's compounding-error problem. The runner's `recorder` hook already
   captures observations from any controller. What's missing is a small "run the student,
   relabel with the expert" command; it is not implemented yet.
4. **Structure suggestion** (fits numpy, 50 ms, small data):
   - longitudinal = `cert.decide` (exact);
   - lateral = small MLP (e.g. 2 × 128) on resampled preview features predicting steer and/or
     path offsets, trained with mirror augmentation, contrast-direction consistency, and
     DAgger rounds;
   - optional residual on the pedal mapping fitted on CARLA data.
5. **Losses and targets.**
   - The acceleration label is nearly bang-bang (§2.5). If you learn it, use regime
     classification plus magnitude.
   - Steering saturates at ±1.
   - Weight by run, not by sample, to undo the 20 Hz correlation.
   - Excluded runs are available as negatives (`quality.exclusion_reasons`) if a method can
     use failure data. Don't mix them into the behaviour-cloning target.
6. **Model selection.** Group-wise CV on `group_id`. Val and test are region hold-outs, which
   measure extrapolation to unseen (radius × preview) combinations, not in-distribution error.

## 5. How to get more or different data

```powershell
python -m autocross data-manifest --profile large --out datasets\large.manifest.json
python -m autocross data-generate --manifest datasets\large.manifest.json --backend bicycle --out datasets\large --workers 6
python -m autocross data-validate datasets\large
python -m autocross data-lean datasets\large
# CARLA (server running, carla==0.9.16 client, Python 3.10-3.12; one worker):
python -m autocross data-generate --manifest datasets\full.manifest.json --backend carla --out datasets\full_carla
```

- Change the design (seed, profiles, contrast factors, held-out regions) in
  `config/datagen.yaml`.
- Change the grip or braking budget of the certificate in `autocross/datagen/expert.py`
  (`A_LAT_FRAC`, `A_BRAKE_PLAN`, …). The certificate stays valid for whatever budget is
  declared; Lean recomputes with the exported constants.

## 6. Known issues and risks

1. **Selection bias from excluded runs (15 % of runs).** Pure pursuit lags on tight turns
   combined with narrow gaps, so those runs hit cones and are excluded. The clean set
   under-represents exactly the hardest cells, although every cell still has a clean example.
   A better lateral tracker in the expert would reduce this. Two things tried and rejected:
   - Stanley steering made it much worse (oscillation, no yaw damping);
   - a lower grip budget (0.6) also made it worse.
2. **Bicycle only.** Bicycle-vs-CARLA transfer is unmeasured for this data. The harness's `compare`
   command targets about 2.5 % lap-time agreement on whole courses, but that says little about
   tight-gap tracking.
3. **One scenario family.** It does not cover what the evaluation course contains (§2.1).
4. **Ideal perception.** Road-model noise is 0. Cone noise is on (seeded) but the expert
   ignores cones. Robustness to noisy lane models is untested.
5. **Conservative horizon.** The worst-case-corner horizon makes the certified speed cautious
   with short previews. That is optimal *for model M* but slower than an expert that "knows"
   the road. It is a deliberate safety and information choice, and it is part of the
   certificate's meaning.
6. **Obstacle drops out of view.** The road model drops an obstacle 3 m after passing it, so
   the expert's plan snaps back to the centerline at that moment. The steering label has a
   small discontinuity there.
7. **Approach-speed hypothesis.** It held for 21 of 25 pairs; the other 4 brake about as much
   (the contrast changes the speed at hand-over, and the expert then often brakes to the same
   certified limit). This is informative, not a data error.

## 7. Suggested next data work (in priority order)

1. Generate a CARLA corpus (same manifest) on the reference machine; compare against the
   bicycle corpus by `scenario_id`.
2. Add families that match the technical course: slalom/esses with offset gates, a hairpin
   after a fast straight, a decreasing-radius turn, alternating obstacle gates. Each needs a
   Python `Family` subclass and its Lean spec plus structural constraint.
3. Add a DAgger relabelling command (student rollout → `expert_action` labels → certificate
   export).
4. Improve the expert's lateral tracker to cut exclusions (e.g. curvature feed-forward with
   yaw-rate damping).
