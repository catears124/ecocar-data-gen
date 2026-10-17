# Algorithm research handoff: learning from the certified imitation corpus

For whoever selects the learning algorithm. This file covers what the data is, what is proven
about it, what is only measured, and the constraints the final controller must meet. The full
data specification is in [`datagen.md`](datagen.md); the precise guarantee is in the README's
**Formal Guarantee** section.

All numbers below come from the release `v1.0.1` `full` corpus: bicycle backend, expert 3.0.0,
certificate model `cert-2`, schema 2. CARLA results, where available, are in the release notes.
They are kept in a separate corpus and never mixed with bicycle data.

## 1. Hard constraints on the final policy (from the harness)

| Constraint | Value | Consequence for the algorithm |
|---|---|---|
| Runtime environment | sandboxed process; **numpy only** (no torch, no onnxruntime, no native code via ctypes), cannot write files or open sockets | the model must run as plain numpy (or pure Python) at inference; weights are read from files next to the controller |
| Compute | 1 CPU core, **50 ms per step** (20 Hz), 1 GB memory, no GPU; `on_start` ≤ 60 s | step budget is generous for small MLPs (~0.05–1 ms) |
| Inputs | `Observation` only. **No map**; course name withheld | train on `policy_observation` only (§2.4); it is a strict subset of what the controller receives |
| Scoring | both backends weighted 50/50 (CARLA, bicycle); technical course 60 %; penalties for cones (−2), off-course (−5), missed gates (−10) | robustness on tight technical elements matters more than raw speed |
| Path metric | `Command.path` (planned path) is scored against the actual trajectory 0.5 s later | predicting a path, not just steer, earns points; the dataset has the expert path per sample (uncertified) |

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
obstacles, closed courses, noisy perception. A policy trained only on this family **will not
cover the evaluation course**.

### 2.2 Sizes (`full` profile, bicycle backend; `dataset_report.json`)

| | train | val | test | total |
|---|---|---|---|---|
| scenarios (accepted = run) | 176 | 12 | 24 | 212 |
| clean runs | 155 | 11 | 18 | 184 |
| **scenario groups** (base + its contrasts; the split unit) | 22 | 2 | 4 | 28 |
| clean samples (20 Hz, all certified) | 24,722 | 1,823 | 2,597 | 29,142 |
| AR(1) effective samples of the certified target (indicative) | 1,025 | 35 | 89 | 1,149 |
| required pairwise cells covered by clean runs | 152 / 154 | 57 / 154 | 71 / 154 | 154 / 154 |

- 72,163 samples were recorded in total. 43,021 are not clean (scripted approach 28,285, after
  the decision window, or from an excluded run). Every one is kept, with its reasons.
- **Do not treat 29 k timesteps as 29 k observations.**
  - Samples within one run form a strongly autocorrelated time series.
  - The independent simulation units are the 184 clean runs.
  - The split unit is the scenario group: only 2 groups in val and 4 in test, so their metrics
    have very wide uncertainty.
  - Only 27,087 of the 29,142 clean samples are distinct: 2,055 repeat an (input, target) exactly
    within the same split, typically a base and its variant before the varied factor becomes visible.
- 28 of 212 runs were excluded (21 boundary cone hits, 9 obstacle cone hits, 4 off-course,
  4 missed gates; one run can have several reasons). Per-run exclusion rate, with Wilson 95 %
  intervals:

| Factor bin | Excluded / runs | 95 % interval |
|---|---|---|
| clearance tight / moderate / wide | 20/80, 7/61, 1/71 | [0.17, 0.35], [0.06, 0.22], [0.00, 0.08] |
| obstacle outside / inside | 19/76, 9/136 | [0.17, 0.36], [0.04, 0.12] |
| preview short / nominal / long | 14/78, 5/56, 9/78 | [0.11, 0.28], [0.04, 0.19], [0.06, 0.21] |
| direction left / right | 21/132, 7/80 | [0.11, 0.23], [0.04, 0.17] |
| split train / val / test | 21/176, 1/12, 6/24 | [0.08, 0.18], [0.01, 0.35], [0.12, 0.45] |

  The clean set under-represents tight gaps and outside obstacles. Every cell still has a clean
  example.

### 2.3 Splits

Splits hold out whole parameter regions of (turn radius × preview):
- `val` = (open radius, nominal preview);
- `test` = (medium radius, short preview);
- everything else is `train`.

Lean checks that a held-out region never appears in another split, that each group and contrast
pair sits in one split, and that no clean sample content appears in two splits. Use
**group-wise cross-validation on `group_id`** inside train for model selection, and keep test
untouched.

### 2.4 Record schema and loading

Use the canonical loader. It validates schema and backend, and never reads the privileged block:

```python
from autocross.datagen.dataset import load_examples, to_arrays
ex = load_examples("ecocar-imitation-full-bicycle-v1.0.1", split="train", expect_backend="bicycle")
# ex[i] = (policy_observation, training_target, certificate_status, metadata)
arrays = to_arrays(ex)   # x, y_long (certified, m/s^2), certified, y_steer, y_throttle, y_brake, group
```

`examples/load_dataset.py` is a runnable NumPy/torch example.

| Field | Use |
|---|---|
| `policy_observation` | **the only inputs**: ego `speed, vx, vy, yaw_rate, ax, ay, steer, throttle, brake`; `road` scalars, preview arrays at 1 m stations (length varies 16–61; resample with a mask), preview polylines, visible obstacles |
| `target.longitudinal` | **certified**: `accel_mm_s2` (integer), `kind` (`greatest_safe` / `min_violation`), `b1_mm2_s2` |
| `target.uncertified` | `steer` [−1, 1] + left, `path` (vehicle frame), `path_frame`, `throttle`, `brake`, `target_speed`: **no guarantee** |
| `privileged.*` | pose, progress, lap, time, cones, corner list, finish distance, oracle distances: **evaluation only, never inputs** |
| `quality.clean`, `quality.exclusion_reasons` | train on `clean` unless you deliberately use failures as negatives |
| `hashes.*` | per-record integrity; `content` identifies an (input, target) example |

### 2.5 Label distribution (clean samples)

- Certificate kinds: 26,825 `greatest_safe` and 2,317 `min_violation` (8 %; full braking because
  the car is already over its braking budget).
- The certified acceleration is close to bang-bang (budget +5.2 m/s², braking budget
  −6.589 m/s², full brake −9.0 m/s²). MSE regression will blur the switching surface. Prefer
  regime classification plus magnitude, or compute it exactly (§4).

### 2.6 Contrast pairs (184)

Each pair changes exactly one factor between base and variant (checked by Lean).
`contrast_report.json` measures whether the expert's action changed, on clean pairs:
- turn radius 20/20, direction (mirror) 26/26, preview 20/20, obstacle side 20/20 and
  clearance 18/18 move in the declared direction;
- approach speed holds in 22/26.

These suit contrastive or consistency losses. Mirror augmentation is exact for the
transformation of inputs and targets (`records.mirror_observation` / `mirror_action`, checked
against the Lean model). The expert's gap-side tie-break at exactly equal gaps is not symmetric.

## 3. What is proven, exactly

See the README **Formal Guarantee**. In short:
- The recorded **longitudinal acceleration** of every clean sample is the greatest integer
  acceleration that keeps a feasible speed plan (point mass, friction box μg/√2, vehicle limits)
  over the visible preview. That plan also stays safe on every unseen road in the declared
  domain.
- No planner with the same information (the preview plus the domain) can be faster at any
  preview station without being unsafe on some in-domain road.
- The Lean kernel proves `Cert.decide c q = recorded` for every one of the 29,142 clean samples.

**Not proven:** steering/path/pedal optimality, CARLA or bicycle dynamics, real-sensor
visibility, anything about a learned policy. Certified labels can still end in cone hits: the
model is a point mass, and the steering is uncertified.

## 4. What this means for the algorithm choice

1. **The certified longitudinal label is a known function of the policy observation.**
   `cert.decide(model_constants(vp), cert.quantise_dict(policy_observation))` computes it exactly,
   in integer Python (~0.04 ms). **Recommendation:** use the certified law directly at inference,
   and spend learning capacity on the uncertified lateral layer.
2. **The whole expert is a pure function of the policy observation.** Plain behaviour cloning
   can at best reproduce it. Learning pays off where the expert is weak or absent:
   - the lateral tracking that causes the exclusions;
   - other scenario families;
   - CARLA dynamics mismatch.
3. **DAgger is cheap.** The expert labels any observation offline. A "run the student, relabel
   with the expert" command is not implemented yet.
4. **Losses.** Weight by run or group, not by sample. Steering saturates at ±1. Excluded runs
   are available as negatives.

## 5. How to get more or different data

```powershell
python -m autocross data-manifest --profile large --out datasets\large.manifest.json
python -m autocross data-generate --manifest datasets\large.manifest.json --backend bicycle --out datasets\large --workers 6
python -m autocross data-validate datasets\large
python -m autocross data-lean datasets\large
python -m autocross data-report datasets\large
# CARLA (server running, carla==0.9.16 client, Python 3.10-3.12; one worker):
python -m autocross data-generate --manifest datasets\full.manifest.json --backend carla --out datasets\full_carla
```

The certificate's constants are derived (`cert.model_constants`, `expert.HORIZON_RADIUS_MM`)
and are not tuning knobs. Changing the scenario domain in `spec.py` changes the horizon radius.

## 6. Known issues and risks

1. **Selection bias** (28 of 212 runs excluded, concentrated in tight gaps and outside
   obstacles; §2.2).
2. **Few independent units.** 28 scenario groups in total; val and test have 2 and 4.
3. **Bicycle only** unless a CARLA corpus is attached to the release.
4. **One scenario family**, ideal perception (road-model noise 0), no occlusion.
5. **Conservative horizon.** The domain's worst-case planned path (2.55 m effective radius)
   makes speeds with short previews cautious. That is the price of the same-information
   guarantee.
6. **Obstacle drops out of view** 3 m after passing it; the steering label has a small
   discontinuity there.
7. **Perception locality.** A policy observation can depend on course geometry up to ~5 m past
   the preview edge (curvature smoothing) and, at ~1e-4 m, on total course length (global
   resampling in the course builder).

## 7. Next: v2 (full-action optimality)

Replace pure pursuit with an exact joint path + speed optimiser over a discretised model: a
lattice of lateral offsets × speeds per preview station, solved by dynamic programming with the
same domain horizon. Each label then carries a Bellman certificate checked in Lean, against one
general theorem that a Bellman-consistent value function lower-bounds every same-information
plan. The guarantee would cover the full action, but only within the discretised model.
