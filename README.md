# mil-sim: EcoCAR Autocross Simulation Harness

A lightweight, CARLA-based simulation framework for testing autonomous **path planning and control** on autocross courses. It is built for the EcoCAR autocross hackathon and as a foundation for model-in-the-loop (MIL) and software-in-the-loop (SIL) testing with ideal ground-truth data.

It mirrors the EIC autonomous autocross event: **the controller never gets a map.** Each control step it receives a processed perception output (a *road model*) describing only the course visible ahead. From that it must plan a path and output steering, throttle and brake, all within an embedded-style compute budget. Teams do not write perception or sensor fusion.

The harness also provides:

- **Two backends:** CARLA 0.9.16 (full physics) and a dynamic bicycle model for fast iteration without CARLA. The bicycle model's parameters are fitted to CARLA telemetry (`fit-bicycle`), and `compare` checks lap-time agreement against a ~2.5% target (the simplified single-track model's transient dynamics differ from CARLA's at roughly this level, so the official score counts each backend for half).
- **Flat autocross pad:** generated on the fly from a small OpenDRIVE road, with knockable physics cones and painted start/finish lines. No Unreal map editing is needed.
- **Sandboxed controllers:** each controller runs in its own process with 1 CPU core, no GPU, a memory ceiling and a 50 ms deadline per step. It cannot read course files or earlier runs from disk.
- **Separate test and evaluation courses:** teams develop on public test and practice courses, and scored runs use unreleased courses kept in a private judges' repository.
- **Scoring:** rules, penalties, metrics, rubric scoring, leaderboard, robustness sweeps and real-time pacing.
- **Full ground-truth telemetry** logged at 40 Hz for every run.

Designed to run on a laptop (tested with an RTX 30-series GPU). A 40 s run takes about 3 s headless in CARLA, and about 1 s on the bicycle model.

---

## Quick start

### Requirements

| | Bicycle backend (development) | CARLA backend (evaluation) |
|---|---|---|
| OS | Windows, Linux or macOS | Windows 10/11 or Linux (CARLA has no macOS build) |
| Python | **3.10–3.12** (3.13+ is not supported by the CARLA client wheels) | same |
| Extra software | none | [CARLA 0.9.16](https://github.com/carla-simulator/carla/releases/tag/0.9.16) (≈ 20 GB unpacked) |
| GPU | not needed | dedicated GPU (tested on RTX 30-series) |
| Python packages | numpy, PyYAML, psutil, matplotlib, pygame, pytest (installed by the setup script) | plus `carla==0.9.16` (installed by the setup script) |

Students only need the bicycle backend. They can develop and test a controller completely without CARLA. The official evaluation runs every controller on **both** backends and counts each for half, so a team that never had CARLA is not heavily penalised (see [Official evaluation](#official-evaluation)).

On Windows, install Python from python.org and tick **"py launcher"**, because the setup script uses it to find a 3.10–3.12 interpreter. On macOS use `brew install python@3.12`, and on Linux your distribution's `python3.12` and `python3.12-venv` packages.

### Setup (one time)

```powershell
git clone https://github.com/dszurek/ecocar-hackathon-sim.git
cd ecocar-hackathon-sim
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1      # Linux: ./scripts/setup.sh
```

The setup script creates `.venv`, installs the harness and, if one exists for your system, the `carla==0.9.16` client, then runs a self-check. A missing CARLA client is not an error: the script prints a note and the bicycle backend works as normal. This is always the case on macOS. If CARLA is not in `%USERPROFILE%\CARLA_0.9.16` or `C:\CARLA_0.9.16`, pass `-CarlaRoot <path>` or set the `CARLA_ROOT` environment variable.

Activate the environment with `.venv\Scripts\Activate.ps1` (Linux: `source .venv/bin/activate`). If PowerShell blocks the activation script, call `.venv\Scripts\python.exe` directly instead of `python`.

### See it run

```powershell
# No CARLA needed: bicycle model with a live top-down view
python -m autocross run -c controllers\baseline_pure_pursuit.py -t tracks\hackathon_test.yaml --backend bicycle --view

# Full CARLA
scripts\start_carla.ps1                  # windowed 1280x720, Low quality; add -Headless for no window
python -m autocross check                # confirms the server is reachable
python -m autocross run -c controllers\baseline_pure_pursuit.py -t tracks\hackathon_test.yaml --realtime
```

In CARLA the spectator follows the car, the start and finish lines are painted on the pad, and the controller's planned path is drawn in cyan. The top-down view shows the course surface on a 10 m ground grid. Every run prints a score summary and writes a folder under `runs/`. Open `report.png` there first.

---

## Writing a controller

Copy `controllers/template_controller.py`. A controller is one Python file with one class that derives from `autocross.api.Controller`:

```python
from autocross.api import Command, Controller

class MyController(Controller):
    def on_start(self, track, vehicle):
        super().on_start(track, vehicle)          # vehicle: limits (mass, wheelbase, max_steer, mu, ...)

    def step(self, obs):                          # called every 50 ms
        road = obs.road                           # what the perception stack sees ahead
        ...
        return Command(steer=s, throttle=t, brake=b,          # steer in [-1, 1], + = left
                       path=my_path, path_frame="vehicle")   # the path you intend to drive
```

### What the controller receives

| | |
|---|---|
| `on_start(track, vehicle)` | **No map** in event mode (`track` arrays are empty). `vehicle` holds the published limits: 1,845 kg, 3.0 m wheelbase, 35° max steer, steering-rate limit, μ ≈ 0.95, 5.2 m/s² acceleration, 9 m/s² braking, and the control period. |
| `obs.road` (`RoadModel`) | The course up to `road_model.range_m` (40 m) ahead, in the vehicle frame (x forward, y left): `lateral_offset`, `heading_error`, `curvature`, `dist_to_left/right_boundary`, `preview_s` / `preview_curvature` / `preview_centerline` / `preview_left/right_boundary`, `corners` (direction, entry/apex/exit distance, min radius, turn angle), `obstacles` (distance, lateral extent), `finish_distance`. |
| Vehicle state | Ground truth: `obs.x, y, yaw, vx, vy, yaw_rate, ax, ay, speed`, and the actuator values actually applied. |
| `obs.cones` (optional) | Raw cone detections (vehicle frame, colour class) within 30 m and a 140° field of view. |

Full definitions are in [`autocross/api.py`](autocross/api.py). The baseline ([`controllers/baseline_pure_pursuit.py`](controllers/baseline_pure_pursuit.py)) shows a complete road-model controller. It uses pure pursuit on the perceived centerline, speed from the curvature preview, and a throttle limit based on how much grip is left for cornering. It ignores obstacles. It finishes cleanly but slowly, and is there to be beaten.

### Rules the harness enforces

- **Compute:** `step()` must return within 50 ms, or the previous command is held and an overrun is logged. The harness measures this itself, as the round trip to your controller process, and also measures all CPU time that process uses, including any background threads. The controller runs on 1 core, with no GPU and a 1 GB memory ceiling. `on_start()` may take up to 60 s.
- **Actuators:** steering is slew-rate limited, and optional actuation delay can be added.
- **Invalid output:** an exception, NaN or wrong return type counts as an error. 20 errors in a row is a DNF (did not finish).
- **Dependencies:** controllers run in the harness's Python environment, so use only packages installed there (numpy is always available).
- **No course data from disk:** the controller process cannot open anything under `tracks/` or `runs/`, or the course being driven (`sandbox.protected_paths`). In event mode `on_start` gets the course name as just `"course"`. Read your own files (lookup tables, weights) from your controller's folder.
- **Sandbox:** the controller process may read files, but it may not write, delete or rename files, start processes, open network connections, import the CARLA client, call native code through `ctypes`, or inspect the harness through the garbage collector. Any of these raises `PermissionError`, which counts as a controller error. Log with `Command.debug` instead: its values appear as `dbg_*` columns in `telemetry.csv`. Run `python -m autocross selftest -c your_controller.py` to check your controller against these rules.

### Submitting

Submit either a single file `<team>.py`, or a folder `<team>/` containing `controller.py` plus any helper modules and data files it reads. The judges put it in `teams/` and run the official evaluation. Before you submit, run:

```powershell
python -m autocross selftest -c my_controller.py            # every public course on the bicycle model
python -m autocross selftest -c my_controller.py --carla    # ... and on CARLA, if you have it
```

It runs every public course with the official settings, and reports finishes, penalties, step time and technical points. It also explains any sandbox-rule violation.

---

## Scoring

Rules are evaluated from ground truth at every physics tick, so results are the same on both backends.

| Penalty (per occurrence) | Points |
|---|---|
| Cone hit (each cone counted once) | −2 |
| Off course (all four wheels outside the cones, once per excursion) | −5 |
| Missed gate (passing a gate station outside the corridor) | −10 |
| Controller error (up to 10) | −1 |

A run is a **DNF** if the car stays off course for more than 5 s, gets more than 15 m from the course, drives the wrong way, is stuck for 8 s, rolls over, or hits the 240 s timeout.

| Category | Weight | Measured as |
|---|---|---|
| Completion time | 30 | vs. a reference time (the course's `reference_time`, or an automatic friction-limited centerline estimate) |
| Path deviation | 15 | mean/peak distance from the path the controller published 0.5 s earlier, near the point it should have reached by now (centerline if no plausible path was published) |
| Compute | 15 | p99 step time vs. deadline, overruns, peak memory |
| Physical realism | 15 | grip over-demand, rear-axle sliding, steering-rate saturation, pedal overlap |
| Innovation | 25 | judged; enter with `score --judged innovation=<0-100>` |

Publish your real plan in `Command.path`, reaching at least a few metres ahead. Each position is compared with the path you published 0.5 s earlier, so a short stub path just ahead of the car scores badly. Paths that fold back or zig-zag more tightly than a car can drive are ignored, and the centerline is used instead.

Weights and thresholds are in [`config/scoring.yaml`](config/scoring.yaml). The four computed categories add up to the run's **technical points** (out of 75), minus penalties.

### Official evaluation

The judges run every submitted controller on the two unreleased evaluation courses (see [below](#test-courses-vs-evaluation-courses)), each on both backends: four runs per team, made in one batch on a single reference machine.

| Course | Weight | | Backend | Weight |
|---|---|---|---|---|
| Main course | 40% | × | CARLA (headless) | 50% |
| Technical course | **60%** | × | Bicycle model | 50% |

**Final score = weighted average of the technical points of the four runs (out of 75) + judged innovation (out of 25).**

A DNF or a crashed controller still counts, with the points of its run. The technical course carries more weight, so a controller that is fast on the test courses but cannot handle the technical elements loses more than one that is slower but clean. The award categories (fastest, most physically realistic, most efficient compute, most precise) use the same weighted averages, and go only to teams that finish all four runs.

To estimate your own score, run `python -m autocross selftest -c my_controller.py` (add `--carla` if you have it) and weight `practice_technical` more heavily than the others.

---

## Commands

All commands are `python -m autocross <command> --help`.

| Command | Purpose |
|---|---|
| `run -c CTRL -t TRACK` | One run. Useful flags: `--backend bicycle`, `--view`, `--realtime [FACTOR]`, `--no-render`, `--team NAME`, `--controller-config cfg.yaml`, `--perturb KEY=VALUE`, `--set section.key=value` |
| `batch -c CTRL` | Robustness sweep over courses × perturbations × seeds ([`config/robustness.yaml`](config/robustness.yaml)): perception and road-model noise, shorter sight distance, low grip, extra mass, actuator latency, steering bias, start offsets |
| `score RUN_DIR` | Re-score a run, optionally adding `--judged innovation=80` |
| `leaderboard [runs] --award` | Best run per team and course, plus the award categories (fastest, most realistic, most efficient, most precise) |
| `gen-track [--seed N] --out F.yaml [--closed] [--obstacles K] --preview` | Generate a random valid course (no `--seed`: secret course) |
| `show-track F.yaml` | Validate a course, print stats and reference time, and save a preview PNG |
| `plot RUN_DIR` | Regenerate `report.png` |
| `calibrate --backend carla` | Measure the vehicle's acceleration, braking and lateral grip limits |
| `compare -c CTRL -t TRACK [-t TRACK ...]` | Run the same controller on CARLA and on the bicycle model and report lap-time difference and speed-profile RMS per course. Exits 1 if any course differs by more than `--tolerance` (default 2.5%). `--carla-runs DIR` reuses earlier CARLA runs |
| `fit-bicycle RUN_DIR... [--write]` | Fit the bicycle model to the telemetry of CARLA runs; `--write` stores the result in `config/sim.yaml` |
| `selftest -c CTRL [--carla]` | Run a controller on every public course with the official settings, and check it against the sandbox rules. Run this before submitting |
| `check` | Verify the install and CARLA connectivity |

### Run outputs (`runs/<timestamp>_<team>_<course>/`)

- `telemetry.csv`: ground truth at physics rate (pose, body velocities and accelerations, measured steering angle, progress, lateral offset, boundary margin, path error, commanded vs. applied actuation, step timing, and `dbg_*` values from `Command.debug`)
- `events.json`: start, finish, cone hits, off-course excursions, missed gates and DNF
- `summary.json`: metrics, score, vehicle parameters, config snapshot, controller and course hashes, and machine info
- `report.png`: course map with the speed-coloured trajectory and events, plus speed, error, actuation and compute plots

---

## Courses

Courses are YAML files built from segments (`straight`, `arc`, `slalom`) or from explicit centerline points. Cones, gates and start/finish lines are generated automatically. `obstacles` places cone groups that are **not** on the map and appear only in perception.

```yaml
name: practice_basic
closed: false            # point-to-point; closed: true + laps: N for loop courses
width: 5.5
segments:
  - {type: straight, length: 35}
  - {type: arc, radius: 15, angle: -110}      # + = left
  - {type: slalom, cycles: 3, wavelength: 26, swing: 2.5}
obstacles:
  - {s: 120, offset: 1.0, width: 2.0}          # station along the course, lateral offset, width
```

### Test courses vs. evaluation courses

| Course | What it is |
|---|---|
| `tracks/hackathon_test.yaml` | the main test course: develop and test here |
| `tracks/practice_basic.yaml` | short open course: corners, a slalom and a hairpin |
| `tracks/practice_obstacles.yaml` | open course with unmapped obstacle groups |
| `tracks/practice_loop.yaml` | closed course, 2 laps |
| `tracks/practice_technical.yaml` | technical course: fast narrow kink, a braking zone into a hairpin, decreasing radius, esses, obstacle gates |

Scoring uses **unreleased evaluation courses** that only the judges have:

- a **main course** with a different layout from `hackathon_test` (different corner sequence, slaloms and obstacle positions) but the same kinds of elements and width;
- a **technical course** that is deliberately hard for controllers that only react to the road right in front of them. `practice_technical` has the same kinds of elements in milder form.

A controller tuned to the exact shape of the test courses will not carry over. Build controllers that work from `obs.road` alone, plan their path and speed ahead, and handle obstacles. Before the event, also test on random courses:

```powershell
python -m autocross gen-track --seed 42 --obstacles 3 --out tracks\my_random.yaml --preview
```

**For judges:** the evaluation courses, the evaluation script (`tracks/private/tools/evaluate.py`) and the reference checks are in a separate private repository, cloned into `tracks/private/`. That folder is git-ignored, and team controllers cannot read anything under `tracks/`. Put team submissions in `teams/` as `<team>.py` or `<team>/controller.py` (also git-ignored). To make a new secret course, leave out `--seed`: `gen-track` then uses a random seed that is never shown or saved. A course made with a small or guessable seed can be regenerated by anyone.

```powershell
python -m autocross gen-track --obstacles 4 --out tracks\private\my_eval.yaml --preview
```

---

## Configuration

[`config/sim.yaml`](config/sim.yaml) holds the official event settings. Any value can be overridden per run with `--set`.

| Section | Contents |
|---|---|
| `sim` | physics step (25 ms), control rate (20 Hz), timeout, seed, `realtime_factor` |
| `carla` | host/port, rendering, spectator, pad margin, painted lines, cone models (boundary, line marker, obstacle), weather, recorder |
| `vehicle` | blueprint, steering limit, CARLA tire friction / brake torque / torque scale, published limits, bicycle-model parameters |
| `road_model` | preview range and step, corner detection, lane-model noise (0 = ideal) |
| `perception` | raw cone sensor range, field of view, noise, dropout, misclassification |
| `track_knowledge` | `perception` (event: no map) or `full` (map given in `on_start`, for debugging) |
| `compute_budget` | enforce/measure mode, cores, deadline, memory, GPU access |
| `sandbox` | paths the controller process may not read (`tracks`, `runs` by default) |
| `rules` | off-course rule, DNF thresholds, gate spacing, cone-hit margin |

**Vehicle calibration.** CARLA's stock Model 3 is unrealistic for this use:

| | Stock CARLA | Configured |
|---|---|---|
| Lateral grip | ≈ 2.4 g | tire friction reduced to give ≈ 0.95 g |
| Braking | ≈ 4 m/s² | brake torque raised to give ≈ 9 m/s² |
| Throttle mid-corner | spins the tires (CARLA has no traction control) | torque scaled to 75% |
| Max road-wheel steer | 70° | 35° |

Re-run `python -m autocross calibrate --backend carla` after changing the vehicle, and update the published limits.

**Matching the bicycle model to CARLA.** Students develop on the bicycle model, and the evaluation scores them on both backends, so the two should agree. The bicycle parameters in `vehicle.bicycle` are fitted to CARLA telemetry: the fitter replays logged CARLA runs in 1.5 s windows, starting the bicycle model from the logged state with the logged actuator commands, and tunes grip, tyre stiffness, drive and brake force, power, drag, steering lag, brake/drive split, yaw inertia and CG height until body velocities, yaw rate and wheel angle match. Mass, wheelbase and CG position are copied from the CARLA vehicle, so controllers also see the same `VehicleParams` on both backends. On the reference machine, with CARLA running:

```powershell
# 1. drive CARLA and the current bicycle model on several courses
python -m autocross compare -c controllers\baseline_pure_pursuit.py -t tracks\hackathon_test.yaml -t tracks\practice_loop.yaml -t tracks\practice_obstacles.yaml -t tracks\practice_basic.yaml
# 2. fit to those CARLA runs and write config\sim.yaml (takes a couple of minutes)
python -m autocross fit-bicycle runs\<stamp>_compare --write
# 3. check agreement again without re-driving CARLA
python -m autocross compare -c controllers\baseline_pure_pursuit.py -t tracks\hackathon_test.yaml -t tracks\practice_loop.yaml -t tracks\practice_obstacles.yaml -t tracks\practice_basic.yaml --carla-runs runs\<stamp>_compare
```

To check that the fit generalises, include CARLA runs from more than one controller (or the baseline with a more aggressive `--controller-config`), and compare on a course the fit did not see. Repeat the fit after changing the CARLA vehicle setup, then commit `config/sim.yaml` so students get the new values.

---

## Toward MIL / SIL

The `Controller` interface is where a real controller plugs in. The simulator, rules, logging and scoring stay the same:

- **MIL:** wrap a Simulink model exported as an FMU (e.g. with `fmpy`) or called through the MATLAB Engine in a `Controller` subclass. Set `compute_budget.mode: measure`, since the loop is lockstep and slow models just slow the run.
- **SIL:** compile Embedded Coder output to a shared library and call its step function via `ctypes`. Keep `mode: enforce` with ECU-like `cpu_cores` / `step_deadline_ms`. This approximates ECU timing only; true timing needs PIL/HIL.
- **Ground truth:** every run's telemetry is suitable as a reference for comparing against vehicle logs. `carla.record: true` also saves a CARLA replay log.
- **Not yet implemented:** camera/lidar sensor models (perception here is derived from ground truth) and an asynchronous real-time loop for HIL.

---

## Imitation-data generation (certified contrastive corpus)

Generates training data for an imitation-learning controller. Each sample is the exact `Observation` a controller received, plus a deterministic expert's label. The label carries a Lean-checked certificate that the longitudinal decision (accelerate or brake) is optimal under a declared model, using only what the car could see. Training is not part of this repository. Full specification: [`docs/datagen.md`](docs/datagen.md). Notes for whoever picks the learning algorithm: [`docs/algorithm_research_handoff.md`](docs/algorithm_research_handoff.md).

```powershell
python -m autocross data-manifest --profile smoke --out datasets\smoke.manifest.json       # or full / large
python -m autocross data-check-manifest datasets\smoke.manifest.json
python -m autocross data-lean datasets\smoke.manifest.json                                 # Lean check of the manifest
python -m autocross data-generate --manifest datasets\smoke.manifest.json --backend bicycle --out datasets\smoke --workers 4
python -m autocross data-validate datasets\smoke
python -m autocross data-lean datasets\smoke                                               # Lean check of the corpus
```

| Command | Purpose |
|---|---|
| `data-manifest --profile P --out M.json` | Finite coverage design (`smoke`: every bin; `full` / `large`: every pair of bins), seeded in-cell draws, matched one-factor contrast pairs, region-held-out splits, recorded rejections |
| `data-check-manifest M.json` | Recompute IDs, hashes, bins, splits, pairs and coverage |
| `data-generate --manifest M.json --backend carla\|bicycle --out DIR` | Run every scenario with the expert and write `samples/*.jsonl`, `runs/`, `tracks/`, `index.json`, `contrast_report.json`. `--workers N` (bicycle only), `--resume`, `--limit N` |
| `data-validate DIR` | Schema, hashes, provenance, exact label replay, applied-action consistency, outcome flags, duplicates, split leakage, realised coverage |
| `data-lean M.json\|DIR` | Export integers, build `lean/` and check: spec equality, validity verdicts, contrast integrity, split rules, coverage exactness, sample leakage, mirror transform, and recompute every certified label. Also runs an axiom audit. Writes `lean_report.json` |

- **Backends.** Every record and `index.json` names the backend. The bicycle corpus is a tooling and logic check, **not CARLA data**. `--backend carla` needs the `carla==0.9.16` client (Python 3.10–3.12) and a running server; otherwise it stops with exit code 3 before writing anything.
- **Lean.** Install [elan](https://github.com/leanprover/elan); `lean/lean-toolchain` pins Lean 4.35.0 (core only, no Mathlib). Lean certifies the properties of its own model, checked on the exported integers. It does not certify CARLA or bicycle physics, the steering/path labels, or how well a policy trained on the data will perform (trust boundary: [`docs/datagen.md`](docs/datagen.md#6-lean-lean-lean-4350-core-only)).

---

## Repository layout

```
autocross/            harness package
  api.py              team-facing types: Controller, Observation, RoadModel, Command, ...
  runner.py           closed loop, actuator model, logging, batch sweeps
  sandbox.py          controller process + compute budget
  perception.py       road model and cone sensor (ground truth + configurable noise)
  monitor.py          race rules and events
  metrics.py, score.py, plot.py, viewer.py
  track.py, trackgen.py, geometry.py
  backends/           carla_backend.py, bicycle.py
  cli.py, calibrate.py, fit.py, config.py
  datagen/            imitation-data generation: spec, design, expert, cert, generate, validate, lean_bridge
config/               sim.yaml, scoring.yaml, robustness.yaml, datagen.yaml
docs/                 datagen.md (data specification), algorithm_research_handoff.md
lean/                 Lean 4 model, proofs and corpus checker (lake build; corpuscheck)
controllers/          baseline_pure_pursuit.py, template_controller.py
tracks/               hackathon_test + practice courses (private/ = judges' evaluation repo, git-ignored)
scripts/              setup and CARLA launch scripts (PowerShell + bash)
tests/                pytest suite (bicycle backend, no CARLA needed)
```

## Development

```powershell
python -m pytest            # 92 tests, ~3 min, no CARLA required (includes anti-cheat and data-pipeline tests; Lean tests skip without lake)
```

The official evaluation should be run on a single reference machine, because the 50 ms deadline depends on the machine.
