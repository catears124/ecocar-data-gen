# VeriDrive — submission manifest

## What is submitted

1. **Certified contrastive imitation corpus and Lean proofs**: VeriDrive's corpus tooling and certified decision models are under `autocross/datagen/`, with released corpus data under `datasets/` when a corpus directory is attached. The Lean 4 models, theorem statements, and checker are under `lean/`.
2. **Controller**: `teams/certline.py`, a single-file `autocross.api.Controller` submission.

The submission format documented by the harness is either one Python file or a folder containing `controller.py` and its data/modules, placed in `teams/`. This release uses the single-file form.

## Controller submission

- Path: `teams/certline.py`
- Size: 57,608 bytes
- SHA-256: `bcdab96e7c02ead0396f8cf2c09079b62ee27f4494d0d8f94533aa492e44cc8e`
- Runtime dependency: NumPy only (the harness provides it); no GPU is used by the controller.
- Release: [`veridrive-v2.0`](https://github.com/catears124/veriDrive/releases/tag/veridrive-v2.0) (byte-identical `certline.py`, race videos).
- Judge commands:

```powershell
python -m autocross selftest -c teams/certline.py
python -m autocross selftest -c teams/certline.py --carla
```

The first command exercises the public courses on the bicycle backend. The second additionally requires a reachable CARLA 0.9.16 server/client and a dedicated GPU as documented by the harness. The controller itself is constrained to the sandbox's one CPU core, 50 ms step deadline, and no GPU.

## Controller and baseline

The submitted controller is `teams/certline.py` (the file above). `controllers/baseline_pure_pursuit.py` is the harness's own reference baseline, kept for scale.

Measured on this branch, bicycle backend, official settings (`selftest`, sequential runs on one machine):

| controller | mean technical, 5 public courses | cones | off | gates | step p99 |
|---|---|---|---|---|---|
| `teams/certline.py` (this branch) | 71.6 / 75 | 0 | 0 | 0 | ≤ 4.0 ms |
| `teams/certline.py` (`main` @ 5ba53e0) | 58.1 / 75 | 0 | 0 | 0 | ≤ 7.3 ms |
| `controllers/baseline_pure_pursuit.py` | 62.3 / 75 | 7 | 0 | 0 | ≤ 3.7 ms |

On 20 generated courses (`gen-track` seeds 101–106, 201–204 used for tuning the new defaults; seeds 301–306, 401–404 held out and run only once, for validation), with compute measured but not enforced: 0 cone hits, 0 off-course, 0 DNF on all 20; the `main` controller had 10 cone hits on the tuning set and 38 on the held-out set. Under the eight perturbations of `config/robustness.yaml` on the five public courses: 3 cone hits in total (2 under `low_grip`, 1 under `actuation_latency`), versus 49 for `main`. These are bicycle-model measurements, not CARLA results.

## Verification commands

Run from the repository root unless noted:

```powershell
python -m autocross data-validate <corpus-dir>
python -m autocross data-lean <corpus-dir>
python -m autocross data-check-manifest <manifest.json>
cd lean
lake build
cd ..
python -m pytest -q
python -m autocross selftest -c teams/certline.py
python -m autocross selftest -c teams/certline.py --carla
```

`data-validate` is the Python hash, schema, replay, certificate, leakage, and coverage check; it does not require CARLA or a GPU. `data-lean` requires the Lean 4.35.0 toolchain and `lake`; it runs the Lean build, axiom audit, compiled checker, and corpus witnesses. `data-check-manifest` checks the manifest without running a simulator. `lake build` requires Lean 4.35.0 and Lake. The pytest suite uses the repository's installed Python test dependencies and does not require a GPU; individual CARLA-dependent tests, if selected by their environment, require CARLA. The bicycle controller selftest requires no CARLA or GPU. The `--carla` selftest requires CARLA 0.9.16 and a suitable dedicated GPU.

## Claim set: proven, measured, assumed/trusted

### Proven by Lean

The following theorem names were checked in `lean/Corpus/`:

- Longitudinal certificate (`lean/Corpus/Cert.lean`): `Corpus.Cert.le_bpass`, `Corpus.Cert.fpass_greatest`, `Corpus.Cert.fpass_feasible`, `Corpus.Cert.accel_greatest`, `Corpus.Cert.robust_extend`, `Corpus.Cert.robust_necessary`, `Corpus.Cert.brake_least`, `Corpus.Cert.brake_minimises_excess`; unconditional classification and exact infeasibility (`lean/Corpus/CertProofs.lean`): `decide_classification`, `feasible_exists_iff`, `decide_gs_iff_feasible`; quantiser bounds (`lean/Corpus/Quant.lean`).
- Certified decision consequences (`lean/Corpus/Cert.lean`): `Corpus.Cert.decide_greatest_safe`, `Corpus.Cert.decide_greatest_safe_feasible`, `Corpus.Cert.decide_greatest_safe_minimises`, `Corpus.Cert.decide_min_violation`, `Corpus.Cert.decide_min_violation_infeasible`, `Corpus.Cert.decide_min_violation_robust`.
- Manifest and mirror properties (`lean/Corpus/Properties.lean`, `lean/Corpus/Mirror.lean`): `Corpus.pairs_within_split`, `Corpus.full_coverage_of_exact`, `Corpus.adjCheck_sound`, `Corpus.noCrossSplit_of_check`, `Corpus.mirrorObs_involutive`, `Corpus.mirrorAct_involutive`, `Corpus.mirror_gap_swap`, `Corpus.chooseSide_mirror`.
- Lateral decision facts (`lean/Corpus/Steer.lean`): `Corpus.Steer.wellFormed_off_length`, `Corpus.Steer.sides_length`, `Corpus.Steer.amps_length`, `Corpus.Steer.decide_kind_invalid`, `Corpus.Steer.decide_no_gap_iff`, `Corpus.Steer.decide_plan_ok_iff`, `Corpus.Steer.clamp_abs_le`, `Corpus.Steer.steer_within_range`; hold-window clearance of `plan_ok` plans under `clearanceDomain`/`holdTargetsOK` (`SteerClear.plan_ok_hold_clearance`, `plan_ok_clearance`); ramp slope/second-difference bounds and mirror equivariance (`SteerShape`).
- Tracker facts (`lean/Corpus/Track.lean`): `Corpus.Track.error_fixed_point`, `Corpus.Track.error_bound_steady`, `Corpus.Track.error_bound_steady_ramp`, `Corpus.Track.corridor_invariant`; delayed small-gain bound and closed-loop consequences (`lean/Corpus/Closed.lean`).
- Bellman/DP facts: option-valued Bellman optimality, attainment and extraction for any finite horizon (`lean/Corpus/BellmanOpt.lean`: `valueO_none_iff`, `valueO_attained`, `extract_spec`, `extracted_policy_optimal`, `bellmanO_unique`); joint offset/speed optimality and composition with the longitudinal certificate (`lean/Corpus/Joint.lean`: `joint_value_spec`, `joint_extract_spec`, `plan_cert_feasible`). The older `Corpus.Dp.dp_optimal`/`speed_dp_optimal` use `Bellman.value`, which is only a lower bound when an action list is empty (`BellmanOpt.dead_mid_old_value`), and their speed model's cap is inert (`Joint.old_speed_cap_inert`); they are kept but superseded by `Joint`.
- Planning levers (`lean/Corpus/Levers.lean`): the joint model for any step relation (`valueP_lower`, `valueP_attained`, `valueP_none_iff`); `lever_value_le` (relaxing a constraint never worsens the optimum); `budget_lever` with `cornering_lever`, `braking_lever`, `accel_lever`, `horizon_lever`; `extra_caps_cost`/`extra_caps_mono` (gap rule, sidestep gate); `box_step_circle`/`circle_value_le_box` (the certified box is a conservative subset of the friction circle); kernel-checked instances `toy_cornering_binds`, `toy_braking_slack`, `toy_circle_gain`.
- Controller (`lean/Corpus/Controller.lean`, `ControllerLong.lean`): clearance interval soundness/completeness, projection and publication inside the interval, speed decision greatest-safe for the controller's own caps, actuation invariants.

All 373 theorems are listed in `lean/Corpus/Audit.lean`; the audit fails if a declared theorem is unlisted. Exact statements: `docs/formal_guarantees.md`; link-by-link status: `docs/verification_matrix.md`; trust boundary: `docs/trust_boundary.md`.

These are model theorems with their stated hypotheses. They are not claims about CARLA tyre physics or a learned policy.

### Measured or reported in repository docs

The release handoff reports, for the `v1.0.1` full bicycle corpus (`docs/algorithm_research_handoff.md`): 212 accepted scenarios, 184 clean runs, 28 scenario groups, and 29,142 clean samples (24,722 train; 1,823 validation; 2,597 test). It reports 72,163 total samples and 43,021 non-clean samples. The same document reports 28 excluded runs of 212 and the split/group counts; these are dataset measurements, not Lean proofs.

The data-generation specification reports the full profile's 184 contrast pairs and 154 pairwise coverage cells (`docs/datagen.md`). It also reports the declared scenario bins and the empirical contrast checks. The handoff reports the clean-label counts (26,825 `greatest_safe`, 2,317 `min_violation`) and the stated model budgets (`+5.2 m/s²`, `−6.589 m/s²`, full brake `−9.0 m/s²`) as corpus/model documentation.

### Assumed or trusted

The declared `timeTable` used by the toy travel-time DP is a calibration/model input, not a vehicle measurement theorem. Vehicle tyre/grip budgets and other published constants are declared model constants; they are not established by the Lean proofs. Pure-pursuit tracking, pedal mapping, simulator dynamics, perception assumptions, and the controller's empirical performance remain trusted or replay-checked layers. Corpus optimality means optimality in the quantised point-mass model at preview stations, not whole-course time-optimal driving.

## Repository map

| Path | Purpose |
|---|---|
| `lean/` | Lean 4 models, proofs, audit, and compiled corpus checker |
| `autocross/datagen/` | Manifest, generation, records, validation, certificates, and canonical dataset loader |
| `datasets/` | Corpus manifests and generated corpus directories (when included) |
| `teams/` | Submission controllers, including `certline.py` |
| `controllers/` | Reference and example controllers |
| `tests/` | Python tests for harness, data, certificates, and validation |
| `docs/` | Data-generation specification and algorithm/research handoff |
| `config/` | Simulation, vehicle, scoring, and data-generation configuration |
| `tracks/` | Public test/practice course definitions |

## Known limitations

The documented corpus has selection bias: excluded runs are concentrated in tight clearances and outside obstacles. There are only 28 independent scenario groups, with 2 validation groups and 4 test groups, so split metrics have wide uncertainty. The release is bicycle-only unless a separate CARLA corpus is attached. It covers one `brake_turn_obstacle` family, not slaloms, chicanes, esses, hairpin sequences, decreasing-radius turns, multiple obstacles, closed courses, or noisy perception; the documented generation uses ideal road-model noise and no occlusion.

The handoff also documents a conservative horizon, an obstacle visibility drop-out discontinuity after passing, and perception locality beyond the preview edge. Failed tracking runs remain in the corpus with exclusion reasons rather than being treated as clean. The joint longitudinal-lateral optimality result (`Joint`) is about a discrete reference model that the submitted controller does not execute; the controller's own path is a heuristic minimum-curvature line, and only its speed decision for that path is proved greatest-safe. Lateral clearance is proved under `clearanceDomain`/`holdTargetsOK`, which the corpus checker does not evaluate per sample. The closed-loop results hold under named assumptions (`DelayedEquation`, `SmallGain`, `RealisedWithin`, `PreviewConsistent`, `NextValid`) that are not proved for the bicycle backend or CARLA. The full list of unresolved obligations is in `docs/proof_closure_report.md`. The Lean certificate is not a claim about simulator dynamics, real sensors, or learned-policy behavior.
