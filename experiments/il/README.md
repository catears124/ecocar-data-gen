# Imitation-learning experiments

This directory contains the reproducible behavioral-cloning experiment. Run all commands from the repository root, with the project environment active. `numpy` is required; training additionally requires the optional PyTorch dependency used by `train_bc`.

## 1. Build and validate a corpus

Start from a datagen manifest (for example, `config/datagen.yaml` or a manifest produced by the datagen tooling). The bicycle backend is deterministic and does not require CARLA:

```powershell
python -m autocross data-generate --manifest config/datagen.yaml --backend bicycle --out datasets/il-corpus --design config/datagen.yaml --workers 1
python -m autocross data-validate datasets/il-corpus
```

For a CARLA-backed corpus, start the CARLA server and use `--backend carla` instead. `--workers 1` is intentional: it avoids scheduling-dependent ordering. A partial smoke corpus can be made with `--limit N`; it is not interchangeable with the full corpus. If a checked-in/generated corpus is already available, skip `data-generate` and run `data-validate` on that directory.

## 2. Train

The following command is the canonical training run (the command writes the learned weights to `model.npz`):

```powershell
python -m experiments.il.train_bc --corpus datasets/il-corpus --condition optimal --budget 200 --seed 0 --out model.npz
```

`--corpus` is a corpus directory, `--condition` selects one of `optimal`, `approx`, `heuristic`, or `cert_control`, `--budget` limits the selected training examples, `--seed` fixes subsampling/initialization, and `--out` is the weights file. Use the same condition, budget, and seed when comparing runs. The trainer uses the corpus `train` split; validation/test records must not be used for fitting.

## 3. Evaluate

Evaluate the resulting policy on held-out corpus examples and the requested tracks:

```powershell
python -m experiments.il.evaluate --corpus datasets/il-corpus --weights model.npz --tracks tracks/practice_basic.yaml,tracks/practice_technical.yaml --conditions optimal --budgets 200 --seeds 0 --output eval.json
```

`--tracks` is a comma-separated list of track YAML paths. `eval.json` records the aggregate metrics and per-track results. Evaluation is deterministic for fixed corpus, weights, condition, budget, seed, track files, and simulator configuration.

## 4. Orchestrate the complete experiment

The orchestration entry point performs the train/evaluate sequence and stores its artifacts under the requested output directory:

```powershell
python -m experiments.il.run_experiment --corpus datasets/il-corpus --tracks tracks --out runs/il
```

Use `--help` for the complete set of optional overrides. Keep the output directory separate for each condition, budget, and seed so that weights and JSON results cannot be accidentally reused.

## Corpus schema and preprocessing

The corpus directory contains `manifest.json` and `examples.jsonl`. Each non-empty JSONL record has:

- `policy_observation`: the canonical policy observation accepted by `autocross.datagen.dataset.policy_features`;
- `condition`: `optimal`, `approx`, `heuristic`, or `cert_control`;
- `split`: `train`, `val`, or `test`;
- `scenario_id` and `course_family`;
- `label`: an object containing `feasible`, `offset_mm`, `accel_mm_s2`, `certificate_path`, and `certificate_ok`.

`scenario_id` is the grouping/statistical unit and MUST NOT cross split boundaries. By default, infeasible labels are dropped before feature construction; the run metadata reports how many were dropped. Features are generated only through `policy_features`, using the fixed preview grid `0,1,...,10` metres. Offset and acceleration labels are converted from millimetres and millimetres/second² to metres and metres/second² for training.

## Determinism, missing inputs, and statistical caveat

Reproduction requires the same corpus bytes, track YAML bytes, simulator/configuration files, dependency versions, and explicit seed. Do not rely on filesystem ordering or parallel generation; use `--workers 1`. Missing corpus files, malformed JSON, missing required fields, invalid split names, or a scenario appearing in multiple splits are hard errors. Missing optional PyTorch makes the training module unavailable rather than silently falling back; missing CARLA is relevant only to CARLA generation/evaluation. Missing track files, weights, or output-parent permissions are likewise fatal inputs, not reasons to substitute another track or model.

Reported confidence intervals and bootstrap summaries MUST resample **courses/tracks**, not individual time steps or JSONL rows. Rows from one course are correlated, and treating them as independent substantially understates uncertainty. The corpus split is by scenario, while track-level evaluation is the independent statistical unit for the experiment; report the number of distinct courses alongside every aggregate metric.
