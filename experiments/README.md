# Experiments

Every number quoted in `docs/timed_controller.md` and `docs/timed_audit.md` is
produced by one of these commands; the machine-readable output is committed
alongside this file (`experiments/bench/*.json`).

Machine: Windows 11, 12 logical cores, no GPU, CPU-only; `numba` is installed but
cannot load inside the controller sandbox (`ctypes.dlopen` is denied), so every
reported controller latency is numpy/CPython.

## Closed-loop campaign

```bash
# public courses + four generated held-out courses, one seed, bicycle backend
python tools/bench.py \
  --controller timed=teams/timed_exact.py \
  --controller certline=teams/certline.py \
  --controller actuator=teams/actuator_exact.py \
  --tracks "tracks/*.yaml" \
  --tracks "datasets/optimal-full/tracks/test-course-*.yaml" \
  --jobs 2 --out experiments/bench/campaign.json
```

`--jobs 2` is used for driving-quality comparisons (parallel workers distort the
latency columns). Latency claims come from single-job runs:

```bash
python tools/bench.py --controller timed=teams/timed_exact.py \
  --tracks tracks/practice_basic.yaml --jobs 1 --out experiments/bench/latency.json
```

`--record-obs` additionally writes `observations.pkl` (the exact observation
stream the controller answered) so that the offline audit and the fidelity report
can replay it.

## Model fidelity

```bash
python tools/timed_fidelity.py experiments/bench/basic/timed_practice_basic_s0
```

Reports the model's prediction error at 1, 2, 4, 6 and 10 control periods against
the measured state, using the executed command sequence.

## Certification audit (binding + Lean)

```bash
python tools/timed_audit.py experiments/bench/basic/timed_practice_basic_s0 \
  --lean-checker lean/.lake/build/bin/timedcheck.exe --limit 40
```

Checks, per control step: the rebuilt instance hash equals the hash the
controller published (`dbg_instance_h52`), the audit's optimal plan equals the
runtime plan (`dbg_plan0..2`), the first command equals the published command
(exact floats), the Python certificate checker accepts, and (sampled) the Lean
checker accepts. A step is only counted as `lean_checked` when all of that holds.

## Reference vectors

```bash
python tools/timed_vectors.py --out lean/timed_vectors.json
```

Writes 10 instances (two full-size) with *every* plan's lexicographic cost from
the scalar reference; `tests/test_timed_model.py` and the Lean tests compare
against it.
