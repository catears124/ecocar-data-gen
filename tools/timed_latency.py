"""Deployment measurement for the ``timed-1`` runtime solver.

    python tools/timed_latency.py --observations experiments/bench/basic/timed_practice_basic_s0 \
        --repeat 3 --out experiments/bench/latency_micro.json

Times ``Solver.solve`` (model construction from a recorded observation, full exact
enumeration, incumbent pruning) and reports p50/p95/p99/max wall time, RSS, and the
node/leaf counts.  Use a recorded observation stream so the instances are the ones
the controller actually solved.  This measures the solver only; the closed-loop
latency (encoder + path construction + IPC) is measured by ``tools/bench.py``.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocross.timed.encode import EncodeError, Encoder  # noqa: E402
from autocross.timed.model import DEFAULT_BLOCKS, DEFAULT_CONSTS, Block  # noqa: E402
from autocross.timed.vector import Solver  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--observations", required=True, help="run dir with observations.pkl")
    ap.add_argument("--config", default=None)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = Path(a.observations)
    if not run.is_absolute():
        run = ROOT / run
    pkl = run / "observations.pkl" if run.is_dir() else run
    steps = pickle.loads(pkl.read_bytes())
    consts = dict(DEFAULT_CONSTS)
    blocks = DEFAULT_BLOCKS
    if a.config:
        import yaml
        cfg = yaml.safe_load((Path(a.config) if Path(a.config).is_absolute() else ROOT / a.config).read_text())
        consts.update({k: int(v) for k, v in (cfg.get("consts") or {}).items()})
        if cfg.get("blocks"):
            blocks = tuple(Block(int(b["steps"]), tuple(int(x) for x in b["steer"]),
                                 tuple(int(x) for x in b["pedal"])) for b in cfg["blocks"])
    enc = Encoder(consts=consts, blocks=blocks)
    insts = []
    for obs, _cmd in steps:
        try:
            insts.append(enc.encode(obs))
        except EncodeError:
            pass
    if a.limit:
        insts = insts[:a.limit]
    times, leaves, ticks = [], [], []
    prev = None
    import psutil
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    peak = rss0
    for _ in range(a.repeat):
        for inst in insts:
            t0 = time.perf_counter()
            plan, J, st = Solver(inst).solve(prev)
            times.append((time.perf_counter() - t0) * 1000.0)
            leaves.append(st["leaves"][-1])
            ticks.append(st["ticks"])
            prev = plan
            peak = max(peak, proc.memory_info().rss)
    t = np.array(times)
    rep = {"instances": len(insts), "solves": len(times), "repeat": a.repeat,
           "model_ms": {"p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
                        "p99": float(np.percentile(t, 99)), "max": float(t.max()),
                        "mean": float(t.mean())},
           "leaves": {"p50": float(np.percentile(leaves, 50)), "max": int(max(leaves))},
           "ticks": {"p50": float(np.percentile(ticks, 50)), "max": int(max(ticks))},
           "peak_rss_mb": peak / 1e6, "baseline_rss_mb": rss0 / 1e6,
           "deadline_ms": 50.0}
    out = Path(a.out) if a.out else run / "timed_latency.json"
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
