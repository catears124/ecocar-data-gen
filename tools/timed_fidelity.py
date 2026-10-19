"""Model-fidelity report for ``timed-1`` against recorded closed-loop telemetry.

    python tools/timed_fidelity.py RUN_DIR [--horizons 1,2,4,6,10] [--out fidelity.json]

Replays the recorded observation stream through the same encoder the controller used,
then for every start step simulates the *actually executed* command sequence for
``h`` control periods with the model and compares against the observations:

  * d   lateral error vs the road model's lateral offset          [m]
  * psi heading error vs the road model's heading error           [rad]
  * r   yaw rate vs measured yaw rate                             [rad/s]
  * vy  lateral velocity vs measured                              [m/s]
  * v   longitudinal speed vs measured                            [m/s]

Reported as p50/p95/max absolute error per horizon.  This measures the model's
predictive accuracy, not its optimality (which is what the Lean checker proves).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocross.timed.encode import Encoder  # noqa: E402
from autocross.timed.model import DEFAULT_BLOCKS, DEFAULT_CONSTS, Block, tick  # noqa: E402

NAMES = ("d", "psi", "r", "vy", "v")


def report(steps, insts, horizons=(1, 2, 4, 6, 10)) -> dict:
    out: dict = {"starts": 0, "horizons": {}}
    emax = max(horizons)
    rows = []
    for k in range(len(steps) - emax):
        inst = insts[k]
        if inst is None:
            continue
        out["starts"] += 1
        st = inst.start
        row = []
        for j in range(emax):
            cmd = steps[k + j][1]
            u = int(round(float(cmd.steer) * 1000))
            p = int(round((float(cmd.throttle) - float(cmd.brake)) * 1000))
            for _ in range(2):
                st, _v = tick(inst, st, u, p)
            ob = steps[k + j + 1][0]
            road = ob.road
            row.append((st.d / 1000.0 - float(road.lateral_offset),
                        st.psi / 1e6 - float(road.heading_error),
                        st.r / 1e6 - float(ob.yaw_rate),
                        st.vy / 1000.0 - float(ob.vy),
                        st.v / 1000.0 - float(ob.vx)))
        rows.append(row)
    arr = np.array(rows) if rows else np.zeros((0, emax, 5))
    for h in horizons:
        e = np.abs(arr[:, h - 1, :])
        out["horizons"][str(h)] = {
            n: {"p50": float(np.median(e[:, i])), "p95": float(np.percentile(e[:, i], 95)),
                "max": float(e[:, i].max())} if e.size else None
            for i, n in enumerate(NAMES)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--horizons", default="1,2,4,6,10")
    ap.add_argument("--config", default=None, help="controller YAML (same consts/blocks as the run)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = Path(a.run_dir)
    if not run.is_absolute():
        run = ROOT / run
    steps = pickle.loads((run / "observations.pkl").read_bytes())
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
        except Exception:  # noqa: BLE001 - an un-encodable step is skipped
            insts.append(None)
    rep = report(steps, insts, tuple(int(x) for x in a.horizons.split(",")))
    rep["run"] = str(run)
    rep["encoded_steps"] = sum(i is not None for i in insts)
    rep["steps"] = len(steps)
    out = Path(a.out) if a.out else run / "timed_fidelity.json"
    out.write_text(json.dumps(rep, indent=1))
    for h, d in rep["horizons"].items():
        print(f"t={0.05 * int(h):.2f}s  " + "  ".join(
            f"{n} p95 {d[n]['p95']:.4f} max {d[n]['max']:.4f}" for n in NAMES))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
