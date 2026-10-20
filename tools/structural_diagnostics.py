"""Structural diagnostics for the time-indexed actuator controller.

Given a recorded run (``observations.pkl``), this tool answers the questions that
drove the redesign, using the same declared model and the same exact solver:

* how often is the model infeasible, and which constraint class is responsible
  (corridor, friction, terminal, or the move-blocked action set)?
* is the baseline's published command representable in the declared action set,
  and is the baseline candidate admissible after it is added?
* how large is the model's prediction error, binned by measured lateral
  acceleration (the regime where racing is decided)?
* how often does widening the action set restore feasibility, i.e. is the
  infeasibility physical or an artifact of the policy class?

The tool never claims that a relaxed model is the real vehicle; relaxations are
diagnostic counterfactuals.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "teams") not in sys.path:
    sys.path.insert(0, str(ROOT / "teams"))

from autocross.timed.encode import Encoder  # noqa: E402
from autocross.timed.model import Block, DEFAULT_BLOCKS, DEFAULT_CONSTS, tick  # noqa: E402
from autocross.timed.vector import DomainOverflow, Solver  # noqa: E402


def _solve(inst):
    try:
        plan, J, _ = Solver(inst).solve()
        return int(J[0]) == 0, int(J[0]), plan
    except DomainOverflow:
        return False, None, None


def _add_option(values, v, lo, hi):
    return tuple(sorted(set(int(x) for x in values) | {max(lo, min(hi, int(v)))}))


def _augment(inst, off, p, steer_extra=(), pedal_extra=()):
    blocks = []
    for b in inst.blocks:
        steer = _add_option(b.steer, off, -2000, 2000)
        steer = tuple(sorted(set(steer) | {int(x) for x in steer_extra}))
        pedal = _add_option(b.pedal, p, -1000, 1000)
        pedal = tuple(sorted(set(pedal) | {int(x) for x in pedal_extra}))
        blocks.append(Block(b.steps, steer, pedal))
    return dataclasses.replace(inst, blocks=tuple(blocks))


def _wide(inst):
    blocks = tuple(Block(b.steps, (-1000, -500, 0, 500, 1000), (-1000, -500, 0, 500, 1000))
                   for b in inst.blocks)
    return dataclasses.replace(inst, blocks=blocks)


def _relax(inst, *, corridor=0, friction=1.0, terminal=1):
    c = dict(inst.consts)
    if corridor:
        wl = tuple(int(x) + int(corridor) for x in inst.wl)
        wr = tuple(int(x) + int(corridor) for x in inst.wr)
    else:
        wl, wr = inst.wl, inst.wr
    if friction != 1.0:
        c["MU"] = int(c["MU"] * friction)
    if terminal != 1:
        c["A_LT"] = int(c["A_LT"] * terminal)
        c["A_BT"] = int(c["A_BT"] * terminal)
        c["RMIN"] = int(c["RMIN"] / terminal)
    return dataclasses.replace(inst, consts=c, wl=wl, wr=wr)


def fidelity(steps, insts, horizon=4):
    rows = []
    for k in range(len(steps) - horizon):
        inst = insts[k]
        if inst is None:
            continue
        st = inst.start
        for j in range(horizon):
            cmd = steps[k + j][1]
            u = int(round(float(cmd.steer) * 1000))
            p = int(round((float(cmd.throttle) - float(cmd.brake)) * 1000))
            for _ in range(2):
                st, _v = tick(inst, st, u, p)
            ob = steps[k + j + 1][0]
            road = ob.road
            rows.append((abs(st.d / 1000.0 - float(road.lateral_offset)),
                         abs(st.psi / 1e6 - float(road.heading_error)),
                         abs(st.r / 1e6 - float(ob.yaw_rate)),
                         abs(st.vy / 1000.0 - float(ob.vy)),
                         abs(st.v / 1000.0 - float(ob.vx)),
                         abs(float(ob.ay))))
    if not rows:
        return {}
    a = np.asarray(rows)
    out = {}
    for lo, hi in ((0, 3), (3, 5), (5, 7), (7, 9), (9, 99)):
        m = (a[:, 5] >= lo) & (a[:, 5] < hi)
        if m.any():
            out[f"{lo}-{hi}"] = {
                "n": int(m.sum()),
                "d_p95": float(np.percentile(a[m, 0], 95)),
                "psi_p95": float(np.percentile(a[m, 1], 95)),
                "r_p95": float(np.percentile(a[m, 2], 95)),
                "v_p95": float(np.percentile(a[m, 4], 95)),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = Path(a.run_dir)
    if not run.is_absolute():
        run = ROOT / run
    steps = pickle.loads((run / "observations.pkl").read_bytes())
    enc = Encoder()
    insts = []
    for obs, _cmd in steps:
        try:
            insts.append(enc.encode(obs))
        except Exception:  # noqa: BLE001
            insts.append(None)

    idxs = [i for i, inst in enumerate(insts) if inst is not None][a.start : a.start + a.limit]
    counts = Counter()
    examples: dict[str, dict] = {}
    for idx in idxs:
        inst = insts[idx]
        cmd = steps[idx][1]
        ok, viol, _plan = _solve(inst)
        counts["steps"] += 1
        counts["feasible"] += int(ok)
        if not ok:
            counts["infeasible"] += 1
            for name, rel in (
                ("relax_corridor", _relax(inst, corridor=100_000)),
                ("relax_friction", _relax(inst, friction=100.0)),
                ("relax_terminal", _relax(inst, terminal=100)),
                ("relax_all", _relax(inst, corridor=100_000, friction=100.0, terminal=100)),
                ("wide_actions", _wide(inst)),
            ):
                rok, _rv, _rp = _solve(rel)
                counts[name] += int(rok)
            if "first" not in examples:
                examples["first"] = {"step": idx, "t": float(steps[idx][0].t)}
        # Baseline command representability and an augmented (baseline-inclusive)
        # action universe.
        u = int(round(float(cmd.steer) * 1000))
        p = int(round((float(cmd.throttle) - float(cmd.brake)) * 1000))
        off = int(u - inst.start.sigma)
        rep = all(any(abs(s - off) < 1 for s in b.steer) and any(pd == p for pd in b.pedal)
                  for b in inst.blocks)
        counts["baseline_representable"] += int(rep)
        aug = _augment(inst, off, p)
        aok, _av, _ap = _solve(aug)
        counts["baseline_augmented_feasible"] += int(aok)
        if not rep and "baseline_not_representable" not in examples:
            examples["baseline_not_representable"] = {
                "step": idx, "off": off, "p": p,
                "blocks": [{"steer": list(b.steer), "pedal": list(b.pedal)}
                           for b in inst.blocks]}
    report = {
        "run": str(run),
        "steps_examined": len(idxs),
        "counts": dict(counts),
        "fractions": {
            "feasible": counts["feasible"] / max(counts["steps"], 1),
            "infeasible": counts["infeasible"] / max(counts["steps"], 1),
            "relax_corridor_recovers": counts["relax_corridor"] / max(counts["infeasible"], 1),
            "relax_friction_recovers": counts["relax_friction"] / max(counts["infeasible"], 1),
            "relax_terminal_recovers": counts["relax_terminal"] / max(counts["infeasible"], 1),
            "relax_all_recovers": counts["relax_all"] / max(counts["infeasible"], 1),
            "wide_actions_recovers": counts["wide_actions"] / max(counts["infeasible"], 1),
            "baseline_representable": counts["baseline_representable"] / max(counts["steps"], 1),
            "baseline_augmented_feasible": counts["baseline_augmented_feasible"] / max(counts["steps"], 1),
        },
        "fidelity_by_ay": fidelity(steps, insts),
        "examples": examples,
    }
    out = Path(a.out) if a.out else run / "structural_diagnostics.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
