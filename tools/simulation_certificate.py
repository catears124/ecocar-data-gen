"""Emit and check a simulator-evaluated finite-candidate decision certificate.

    python tools/simulation_certificate.py RUN_DIR --step 160 --out tmp/simcert.json

The certificate declares the measured state, vehicle constants, preview, the
finite candidate command sequences, every candidate's simulator-exact cost, the
selected first command, and the baseline candidate.  The checker reruns the
harness's bicycle transition function and verifies the selection.

Scope: best among the declared finite candidate set in the bicycle simulator at
the recorded t.  Not optimality over all continuous controls; see
``docs/simulation_optimality.md``.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "teams"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from autocross.api import TrackInfo, VehicleParams  # noqa: E402
from autocross.config import load_sim_config  # noqa: E402
from autocross.simcert.certificate import SCHEMA, SCHEMA_VERSION, check_certificate  # noqa: E402
from autocross.simcert.sim import first_command, rollout, rollout_policies  # noqa: E402
from sim_opt import SimOptController  # noqa: E402


def vehicle_params() -> VehicleParams:
    cfg = load_sim_config()
    b = dict(cfg["vehicle"]["bicycle"])
    vc = cfg["vehicle"]
    return VehicleParams(
        float(b["mass"]), float(b["wheelbase"]), float(b["track_width"]),
        float(b["length"]), float(b["width"]), float(b["cg_to_front"]),
        float(b["wheelbase"] - b["cg_to_front"]),
        float(np.radians(vc["max_steer_deg"])), float(vc["steer_rate_limit"]),
        float(vc["mu"]), float(vc["max_accel"]), float(vc["max_decel"]),
        float(vc["max_speed"]), 0.05, 0)


def empty_track() -> TrackInfo:
    return TrackInfo("course", False, 0, np.zeros((0, 2)), np.zeros(0), np.zeros(0),
                     np.zeros((0, 2)), np.zeros(0, int), np.zeros((2, 2)), np.zeros((2, 2)),
                     np.zeros((0, 2, 2)), False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--step", type=int, default=160)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    run = Path(a.run_dir)
    if not run.is_absolute():
        run = ROOT / run
    steps = pickle.loads((run / "observations.pkl").read_bytes())
    if not (0 <= a.step < len(steps)):
        raise SystemExit(f"step {a.step} outside 0..{len(steps) - 1}")
    obs, _cmd = steps[a.step]
    vp = vehicle_params()
    ctrl = SimOptController({})
    ctrl.on_start(empty_track(), vp)
    base = ctrl._baseline.step(obs)
    road = obs.road
    cands = ctrl._candidates(base)
    inst = ctrl._instance(obs, road, cands)
    res = rollout(inst)
    dummy = tuple(tuple((0.0, 0.0, 0.0) for _ in range(ctrl.H)) for _ in ctrl.policies)
    inst_p = ctrl._instance(obs, road, dummy)
    res_p = rollout_policies(inst_p, ctrl.policies)
    seqs = list(cands)
    for j in range(len(ctrl.policies)):
        seqs.append(tuple(tuple(float(v) for v in row) for row in res_p["realized"][j]))
    inst_all = ctrl._instance(obs, road, tuple(seqs))
    inst_all = inst_all.__class__(**{**inst_all.__dict__, "baseline_index": len(cands) - 1})
    res_all = rollout(inst_all)
    costs = [list(c) for c in res_all["costs"]]
    selected = min(range(len(costs)), key=lambda k: costs[k])
    first = first_command(inst_all, selected)
    cert = {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        "model": "bicycle-sim-1",
        "step": int(a.step), "t": float(obs.t),
        "vehicle": ctrl.vcfg,
        "state0": [float(v) for v in inst_all.state0],
        "app0": [float(v) for v in inst_all.app0],
        "acc0": [float(v) for v in inst_all.acc0],
        "preview_cl": [[float(x), float(y)] for x, y in road.preview_centerline],
        "preview_s": [float(x) for x in road.preview_s],
        "preview_wl": [float(x) for x in road.preview_width_left],
        "preview_wr": [float(x) for x in road.preview_width_right],
        "preview_k": [float(x) for x in road.preview_curvature],
        "obstacles": [[float(o.distance), float(o.lateral_min), float(o.lateral_max)]
                      for o in (road.obstacles or [])],
        "half_width": float(inst_all.half_width), "half_length": float(inst_all.half_length),
        "margin": float(inst_all.margin), "t_tail": float(inst_all.t_tail),
        "max_steer_rate": float(inst_all.max_steer_rate),
        "candidates": [[[float(v) for v in c] for c in seq] for seq in seqs],
        "costs": costs,
        "selected": int(selected),
        "first_command": [float(v) for v in first],
        "baseline_index": int(len(cands) - 1),
        "baseline_cost": costs[len(cands) - 1],
        "selected_cost": costs[selected],
        "control_resolution": {"steer": 0.2, "pedal": 0.4},
        "scope": ("best among the declared finite candidate sequences, evaluated by the "
                  "harness bicycle transition function; not all continuous controls"),
    }
    # Measured first-command sensitivity.  This is a declared/measured
    # Lipschitz estimate, not a formally validated bound; the certificate's
    # epsilon is conditional on it.
    def scalar(seq):
        inst2 = ctrl._instance(obs, road, (tuple(seq),))
        r2 = rollout(inst2)
        c2 = r2["costs"][0]
        return float(c2[0]) * 1_000_000.0 - float(c2[1])
    base_score = scalar(seqs[selected])
    best = 0.0
    delta = 0.2
    for comp in range(3):
        for sign in (-1.0, 1.0):
            seq2 = list(seqs[selected])
            row = list(seq2[0])
            row[comp] = float(np.clip(row[comp] + sign * delta, -1.0, 1.0))
            seq2[0] = tuple(row)
            best = max(best, abs(scalar(seq2) - base_score) / delta)
    lipschitz_milli = int(round(best * ctrl.H * 1000.0))
    resolution_milli = 400
    cert["lipschitz_milli"] = lipschitz_milli
    cert["resolution_milli"] = resolution_milli
    cert["epsilon_milli"] = lipschitz_milli * resolution_milli // 1000
    cert["epsilon_scope"] = ("measured first-command sensitivity x horizon; conditional, "
                             "not a formally validated Lipschitz bound")
    report = check_certificate(cert)
    cert["check_ok"] = bool(report.get("ok"))
    cert["check_errors"] = report.get("errors", [])
    out = Path(a.out) if a.out else run / f"simcert_{a.step:06d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cert, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"out": str(out), "selected": selected, "cost": costs[selected],
                      "baseline": costs[len(cands) - 1], "first": first,
                      "check_ok": report.get("ok"), "errors": report.get("errors"),
                      "candidates": len(costs)}, indent=1))
    if a.check and not report.get("ok"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
