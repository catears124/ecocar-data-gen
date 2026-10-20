"""Tests for the simulator-exact finite-candidate certificate."""
from __future__ import annotations

import copy

import numpy as np

from autocross.config import load_sim_config
from autocross.simcert.certificate import SCHEMA, SCHEMA_VERSION, check_certificate
from autocross.simcert.sim import SimInstance, rollout


def _instance():
    cfg = load_sim_config()
    vcfg = cfg["vehicle"]
    b = dict(vcfg["bicycle"])
    vcfg = {"bicycle": b, "max_steer_deg": float(vcfg["max_steer_deg"])}
    H = 3
    cands = (tuple([(0.0, 0.0, 0.0)] * H), tuple([(0.0, 0.4, 0.0)] * H))
    n = 12
    return SimInstance(
        vehicle=vcfg, state0=(0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0),
        app0=(0.0, 0.0, 0.0), acc0=(0.0, 0.0), candidates=cands,
        preview_cl=tuple((float(i), 0.0) for i in range(n)),
        preview_s=tuple(float(i) for i in range(n)),
        preview_wl=tuple([3.0] * n), preview_wr=tuple([3.0] * n),
        preview_k=tuple([0.0] * n), obstacles=(),
        half_width=1.08, half_length=2.4, margin=0.25, baseline_index=0,
    )


def _cert(inst, result, selected):
    return {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        "vehicle": inst.vehicle, "state0": list(inst.state0), "app0": list(inst.app0),
        "acc0": list(inst.acc0),
        "preview_cl": [list(p) for p in inst.preview_cl],
        "preview_s": list(inst.preview_s), "preview_wl": list(inst.preview_wl),
        "preview_wr": list(inst.preview_wr), "preview_k": list(inst.preview_k),
        "obstacles": [list(o) for o in inst.obstacles],
        "half_width": inst.half_width, "half_length": inst.half_length,
        "margin": inst.margin, "t_tail": inst.t_tail,
        "max_steer_rate": inst.max_steer_rate,
        "candidates": [[list(c) for c in seq] for seq in inst.candidates],
        "costs": [list(c) for c in result["costs"]],
        "selected": selected, "first_command": list(inst.candidates[selected][0]),
        "baseline_index": inst.baseline_index,
        "baseline_cost": list(result["costs"][inst.baseline_index]),
        "lipschitz_milli": 1000, "resolution_milli": 400, "epsilon_milli": 400,
    }


def test_certificate_accepts_first_minimum():
    inst = _instance()
    result = rollout(inst)
    selected = min(range(len(result["costs"])), key=lambda k: result["costs"][k])
    cert = _cert(inst, result, selected)
    report = check_certificate(cert)
    assert report["ok"], report


def test_certificate_rejects_wrong_selection_and_cost():
    inst = _instance()
    result = rollout(inst)
    selected = min(range(len(result["costs"])), key=lambda k: result["costs"][k])
    bad = _cert(inst, result, selected)
    bad["selected"] = (selected + 1) % len(result["costs"])
    assert not check_certificate(bad)["ok"]
    bad2 = copy.deepcopy(_cert(inst, result, selected))
    bad2["costs"][0][1] += 1
    assert not check_certificate(bad2)["ok"]
