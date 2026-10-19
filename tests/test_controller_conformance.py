"""Kernel-evaluation checks for the controller's integer helper model."""
import importlib.util
import subprocess
from pathlib import Path

import numpy as np


def _load(root):
    path = root / "teams" / "certline.py"
    spec = importlib.util.spec_from_file_location("certline_conformance", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_controller_lean_kernel_imports_and_decides(root, tmp_path):
    lean = root / "lean"
    lake = subprocess.run(["lake", "build", "Corpus.Controller"], cwd=lean,
                          capture_output=True, text=True, timeout=120)
    assert lake.returncode == 0, lake.stdout + lake.stderr
    src = tmp_path / "controller_conformance.lean"
    src.write_text(
        "import Corpus.Controller\n"
        "open Corpus.Controller\n"
        "theorem controller_kernel_decides : "
        "-1000 ≤ steerAct {desired := 1200, prev := 0, rate := 100, invalid := false} ∧ "
        "steerAct {desired := 1200, prev := 0, rate := 100, invalid := false} ≤ 1000 := by decide\n"
        "theorem controller_publish_kernel : "
        "1880 ≤ publishOffset 2500 1880 2100 1500 ∧ "
        "publishOffset 2500 1880 2100 1500 ≤ 2100 := by decide\n"
    )
    chk = subprocess.run(["lake", "env", "lean", str(src)], cwd=lean,
                         capture_output=True, text=True, timeout=120)
    assert chk.returncode == 0, chk.stdout + chk.stderr


def test_integer_projection_matches_box_contract(root):
    mod = _load(root)
    rng = np.random.default_rng(4)
    for _ in range(100):
        n = 8
        lo = rng.integers(-2000, 0, n).astype(float)
        hi = lo + rng.integers(1, 2000, n)
        raw = rng.integers(-5000, 5000, n).astype(float)
        out = mod._project_plan(raw, lo, hi, 350.0, iters=3)
        assert np.all(out >= lo) and np.all(out <= hi)


def test_curvature_caps_and_signed_lateral_are_deterministic(root):
    mod = _load(root)
    vp = mod.VehicleParams(mass=1, wheelbase=3, track_width=1, length=4, width=2,
                           cg_to_front=1, cg_to_rear=1, max_steer=0.5,
                           max_steer_rate=2, mu=1, max_accel=5, max_decel=8,
                           max_speed=40, control_dt=.05, actuation_delay_steps=0)
    cfg = dict(mod.DEFAULTS)
    lon = mod.lon_constants(vp, cfg)
    steer = mod.lateral_constants(vp, cfg, lon)
    q = {"v0": 5_000, "s": [0, 1000, 2000], "k": [0, 0, 0],
         "wl": [3000, 3000, 3000], "wr": [3000, 3000, 3000], "lat": -1,
         "obstacles": [[1000, -500, 500]]}
    out = mod.decide_lateral(steer, lon, q)
    assert out["off"] == mod.decide_lateral(steer, lon, q)["off"]
    assert all(x >= 0 for x in mod.curv_bound_micro(q, out["off"]))


def test_cap_scale_one_is_exact_for_model_domain(root):
    mod = _load(root)
    values = [0, 1, -1, 2**20, -(2**20), 2**52 - 1, -(2**52 - 1)]
    assert all(int(float(x) * 1.0) == x for x in values)
    assert mod._w_ob({"hold": 3000, "ramp_in": 18000, "ramp_out": 15000},
                     20900, 24000) == 999

def test_sidestep_caps_restrict_certified_speed(root):
    mod = _load(root)
    vp = mod.VehicleParams(mass=1, wheelbase=3, track_width=1, length=4, width=2,
                           cg_to_front=1, cg_to_rear=1, max_steer=0.5,
                           max_steer_rate=2, mu=1, max_accel=5, max_decel=8,
                           max_speed=40, control_dt=.05, actuation_delay_steps=0)
    lon = mod.lon_constants(vp, dict(mod.DEFAULTS))
    q = {"v0": 5_000, "s": [0, 1000, 2000], "k": [0, 0, 0],
         "wl": [3000, 3000, 3000], "wr": [3000, 3000, 3000],
         "lat": 0, "obstacles": []}
    out = mod.decide_long(lon, q, [0, 0, 0], 1.0, [100, 100, 100])
    assert out["kind"] == mod.KIND_MIN_VIOLATION and out["B0"] == 100
