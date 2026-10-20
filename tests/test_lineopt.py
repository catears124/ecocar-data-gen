"""Unit tests for the hierarchical line/speed planner and the exact lattice DP."""
from __future__ import annotations

import math

import numpy as np

from autocross.lineopt.dp import evaluate_plan, solve_fast
from autocross.lineopt.path import (path_curvatures, plan_path, profile_time_us,
                                    speed_profile)
from autocross.optimal.model import Model, State
from autocross.optimal.solver import solve


def _consts():
    lon = dict(a_lat=6730, a_acc=5200, a_brk=6730, a_max=9000, v_max=40000,
               r_h=2078, hold=3000, ramp_in=18000, ramp_out=15000, r_sw_m=15,
               veh_width=2160)
    return {"lon": lon, "margin": 300, "step": 1000, "kappa_max": 203000}


def _model(n=7, offsets=(-1000, -500, 0, 500, 1000), k=None, wl=2750, speeds=None,
           start_b=8_000_000, horizon=None, terminal=16_000_000):
    k = list(k or [0] * n)
    wl = [wl] * n
    q = {"base": {"v0": int(math.isqrt(start_b)), "s": [i * 1000 for i in range(n)],
                  "k": k, "wl": wl, "wr": list(wl), "obstacles": []}, "lat": 0}
    speeds = speeds or tuple(int(round(v * 1000)) ** 2 for v in (2, 4, 6, 8, 10))
    return Model(consts=_consts(), q=q, offsets=tuple(offsets), speeds=tuple(speeds),
                 horizon=(n - 1 if horizon is None else horizon),
                 start=State(0, 0, 0, start_b), terminal_cap=terminal)


def test_fast_dp_matches_scalar_value_and_plan():
    model = _model()
    scalar = solve(model)
    fast = solve_fast(model)
    assert scalar.cost == fast.value
    if scalar.cost is not None:
        assert [(s.station, s.previous, s.current, s.b) for s in scalar.plan] == \
               [(s.station, s.previous, s.current, s.b) for s in fast.plan]
        assert evaluate_plan(model, fast.plan) == fast.value


def test_fast_dp_matches_scalar_on_curvature_and_corridor():
    model = _model(n=8, k=[0, 40_000, 80_000, 40_000, 0, -40_000, -80_000, -40_000],
                   wl=2400, offsets=(-800, -400, 0, 400, 800))
    scalar = solve(model)
    fast = solve_fast(model)
    assert scalar.cost == fast.value
    if fast.value is not None:
        assert evaluate_plan(model, fast.plan) == fast.value


def test_path_dp_starts_at_observed_offset_and_respects_corridor():
    model = _model(n=12, offsets=tuple(range(-1200, 1201, 200)))
    # Put the observed start on the lattice; the synthetic model uses offset 0.
    pp = plan_path(model)
    assert pp.cost_us is not None
    assert len(pp.offsets) == model.horizon + 1
    assert pp.offsets[0] == model.start.current
    for i, off in enumerate(pp.offsets):
        assert model.corridor_ok(i, off)
    curv = path_curvatures(model, pp.offsets)
    assert int(np.max(curv)) <= model.kappa_max


def test_speed_profile_respects_terminal_cap_and_positive_speeds():
    model = _model(n=41, offsets=(0,), k=[0] * 41, horizon=40, terminal=11_900_000,
                   speeds=(4_000_000, 16_000_000, 36_000_000, 64_000_000,
                           100_000_000, 144_000_000, 196_000_000, 256_000_000))
    b = speed_profile(model, tuple([0] * 41), 5.9, v_max_mps=20.0)
    assert np.all(b > 0)
    assert float(b[-1]) <= 11_900_000 / 1e6 + 1e-6
    # Acceleration is bounded by the declared budget (with the circle reserve).
    for i in range(1, len(b)):
        dv2 = b[i] - b[i - 1]
        assert dv2 <= 2 * 5.2 * 1.0 + 1e-6 or b[i] < 1.0
    assert profile_time_us(model, tuple([0] * 41), b) > 0
