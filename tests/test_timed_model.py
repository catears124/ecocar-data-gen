"""Tests for the ``timed-1`` model, the numpy solver and the encoder contract.

The Lean side (certificates accepted by ``timedcheck``, corruptions rejected) is in
``tests/test_timed_lean.py``; these tests check the arithmetic and the solver
against an independent exhaustive enumeration of every plan.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from autocross.timed import model as tm
from autocross.timed import vector as tv
from autocross.timed.encode import EncodeError, Encoder, delta_update, lag
from autocross.timed.model import Block


def tiny_blocks():
    return (Block(1, (-100, 0, 100), (-1000, 0, 1000)),
            Block(2, (-200, 0, 200), (-1000, 0, 1000)))


def make(*, n=12, k=None, wl=3000, wr=3000, obstacles=(), start=None, p_prev=0,
         blocks=None, consts=None):
    c = dict(tm.DEFAULT_CONSTS)
    c.update(consts or {})
    kk = list(k) if k is not None else [0] * n
    st = start or tm.State(0, 0, 0, 9000, 0, 0, 0, 0)
    return tm.Instance(c, tuple(kk), (wl,) * n if isinstance(wl, int) else tuple(wl),
                       (wr,) * n if isinstance(wr, int) else tuple(wr), tuple(obstacles), st,
                       p_prev, blocks or tiny_blocks())


def test_instance_roundtrip_and_hash():
    m = make(k=[0, 0, 40000, 60000, 60000, 0, 0, 0, 0, 0, 0, 0])
    r = tm.Instance.from_json(m.to_json())
    assert r.to_json() == m.to_json()
    assert r.sha256() == m.sha256()
    bad = m.to_json()
    bad["consts"]["MASS"] += 1
    assert tm.Instance.from_json(bad).sha256() != m.sha256()


def test_well_formed_rejects_bad_domains():
    m = make()
    assert tm.well_formed(m) == []
    broken = m.to_json()
    broken["consts"]["STEP"] = 0
    assert any("STEP" in e for e in tm.well_formed(tm.Instance.from_json(broken)))
    two = m.__class__(**{**m.__dict__, "blocks": (Block(2, (0,), (0,)),)})
    assert any("first block" in e for e in tm.well_formed(two))


@pytest.mark.parametrize("kwargs", [
    {},
    {"start": tm.State(0, 300, -20000, 12000, 0, 0, 0, 0)},
    {"start": tm.State(0, 400, 0, 10000, 0, 0, 300, 180000)},
    {"wl": 2500, "wr": 3500, "k": [0, 0, 40000, 60000, 60000] + [0] * 7},
    {"wl": 3500, "wr": 2500, "k": [0, 0, -35000, -55000] + [0] * 8},
    {"obstacles": ((3000, 5000, 1200, 4000),)},
    {"obstacles": ((3000, 5000, -4000, -1200),)},
    {"start": tm.State(0, 0, 0, 1500, 0, 0, 0, 0)},
])
def test_exhaustive_matches_vector_solver(kwargs):
    m = make(**kwargs)
    plan, J = tm.solve_exhaustive(m)
    vplan, vJ, _ = tv.solve(m)
    assert vJ == J, (vJ, J)
    assert vplan == plan, (vplan, plan)
    assert tm.evaluate(m, vplan) == J


def test_pruning_does_not_change_the_result():
    rng = random.Random(0x71)
    for _ in range(6):
        m = make(k=[rng.randint(-80000, 80000) for _ in range(12)],
                 start=tm.State(0, rng.randint(-1200, 1200), rng.randint(-120000, 120000),
                                rng.randint(2000, 18000), rng.randint(-400, 400),
                                rng.randint(-300000, 300000), rng.randint(-300, 300),
                                rng.randint(-150000, 150000)), p_prev=rng.randint(-1000, 1000))
        base_plan, base_J, _ = tv.solve(m)
        pruned = tv.solve(m, incumbent=base_plan)
        assert pruned[0] == base_plan and pruned[1] == base_J


def test_violation_dominates_progress():
    """A slower plan that satisfies the constraints beats a faster plan that does not."""
    m = make(k=[0] * 4 + [120000] * 8, wl=1500, wr=1500)
    assert tm.well_formed(m) == []
    plan, (V, negP, E) = tm.solve_exhaustive(m)
    assert V == 0, (plan, V)          # a feasible plan exists and is chosen first
    for p in tm.all_plans(m):
        c = tm.evaluate(m, p)
        assert c[0] >= V


def test_negative_steering_is_used_when_it_is_optimal():
    m = make(start=tm.State(0, 500, 0, 11000, 0, 0, 0, 0), wl=1600, wr=3000)
    plan, J = tm.solve_exhaustive(m)
    u, p = tm.first_command(m, plan)
    st = tm.State(0, m.start.d, m.start.psi, m.start.v, m.start.vy, m.start.r,
                  tm.clamp(u, -1000, 1000), m.start.delta)
    assert u < 0, (plan, u)


def test_infeasible_instance_returns_minimum_violation():
    m = make(obstacles=((0, 12000, -2000, 2000),))     # obstacle covers the corridor
    plan, J = tm.solve_exhaustive(m)
    assert J[0] > 0
    assert J == min(tm.evaluate(m, p) for p in tm.all_plans(m))


def test_first_command_matches_published_floats():
    m = make()
    plan, _J = tm.solve_exhaustive(m)
    u, p = tm.first_command(m, plan)
    steer, thr, brk = tm.command_floats(u, p)
    assert steer == u / 1000.0
    assert (thr, brk) == ((p / 1000.0, 0.0) if p > 0 else ((0.0, -p / 1000.0) if p < 0 else (0.0, 0.0)))


def test_delta_estimator_replays_the_harness_limiter():
    c = dict(tm.DEFAULT_CONSTS)
    d = delta_update(c, 0, 0, 1000)
    # two lag steps: sigma moves at most SLEW in the first tick, then to the command
    mid = tm.clamp(0 + tm.clamp(1000, -c["SLEW"], c["SLEW"]), -1000, 1000)
    assert mid == c["SLEW"]
    want = lag(c, lag(c, 0, mid), 1000)
    assert d == want
    assert 0 < d < c["MAX_STEER"]


def test_encoder_is_deterministic_and_rejects_bad_preview():
    enc = Encoder()
    obs = _fake_obs(step=0, steer=0.0)
    a = enc.encode(obs)
    enc2 = Encoder()
    b = enc2.encode(_fake_obs(step=0, steer=0.0))
    assert a.sha256() == b.sha256()
    with pytest.raises(EncodeError):
        Encoder().encode(_fake_obs(step=0, steer=0.0, step_m=1.5))


def _fake_obs(*, step=0, steer=0.0, step_m=1.0, n=41, lateral=0.0, heading=0.0):
    class Road:
        valid = True
        preview_s = np.arange(0.0, step_m * n, step_m)
        preview_curvature = np.zeros(n)
        preview_centerline = np.column_stack((np.arange(0.0, step_m * n, step_m), np.zeros(n)))
        preview_width_left = np.full(n, 3.0)
        preview_width_right = np.full(n, 3.0)
        lateral_offset = lateral
        heading_error = heading
        obstacles = []

    class Obs:
        pass

    o = Obs()
    o.step = step
    o.steer = steer
    o.road = Road()
    o.vx, o.vy, o.yaw_rate = 10.0, 0.0, 0.0
    o.throttle, o.brake = 0.0, 0.0
    return o


def _vectors():
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "lean" / "timed_vectors.json"
    if not p.exists():
        pytest.skip("lean/timed_vectors.json not generated")
    return json.loads(p.read_text())["cases"]


@pytest.mark.parametrize("case", [c["name"] for c in _vectors()])
def test_reference_vectors_match_the_scalar_model(case):
    """Every stored plan cost is reproduced by the scalar model, and the vector
    solver returns the same optimum and first command."""
    import json
    from pathlib import Path
    doc = json.loads((Path(__file__).resolve().parents[1] / "lean" / "timed_vectors.json").read_text())
    c = next(x for x in doc["cases"] if x["name"] == case)
    m = tm.Instance.from_json(c["instance"])
    assert tm.well_formed(m) == []
    plans = [tuple(p) for p in c["plans"]]
    assert plans == list(tm.all_plans(m))
    costs = [tuple(x) for x in c["costs"]]
    # the scalar reference must reproduce its own stored costs on a sample
    step = max(1, len(plans) // 40)
    for i in range(0, len(plans), step):
        assert tm.evaluate(m, plans[i]) == costs[i], (plans[i], tm.evaluate(m, plans[i]), costs[i])
    vplan, vJ, _ = tv.solve(m)
    assert vJ == tuple(c["best_cost"])
    assert list(vplan) == c["best"]
    u, p = tm.first_command(m, vplan)
    assert [u, p] == c["first"]


def test_violation_pruning_is_sound_from_outside_the_corridor():
    """A prune must never change the optimum, even from an infeasible start state.

    Progress cannot be bounded cheaply from outside the corridor (the progress
    rate is only clamped, not bounded), so only violation is used for pruning.
    This is the regression test for a progress bound that was silently
    inadmissible: with a deliberately bad incumbent it returned a worse plan.
    """
    rng = random.Random(0xB0A2D)
    checked = 0
    for _ in range(10):
        m = make(n=8, k=[30000] * 8, wl=1800, wr=1800,
                 start=tm.State(0, rng.choice((-3000, -2500, 2500, 3000)),
                                rng.choice((-80000, 0, 80000)), 11000, 0, 0, 0, 0))
        plan, J, _ = tv.solve(m)
        for incumbent in (plan, tuple((a + 1) % b.n for a, b in zip(plan, m.blocks)),
                          tuple(0 for _ in m.blocks)):
            pp, JJ, _ = tv.solve(m, incumbent=incumbent)
            assert JJ == J, (incumbent, JJ, J)
            assert pp == plan, (incumbent, pp, plan)
            checked += 1
    assert checked == 30
