"""Independent tests for the finite actuator model and its certificate."""
from __future__ import annotations

import itertools
import random

import pytest

from autocross.actuator.certificate import (
    CORRUPTIONS, certificate_from_diag, check_certificate, corrupt,
)
from autocross.actuator.model import ActModel, ActState, COST_TIME_WEIGHT, isqrt
from autocross.actuator.solver import assert_vector_parity, solve, solve_sparse


def make_model(*, horizon=2, offsets=(-500, 0, 500), psis=(-40_000, 0, 40_000),
               speeds=(4_000_000, 9_000_000, 16_000_000),
               deltas=(-200_000, 0, 200_000), u=(-1000, 0, 1000),
               a=(-3000, 0, 3000), b0=9_000_000, terminal_distance=0,
               curvature=(0, 0, 0, 0, 0)):
    n = horizon + 1
    return ActModel(
        consts={"lon": {"a_lat": 6589, "a_acc": 5200, "a_brk": 6589, "v_max": 40000,
                        "r_h": 1809, "hold": 3000, "veh_width": 2160},
                "margin": 300, "step": 1000, "max_steer": 610865,
                "steer_rate_urad_s": 1221730, "steer_tau_us": 60000,
                "wheelbase_mm": 3000},
        q={"base": {"s": [1000 * i for i in range(n)],
                    "k": list(curvature)[:n] + [0] * max(0, n - len(curvature)),
                    "wl": [5000] * n, "wr": [5000] * n, "obstacles": []}, "lat": 0},
        offsets=offsets, psis=psis, speeds=speeds, deltas=deltas,
        steer_inputs=u, acc_inputs=a, horizon=horizon,
        start=ActState(0, 0, 0, b0, 0), terminal_distance=terminal_distance)


def exhaustive(model):
    best = None
    for acts in itertools.product(itertools.product(model.steer_inputs, model.acc_inputs),
                                  repeat=model.horizon):
        s = model.start
        total = 0
        ok = True
        for u, a in acts:
            t = model.action_ok(s, u, a)
            if t is None:
                ok = False
                break
            total += model.action_cost(s, t)
            s = t
        if ok:
            c = model.terminal_cost(s)
            if c is not None:
                total += COST_TIME_WEIGHT * c
                if best is None or total < best:
                    best = total
    return best


@pytest.mark.parametrize("kwargs", [
    {}, {"horizon": 1}, {"b0": 4_000_000},
    {"curvature": (0, 0, 2000, 5000, 0)},
])
def test_tiny_solver_matches_exhaustive(kwargs):
    m = make_model(**kwargs)
    _t, cost, _u, _d = solve(m)
    assert cost == exhaustive(m)


def test_vector_and_scalar_agree_and_certificate_accepts():
    m = make_model(horizon=2)
    assert assert_vector_parity(m, 150) > 100
    b, cost, a, d = solve_sparse(m)
    cert = certificate_from_diag(m, d, claimed=cost, first_action=a)
    rep = check_certificate(cert)
    assert rep["ok"], rep
    if cost is not None:
        assert b is not None and a is not None
        assert m.action_ok(m.start, *a) == b
        assert m.action_cost(m.start, b) + d["values"][1][b] == cost


def test_adversarial_mutations_are_rejected():
    m = make_model(horizon=2)
    _b, cost, a, d = solve_sparse(m)
    cert = certificate_from_diag(m, d, claimed=cost, first_action=a)
    assert check_certificate(cert)["ok"]
    for kind in CORRUPTIONS:
        assert not check_certificate(corrupt(cert, kind))["ok"], kind


def test_random_tiny_models_agree_with_exhaustive():
    rng = random.Random(0xAC7)
    for _ in range(8):
        m = make_model(horizon=rng.choice((1, 2)),
                       b0=rng.choice((4_000_000, 9_000_000)),
                       speeds=rng.choice(((4_000_000, 9_000_000, 16_000_000),
                                          (1_000_000, 4_000_000, 16_000_000))))
        _b, cost, _a, _d = solve(m)
        assert cost == exhaustive(m)
