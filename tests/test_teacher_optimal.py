"""Independent tests for the exact finite-horizon teacher."""
from __future__ import annotations

from fractions import Fraction
import copy
import random

import pytest

from autocross.optimal.certificate import check_certificate, corrupt, certificate_from_solution
from autocross.optimal.model import Model, State, cdiv, isqrt, time_cost
from autocross.optimal.solver import enumerate_plans, solve


def make_model(*, seed=0, horizon=3, offsets=(-1000, 0, 1000), speeds=(1, 4, 9),
               b0=4, terminal_cap=None, acc=2, brk=2, wl=4000, wr=4000,
               obstacles=(), kappa_max=10**12, a_lat=10**9):
    n = horizon + 1
    lon = {"a_lat": a_lat, "a_acc": acc, "a_brk": brk, "a_max": 10**9,
           "v_max": 100000, "r_h": 1000, "hold": 0, "ramp_in": 0,
           "ramp_out": 0, "r_sw_m": 15, "veh_width": 2000}
    q = {"base": {"v0": b0, "s": [i * 1000 for i in range(n)],
                   "k": [0] * n, "wl": [wl] * n, "wr": [wr] * n,
                   "obstacles": [list(x) for x in obstacles]}, "lat": 0}
    consts = {"lon": lon, "margin": 0, "step": 1000, "kappa_max": kappa_max}
    return Model(consts, q, tuple(offsets), tuple(speeds), horizon,
                 State(0, 0, 0, b0),
                 max(speeds) if terminal_cap is None else terminal_cap)


def exhaustive(model):
    plans = list(enumerate_plans(model))
    return min((cost, tuple(path)) for path, cost in plans) if plans else None


def independent_bellman(model):
    def value(s):
        if s.station == model.horizon:
            return model.terminal_value(s)
        vals = []
        for t, c in model.actions(s):
            v = value(t)
            if v is not None:
                vals.append(c + v)
        return min(vals) if vals else None
    return value(model.start)


@pytest.mark.parametrize("kwargs", [
    {"b0": 1, "terminal_cap": 9}, {"b0": 3, "terminal_cap": 9},
    {"b0": 37_000_000, "terminal_cap": 9}, {"b0": 9, "acc": 1, "brk": 1},
    {"b0": 4, "acc": 1, "brk": 20}, {"b0": 4, "acc": 20, "brk": 1},
    {"b0": 4, "terminal_cap": 1}, {"b0": 4, "terminal_cap": 0},
    {"b0": 4, "wl": 1000, "wr": 1000},
    {"b0": 4, "wl": 1500, "wr": 1500},
    {"b0": 4, "obstacles": ((1000, -100, 100),)},
    {"b0": 4, "obstacles": ((1000, -3000, -2500),)},
    {"b0": 1, "offsets": (-1000, 0, 1000), "speeds": (1, 4, 9), "horizon": 2},
    {"b0": 2, "offsets": (-1000, 0, 1000), "speeds": (1, 4, 9), "horizon": 2},
    {"b0": 5, "offsets": (-1000, 0, 1000), "speeds": (1, 4, 9), "horizon": 2},
    {"b0": 8, "offsets": (-1000, 0, 1000), "speeds": (1, 4, 9), "horizon": 2},
    {"b0": 37_000_000, "acc": 0, "brk": 0},
    {"b0": 4, "terminal_cap": 4, "horizon": 2},
    {"b0": 4, "terminal_cap": 9, "horizon": 2},
    {"b0": 4, "terminal_cap": 100, "horizon": 2},
    {"b0": 4, "kappa_max": 0},
])
def test_tiny_solver_matches_exhaustive(kwargs):
    model = make_model(**kwargs)
    plans = list(enumerate_plans(model))
    expected = min((c for _, c in plans), default=None)
    sol = solve(model)
    assert sol.cost == expected
    assert sol.cost == independent_bellman(model)


def test_seeded_random_models_match_two_independent_references():
    rng = random.Random(0x71EA)
    for _ in range(20):
        horizon = rng.choice((2, 3))
        speeds = tuple(sorted(rng.sample((1, 4, 9, 16), 3)))
        offsets = (-1000, 0, 1000)
        b0 = rng.choice((1, 2, 3, 4, 7, 37_000_000))
        obstacles = () if rng.randrange(2) else ((1000, -100, 100),)
        model = make_model(horizon=horizon, speeds=speeds, b0=b0,
                           acc=rng.choice((1, 2, 10)), brk=rng.choice((1, 2, 10)),
                           terminal_cap=rng.choice((0, 4, 9, 100)),
                           wl=rng.choice((1000, 1500, 4000)),
                           wr=rng.choice((1000, 1500, 4000)), obstacles=obstacles)
        sol = solve(model)
        assert sol.cost == independent_bellman(model)
        ex = exhaustive(model)
        assert sol.cost == (None if ex is None else ex[0])


def test_certificates_and_adversarial_mutations():
    models = [make_model(horizon=2, b0=4, terminal_cap=9),
              make_model(horizon=2, b0=37_000_000, terminal_cap=9),
              make_model(horizon=2, b0=4, terminal_cap=0),
              make_model(horizon=2, b0=4, obstacles=((1000, -100, 100),))]
    feasible = []
    for model in models:
        cert = certificate_from_solution(solve(model))
        assert check_certificate(cert)["ok"]
        if cert["claimed"] is not None:
            feasible.append(cert)
    assert feasible
    cert = feasible[0]
    for kind in ("value", "drop_row", "wrong_plan", "forge_cost"):
        assert not check_certificate(corrupt(cert, kind))["ok"]
    # Root is needed by the Lean-facing schema, and is independently checked by
    # mutating a value that participates in the encoded speed square roots.
    root_bad = copy.deepcopy(cert)
    root_bad["roots"][0][1] += 1
    assert not check_certificate(root_bad)["ok"]
    closure_bad = copy.deepcopy(cert)
    closure_bad["states"][1].pop()
    assert not check_certificate(closure_bad)["ok"]
    duplicate = copy.deepcopy(cert)
    duplicate["tables"][1].append(copy.deepcopy(duplicate["tables"][1][0]))
    assert not check_certificate(duplicate)["ok"]
    successor = copy.deepcopy(cert)
    successor["plan"][1]["current"] = 1000 if successor["plan"][1]["current"] != 1000 else -1000
    assert not check_certificate(successor)["ok"]


def test_plan_is_first_lexicographic_optimum():
    for b0 in (1, 3, 4, 9, 37_000_000):
        model = make_model(horizon=2, b0=b0, terminal_cap=9)
        sol = solve(model)
        plans = list(enumerate_plans(model))
        if not plans:
            assert sol.cost is None
            continue
        best = min(c for _, c in plans)
        first = next(tuple(path) for path, cost in plans if cost == best)
        assert sol.cost == best and tuple(sol.plan) == first
        assert sum(time_cost(model.gap(s.station), s.b, t.b)
                   for s, t in zip(sol.plan, sol.plan[1:])) == sol.cost
        assert all(model.step_ok(s, t) for s, t in zip(sol.plan, sol.plan[1:]))


def test_time_cost_is_exact_ceiling_and_physical_tolerance():
    rng = random.Random(903)
    for _ in range(200):
        d = rng.randrange(1, 5001)
        b = rng.randrange(1, 100_000)
        w = rng.randrange(1, 100_000)
        rb, rw = isqrt(b), isqrt(w)
        declared = time_cost(d, b, w)
        assert declared == cdiv(2 * d * 1_000_000, rb + rw)
        exact = Fraction(2 * d * 1_000_000, rb + rw)
        assert 0 <= declared - exact < 1
