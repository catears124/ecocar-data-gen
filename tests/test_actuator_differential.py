"""Differential verification of every actuator-1 solver and checker.

The reference is a scalar exhaustive enumeration of every action sequence
through ``ActModel.action_ok``/``action_cost``/``terminal_cost`` (model.py is
the definition).  The vectorised ``solve``, the scalar sparse ``solve_sparse``,
the compiled ``solve_numba``, the Python ``check_certificate`` and the compiled
Lean ``actuatorcheck`` must agree with it on the optimal value, on
infeasibility, and on an admissible optimal first action (all three solvers use
the same declared tie-break, so they must return the identical action).

Regression counterexamples found while auditing (concrete numbers below):

* negative-steer optimum: the old numba wrapper used ``u < 0`` as its
  infeasibility sentinel and reported the instance of
  ``test_unique_negative_steer_optimum`` infeasible; its unique optimum is
  ``(-1000, 3000)`` with value 288_215_000.
* objective: the old numba kernel charged ``5000*|d'| + 5000*|psi'|//100``
  (no reference) and returned 254_500_000 on
  ``test_reference_tracking_objective``; the declared optimum is 252_500_000.
* midpoint: the old numba kernel tested the grid-snapped midpoint, which
  admits the obstacle-straddling step of ``test_unsnapped_midpoint_obstacle``.
* lateral acceleration: the old vectorised ``solve`` compared the floored
  ``b*|kappa|//1e9 <= a_lat`` instead of ``b*|kappa| <= a_lat*1e9`` and
  returned 539_715_000 with ``(0, 3000)`` on
  ``test_lateral_bound_is_checked_before_flooring``; the optimum is
  539_715_200 with ``(1000, 3000)`` (the old value was infeasible-optimistic).
"""
from __future__ import annotations

import json
import random
import subprocess

import pytest

from conftest import lean_exe, numba_or_skip
from autocross.actuator.certificate import (
    CORRUPTIONS, certificate_for_action, certificate_from_diag, check_certificate, corrupt,
    domain_errors,
)
from autocross.actuator.model import COST_TIME_WEIGHT, ActModel, ActState, isqrt, snap
from autocross.actuator.solver import solve, solve_sparse

pytestmark = pytest.mark.verification

LON = {"a_lat": 6589, "a_acc": 5200, "a_brk": 6589, "v_max": 40000,
       "r_h": 1809, "hold": 3000, "veh_width": 2160}
HALF = LON["veh_width"] // 2 + 300  # half width + margin


def build(*, horizon=2, gap=1000, gaps=None, offsets=(-500, 0, 500),
          psis=(-40_000, 0, 40_000), speeds=(4_000_000, 9_000_000, 16_000_000),
          deltas=(-200_000, 0, 200_000), u=(-1000, 0, 1000), a=(-3000, 0, 3000),
          start=(0, 0, 9_000_000, 0), k=None, wl=None, wr=None, obstacles=(),
          ref_d=(), ref_psi=(), terminal_distance=0, terminal_min_b=4_000_000) -> ActModel:
    n = horizon + 1
    gaps = list(gaps) if gaps is not None else [gap] * horizon
    s = [0]
    for g in gaps[:horizon]:
        s.append(s[-1] + g)
    k = list(k or ())[:n] + [0] * max(0, n - len(k or ()))
    return ActModel(
        consts={"lon": dict(LON), "margin": 300, "step": gap, "max_steer": 610865,
                "steer_rate_urad_s": 1221730, "steer_tau_us": 60000, "wheelbase_mm": 3000},
        q={"base": {"s": s, "k": k,
                    "wl": list(wl) if wl is not None else [5000] * n,
                    "wr": list(wr) if wr is not None else [5000] * n,
                    "obstacles": [list(o) for o in obstacles]}, "lat": 0},
        offsets=tuple(offsets), psis=tuple(psis), speeds=tuple(speeds), deltas=tuple(deltas),
        steer_inputs=tuple(u), acc_inputs=tuple(a), horizon=horizon,
        start=ActState(0, *start), terminal_distance=terminal_distance,
        ref_d=tuple(ref_d), ref_psi=tuple(ref_psi), terminal_min_b=terminal_min_b)


def mirror(m: ActModel) -> ActModel:
    """Left/right mirror: negate d, psi, delta, curvature, references; swap widths."""
    base = m.q["base"]
    q = {"base": {**base, "k": [-x for x in base["k"]], "wl": list(base["wr"]), "wr": list(base["wl"]),
                  "obstacles": [[o[0], -o[2], -o[1]] for o in base["obstacles"]]}, "lat": m.q["lat"]}
    st = m.start
    return m.__class__(**{**m.__dict__, "q": q,
                          "start": ActState(0, -st.d, -st.psi, st.b, -st.delta),
                          "ref_d": tuple(-x for x in m.ref_d),
                          "ref_psi": tuple(-x for x in m.ref_psi)})


def exhaustive(m: ActModel):
    """Scalar enumeration of every action sequence: (optimum, optimal first actions)."""
    def best_from(s, depth):
        if depth == m.horizon:
            c = m.terminal_cost(s)
            return None if c is None else COST_TIME_WEIGHT * c
        best = None
        for u in m.steer_inputs:
            for a in m.acc_inputs:
                t = m.action_ok(s, u, a)
                if t is None:
                    continue
                rest = best_from(t, depth + 1)
                if rest is None:
                    continue
                v = m.action_cost(s, t) + rest
                if best is None or v < best:
                    best = v
        return best

    per_first = {}
    for u in m.steer_inputs:
        for a in m.acc_inputs:
            t = m.action_ok(m.start, u, a)
            if t is None:
                continue
            rest = best_from(t, 1)
            if rest is not None:
                per_first[(u, a)] = m.action_cost(m.start, t) + rest
    if not per_first:
        return None, [], per_first
    best = min(per_first.values())
    # Declared order: steering outer, acceleration inner.
    firsts = [ua for ua in per_first if per_first[ua] == best]
    return best, firsts, per_first


def replay(m: ActModel, plan, acts) -> int:
    assert len(plan) == m.horizon + 1 and len(acts) == m.horizon and plan[0] == m.start
    total = 0
    for i, (u, a) in enumerate(acts):
        t = m.action_ok(plan[i], u, a)
        assert t == plan[i + 1], (i, u, a)
        total += m.action_cost(plan[i], t)
    c = m.terminal_cost(plan[-1])
    assert c is not None
    return total + COST_TIME_WEIGHT * c


def lean_accepts(exe, cert, path) -> bool:
    path.write_text(json.dumps(cert), encoding="utf-8")
    r = subprocess.run([str(exe), str(path)], capture_output=True, text=True, timeout=120)
    ok = r.returncode == 0
    assert ok == ('"ok":true' in r.stdout), (r.returncode, r.stdout, r.stderr)
    return ok


@pytest.fixture(scope="module")
def numba_solve():
    return numba_or_skip()


@pytest.fixture(scope="module")
def lean():
    return lean_exe("actuatorcheck")


def agree(m: ActModel, numba_solve, *, lean=None, tmp=None):
    """Every solver and checker agrees with exhaustive enumeration; returns (value, action)."""
    best, firsts, per_first = exhaustive(m)
    results = {"solve": solve(m), "sparse": solve_sparse(m), "numba": numba_solve(m)}
    for name, (t, c, act, d) in results.items():
        assert c == best, (name, c, best)
        if best is None:
            assert t is None and act is None and not d["feasible"], name
            continue
        assert act in firsts, (name, act, firsts)
        assert act == firsts[0], (name, act, firsts)  # declared tie-break
        assert m.action_ok(m.start, *act) == t, name
        assert d["plan_actions"][0] == act, name
        assert replay(m, d["plan"], d["plan_actions"]) == best, name
    # Certificates exist only on the checker domain (start d/psi/delta on the
    # declared grids); Python and Lean must give the same verdict either way.
    in_domain = not domain_errors(m)
    _t, sc, sa, sd = results["sparse"]
    cert = certificate_from_diag(m, sd, claimed=sc, first_action=sa)
    rep = check_certificate(cert)
    assert rep["ok"] == in_domain, rep["errors"]
    if lean is not None:
        assert lean_accepts(lean, cert, tmp / "sparse.json") == in_domain
    if best is None:
        assert all(certificate_for_action(m, (u, a), sparse=results["sparse"]) is None
                   for u in m.steer_inputs for a in m.acc_inputs)
        return None, None
    act = results["numba"][2]
    bound = certificate_for_action(m, act, sparse=results["sparse"])
    assert bound is not None and bound["firstAction"] == list(act) and bound["claimed"] == best
    rep = check_certificate(bound)
    assert rep["ok"] == in_domain, rep["errors"]
    if lean is not None:
        assert lean_accepts(lean, bound, tmp / "bound.json") == in_domain
    for ua, v in per_first.items():
        if v > best:  # admissible but suboptimal: no certificate may be issued
            assert certificate_for_action(m, ua, sparse=results["sparse"]) is None, ua
    return best, act


# ------------------------------------------------------------------ regression counterexamples

def negative_steer_model():
    return build(horizon=1, k=(-5000, -5000), ref_d=(-500, 0), ref_psi=(40000, -40000),
                 start=(500, -40000, 9_000_000, -200_000))


def test_unique_negative_steer_optimum(numba_solve, lean, tmp_path):
    """Old ``solve_numba`` returned ``(None, None, None)`` here (``u < 0`` sentinel)."""
    m = negative_steer_model()
    best, firsts, _ = exhaustive(m)
    assert best == 288_215_000 and firsts == [(-1000, 3000)]
    value, act = agree(m, numba_solve, lean=lean, tmp=tmp_path)
    assert value == 288_215_000 and act == (-1000, 3000) and act[0] < 0


def test_reference_tracking_objective(numba_solve):
    """Old kernel objective ``5000*|d'| + 5000*|psi'|//100`` ignored ref_d/ref_psi.

    With it the optimum of this instance is 254_500_000 (what the old kernel
    returned); the declared objective's optimum is 252_500_000.
    """
    m = build(horizon=1, k=(5000, 5000), ref_d=(0, 0), ref_psi=(40000, -40000),
              start=(500, -40000, 16_000_000, 0))

    def old_cost(s, t):
        return (COST_TIME_WEIGHT * (-((-(2 * m.gap(s.station) * 1_000_000))
                                      // (isqrt(s.b) + isqrt(t.b))))
                + abs(t.delta - s.delta) // 1000 + 5000 * abs(t.d) + 5000 * abs(t.psi) // 100)

    old = min(old_cost(m.start, t) + COST_TIME_WEIGHT * m.terminal_cost(t)
              for t, _c, _u, _a in m.actions(m.start) if m.terminal_cost(t) is not None)
    assert old == 254_500_000
    value, _act = agree(m, numba_solve)
    assert value == 252_500_000


def midpoint_model(obstacles):
    return build(horizon=1, gap=20000, offsets=(-3000, 0, 3000), psis=(-80000, 0, 80000),
                 start=(0, 80000, 9_000_000, 0), ref_d=(0, 3000), obstacles=obstacles)


def test_unsnapped_midpoint_obstacle(numba_solve, lean, tmp_path):
    """An obstacle forbids d in (100, 2900) at station 1.  Action (0, 3000) moves
    d 0 -> 3000; only its exact midpoint 1500 is blocked (the snapped midpoint
    3000 is free), so the model rejects it and the optimum steers left."""
    free = midpoint_model(())
    m = midpoint_model([(20000, 100 + HALF, 2900 - HALF)])
    t = free.action_ok(free.start, 0, 3000)
    assert t is not None and t.d == 3000
    assert m.lateral_ok(1, t.d) and not m.lateral_ok(1, (t.d + 0) // 2)
    assert m.lateral_ok(1, snap((t.d + 0) // 2, m.offsets))
    assert m.action_ok(m.start, 0, 3000) is None
    assert agree(free, numba_solve)[0] == 5_718_286_000
    value, act = agree(m, numba_solve, lean=lean, tmp=tmp_path)
    assert (value, act) == (5_733_286_200, (-1000, 3000))


def test_lateral_bound_is_checked_before_flooring(numba_solve):
    """b*|kappa| = 16e6*411850 = 6.5896e12 exceeds a_lat*1e9 = 6.589e12 but floors to a_lat."""
    m = build(horizon=2, k=(0, 411850, 0), start=(0, -40000, 9_000_000, 0))
    s1 = ActState(1, 0, -40000, 16_000_000, 0)
    t = m.next_state(s1, 0, 0)
    assert t is not None
    kp = m.k[1] + (t.psi - s1.psi) * 1000 // m.gap(1)
    a_lat = LON["a_lat"] * 1_000_000_000
    assert a_lat < t.b * abs(kp) < a_lat + 1_000_000_000
    assert m.action_ok(s1, 0, 0) is None
    value, act = agree(m, numba_solve)
    assert (value, act) == (539_715_200, (1000, 3000))


# ------------------------------------------------------------------ structured cases

MIRROR_CASES = {
    "left_curve_refs": dict(horizon=2, k=(15000, 15000, 0), ref_d=(0, 500, 500),
                            ref_psi=(0, 40000, 0), start=(0, 0, 9_000_000, 0)),
    "offset_start": dict(horizon=2, start=(500, 0, 9_000_000, 0), ref_d=(0, -500, -500)),
    "asym_corridor": dict(horizon=2, wl=[5000, 1900, 1900], wr=[5000, 5000, 5000],
                          start=(500, 40000, 9_000_000, 200_000)),
    "obstacle_left": dict(horizon=2, obstacles=[(1000, 0, 1500)], start=(0, 0, 9_000_000, 0)),
    "heading_negative": dict(horizon=3, start=(0, -40000, 9_000_000, -200_000),
                             a=(-3000, 3000)),
}


@pytest.mark.parametrize("name", sorted(MIRROR_CASES))
def test_mirrored_instances(name, numba_solve):
    """Mirrored instances have equal values and mirrored optimal first steers.

    The model is mirror-symmetric only up to its rounding: ``snap`` breaks ties
    toward the larger grid value and ``//`` floors (see
    ``test_mirror_asymmetry_is_rounding``).  These instances avoid such ties.
    """
    m = build(**MIRROR_CASES[name])
    mm = mirror(m)
    v, act = agree(m, numba_solve)
    vm, actm = agree(mm, numba_solve)
    assert v == vm
    _b, firsts, _ = exhaustive(m)
    _b, firsts_m, _ = exhaustive(mm)
    assert sorted((-u, a) for u, a in firsts) == sorted(firsts_m)
    if v is not None and len(firsts) == 1:
        assert actm == (-act[0], act[1])


def test_mirror_asymmetry_is_rounding(numba_solve):
    """With curvature 20000 the stage-1 heading lands exactly on a snap tie:
    snap(+20000) = +40000 but snap(-20000) = 0, so the mirrored instance is
    genuinely different (all solvers still agree on each)."""
    m = build(horizon=2, k=(20000, 20000, 0), ref_d=(0, 500, 500), ref_psi=(0, 40000, 0))
    assert snap(20000, m.psis) == 40000 and snap(-20000, m.psis) == 0
    v, _ = agree(m, numba_solve)
    vm, _ = agree(mirror(m), numba_solve)
    assert (v, vm) == (542_715_000, 540_715_400)


STRUCTURED = {
    "both_signs_refs": dict(horizon=2, k=(-15000, 10000, 0), ref_d=(0, -500, 500),
                            ref_psi=(0, -40000, 40000)),
    "asym_corridor_right": dict(horizon=2, wl=[5000] * 3, wr=[1700, 1700, 1700],
                                start=(-250, 0, 9_000_000, 0), offsets=(-500, -250, 0, 250, 500)),
    "obstacle_right": dict(horizon=2, obstacles=[(1000, -1500, 0)], start=(0, 0, 9_000_000, 0)),
    "obstacle_both": dict(horizon=2, obstacles=[(1000, -2500, -1900), (2000, 1900, 2500)],
                          offsets=(-750, -250, 0, 250, 750)),
    "nonuniform_grids": dict(horizon=3, gaps=(800, 1500, 600),
                             offsets=(-750, -300, 0, 100, 600), psis=(-80000, -10000, 0, 30000),
                             speeds=(2_250_000, 4_000_000, 10_000_000, 12_250_000),
                             deltas=(-300_000, -50_000, 0, 120_000), u=(-1000, 300),
                             a=(-6000, 0, 5200), start=(100, 30000, 10_000_000, -50_000),
                             terminal_min_b=2_000_000),
    "terminal_distance": dict(horizon=2, terminal_distance=8000, k=(5000, -5000, 0)),
    "edge_offset_hi": dict(horizon=2, offsets=(-500, 0, 500), wl=[1880] * 3,
                           start=(500, 40000, 16_000_000, 200_000)),
    "edge_offset_lo": dict(horizon=2, offsets=(-500, 0, 500), wr=[1880] * 3,
                           start=(-500, -40000, 16_000_000, -200_000)),
    "edge_speed_min": dict(horizon=2, start=(0, 0, 4_000_000, 0)),
    "edge_speed_max": dict(horizon=2, start=(0, 0, 16_000_000, 0)),
    "off_grid_start": dict(horizon=2, start=(130, -17000, 7_000_000, 90_000)),
    "ties": dict(horizon=1, k=(5000, 5000), ref_psi=(40000, -40000), start=(500, -40000, 16_000_000, 0)),
    "horizon3_refs": dict(horizon=3, ref_d=(0, 500, 0, -500), ref_psi=(0, 0, 40000, -40000),
                          k=(0, 10000, -10000, 0)),
}


@pytest.mark.parametrize("name", sorted(STRUCTURED))
def test_structured_instances(name, numba_solve, lean, tmp_path):
    agree(build(**STRUCTURED[name]), numba_solve, lean=lean, tmp=tmp_path)


def test_both_steering_signs_are_optimal_somewhere(numba_solve):
    signs = set()
    for name in sorted(STRUCTURED):
        v, act = agree(build(**STRUCTURED[name]), numba_solve)
        if act is not None:
            signs.add((act[0] > 0) - (act[0] < 0))
    v, act = agree(negative_steer_model(), numba_solve)
    signs.add(-1)
    assert {-1, 1} <= signs


INFEASIBLE = {
    "terminal_speed_unreachable": dict(horizon=2, terminal_min_b=20_000_000),
    "corridor_closed_station1": dict(horizon=2, wl=[5000, 1000, 5000], wr=[5000, 1000, 5000]),
    "corridor_closed_terminal": dict(horizon=2, wl=[5000, 5000, 1300], wr=[5000, 5000, 1300]),
    "obstacle_blocks_all": dict(horizon=1, obstacles=[(1000, -3000, 3000)]),
    "start_outside_reach": dict(horizon=1, wl=[5000, 1700], wr=[5000, -900],
                                start=(-500, -40000, 16_000_000, -200_000)),
}


@pytest.mark.parametrize("name", sorted(INFEASIBLE))
def test_infeasible_instances(name, numba_solve, lean, tmp_path):
    m = build(**INFEASIBLE[name])
    assert exhaustive(m)[0] is None
    assert agree(m, numba_solve, lean=lean, tmp=tmp_path) == (None, None)


def test_dense_solvers_refuse_int64_unsafe_instances(numba_solve):
    m = build(horizon=1, terminal_distance=10 ** 13)
    with pytest.raises(OverflowError):
        solve(m)
    with pytest.raises(OverflowError):
        numba_solve(m)


# ------------------------------------------------------------------ randomized campaign

def random_model(rng: random.Random) -> ActModel:
    h = rng.choice((1, 1, 2, 2, 2, 3))
    offsets = rng.choice(((-500, 0, 500), (-750, -250, 0, 250, 750), (-600, -100, 0, 350),
                          (-1000, 0, 1000)))
    psis = rng.choice(((-40_000, 0, 40_000), (-80_000, -20_000, 0, 50_000), (-60_000, 0, 60_000)))
    speeds = rng.choice(((4_000_000, 9_000_000, 16_000_000), (1_000_000, 4_000_000, 16_000_000),
                         (2_250_000, 6_250_000, 7_000_000, 12_250_000)))
    deltas = rng.choice(((-200_000, 0, 200_000), (-300_000, -100_000, 0, 250_000)))
    us = rng.choice(((-1000, 0, 1000), (-1000, 1000), (-700, 0, 400), (-1000, -500, 500)))
    accs = rng.choice(((-3000, 0, 3000), (-6000, 5200), (-6000, -3000, 0, 3000)))
    if h == 3:
        us = us[:2] if len(us) * len(accs) > 9 else us
        accs = accs[:3]
    n = h + 1
    k = [rng.choice((-60000, -20000, -5000, 0, 0, 5000, 20000, 60000)) for _ in range(n)]
    wl = [rng.choice((1700, 1900, 2200, 5000, 5000)) for _ in range(n)]
    wr = [rng.choice((1700, 1900, 2200, 5000, 5000)) for _ in range(n)]
    obstacles = []
    if rng.random() < 0.35:
        lmin = rng.choice((-2500, -1800, -400, 300, 1500))
        obstacles.append((rng.choice((1000, 2000, 3000)), lmin, lmin + rng.choice((200, 700))))
    ref_d = tuple(rng.choice((-500, -250, 0, 250, 500)) for _ in range(rng.choice((0, n, n - 1))))
    ref_psi = tuple(rng.choice((-40000, 0, 40000)) for _ in range(rng.choice((0, n))))
    start = (rng.choice(offsets + (rng.randint(-700, 700),)),
             rng.choice(psis + (rng.randint(-60000, 60000),)),
             rng.choice(speeds + (rng.randint(1_000_000, 16_000_000),)),
             rng.choice(deltas + (rng.randint(-250_000, 250_000),)))
    return build(horizon=h, gaps=[rng.choice((700, 1000, 1500, 2500)) for _ in range(h)],
                 offsets=offsets, psis=psis, speeds=speeds, deltas=deltas, u=us, a=accs,
                 start=start, k=k, wl=wl, wr=wr, obstacles=obstacles, ref_d=ref_d,
                 ref_psi=ref_psi, terminal_distance=rng.choice((0, 0, 5000)),
                 terminal_min_b=rng.choice((1_000_000, 4_000_000)))


CAMPAIGN = 500
LEAN_EVERY = 10


def test_randomized_campaign(numba_solve, lean, tmp_path):
    rng = random.Random(0xAC71)
    feasible = negative = certified = 0
    for i in range(CAMPAIGN):
        m = random_model(rng)
        certified += exhaustive(m)[0] is not None and not domain_errors(m)
        try:
            v, act = agree(m, numba_solve, lean=lean if i % LEAN_EVERY == 0 else None, tmp=tmp_path)
        except AssertionError as e:
            raise AssertionError(f"campaign instance {i}: {m.as_instance()}") from e
        if v is not None:
            feasible += 1
            negative += act[0] < 0
    # The campaign must exercise both feasible and infeasible instances and
    # negative-steer optima.
    assert 0.3 * CAMPAIGN < feasible < CAMPAIGN
    assert negative > 10
    assert certified > 0.2 * CAMPAIGN


# ------------------------------------------------------------------ Lean acceptance / rejection

LEAN_CERT_CASES = {
    "negative_steer": negative_steer_model,
    "refs_h3": lambda: build(**STRUCTURED["horizon3_refs"]),
    "nonuniform": lambda: build(**STRUCTURED["nonuniform_grids"]),
    "obstacle_midpoint": lambda: midpoint_model([(20000, 100 + HALF, 2900 - HALF)]),
}


@pytest.mark.parametrize("case", sorted(LEAN_CERT_CASES))
def test_lean_accepts_bound_certificates_and_rejects_corruptions(case, numba_solve, lean, tmp_path):
    m = LEAN_CERT_CASES[case]()
    _t, cost, act, _d = numba_solve(m)
    cert = certificate_for_action(m, act)
    assert cert is not None and cert["firstAction"] == list(act) and cert["claimed"] == cost
    assert check_certificate(cert)["ok"]
    assert lean_accepts(lean, cert, tmp_path / "ok.json")
    for kind in CORRUPTIONS:
        bad = corrupt(cert, kind)
        assert not check_certificate(bad)["ok"], kind
        if kind == "sha":
            # The transport hash is a Python-side integrity field; the Lean
            # checker recomputes the semantics and does not read it.
            continue
        assert not lean_accepts(lean, bad, tmp_path / f"{kind}.json"), kind
