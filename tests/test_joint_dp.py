"""Joint offset/speed DP (`Corpus.Joint`) and option-valued Bellman (`Corpus.BellmanOpt`).

An independent Python reference enumerates every joint plan of the concrete instance
(`Corpus.Joint.toyC`/`toyQ v0`/`toyGrid`/`toyTime`) using the production integer
mirrors (`autocross.datagen.cert`, `autocross.datagen.steer`) for the road curvature,
caps, corridor and obstacle clearance; the Lean kernel must reproduce the optimum (or
the infeasibility) of the DP for several start speeds, and must refuse a wrong value.
The dead-end tests pin the defect of the old `bestCost [] = 0` convention."""
from __future__ import annotations

import itertools
import subprocess

import pytest

from autocross.datagen import cert, steer
from conftest import lake_or_skip

BUCKETS = [-1000, -750, -500, -250, 0, 250, 500, 750, 1000]
LON = {"a_lat": 6000, "a_acc": 2000, "a_brk": 3000, "a_max": 9000, "v_max": 20000, "r_h": 4000,
       "hold": 500, "ramp_in": 1000, "ramp_out": 1000, "r_sw_m": 15, "veh_width": 2000}
C = {"lon": LON, "margin": 200, "step": 1000, "kappa_max": 1000000}
GRID = [0, 9000000, 10500000, 16000000, 20000000]


def _q(v0: int) -> dict:
    base = {"v0": v0, "s": [0, 1000, 2000], "k": [0, 0, 0], "wl": [4000] * 3, "wr": [4000] * 3,
            "obstacles": [[2000, -1500, -300]]}
    return {"base": base, "lat": 0}


def _time(_i: int, b: int, w: int) -> int:
    return 1000000 // (1 + (b + w) // 1000000)


def _station_y(q: dict, i: int, lon: dict = LON) -> int:
    base, y = q["base"], 0
    for ob in base["obstacles"]:
        _, yb, _ = cert.obstacle_plan(lon, base, ob)
        if ob[0] - lon["hold"] - lon["ramp_in"] <= base["s"][i] <= ob[0] + lon["hold"] + lon["ramp_out"]:
            y = max(y, yb)
    return y


def _cdiv(a: int, b: int) -> int:
    return -((-a) // b)


def _path_cap(q: dict, i: int, p: int, x: int, n: int, lon: dict = LON) -> tuple[int, int]:
    base = q["base"]
    vmax2 = lon["v_max"] ** 2
    k = cert.kbar(abs(base["k"][i]), _station_y(q, i, lon), lon["r_sw_m"])
    cap_i = vmax2 if k == 0 else min(vmax2, lon["a_lat"] * 10 ** 9 // k)
    curv = abs(k) + _cdiv(10 ** 9 * abs(n - 2 * x + p), C["step"] ** 2)
    cap = cap_i if curv <= 0 else min(cap_i, lon["a_lat"] * 10 ** 9 // curv)
    return curv, cap


def _step_ok(q: dict, s: tuple, t: tuple, lon: dict = LON, circle: int | None = None) -> bool:
    i, p, x, b = s
    j, tp, n, w = t
    base = q["base"]
    if j != i + 1 or tp != x or n not in BUCKETS or j >= len(base["s"]):
        return False
    if not (steer.lo(C, q, j) <= n <= steer.hi(C, q, j)):
        return False
    if steer.clearAt(C, q, base["s"][j], j, n) < 0:
        return False
    curv, cap = _path_cap(q, i, p, x, n, lon)
    d = base["s"][i + 1] - base["s"][i]
    if curv > C["kappa_max"] or w not in GRID or w < 0:
        return False
    if circle is not None:          # Corpus.Levers.circleOK: friction circle of radius `circle`
        lat = b * max(curv, 0) * (2 * d)
        lng = (w - b) * 10 ** 9
        rad = circle * 10 ** 9 * (2 * d)
        return 0 <= b <= lon["v_max"] ** 2 and lat * lat + lng * lng <= rad * rad
    return (b <= cap and w <= b + 2 * lon["a_acc"] * d and b <= w + 2 * lon["a_brk"] * d)


def reference_optimum(v0: int, lon: dict = LON, circle: int | None = None) -> int | None:
    """Independent enumeration of the joint model; `lon` raises budgets
    (`Corpus.Levers.raise`), `circle` switches to `Corpus.Levers.circleOK`.  The
    terminal cap is the model's own (`Joint.terminalCap`), the base one for the circle."""
    q = _q(v0)
    terminal = cert.caps(lon if circle is None else LON, q["base"])[0][-1]
    start = (0, 0, 0, v0 * v0)
    best = None
    for n1, w1, n2, w2 in itertools.product(BUCKETS, GRID, BUCKETS, GRID):
        t1, t2 = (1, 0, n1, w1), (2, n1, n2, w2)
        if _step_ok(q, start, t1, lon, circle) and _step_ok(q, t1, t2, lon, circle) and w2 <= terminal:
            cost = _time(0, start[3], w1) + _time(1, w1, w2)
            best = cost if best is None else min(best, cost)
    return best


def test_reference_reproduces_documented_instance():
    assert reference_optimum(4000) == 82491           # value of Corpus.Joint.toy_value
    assert reference_optimum(6000) is None            # Corpus.Joint.toy_infeasible


def _kernel(src: str) -> subprocess.CompletedProcess:
    lake = lake_or_skip()
    from autocross.datagen.lean_bridge import LEAN_DIR, build
    build()
    f = LEAN_DIR / "JointParity.lean"
    f.write_text(src, encoding="utf-8")
    try:
        return subprocess.run([lake, "env", "lean", f.name], cwd=LEAN_DIR, capture_output=True, text=True)
    finally:
        f.unlink(missing_ok=True)


def test_lean_dp_equals_python_enumeration_for_several_start_speeds():
    lines = ["import Corpus", "set_option maxRecDepth 200000", "open Corpus Corpus.Joint"]
    for k, v0 in enumerate((0, 3000, 4000, 4500, 6000)):
        ref = reference_optimum(v0)
        rhs = "none" if ref is None else f"some ({ref})"
        lines.append(f"theorem t{k} : BellmanOpt.valueO (terminal toyC (toyQ ({v0}))) "
                     f"(stages toyC (toyQ ({v0})) toyGrid toyTime) (start (toyQ ({v0}))) = {rhs} := by decide")
    r = _kernel("\n".join(lines) + "\n")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]


LEVERS = [("lat", 5000), ("lat", 7000), ("lat", 8000), ("lat", 12000), ("brk", 9000),
          ("circle", 6500), ("circle", 7000), ("circle", 9000)]


def _lever_ref(kind: str, x: int) -> int | None:
    if kind == "circle":
        return reference_optimum(4000, circle=x)
    return reference_optimum(4000, lon=dict(LON, **{{"lat": "a_lat", "brk": "a_brk"}[kind]: x}))


def test_lever_references_are_monotone():
    base = reference_optimum(4000)
    order = lambda v: float("inf") if v is None else v                      # none = +inf
    lat = [_lever_ref("lat", a) for a in (5000, 6000, 7000, 8000, 12000)]
    assert all(order(a) >= order(b) for a, b in zip(lat, lat[1:]))         # Levers.cornering_lever
    assert order(_lever_ref("brk", 9000)) <= order(base)                   # Levers.braking_lever
    assert order(_lever_ref("circle", 7000)) <= order(base)                # Levers.circle_value_le_box
    assert lat[0] is None and lat[3] < base                                # the lever binds here


def test_lean_lever_values_equal_python_enumeration():
    """Kernel `decide` on `Corpus.Levers` models equals the independent enumeration."""
    lines = ["import Corpus", "set_option maxRecDepth 200000", "open Corpus Corpus.Joint Corpus.Levers"]
    for k, (kind, x) in enumerate(LEVERS):
        ref = _lever_ref(kind, x)
        rhs = "none" if ref is None else f"some ({ref})"
        if kind == "circle":
            model = f"(circleOK toyC (toyQ 4000) toyGrid {x}) 3 (terminalCap toyC (toyQ 4000))"
        else:
            c = f"(toyLat {x})" if kind == "lat" else f"(toyBrk {x})"
            model = f"(stepOK {c} (toyQ 4000) toyGrid) 3 (terminalCap {c} (toyQ 4000))"
        lines.append(f"theorem l{k} : valueP {model} toyGrid toyTime 2 (start (toyQ 4000)) = {rhs} := by decide")
    r = _kernel("\n".join(lines) + "\n")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]


def test_lean_kernel_refutes_a_wrong_optimum():
    src = ("import Corpus\nset_option maxRecDepth 200000\nopen Corpus Corpus.Joint\n"
           "theorem bad : BellmanOpt.valueO (terminal toyC (toyQ 4000)) (stages toyC (toyQ 4000) toyGrid toyTime)"
           " (start (toyQ 4000)) = some 82490 := by decide\n")
    r = _kernel(src)
    assert r.returncode != 0


# ---------------------------------------------------------------- dead ends (BellmanOpt)
MID = [{0: [(1, 1), (2, 5)]}, {2: [(3, 1)]}]          # state 1 is a dead end at stage 2


def _old_value(stages, s):
    """`Corpus.Bellman.value` with `bestCost [] = 0`."""
    if not stages:
        return 0
    acts = stages[0].get(s, [])
    if not acts:
        return 0
    return min(c + _old_value(stages[1:], t) for t, c in acts)


def _opt_value(stages, s):
    """`Corpus.BellmanOpt.valueO`: None = infeasible."""
    if not stages:
        return 0
    vals = [c + v for t, c in stages[0].get(s, []) if (v := _opt_value(stages[1:], t)) is not None]
    return min(vals) if vals else None


def test_old_convention_rewards_a_dead_end_and_option_value_does_not():
    assert _old_value(MID, 0) == 1           # Corpus.BellmanOpt.dead_mid_old_value
    assert _opt_value(MID, 0) == 6           # Corpus.BellmanOpt.dead_mid_valueO
    assert _opt_value(MID, 1) is None        # infeasible start
    # every complete trajectory costs >= the option value, and one attains it
    costs = [c1 + c2 for t1, c1 in MID[0][0] for _, c2 in MID[1].get(t1, [])]
    assert min(costs) == 6 and _old_value(MID, 0) < min(costs)


def test_lean_kernel_checks_dead_end_counterexamples():
    src = ("import Corpus\nopen Corpus.BellmanOpt\n"
           "example : valueO midTerminal midStages 0 = some 6 := by decide\n"
           "example : Corpus.Bellman.value (oldModel midTerminal midStages) midStages 0 = 1 := by decide\n"
           "example : valueO midTerminal midStages 1 = none := by decide\n")
    r = _kernel(src)
    assert r.returncode == 0, r.stdout[-2000:]
