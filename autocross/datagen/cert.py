"""Certified longitudinal decision (model ``cert-2``): exact integer arithmetic, mirrored
one-to-one by ``lean/Corpus/Cert.lean``. Changes from v1.0.0's ``cert-1``: explicit domain guards
(non-negative speed, sane constants, non-negative caps), exact decimal quantisation, every
constant DERIVED (grip and acceleration budgets from the friction circle, speed cap max_speed,
horizon from the declared scenario domain; see ``model_constants``) and no ad-hoc gap-speed rule.

Model M (everything the decision may depend on is in the observation):
  * Stations: the road-model preview stations s_0 = 0 < s_1 < ... < s_n (mm). Nothing beyond
    s_n is known except the declared scenario domain.
  * Unknowns: b_i = v_i^2 (mm^2/s^2) at each station, b_i >= 0 (point-mass speed profile along
    the planned path).
  * Grip: b_i * kbar_i <= A_LAT * 10^9, where kbar_i (micro-1/m) is the model's bound on the
    curvature of the expert's planned path at station i: |road curvature| plus the curvature of
    the obstacle sidestep (smoothstep ramp of length >= R_SW: max |y''| = 6|y|/R_SW^2), divided by
    the parallel-curve factor (1 - |kappa| |y|). That this bounds the curvature of the path the
    (uncertified) lateral planner emits is a modelling assumption.
  * Friction: A_LAT = mu*g/sqrt(2) and A_ACC, A_BRK <= mu*g/sqrt(2): the largest equal-axis box
    inside the friction circle, so every plan of M respects the friction circle of a point mass.
    This is sound but not tight: plans that use the full circle (e.g. trail braking) are outside M.
  * b_i <= V_MAX^2, V_MAX = the vehicle's max_speed.
  * Terminal constraint b_n <= A_LAT * R_H, R_H = ``domain_horizon_mm``: the effective radius of
    the tightest planned path the declared scenario domain allows at any station (tightest turn
    radius with the largest sidestep). Every unseen station of every in-domain road therefore has
    cap >= A_LAT * R_H, so a profile satisfying the terminal constraint can be continued safely
    on EVERY in-domain road (``decide_robust_extend``), and a road whose tightest point starts
    right after the preview forces it. This is the only way the unseen road enters.
  * Accel/brake budget: b_{i+1} <= b_i + 2 A_ACC ds_i and b_i <= b_{i+1} + 2 A_BRK ds_i with
    A_ACC = min(max_accel, box), A_BRK = min(max_decel, box).
  * Start: b_0 = v0^2 (current speed).

Certificates (theorems in ``Corpus.Cert``):
  * ``greatest_safe`` (b_0 <= B_0): the forward-backward profile F is feasible and every feasible
    profile is pointwise <= F (``decide_greatest_safe``, ``decide_greatest_safe_feasible``);
    F minimises every objective that does not increase when speeds rise pointwise, e.g. traversal
    time (``decide_greatest_safe_minimises``). The recorded integer acceleration
    a = floor((F_1 - b_0) / (2 ds_0)) mm/s^2 lies in [-A_BRK, A_ACC], holding it over the first
    segment starts a feasible profile, and no larger integer acceleration does
    (``decide_accel_greatest``).
  * ``min_violation`` (b_0 > B_0: already too fast for the budget): no feasible profile exists
    (``decide_min_violation_infeasible``); full braking at the vehicle's maximum deceleration
    gives the pointwise-smallest speeds of any dynamically possible profile, so it minimises every
    violation measure that is monotone in the speeds, e.g. the total excess over upper speed caps
    (``decide_min_violation``, ``brake_minimises_excess``). Not arbitrary violation measures.
  * ``no_feasible_gap``: a visible obstacle leaves no gap wide enough; no certificate.
  * ``invalid_input``: outside the model's domain; no certificate.
The certificate covers the longitudinal decision only. Path/steering/pedal mapping are not certified.

Quantisation: every input is ``floor(x * scale + 1/2)`` computed EXACTLY on the shortest decimal
representation of the float (``repr``), which is also the JSON text written to the corpus; the
Lean checker computes the same integers from the JSON numbers of the recorded sample."""
from __future__ import annotations

import math
from functools import lru_cache
from fractions import Fraction

CERT_MODEL = "cert-2"
KIND_GREATEST_SAFE, KIND_MIN_VIOLATION, KIND_NO_GAP, KIND_INVALID = 0, 1, 2, 3
KIND_NAMES = {0: "greatest_safe", 1: "min_violation", 2: "no_feasible_gap", 3: "invalid_input"}
KIND_IDS = {v: k for k, v in KIND_NAMES.items()}
CERTIFIED_KINDS = (KIND_GREATEST_SAFE, KIND_MIN_VIOLATION)
MM, MICRO = 1000, 1_000_000


G = 9.81  # m/s^2


def kbar(ka: int, y: int, r_sw_m: int) -> int:
    """Model bound (micro-1/m) on the curvature of the planned path at a station with road curvature
    |ka| (micro-1/m) and sidestep offset |y| (mm); exactly the arithmetic of ``caps``/``capAt``."""
    ksw = _cdiv(6000 * y, r_sw_m * r_sw_m)
    den = max(1_000_000 - _cdiv(ka * y, 1000), 200_000)
    return _cdiv((ka + ksw) * 1_000_000, den)


def domain_horizon_mm(r_min_m: float, y_max_m: float, r_sw_m: int) -> int:
    """Effective radius (mm) of the tightest planned path the declared scenario domain allows at ANY
    station: tightest centerline radius together with the largest sidestep offset. Every station
    of every in-domain continuation therefore has grip cap >= a_lat * r_h (``a_lat * floor(1e9/k)
    <= floor(a_lat * 1e9 / k)``), which is what the terminal constraint b_n <= a_lat * r_h needs."""
    k = kbar(_cdiv(10 ** 9, int(round(r_min_m * 1000))), int(round(y_max_m * 1000)), r_sw_m)
    return 10 ** 9 // k


def friction_box_mm(mu: float) -> int:
    """Half-side (mm/s^2) of the largest equal-axis box inscribed in the friction circle of radius
    mu*g: floor(floor(mu*g) / sqrt(2)), exact integer arithmetic. Any (a_long, a_lat) inside the box
    satisfies a_long^2 + a_lat^2 <= (mu g)^2."""
    m = math.floor(Fraction(repr(float(mu))) * Fraction(repr(G)) * 1000)
    return math.isqrt(m * m // 2)


def model_constants(vp, r_h_mm: int, hold: float, ramp_in: float, ramp_out: float, r_sw: int) -> dict:
    """Integer constants of M (mm, mm/s, mm/s^2, metres for R_SW). Every value is DERIVED:
      * a_lat = friction box mu*g/sqrt(2); a_acc = min(max_accel, box); a_brk = min(max_decel, box):
        lateral and longitudinal budgets of the largest equal-axis box inside the friction circle,
        so every plan of M is within the friction circle of a point mass (sound, not tight);
      * a_max = max_decel: the full-brake label of the min_violation branch;
      * v_max = max_speed, veh_width = width (published VehicleParams);
      * r_h from the declared scenario domain; hold / ramps / R_SW from the lateral planner geometry."""
    box = friction_box_mm(vp.mu)
    return {
        "a_lat": box, "a_acc": min(int(round(vp.max_accel * 1000)), box),
        "a_brk": min(int(round(vp.max_decel * 1000)), box), "a_max": int(round(vp.max_decel * 1000)),
        "v_max": int(round(vp.max_speed * 1000)), "r_h": int(r_h_mm),
        "hold": int(round(hold * 1000)), "ramp_in": int(round(ramp_in * 1000)), "ramp_out": int(round(ramp_out * 1000)),
        "r_sw_m": int(r_sw), "veh_width": int(round(vp.width * 1000)),
    }


CONST_KEYS = ("a_lat", "a_acc", "a_brk", "a_max", "v_max", "r_h", "hold", "ramp_in", "ramp_out", "r_sw_m", "veh_width")


@lru_cache(maxsize=None)
def qfix(x, scale: int) -> int:
    """Memoised round-half-up quantisation; cache hits are value-identical to the exact formula.

    The result is computed on the shortest decimal repr of the float and raises ``ValueError``
    for bool, ``None``, NaN, inf, or non-numbers.
    """
    if isinstance(x, bool) or x is None:
        raise ValueError("not a number")
    x = float(x)
    if not math.isfinite(x):
        raise ValueError("non-finite")
    return math.floor(Fraction(repr(x)) * scale + Fraction(1, 2))


def quantise_dict(obs: dict) -> dict | None:
    """Fixed-point inputs of M from a serialised (policy) observation dict; None = invalid input.
    Uses only: speed, road.valid, road.preview_s / preview_curvature / preview_width_left /
    preview_width_right, road.obstacles[*].distance / lateral_min / lateral_max."""
    r = obs.get("road")
    if r is None or r.get("valid") is not True:
        return None
    try:
        q = {"v0": qfix(obs["speed"], MM), "s": [qfix(x, MM) for x in r["preview_s"]],
             "k": [qfix(x, MICRO) for x in r["preview_curvature"]], "wl": [qfix(x, MM) for x in r["preview_width_left"]],
             "wr": [qfix(x, MM) for x in r["preview_width_right"]],
             "obstacles": [[qfix(o["distance"], MM), qfix(o["lateral_min"], MM), qfix(o["lateral_max"], MM)]
                           for o in r["obstacles"]]}
    except (ValueError, TypeError, KeyError):
        return None
    return q if len(q["s"]) >= 2 else None


def quantise(obs) -> dict | None:
    """Fixed-point inputs of M from an Observation object (same integers as ``quantise_dict`` of
    its serialisation: floats serialise as their repr)."""
    r = obs.road
    if r is None or not r.valid:
        return None
    try:
        q = {"v0": qfix(obs.speed, MM), "s": [qfix(x, MM) for x in r.preview_s],
             "k": [qfix(x, MICRO) for x in r.preview_curvature], "wl": [qfix(x, MM) for x in r.preview_width_left],
             "wr": [qfix(x, MM) for x in r.preview_width_right],
             "obstacles": [[qfix(o.distance, MM), qfix(o.lateral_min, MM), qfix(o.lateral_max, MM)] for o in r.obstacles]}
    except (ValueError, TypeError):
        return None
    return q if len(q["s"]) >= 2 else None


def _cdiv(a: int, b: int) -> int:
    return -((-a) // b)


def _nearest_index(s: list[int], d: int) -> int:
    """First station at or beyond d (last station if d is beyond the preview)."""
    for i, x in enumerate(s):
        if x >= d:
            return i
    return len(s) - 1


def obstacle_plan(c: dict, q: dict, ob: list[int]) -> tuple[int, int, int]:
    """(side +1 left / -1 right / 0 tie, |target offset| bound (mm), gap width (mm)) for one
    obstacle, from the widths at the first station at or beyond it."""
    d, lmin, lmax = ob
    i = _nearest_index(q["s"], d)
    gl, gr = q["wl"][i] - lmax, lmin + q["wr"][i]
    yl, yr = _cdiv(abs(lmax + q["wl"][i]), 2), _cdiv(abs(lmin - q["wr"][i]), 2)
    if gl > gr:
        return 1, yl, gl
    if gr > gl:
        return -1, yr, gr
    return 0, max(yl, yr), gl


def caps(c: dict, q: dict) -> tuple[list[int], bool]:
    """Per-station upper bounds on b and whether every visible gap is wide enough for the car."""
    n = len(q["s"])
    y = [0] * n
    feasible = True
    for ob in q["obstacles"]:
        _, yb, gap = obstacle_plan(c, q, ob)
        if gap < c["veh_width"]:
            feasible = False
        for i, s in enumerate(q["s"]):
            if ob[0] - c["hold"] - c["ramp_in"] <= s <= ob[0] + c["hold"] + c["ramp_out"]:
                y[i] = max(y[i], yb)
    vmax2 = c["v_max"] * c["v_max"]
    out = []
    for i in range(n):
        k = kbar(abs(q["k"][i]), y[i], c["r_sw_m"])
        out.append(vmax2 if k == 0 else min(vmax2, c["a_lat"] * 10 ** 9 // k))
    out[-1] = min(out[-1], c["a_lat"] * c["r_h"])
    return out, feasible


def backward(c: dict, cap: list[int], s: list[int]) -> list[int]:
    b = list(cap)
    for i in range(len(b) - 2, -1, -1):
        b[i] = min(cap[i], b[i + 1] + 2 * c["a_brk"] * (s[i + 1] - s[i]))
    return b


def forward(c: dict, bw: list[int], s: list[int], b0: int) -> list[int]:
    f = [b0]
    for i in range(1, len(bw)):
        f.append(min(bw[i], f[-1] + 2 * c["a_acc"] * (s[i] - s[i - 1])))
    return f


def well_formed(q: dict | None) -> bool:
    """Lean ``wellFormed``: >= 2 stations from 0, strictly increasing, aligned vectors, obstacle
    triples, non-negative speed."""
    if q is None:
        return False
    s = q["s"]
    n = len(s)
    return (n >= 2 and s[0] == 0 and all(s[i] < s[i + 1] for i in range(n - 1))
            and len(q["k"]) == n and len(q["wl"]) == n and len(q["wr"]) == n
            and all(len(ob) == 3 for ob in q["obstacles"]) and q["v0"] >= 0)


def consts_ok(c: dict) -> bool:
    """Lean ``constsOK``: the hypotheses the theorems need from the constants."""
    return c["a_acc"] >= 0 and c["a_brk"] >= 0 and c["a_max"] > 0 and c["r_sw_m"] > 0


def decide(c: dict, q: dict | None) -> dict:
    """Certified longitudinal decision for the quantised observation ``q`` (Lean ``Cert.decide``)."""
    invalid = {"kind": KIND_INVALID, "a": -c["a_max"], "b1": None, "b0": None, "B0": None}
    if not well_formed(q) or not consts_ok(c):
        return invalid
    cap, feasible = caps(c, q)
    if any(x < 0 for x in cap):
        return invalid
    b0 = q["v0"] * q["v0"]
    if not feasible:
        return {"kind": KIND_NO_GAP, "a": -c["a_max"], "b1": None, "b0": b0, "B0": None}
    bw = backward(c, cap, q["s"])
    if b0 > bw[0]:
        return {"kind": KIND_MIN_VIOLATION, "a": -c["a_max"], "b1": None, "b0": b0, "B0": bw[0]}
    f = forward(c, bw, q["s"], b0)
    a = (f[1] - b0) // (2 * (q["s"][1] - q["s"][0]))
    return {"kind": KIND_GREATEST_SAFE, "a": a, "b1": f[1], "b0": b0, "B0": bw[0]}


def domain_violations(c: dict, q: dict | None) -> int:
    """Number of VISIBLE stations whose grip cap is below the horizon cap a_lat * r_h, i.e. where the
    observed road is tighter than the declared domain allows. The robustness theorems assume every
    unseen station satisfies this; a visible violation means the domain assumption is wrong."""
    if not well_formed(q) or not consts_ok(c):
        return 0
    n = len(q["s"])
    y = [0] * n
    for ob in q["obstacles"]:
        _, yb, _ = obstacle_plan(c, q, ob)
        for i, s in enumerate(q["s"]):
            if ob[0] - c["hold"] - c["ramp_in"] <= s <= ob[0] + c["hold"] + c["ramp_out"]:
                y[i] = max(y[i], yb)
    bad = 0
    for i in range(n):
        k = kbar(abs(q["k"][i]), y[i], c["r_sw_m"])
        if k > 0 and c["a_lat"] * 10 ** 9 // k < c["a_lat"] * c["r_h"]:
            bad += 1
    return bad
