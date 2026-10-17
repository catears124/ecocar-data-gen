"""Certified longitudinal decision (model ``cert-1``): exact integer arithmetic, mirrored
one-to-one by ``lean/Corpus/Cert.lean``.

Model M (everything the decision may depend on is in the observation):
  * Stations: the road-model preview stations s_0 = 0 < s_1 < ... < s_n (mm). Nothing beyond
    s_n is known.
  * Unknowns: b_i = v_i^2 (mm^2/s^2) at each station.
  * Grip: b_i * kbar_i <= A_LAT * 10^9, where kbar_i (micro-1/m) bounds the curvature of the
    expert's planned path at station i: |road curvature| plus the curvature of the obstacle
    sidestep (smoothstep ramp of length >= R_SW: max |y''| = 6|y|/R_SW^2), divided by the
    parallel-curve factor (1 - |kappa| |y|).
  * Beside a visible obstacle (|s - d_obs| <= HOLD): b_i <= (GAP_BASE + GAP_PER_M * slack)^2,
    slack = free gap width - vehicle width.
  * b_i <= V_MAX^2.
  * Horizon: b_n <= A_LAT * R_H (a corner of radius R_H could start right after the preview).
  * Accel/brake budget: b_{i+1} <= b_i + 2 A_ACC ds_i and b_i <= b_{i+1} + 2 A_BRK ds_i.
  * Start: b_0 = v0^2 (current speed).

Certificates (theorems in ``Corpus.Cert``):
  * ``greatest_safe`` (b_0 <= B_0): the forward-backward profile F is feasible and every feasible
    profile is pointwise <= F. The label is the acceleration to F_1 over the first station:
    no command that keeps the plan feasible for every unseen continuation can be faster, and F
    minimises any traversal time that decreases with speed.
  * ``min_violation`` (b_0 > B_0: already too fast for the budget): full braking at the
    vehicle's maximum deceleration gives the pointwise-smallest speeds of any dynamically
    possible profile, so it minimises every constraint violation.
  * ``no_feasible_gap``: a visible obstacle leaves no gap wide enough; no certificate.
The certificate covers the longitudinal decision only. Path/steering are not certified."""
from __future__ import annotations

CERT_MODEL = "cert-1"
KIND_GREATEST_SAFE, KIND_MIN_VIOLATION, KIND_NO_GAP, KIND_INVALID = 0, 1, 2, 3
KIND_NAMES = {0: "greatest_safe", 1: "min_violation", 2: "no_feasible_gap", 3: "invalid_input"}


def model_constants(vp, a_lat_frac: float, a_brake: float, v_max: float, horizon_r: float, hold: float,
                    ramp_in: float, ramp_out: float, r_sw: float, gap_base: float, gap_per_m: float,
                    gap_margin: float) -> dict:
    """Integer constants of M for one vehicle (mm, mm/s, mm/s^2, metres for R_SW)."""
    g = 9.81
    return {
        "a_lat": int(round(a_lat_frac * vp.mu * g * 1000)), "a_acc": int(round(vp.max_accel * 1000)),
        "a_brk": int(round(a_brake * 1000)), "a_max": int(round(vp.max_decel * 1000)),
        "v_max": int(round(v_max * 1000)), "r_h": int(round(horizon_r * 1000)),
        "hold": int(round(hold * 1000)), "ramp_in": int(round(ramp_in * 1000)), "ramp_out": int(round(ramp_out * 1000)),
        "r_sw_m": int(r_sw), "gap_base": int(round(gap_base * 1000)), "gap_per_m": int(round(gap_per_m)),
        "veh_width": int(round(vp.width * 1000)), "gap_margin": int(round(gap_margin * 1000)),
    }


def quantise(obs) -> dict | None:
    """Fixed-point inputs of M from an Observation (mm, micro-1/m, mm/s)."""
    r = obs.road
    if r is None or not r.valid or len(r.preview_s) < 2:
        return None
    q = lambda x, k: int(round(float(x) * k))  # noqa: E731
    return {"v0": q(obs.speed, 1000), "s": [q(x, 1000) for x in r.preview_s],
            "k": [q(x, 1e6) for x in r.preview_curvature], "wl": [q(x, 1000) for x in r.preview_width_left],
            "wr": [q(x, 1000) for x in r.preview_width_right],
            "obstacles": [[q(o.distance, 1000), q(o.lateral_min, 1000), q(o.lateral_max, 1000)] for o in r.obstacles]}


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
    """Per-station upper bounds on b and whether every visible gap is feasible."""
    n = len(q["s"])
    y = [0] * n
    gcap = [None] * n
    feasible = True
    for ob in q["obstacles"]:
        _, yb, gap = obstacle_plan(c, q, ob)
        if gap < c["veh_width"] + 2 * c["gap_margin"]:
            feasible = False
        slack = max(gap - c["veh_width"], 0)
        vg = c["gap_base"] + c["gap_per_m"] * slack
        for i, s in enumerate(q["s"]):
            if ob[0] - c["hold"] - c["ramp_in"] <= s <= ob[0] + c["hold"] + c["ramp_out"]:
                y[i] = max(y[i], yb)
            if ob[0] - c["hold"] <= s <= ob[0] + c["hold"]:
                gcap[i] = vg * vg if gcap[i] is None else min(gcap[i], vg * vg)
    vmax2 = c["v_max"] * c["v_max"]
    out = []
    for i in range(n):
        ka = abs(q["k"][i])
        ksw = _cdiv(6000 * y[i], c["r_sw_m"] * c["r_sw_m"])
        den = max(1_000_000 - _cdiv(ka * y[i], 1000), 200_000)
        kbar = _cdiv((ka + ksw) * 1_000_000, den)
        cap = vmax2 if kbar == 0 else min(vmax2, c["a_lat"] * 10 ** 9 // kbar)
        if gcap[i] is not None:
            cap = min(cap, gcap[i])
        out.append(cap)
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


def decide(c: dict, q: dict | None) -> dict:
    """Certified longitudinal decision for the quantised observation ``q``."""
    if q is None or any(q["s"][i + 1] <= q["s"][i] for i in range(len(q["s"]) - 1)) or q["s"][0] != 0:
        return {"kind": KIND_INVALID, "a": -c["a_max"], "b1": None, "b0": None, "B0": None}
    cap, feasible = caps(c, q)
    b0 = q["v0"] * q["v0"]
    if not feasible:
        return {"kind": KIND_NO_GAP, "a": -c["a_max"], "b1": None, "b0": b0, "B0": None}
    bw = backward(c, cap, q["s"])
    if b0 > bw[0]:
        return {"kind": KIND_MIN_VIOLATION, "a": -c["a_max"], "b1": None, "b0": b0, "B0": bw[0]}
    f = forward(c, bw, q["s"], b0)
    a = (f[1] - b0) // (2 * (q["s"][1] - q["s"][0]))
    return {"kind": KIND_GREATEST_SAFE, "a": a, "b1": f[1], "b0": b0, "B0": bw[0]}
