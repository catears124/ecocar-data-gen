"""Lean-certified lateral/steering decision model ``steer-1``.

The model consumes the same quantised road preview as ``cert-2`` plus the vehicle's
quantised lateral-offset magnitude (millimetres; the magnitude is non-negative).  An exact
gap tie therefore resolves deterministically to the left side.  All decision arithmetic is
integer arithmetic: offsets are millimetres, station distances are millimetres,
curvature is micro-1/m, and steering is milli-units of the published steering
range.  ``ramp`` is the floor-divided smoothstep profile.  With a declared
station step of 1000 mm, the discrete second difference ``d2`` represents
``d2 / step**2 * 10**9 = 1000*d2`` micro-1/m; the model keeps this fixed
integer scale for every declared station step.  Steering uses
``clip(cdiv(curv * 1000, max(kappa_max, 1)), -1000, 1000)``.  The trusted
vehicle bridge derives ``kappa_max = floor(max_steer * 10**9 / wheelbase_mm)``:
``wheelbase_m * (curv * 10**-6 / m)`` is the linearised bicycle steering angle.

The certified branch is the complete obstacle-side choice, offset profile,
curvature demand, and feed-forward steering demand.  Vehicle-parameter
conversion and the bicycle-model interpretation of ``kappa_max`` remain the
trusted bridge, as in ``cert-2``.
"""
from __future__ import annotations

from fractions import Fraction

from autocross.datagen import cert

CERT_MODEL = "steer-1"
KIND_PLAN_OK, KIND_NO_GAP, KIND_INVALID = 0, 1, 2
KIND_NAMES = {0: "plan_ok", 1: "no_feasible_gap", 2: "invalid_input"}

# The lateral geometry is the declared brake-turn-obstacle geometry used by the
# existing expert.  Keep the longitudinal constants produced by cert-2 in one
# nested object rather than duplicating their definitions here.
_MARGIN = 300
_STEP = 1000
_HOLD_M = 3.0
_RAMP_IN_M = 18.0
_RAMP_OUT_M = 15.0
_R_SW_M = 15
# The declared domain constants come from the scenario family spec, NOT from `expert`: the
# expert imports this module, so importing it back would be a cycle (and a silently wrong
# horizon fallback would make `model_constants` disagree with the generator/validator).
def _horizon_mm() -> int:
    from autocross.datagen.spec import get_family

    fam = get_family("brake_turn_obstacle")
    r_min = min(b.lo for b in fam.factor("turn_radius_m").bins)
    y_max = max(b.hi for b in fam.factor("road_width_m").bins) / 2.0
    return cert.domain_horizon_mm(r_min, y_max, _R_SW_M)


_HORIZON_MM = _horizon_mm()

CONST_KEYS = cert.CONST_KEYS

# Public names used in the Lean mirror and by construction tests.  cert.py in
# this checkout retains the historical private spellings, so aliases are kept
# here without making a second arithmetic implementation.
cdiv = getattr(cert, "cdiv", cert._cdiv)
iabs = getattr(cert, "iabs", abs)
nearestIndex = getattr(cert, "nearestIndex", cert._nearest_index)
plan = getattr(cert, "plan", cert.obstacle_plan)

def _qfix(x: object, scale: int) -> int:
    return cert.qfix(x, scale)

def quantise_from_base(base: dict | None, lateral_offset) -> dict | None:
    """Attach the magnitude of a policy lateral offset to an already-quantised cert-2 input."""
    if base is None:
        return None
    try:
        lat = abs(_qfix(lateral_offset, 1000))
    except (ValueError, TypeError, AttributeError, KeyError):
        return None
    return {"base": base, "lat": lat}


def quantise_dict(obs: dict) -> dict | None:
    """Quantise the cert-2 observation and its policy-visible lateral offset."""
    try:
        base = cert.quantise_dict(obs)
    except (AttributeError, TypeError):
        return None
    try:
        lateral_offset = obs["road"]["lateral_offset"]
    except (KeyError, TypeError):
        return None
    return quantise_from_base(base, lateral_offset)


def quantise(obs) -> dict | None:
    """Quantise an Observation object to the Lean ``Steer.Q`` shape."""
    try:
        base = cert.quantise(obs)
    except (AttributeError, TypeError):
        return None
    try:
        lateral_offset = obs.road.lateral_offset
    except (AttributeError, TypeError):
        return None
    return quantise_from_base(base, lateral_offset)


def _lon(c: dict) -> dict:
    return c["lon"]


def model_constants(vp) -> dict:
    """Derive lateral constants from ``VehicleParams`` using exact integer outputs.

    ``kappa_max = floor(max_steer * 10**9 / wheelbase_mm)``.  Since curvature is
    represented in micro-1/m and wheelbase in millimetres, this is precisely the
    largest integer curvature for which the linear bicycle demand
    ``delta = wheelbase_m * kappa`` is no greater than ``max_steer`` radians.
    """
    lon = cert.model_constants(vp, _HORIZON_MM, _HOLD_M, _RAMP_IN_M, _RAMP_OUT_M, _R_SW_M)
    wb_mm = int(round(vp.wheelbase * 1000))
    if wb_mm <= 0:
        kappa_max = 0
    else:
        # Fraction(repr(x)) follows cert.qfix's exact decimal convention and
        # avoids introducing a binary-float decision into this derived integer.
        kappa_max = int(Fraction(repr(vp.max_steer)) * 1_000_000_000 // wb_mm)
    return {"lon": lon, "margin": _MARGIN, "step": _STEP, "kappa_max": kappa_max}


def _consts_ok(c: dict) -> bool:
    try:
        lon = _lon(c)
        return (cert.consts_ok(lon) and c["margin"] >= 0 and c["step"] > 0
                and c["kappa_max"] >= 0)
    except (KeyError, TypeError):
        return False


def well_formed(c: dict | None, q: dict | None) -> bool:
    """The decidable domain guard shared by Python and ``Corpus.Steer``."""
    if not isinstance(c, dict) or not isinstance(q, dict):
        return False
    try:
        base = q["base"]
        if not cert.well_formed(base) or not _consts_ok(c) or q["lat"] < 0:
            return False
        s = base["s"]
        step = c["step"]
        if any(s[i + 1] - s[i] != step for i in range(len(s) - 1)):
            return False
        obstacles = base["obstacles"]
        if any(not isinstance(ob, list) or len(ob) != 3 for ob in obstacles):
            return False
        hold = c["lon"]["hold"]
        for i, ob in enumerate(obstacles):
            d = ob[0]
            for other in obstacles[i + 1:]:
                e = other[0]
                if not (e - hold > d + hold or d - hold > e + hold):
                    return False
        return True
    except (KeyError, TypeError, IndexError):
        return False


def _target_and_gap(c: dict, q: dict, ob: list[int]) -> tuple[int, int, int, int]:
    """Return ``(left_target, right_target, left_gap, right_gap)``."""
    base = q["base"]
    d, lmin, lmax = ob
    i = nearestIndex(base["s"], d)
    wl, wr = base["wl"][i], base["wr"][i]
    yl = cdiv(iabs(lmax + wl), 2)
    yr = cdiv(iabs(lmin - wr), 2)
    return yl, yr, wl - lmax, lmin + wr

def sideAmp(q: dict, d: int, lmin: int, lmax: int) -> tuple[int, int]:
    """Lean ``sideAmp``: choose ``(+/-1, absolute target)`` for one obstacle."""
    base = q["base"]
    i = nearestIndex(base["s"], d)
    wl, wr = base["wl"][i], base["wr"][i]
    yl = cdiv(iabs(lmax + wl), 2)
    yr = cdiv(iabs(lmin - wr), 2)
    gl, gr = wl - lmax, lmin + wr
    if gl > gr:
        return 1, yl
    if gr > gl:
        return -1, yr
    return (1, yl) if q["lat"] >= 0 else (-1, yr)


def sideAmpOb(q: dict, ob: list[int]) -> tuple[int, int]:
    """Lean ``sideAmpOb`` accessor (malformed triples return ``(0, 0)``)."""
    if not isinstance(ob, list) or len(ob) != 3:
        return 0, 0
    return sideAmp(q, ob[0], ob[1], ob[2])


def choose_side(c: dict, q: dict, ob: list[int]) -> tuple[int, int]:
    """Choose side and absolute target offset using the specified tie rule."""
    del c
    return sideAmpOb(q, ob)


def ramp(A: int, x: int, L: int) -> int:
    """Floor-divided cubic smoothstep, for ``0 <= x <= L`` and ``A >= 0``."""
    if L <= 0:
        return 0
    return (A * (3 * x * x * L - 2 * x * x * x)) // (L * L * L)


def offOb(c: dict, d: int, side: int, amp: int, si: int) -> int:
    """Signed offset contributed by one obstacle at one station."""
    hold, rin, rout = c["lon"]["hold"], c["lon"]["ramp_in"], c["lon"]["ramp_out"]
    if si < d - hold - rin:
        return 0
    if si < d - hold:
        return side * ramp(amp, si - d + hold + rin, rin)
    if si <= d + hold:
        return side * amp
    if si <= d + hold + rout:
        return side * (amp - ramp(amp, si - d - hold, rout))
    return 0


def wOb(c: dict, d: int, si: int) -> int:
    """Integer [0,1000] weight of one obstacle at one station."""
    hold, rin, rout = c["lon"]["hold"], c["lon"]["ramp_in"], c["lon"]["ramp_out"]
    if si < d - hold - rin:
        return 0
    if si < d - hold:
        return ramp(1000, si - d + hold + rin, rin)
    if si <= d + hold:
        return 1000
    if si <= d + hold + rout:
        return 1000 - ramp(1000, si - d - hold, rout)
    return 0

def rawOffAt(c: dict, q: dict, si: int) -> int:
    """Combine obstacle offsets by the first strictly largest weight."""
    best_w, best_off = 0, 0
    for ob in q["base"]["obstacles"]:
        if not isinstance(ob, list) or len(ob) != 3:
            continue
        d, lmin, lmax = ob
        w = wOb(c, d, si)
        side, amp = sideAmp(q, d, lmin, lmax)
        if best_w < w:
            best_w, best_off = w, offOb(c, d, side, amp, si)
    return best_off


def halfw(c: dict) -> int:
    return c["lon"]["veh_width"] // 2


def lo(c: dict, q: dict, i: int) -> int:
    return -(q["base"]["wr"][i] - halfw(c) - c["margin"])


def hi(c: dict, q: dict, i: int) -> int:
    return q["base"]["wl"][i] - halfw(c) - c["margin"]


def clampAt(c: dict, q: dict, i: int, value: int) -> int:
    low, high = lo(c, q, i), hi(c, q, i)
    if low <= high:
        return min(max(value, low), high)
    return (low + high) // 2


def offAt(c: dict, q: dict, si: int, i: int) -> int:
    return clampAt(c, q, i, rawOffAt(c, q, si))


def offs(c: dict, q: dict) -> list[int]:
    return [offAt(c, q, si, i) for i, si in enumerate(q["base"]["s"])]


def secondAt(xs: list[int], i: int) -> int:
    if i == 0 or i + 1 >= len(xs):
        return 0
    return iabs(xs[i + 1] - 2 * xs[i] + xs[i - 1])


def curvAt(q: dict, xs: list[int], i: int) -> int:
    return iabs(q["base"]["k"][i] if i < len(q["base"]["k"]) else 0) + 1000 * secondAt(xs, i)


def curvs(q: dict, xs: list[int]) -> list[int]:
    return [curvAt(q, xs, i) for i in range(len(xs))]





def steerAt(c: dict, curv: int) -> int:
    km = max(c["kappa_max"], 1)
    return max(-1000, min(1000, cdiv(curv * 1000, km)))


def steers(c: dict, ks: list[int]) -> list[int]:
    return [steerAt(c, k) for k in ks]


def sides(q: dict) -> list[int]:
    return [sideAmpOb(q, ob)[0] for ob in q["base"]["obstacles"]]


def amps(q: dict) -> list[int]:
    return [sideAmpOb(q, ob)[1] for ob in q["base"]["obstacles"]]


def hasNoGap(c: dict, q: dict) -> bool:
    for ob in q["base"]["obstacles"]:
        if not isinstance(ob, list) or len(ob) != 3:
            return True
        _, _, gl, gr = _target_and_gap(c, q, ob)
        if max(gl, gr) < c["lon"]["veh_width"] + 2 * c["margin"]:
            return True
    return False

def _plan(c: dict, q: dict) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    side_list, amp_list = sides(q), amps(q)
    off_list = offs(c, q)
    curv_list = curvs(q, off_list)
    return side_list, amp_list, off_list, curv_list, steers(c, curv_list)
def decide(c: dict, q: dict | None) -> dict:
    """Recompute the complete integer lateral decision for the quantised input."""
    if not well_formed(c, q):
        return {"kind": KIND_INVALID, "side": [], "amp": [], "off": [], "curv": [], "steer": []}
    side_list, amp_list, off_list, curv_list, steer_list = _plan(c, q)
    return {"kind": KIND_NO_GAP if hasNoGap(c, q) else KIND_PLAN_OK, "side": side_list,
            "amp": amp_list, "off": off_list, "curv": curv_list, "steer": steer_list}


__all__ = ["CERT_MODEL", "KIND_PLAN_OK", "KIND_NO_GAP", "KIND_INVALID", "KIND_NAMES",
           "CONST_KEYS", "quantise_dict", "quantise", "quantise_from_base", "model_constants", "well_formed", "decide",
           "rawOffAt", "halfw", "lo", "hi", "clampAt", "offAt", "offs", "secondAt", "curvAt", "curvs",
           "steerAt", "steers", "sides", "amps", "hasNoGap", "choose_side"]
