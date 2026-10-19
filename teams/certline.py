"""
we'd like to thank our good friends claude and deepseek
 
Two certified integer decision laws from the EcoCAR certified-imitation work are vendored here
byte-for-byte in their arithmetic and evaluated every control step on the *live* road model:

* **``cert-2`` (longitudinal).** ``_decide_long`` walks a forward/backward speed profile on the
  *planned path* under a per-station grip bound ``b_i*kappa_i <= a_lat*1e9``, an
  acceleration/braking budget, a speed cap, and a terminal horizon bound ``b_n <= a_lat*r_h``
  (the tightest planned path the declared road domain allows may start just past the preview).
  The output is the greatest integer acceleration whose end point still admits a feasible
  profile; when the car is already over budget it is full braking (the pointwise-least speeds).
  Lean proves, for this function: greatest-safe, same-information optimality, and the
  min-violation property (see ``lean/Corpus/Cert.lean``).
* **``steer-2`` (lateral).** ``_decide_lateral`` chooses the obstacle pass side, builds the
  integer offset profile (smoothstep ramps into the wider free gap, clamped to the cone
  corridor), and derives the planned path's curvature demand ``curv_i = |kappa_road_i| +
  1000*|d2 offset_i|`` plus the linearised steering feed-forward ``curv_i*1000/kappa_max``.
  Lean proves the branch complete, the gap-side tie rule (``lean/Corpus/Steer.lean``) and, under
  the decidable domain predicates ``clearanceDomain``/``holdTargetsOK``, hold-window clearance of
  every ``plan_ok`` plan (``lean/Corpus/SteerClear.lean``).

The two are made mutually consistent here: the longitudinal grip bound uses the curvature of the
path *this controller actually publishes* (centreline curvature plus the second difference of the
published offsets, max over a +-1 station window), so the plan and its speed profile describe
the same path. ``lean/Corpus/Controller.lean`` and ``ControllerLong.lean`` prove the clearance
interval, projection, publication order and speed decision of this file's integer functions.

Above them sits the (uncertified) tracking/actuation layer: path-error state feedback (cross-track
and heading error to the published path, taken at the pose the car will have once the command
acts, plus the path's own curvature as feed-forward), a grip clamp so the commanded steer never
asks for more than ``grip_clamp`` of ``mu*g``, a command rate limit inside the actuator's slew
limit, and a pedal map that never overlaps the pedals and leaves grip for cornering. Each plan
starts on the previous plan at the car's new course station, so the published path is a stable
reference rather than one re-anchored on the car every step.

Budgets (``lat_frac``/``acc_frac``/``brk_frac``/``horizon_r``) are declared model constants, not
tyre measurements: the sharper the declared budget the faster and the less conservative the
certificate. Defaults sit inside the published tyre limits. Tune with ``--controller-config``.

Trust boundary: the certified label covers the longitudinal decision (and, for ``steer-2``, the
lateral plan) of its own model. Steering, pedal mapping and path tracking are not certified, and
nothing here is a claim about CARLA or bicycle dynamics.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from autocross.api import Command, Controller, Observation, TrackInfo, VehicleParams

G = 9.81
MM, MICRO = 1000, 1_000_000

KIND_GREATEST_SAFE, KIND_MIN_VIOLATION, KIND_NO_GAP, KIND_INVALID = 0, 1, 2, 3
KIND_PLAN_OK, KIND_PLAN_NO_GAP, KIND_PLAN_INVALID = 0, 1, 2

# The declared budgets default to the FRICTION BOX of the certified model: the largest equal-axis
# box inside the friction circle, mu*g/sqrt(2) = 0.7071 * mu * g. Every plan of the model then stays
# inside the point-mass friction circle even while accelerating or braking (the circle is what a
# tyre actually has; an equal-axis box is the safe, provable subset of it), and the tracker keeps
# the remaining (1 - 1/sqrt(2)) of the circle as margin for corrections that the plan cannot see.
BOX_FRAC = 1.0 / math.sqrt(2.0)

# Declared road domain of the *event*, used for the terminal horizon constraint r_h. The evaluation
# courses are produced by ``autocross.trackgen`` (the README: the judges make secret courses with
# ``gen-track``), so the domain is that generator's *union over open and closed*, not the shapes of
# the public courses and not the training family:
#   * open: tightest element is a hairpin of radius 9.0 m (uniform 9-11.5 m) at ``width + 1.0``;
#   * closed (``gen-track --closed``): reaches an implied minimum radius of 6.55 m -- measured over
#     40 seeds, p05 6.72 m, 30% of courses below 9 m -- so an open-only floor of 9 m is NOT sound
#     on a loop course;
#   * the lateral plan may sidestep into the full half-width, y_max = 6.5/2.
# The union floor (6.5 m) is therefore the declared r_min, and a decision from it is sound whichever
# of open/closed the unreleased courses turn out to be. r_h is the effective radius of that tightest
# planned path by the same formula cert-2 uses (sidestep curvature 6y/R_SW^2, parallel-curve
# corrected), computed exactly as ``autocross.datagen.cert.domain_horizon_mm``. The corpus expert
# declares the same y_max and R_SW but r_min = 6.0 m (the measured floor minus a 0.5 m margin,
# r_h = 1.809 m); this controller uses the measured floor itself (r_h = 2.078 m). Both are
# declarations from a 40-seed measurement, not proofs about trackgen.
DOMAIN_R_MIN_M = 6.5                        # union floor: trackgen closed measured floor
DOMAIN_WIDTH_M = 5.5 + 1.0                  # trackgen hairpin width = base width + 1.0


def _cdiv(a: int, b: int) -> int:
    return -((-a) // b)


def domain_horizon_mm(r_min_m: float, y_max_m: float, r_sw_m: int) -> int:
    """Effective radius (mm) of the tightest planned path the declared domain allows, exactly as
    ``autocross.datagen.cert.domain_horizon_mm`` computes it."""
    k = _cdiv(10 ** 9, int(round(r_min_m * 1000)))
    y = int(round(y_max_m * 1000))
    ksw = _cdiv(6000 * y, r_sw_m * r_sw_m)
    den = max(1_000_000 - _cdiv(k * y, 1000), 200_000)
    return 10 ** 9 // _cdiv((k + ksw) * 1_000_000, den)


DOMAIN_HORIZON_MM = domain_horizon_mm(DOMAIN_R_MIN_M, DOMAIN_WIDTH_M / 2.0, 15)

DEFAULTS = dict(
    # --- declared budgets of the certified models -------------------------------------------
    lat_frac=BOX_FRAC,      # a_lat = lat_frac * mu * g (lateral grip budget of the plan)
    acc_frac=BOX_FRAC,      # a_acc = min(max_accel, acc_frac * mu * g)
    brk_frac=0.50,          # a_brk = min(max_decel, brk_frac * mu * g). Below the box on purpose:
                            #   planning to brake at half the circle starts braking earlier and
                            #   leaves the tracker lateral grip on corner entry (bicycle sweep over
                            #   15 courses: 0.71 -> 48 cone hits, 0.50 -> 0, lap time unchanged)
    horizon_r=DOMAIN_HORIZON_MM / 1000.0,   # terminal constraint b_n <= a_lat * r_h [m]; the
                            #   declared domain's tightest planned path (2.078 m), see above
    gap_speed=1.0,          # 1 = declared gap-speed rule beside a visible obstacle (see below)
    gap_base=7.0,           # gap-speed rule: v <= gap_base + gap_per_m * slack [m/s]
    gap_per_m=8.0,          #   slack = free gap width - car width - 2*gap_margin [m]
    hold=3.0,               # obstacle half-length of the lateral plan [m]
    ramp_in=18.0,           # sidestep ramp lengths of the lateral plan [m]
    ramp_out=15.0,
    margin=0.30,            # lateral plan keeps the car body this far inside the cones [m]
    gap_margin=0.25,        # free space required each side of the car for a gap to count [m]
    max_offset=1.50,        # cap on |sidestep offset| [m]; the corridor clamp binds first
    # --- path ---------------------------------------------------------------------------------
    kappa_floor=0.0020,     # 1/m, curvature noise floor used for the speed caps
    kappa_window=1,         # +-stations: the cap uses the max over this window (a sound bound)
    kappa_safety=1.00,      # explicit safety factor on the cap curvature (1.0 = pure bound)
    cap_margin=1.00,        # b_i cap scaled by this; 1.0 = the declared budget exactly
                            #   (the box above already reserves the tracker's margin)
    # --- tracking (uncertified) ---------------------------------------------------------------
    k_ff=0.35,              # gain on the curvature feed-forward steering
    k_pp=1.3,               # gain on the pure-pursuit term
    k_psi=0.40,             # gain on the path-tangent heading error at the look-ahead (mode 0)
    k_r=0.30,               # yaw-rate damping gain [s]
    k_obs_track=1.50,       # obstacle-hold cross-track correction (uncertified tracker reserve)
    k_v=0.50,               # speed-error gain on top of the certified acceleration [1/s]
    ld_base=2.0,            # look-ahead [m]
    ld_gain=0.30,           # look-ahead per m/s
    ld_min=3.0,
    ld_max=8.0,
    ld_ff=3.0,              # look-ahead for the curvature feed-forward and yaw-rate reference [m]
    trk_mode=1.0,           # 1 = path-error state feedback (below), 0 = legacy pure-pursuit blend
    trk_omega=3.25,         # closed-loop natural frequency of the lateral error [rad/s]
    trk_zeta=0.9,           # ... and its damping ratio
    trk_vmin=3.0,           # speed floor of the gain schedule [m/s]
    trk_lag=0.10,           # steering lag the errors are predicted over [s] (actuator + ZOH)
    trk_lead=0.0,           # feed-forward curvature taken this far ahead of the projection [s]
    line_anchor=1.0,        # 1 = start each plan on the previous plan (course frame), 0 = on the car
    thr_reserve=0.5,        # throttle cap 1 - (thr_reserve * |ay| / mu g)^2 (floor 0.15)
    grip_clamp=1.00,        # commanded steer never asks for more than this * mu*g laterally
    recover_lat=0.70,       # recovery guard: |lateral offset| / width beyond which it engages
    recover_head=0.30,      # recovery guard: |heading error| [rad] beyond which it engages
    k_rec_head=1.40,        # recovery guard: steer per rad of heading error [rad/rad]
    k_rec_lat=0.20,         # recovery guard: steer per metre of lateral offset [rad/m]
    brake_deadband=0.2,     # m/s^2 of demanded deceleration left to drag before braking
    recover_brake=0.0,      # brake pedal while the recovery guard is engaged
    recover_throttle=0.25,  # throttle held while recovering below recover_vmin (never stall: a
    recover_vmin=4.0,       #   stopped car is a 'stuck' DNF after 8 s)
    # --- racing line: min-curvature offsets *inside* the certified clearance interval ----------
    line_on=1.0,            # 1 = plan the min-curvature line, 0 = the certified steer-2 plan
    line_shrink=0.15,       # plan this far inside the certified interval [m] (strictly safer than
                            #   the certified plan, which rides the interval it clamps to)
    line_w_prev=0.7,        # stickiness to the previous plan (course frame): keeps the published
                            #   path stable without freezing it
    line_w_center=0.001,    # weak pull to the centerline
    line_ramp=0.35,         # fallback plan: max lateral rate per station [m/m] (a fold the harness
                            #   would reject is the one thing the tracker must never be handed)
    line_iters=8,           # active-set iterations of the box QP
    line_eps=1e-6,          # Tikhonov term so the free block is always invertible
    # --- obstacle memory (the road model drops an obstacle 3 m behind the car while the rear axle
    #     is still alongside it; the plan must not pull back into it) ---------------------------
    obs_pre=4.5,            # keep a remembered obstacle this far before its station [m]
    obs_post=4.5,           # ... and this far past it [m]; the pass side is chosen once and kept
    # --- robustness: all three are inactive in nominal conditions (delay 0, bias 0, grip true) --
    delay_comp=1.0,         # steer from the pose the queued commands will produce
    bias_est=1.0,           # subtract the measured steering bias (obs.steer vs what we sent)
    grip_adapt=1.0,         # scale the budgets down when the yaw response proves saturation
    sat_ratio=0.68,         # yaw rate / kinematic yaw rate below which the tyres are saturated
    sat_ay=4.5,             # only judge saturation when cornering at least this hard [m/s^2]
    bias_gain=0.15,         # bias estimator gain
    bias_lim=0.15,          # |bias| cap [steer fraction]
    beta_ease=1.0,          # shed pedals on measured side-slip (obs.vy/obs.vx)
    beta_ease_lim=0.05,     # |side slip| beyond which the pedals are eased [rad]
)


# --------------------------------------------------------------------------- cert-2 (vendored)
def qfix(x, scale: int) -> int:
    """round-half-up(x*scale), exact on the shortest decimal repr; raises on None/NaN/inf.

    Integer-only: ``repr(x)`` is split into an integer mantissa ``M`` and a decimal exponent ``E``
    (``x = M·10^E`` exactly as written), so ``floor(M·scale·10^E + 1/2)`` is computed with integer
    floor division -- the same value as ``floor(Fraction(repr(x))*scale + 1/2)``, ~20x faster.
    """
    if isinstance(x, bool) or x is None:
        raise ValueError("not a number")
    x = float(x)
    if not math.isfinite(x):
        raise ValueError("non-finite")
    mant, _, exp = repr(x).partition("e")
    ip, _, fp = mant.partition(".")
    m = int(ip + fp)
    e = (int(exp) if exp else 0) - len(fp)
    if e >= 0:
        return m * scale * 10 ** e
    d = 10 ** (-e)
    return (2 * m * scale + d) // (2 * d)


def _nearest_index(s: list[int], d: int) -> int:
    for i, x in enumerate(s):
        if x >= d:
            return i
    return len(s) - 1


def quantise(obs) -> dict | None:
    """Fixed-point inputs of both models from an Observation (mm, micro-1/m, mm/s)."""
    r = getattr(obs, "road", None)
    if r is None or not r.valid:
        return None
    try:
        q = {"v0": qfix(obs.speed, MM), "s": [qfix(x, MM) for x in r.preview_s],
             "k": [qfix(x, MICRO) for x in r.preview_curvature],
             "wl": [qfix(x, MM) for x in r.preview_width_left],
             "wr": [qfix(x, MM) for x in r.preview_width_right],
             "lat": qfix(r.lateral_offset, MM),
             "obstacles": [[qfix(o.distance, MM), qfix(o.lateral_min, MM), qfix(o.lateral_max, MM)]
                           for o in r.obstacles]}
    except (ValueError, TypeError):
        return None
    return q if len(q["s"]) >= 2 else None


def well_formed(q: dict | None) -> bool:
    """The decidable domain guard (``wellFormed``)."""
    if q is None:
        return False
    s = q["s"]
    n = len(s)
    return (n >= 2 and s[0] == 0 and all(s[i] < s[i + 1] for i in range(n - 1))
            and len(q["k"]) == n and len(q["wl"]) == n and len(q["wr"]) == n
            and all(len(ob) == 3 for ob in q["obstacles"]) and q["v0"] >= 0)


def consts_ok(c: dict) -> bool:
    return c["a_acc"] >= 0 and c["a_brk"] >= 0 and c["a_max"] > 0 and c["r_sw_m"] > 0


def lon_constants(vp: VehicleParams, cfg: dict) -> dict:
    """Declared integer constants of the longitudinal model (mm, mm/s^2)."""
    a_lat = int(round(cfg["lat_frac"] * vp.mu * G * 1000))
    return {"a_lat": a_lat,
            "a_acc": min(int(round(vp.max_accel * 1000)), int(round(cfg["acc_frac"] * vp.mu * G * 1000))),
            "a_brk": min(int(round(vp.max_decel * 1000)), int(round(cfg["brk_frac"] * vp.mu * G * 1000))),
            "a_max": int(round(vp.max_decel * 1000)), "v_max": int(round(vp.max_speed * 1000)),
            "r_h": int(round(cfg["horizon_r"] * 1000)), "hold": int(round(cfg["hold"] * 1000)),
            "ramp_in": int(round(cfg["ramp_in"] * 1000)), "ramp_out": int(round(cfg["ramp_out"] * 1000)),
            "r_sw_m": int(min(cfg["ramp_in"], cfg["ramp_out"])),
            "veh_width": int(round(vp.width * 1000)),
            "gap_base": int(round(cfg["gap_base"] * 1000)),
            "gap_per_m": int(round(cfg["gap_per_m"])),
            "gap_margin": int(round(cfg["gap_margin"] * 1000)),
            "gap_speed": float(cfg["gap_speed"])}


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


def decide_long(c: dict, q: dict | None, kappa_micro: list[int], cap_scale: float = 1.0,
                extra_caps: list[int] | None = None) -> dict:
    """Certified longitudinal decision on a path whose curvature bound is ``kappa_micro``.

    ``extra_caps`` are optional per-station speed² caps in mm²/s².  They are
    intersected with the curvature/gap caps before the unchanged Cert-2
    backward/forward passes.  The sidestep gate supplies these caps using
    ``v² <= a_lat D²/(4 Δy)``: a bang-bang lateral move of ``Δy`` at
    acceleration ``a_lat`` takes ``2*sqrt(Δy/a_lat)`` seconds, which must fit
    before the hold-entry distance ``D``.
    """
    invalid = {"kind": KIND_INVALID, "a": -c["a_max"], "b1": None, "b0": None, "B0": None}
    if (not well_formed(q) or not consts_ok(c) or len(kappa_micro) != len(q["s"])
            or (extra_caps is not None and len(extra_caps) != len(q["s"]))):
        return invalid
    vmax2 = c["v_max"] * c["v_max"]
    cap = [vmax2 if k <= 0 else min(vmax2, c["a_lat"] * 10 ** 9 // k) for k in kappa_micro]
    if extra_caps is not None:
        cap = [min(x, int(y)) for x, y in zip(cap, extra_caps)]
    # Declared gap-speed rule (the obstacle-passing constraint of the certified family model): beside
    # a visible obstacle the speed is capped by the free gap, so passing cannot rely on the line
    # being held exactly. A gap too narrow for the car makes the plan infeasible (no_feasible_gap).
    feasible = True
    if c.get("gap_speed"):
        for ob in q["obstacles"]:
            i = _nearest_index(q["s"], ob[0])
            gl, gr = q["wl"][i] - ob[2], ob[1] + q["wr"][i]
            gap = max(gl, gr)
            if gap < c["veh_width"] + 2 * c["gap_margin"]:
                feasible = False
            vg = c["gap_base"] + c["gap_per_m"] * max(gap - c["veh_width"] - 2 * c["gap_margin"], 0)
            for j, sj in enumerate(q["s"]):
                if abs(sj - ob[0]) <= c["hold"]:
                    cap[j] = min(cap[j], vg * vg)
    cap[-1] = min(cap[-1], c["a_lat"] * c["r_h"])
    cap = [int(x * cap_scale) for x in cap]
    # Same branch order as the Lean specification `Corpus.Controller.decideCaps` (and as
    # `cert.decide`): a negative cap is an invalid input before the gap check is consulted.
    if any(x < 0 for x in cap):
        return invalid
    if not feasible:
        return {"kind": KIND_NO_GAP, "a": -c["a_max"], "b1": None, "b0": q["v0"] * q["v0"], "B0": None}
    b0 = q["v0"] * q["v0"]
    bw = backward(c, cap, q["s"])
    if b0 > bw[0]:
        return {"kind": KIND_MIN_VIOLATION, "a": -c["a_max"], "b1": None, "b0": b0, "B0": bw[0]}
    f = forward(c, bw, q["s"], b0)
    a = (f[1] - b0) // (2 * (q["s"][1] - q["s"][0]))
    return {"kind": KIND_GREATEST_SAFE, "a": a, "b1": f[1], "b0": b0, "B0": bw[0]}


# --------------------------------------------------------------------------- steer-2 (vendored)
def _ramp(A: int, x: int, L: int) -> int:
    """Floor-divided cubic smoothstep for ``0 <= x <= L``."""
    if L <= 0:
        return 0
    return (A * (3 * x * x * L - 2 * x * x * x)) // (L * L * L)


def _side_amp(q: dict, d: int, lmin: int, lmax: int) -> tuple[int, int]:
    i = _nearest_index(q["s"], d)
    wl, wr = q["wl"][i], q["wr"][i]
    yl, yr = _cdiv(abs(lmax + wl), 2), _cdiv(abs(lmin - wr), 2)
    gl, gr = wl - lmax, lmin + wr
    if gl > gr:
        return 1, yl
    if gr > gl:
        return -1, yr
    return (1, yl) if q["lat"] >= 0 else (-1, yr)


def _off_ob(c: dict, d: int, side: int, amp: int, si: int) -> int:
    hold, rin, rout = c["hold"], c["ramp_in"], c["ramp_out"]
    if si < d - hold - rin:
        return 0
    if si < d - hold:
        return side * _ramp(amp, si - d + hold + rin, rin)
    if si <= d + hold:
        return side * amp
    if si <= d + hold + rout:
        return side * (amp - _ramp(amp, si - d - hold, rout))
    return 0


def _w_ob(c: dict, d: int, si: int) -> int:
    hold, rin, rout = c["hold"], c["ramp_in"], c["ramp_out"]
    if si < d - hold - rin:
        return 0
    if si < d - hold:
        return _ramp(1000, si - d + hold + rin, rin)
    if si <= d + hold:
        return 1000
    if si <= d + hold + rout:
        # Keep ramp-out weights strictly below the in-hold weight.  Otherwise a
        # small ramp-out position ties an earlier obstacle's hold and the
        # first-strictly-largest supplier remains selected.
        return min(999, 1000 - _ramp(1000, si - d - hold, rout))
    return 0


def lateral_constants(vp: VehicleParams, cfg: dict, lon: dict) -> dict:
    """``steer-2`` constants: kappa_max is the linearised bicycle limit of the published steer.
    Carries the longitudinal geometry it needs (hold, ramps, vehicle width) so the lateral
    decision is a pure function of one constant object, as in the Lean mirror."""
    wb_mm = int(round(vp.wheelbase * 1000))
    kappa_max = 0 if wb_mm <= 0 else int(Fraction(repr(float(vp.max_steer))) * 1_000_000_000 // wb_mm)
    return {"margin": int(round(cfg["margin"] * 1000)), "step": 1000, "kappa_max": kappa_max,
            "veh_width": lon["veh_width"], "hold": lon["hold"], "ramp_in": lon["ramp_in"],
            "ramp_out": lon["ramp_out"]}


def _halfw(c: dict) -> int:
    return c["veh_width"] // 2


def _bounds(c: dict, q: dict, i: int) -> tuple[int, int]:
    """Corridor bounds (mm) for the lateral offset at station ``i``."""
    return -(q["wr"][i] - _halfw(c) - c["margin"]), q["wl"][i] - _halfw(c) - c["margin"]


def _clamp_at(c: dict, q: dict, i: int, value: int) -> int:
    low, high = _bounds(c, q, i)
    return min(max(value, low), high) if low <= high else (low + high) // 2


def _raw_off(c: dict, q: dict, si: int) -> int:
    """Combine the obstacle sidesteps by first strictly-largest weight."""
    best_w, best_off = 0, 0
    for ob in q["obstacles"]:
        d, lmin, lmax = ob
        w = _w_ob(c, d, si)
        side, amp = _side_amp(q, d, lmin, lmax)
        if best_w < w:
            best_w, best_off = w, _off_ob(c, d, side, amp, si)
    return best_off


def _second_at(xs: list[int], i: int) -> int:
    if i == 0 or i + 1 >= len(xs):
        return 0
    return abs(xs[i + 1] - 2 * xs[i] + xs[i - 1])


def curv_bound_micro(q: dict, offsets: list[int]) -> list[int]:
    """``steer-2`` curvature demand of an offset profile: ``|k_road| + 1000*|d2 offset|`` (micro-1/m).
    An upper bound on the true curvature of the offset path, exactly as the certified model uses it."""
    return [abs(q["k"][i]) + 1000 * _second_at(offsets, i) for i in range(len(offsets))]


def _has_no_gap(c: dict, q: dict) -> bool:
    for ob in q["obstacles"]:
        d, lmin, lmax = ob
        i = _nearest_index(q["s"], d)
        gl, gr = q["wl"][i] - lmax, lmin + q["wr"][i]
        if max(gl, gr) < c["veh_width"] + 2 * c["margin"]:
            return True
    return False


def decide_lateral(c: dict, lon: dict, q: dict | None) -> dict:
    """Complete ``steer-2`` decision: pass sides, offset profile (mm), curvature demand, steering."""
    if q is None or not well_formed(q) or not consts_ok(lon):
        return {"kind": KIND_PLAN_INVALID, "off": [], "curv": [], "steer": [], "side": []}
    offs = [_clamp_at(c, q, i, _raw_off(c, q, si)) for i, si in enumerate(q["s"])]
    curv = curv_bound_micro(q, offs)
    km = max(c["kappa_max"], 1)
    steer = [max(-1000, min(1000, _cdiv(k * 1000, km))) for k in curv]
    side = [_side_amp(q, ob[0], ob[1], ob[2])[0] for ob in q["obstacles"]]
    return {"kind": KIND_PLAN_NO_GAP if _has_no_gap(c, q) else KIND_PLAN_OK,
            "off": offs, "curv": curv, "steer": steer, "side": side}


# --------------------------------------------------------------------------- helpers
def _plausible(path: np.ndarray, max_curvature: float = 0.45, min_length: float = 2.0) -> bool:
    """As the harness's ``geometry.path_is_plausible``: long enough, finite, no tight folds."""
    p = np.asarray(path, float)
    if len(p) < 3 or not np.isfinite(p).all():
        return False
    d = np.diff(p, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    if seg.sum() < min_length or (seg <= 1e-9).any():
        return False
    h = np.arctan2(d[:, 1], d[:, 0])
    dh = np.abs((np.diff(h) + np.pi) % (2.0 * np.pi) - np.pi)
    return bool(np.all(dh <= max_curvature * np.maximum(seg[1:], 1e-6) + 1e-9))


# --------------------------------------------------------------------------- racing line
def _remember(c: dict, mem: list[dict], obstacles_mm: list[list[int]], q: dict,
              progress_mm: int, pre_mm: int, post_mm: int) -> None:
    """Track obstacles in course coordinates, each with a pass side chosen once.

    The road model drops an obstacle about 3 m behind the car while the rear axle is still
    alongside it, so a plan built from the visible set alone pulls back to the centerline exactly
    where the car has to stay offset. A remembered obstacle is matched by station and lateral
    overlap, keeps the side steer-2 gave it when it was first seen (no left/right flip-flopping
    between steps), and is dropped ``post_mm`` after the car has passed its station.
    """
    for d, lmin, lmax in obstacles_mm:
        st = progress_mm + d
        for o in mem:
            if abs(o["st"] - st) < 1000 and lmin < o["lmax"] + 500 and lmax > o["lmin"] - 500:
                o["st"], o["lmin"], o["lmax"] = st, lmin, lmax
                break
        else:
            mem.append({"st": st, "lmin": lmin, "lmax": lmax,
                        "side": _side_amp(q, d, lmin, lmax)[0]})
    mem[:] = [o for o in mem if progress_mm - post_mm <= o["st"] <= progress_mm + pre_mm + 60000]
    for o in mem:
        o["d"] = o["st"] - progress_mm




def _sidestep_caps(c: dict, q: dict, mem: list[dict], offsets_m: np.ndarray) -> list[int]:
    """Integer speed² caps needed to reach each remembered hold entry.

    At every step ``q`` is re-quantised and ``mem`` is in vehicle-relative
    stations.  For an entry ``D = d - hold > 0`` and lateral displacement
    ``dy``, the bang-bang minimum time is ``2*sqrt(dy/a_lat)``; requiring
    ``D/v`` to exceed that time gives ``v² <= a_lat*D²/(4*dy)``.
    """
    caps = [2 ** 120] * len(q["s"])
    if not mem or len(offsets_m) != len(q["s"]) or c["a_lat"] <= 0:
        return caps
    for o in mem:
        entry = int(o["d"]) - int(c["hold"])
        if entry <= 0:
            continue
        target = int(round(float(np.interp(entry, q["s"], offsets_m)) * MM))
        dy = abs(target - int(q["lat"]))
        if dy <= 0:
            continue
        gate = c["a_lat"] * entry * entry // (4 * dy)
        for i, si in enumerate(q["s"]):
            if si < entry:
                caps[i] = min(caps[i], gate)
    return caps


def _spare_boxes(c: dict, q: dict, mem: list[dict], i: int) -> tuple[int, int]:
    """Certified clearance interval (mm) at station ``i``.

    The corridor walls of steer-2 (``_bounds``) intersected with the half-space constraint of
    every obstacle whose hold window covers the station: passing on the obstacle's left makes it a
    right-hand wall at ``lmax + halfw + margin``, passing on its right makes it a left-hand wall at
    ``lmin - halfw - margin``. This is exactly the pair of terms in steer-2's ``clearAt``, so
    ``lo <= d <= hi`` at a station is the statement "the plan keeps the certified clearance here".
    """
    lo, hi = _bounds(c, q, i)
    si = q["s"][i]
    for o in mem:
        if not (o["d"] - c["hold"] <= si <= o["d"] + c["hold"]):
            continue
        if o["side"] > 0:
            lo = max(lo, o["lmax"] + _halfw(c) + c["margin"])
        elif o["side"] < 0:
            hi = min(hi, o["lmin"] - _halfw(c) - c["margin"])
    return lo, hi


def _box_qp(H, g, lo, hi, pin, iters: int, eps: float):
    """min 0.5 d'Hd + g'd  s.t.  lo <= d <= hi, d[pin] fixed. Small active-set loop."""
    n = len(g)
    fixed = np.array(pin, bool)
    val = 0.5 * (lo + hi)
    val[fixed] = lo[fixed]
    d = val.copy()
    for _ in range(max(int(iters), 1)):
        free = ~fixed
        d[fixed] = val[fixed]
        if not free.any():
            break
        Hf = H[np.ix_(free, free)] + eps * np.eye(int(free.sum()))
        rhs = -g[free] - H[np.ix_(free, fixed)] @ d[fixed]
        try:
            d[free] = np.linalg.solve(Hf, rhs)
        except np.linalg.LinAlgError:
            return None
        vlo = free & (d < lo - 1e-9)
        vhi = free & (d > hi + 1e-9)
        if not (vlo.any() or vhi.any()):
            break
        fixed |= vlo | vhi
        val[vlo], val[vhi] = np.maximum(val[vlo], lo[vlo]), np.minimum(val[vhi], hi[vhi])
        val[vlo], val[vhi] = lo[vlo], hi[vhi]
    return np.clip(d, lo, hi)


def _plan_boxes(c: dict, q: dict, mem: list[dict], cfg: dict,
                return_uncertified: bool = False):
    """Return the corridor/obstacle boxes, shrunk where there is room.

    A contradictory box is deliberately retained as ``lo > hi`` instead of being
    replaced by a guessed steer-2 point.  Such a station has no certified
    clearance witness and is reported to the caller as an uncertified recovery
    station.  ``return_uncertified=False`` preserves the two-array helper API.
    """
    n = len(q["s"])
    shrink = int(round(float(cfg["line_shrink"]) * 1000.0))
    lo = np.empty(n)
    hi = np.empty(n)
    uncertified = np.zeros(n, dtype=bool)
    for i in range(n):
        a, b = _spare_boxes(c, q, mem, i)
        if a > b:
            uncertified[i] = True
        elif b - a > 2 * shrink:
            a, b = a + shrink, b - shrink
        lo[i], hi[i] = a, b
    if return_uncertified:
        return lo, hi, uncertified
    return lo, hi


def _clamp_valid(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Clamp only nonempty intervals; contradictory boxes remain recovery values."""
    out = np.asarray(x, dtype=float).copy()
    valid = np.asarray(lo) <= np.asarray(hi)
    out[valid] = np.minimum(np.maximum(out[valid], lo[valid]), hi[valid])
    return out


def _publish_offsets(off_mm: np.ndarray, lo_mm: np.ndarray, hi_mm: np.ndarray,
                     max_offset_m: float) -> np.ndarray:
    """Fixed publish order: absolute cap first, certified interval clamp last."""
    out = np.clip(np.asarray(off_mm, dtype=float), -1000.0 * max_offset_m,
                  1000.0 * max_offset_m)
    return _clamp_valid(out, np.asarray(lo_mm, dtype=float), np.asarray(hi_mm, dtype=float)) / 1000.0


def _project_plan(off_mm: np.ndarray, lo_mm: np.ndarray, hi_mm: np.ndarray, ramp_mm: float,
                  iters: int = 2) -> np.ndarray:
    """Rate-limit a plan and re-clamp every pass into each nonempty interval.

    Contradictory intervals are retained for explicit uncertified recovery.  For
    every station whose interval is nonempty, the returned integer/mm value is
    inside that interval.
    """
    d = np.array(off_mm, float).copy()
    lo_mm, hi_mm = np.asarray(lo_mm, float), np.asarray(hi_mm, float)
    for _ in range(max(int(iters), 1)):
        for i in range(1, len(d)):                    # forward
            d[i] = min(max(d[i], d[i - 1] - ramp_mm), d[i - 1] + ramp_mm)
            if lo_mm[i] <= hi_mm[i]:
                d[i] = min(max(d[i], lo_mm[i]), hi_mm[i])
        for i in range(len(d) - 2, -1, -1):           # backward
            d[i] = min(max(d[i], d[i + 1] - ramp_mm), d[i + 1] + ramp_mm)
            if lo_mm[i] <= hi_mm[i]:
                d[i] = min(max(d[i], lo_mm[i]), hi_mm[i])
    return _clamp_valid(d, lo_mm, hi_mm)




def _racing_line(c: dict, q: dict, cl: np.ndarray, nrm: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                 prev: tuple | None, cfg: dict, s_abs: np.ndarray):
    """Minimum-curvature offsets (m) inside the certified clearance interval, or ``None``.

    Objective: the classic sum of squared second differences of the *path points*
    ``P_i = cl_i + n_i d_i`` -- i.e. ``min sum |P_{i+1} - 2 P_i + P_{i-1}|^2``, a curvature proxy
    that is quadratic in the offsets ``d``. Constraint: the certified interval of ``_plan_boxes``,
    pinned at station 0 to the lateral offset the car actually has (continuity of the published
    path). Because every station ends inside the certified interval, the plan satisfies steer-2's
    clearance predicate station by station -- the hypothesis of the certified terminal horizon
    bound -- so the longitudinal decision keeps its meaning on this path. It is also strictly
    smoother than the certified plan where that plan folds (opposing obstacles close together).
    """
    n = len(q["s"])
    if n < 6:
        return None
    lo = lo / 1000.0
    hi = hi / 1000.0
    pin = lo >= hi - 1e-9
    dprev = None
    if prev is not None:                            # previous plan, resampled at today's stations
        s_prev, d_prev = prev
        if s_prev[0] <= s_abs[0] <= s_prev[-1]:
            dprev = np.interp(s_abs, s_prev, d_prev)
    start = q["lat"] / 1000.0                       # the path starts where the car is ...
    if dprev is not None and float(cfg["line_anchor"]) > 0.5:
        start = float(dprev[0])                     # ... or where the previous plan put it
        if lo[0] <= hi[0]:
            start = min(max(start, lo[0]), hi[0])
    lo[0] = hi[0] = start
    pin = np.array(pin, bool)
    pin[0] = True
    D = np.zeros((n - 2, n))
    ix = np.arange(n - 2)
    D[ix, ix], D[ix, ix + 1], D[ix, ix + 2] = 1.0, -2.0, 1.0
    Ax, Ay = D * nrm[:, 0], D * nrm[:, 1]
    H = Ax.T @ Ax + Ay.T @ Ay
    g = Ax.T @ (D @ cl[:, 0]) + Ay.T @ (D @ cl[:, 1])
    if dprev is not None:                           # stickiness: keep the published path stable
        w = float(cfg["line_w_prev"]) * np.exp(-(s_abs - s_abs[0]) / 12.0)
        H = H + np.diag(w)
        g = g - w * dprev
    if float(cfg["line_w_center"]) > 0.0:
        H = H + float(cfg["line_w_center"]) * np.eye(n)
    d = _box_qp(H, g, lo, hi, pin, int(cfg["line_iters"]), float(cfg["line_eps"]))
    if d is None or not np.isfinite(d).all():
        return None
    return np.clip(d, lo, hi)


def _path_errors(path: np.ndarray, s: np.ndarray, px: float, py: float, pyaw: float):
    """Signed cross-track error (+ = car left of the path), heading error (+ = car turned left of
    the path) and station of the projection of the pose ``(px, py, pyaw)`` (vehicle frame) onto
    the polyline ``path`` sampled at stations ``s``."""
    c_, s_ = math.cos(pyaw), math.sin(pyaw)
    rel = path - np.array([px, py])
    p = np.column_stack((rel[:, 0] * c_ + rel[:, 1] * s_, -rel[:, 0] * s_ + rel[:, 1] * c_))
    a, d = p[:-1], np.diff(p, axis=0)
    dd = np.maximum(np.einsum("ij,ij->i", d, d), 1e-12)
    t = np.clip(-np.einsum("ij,ij->i", a, d) / dd, 0.0, 1.0)
    q = a + d * t[:, None]
    i = int(np.argmin(np.einsum("ij,ij->i", q, q)))
    dx, dy = float(d[i, 0]), float(d[i, 1])
    e_y = (dx * (-float(q[i, 1])) - dy * (-float(q[i, 0]))) / math.sqrt(float(dd[i]))
    e_psi = -math.atan2(dy, dx)
    return e_y, e_psi, float(s[i] + t[i] * (s[i + 1] - s[i]))


# --------------------------------------------------------------------------- controller
class CertifiedRacingLine(Controller):
    """``cert-2`` speed plan + ``steer-2`` lateral plan, tracked within the actuator limits."""

    def on_start(self, track: TrackInfo, vehicle: VehicleParams) -> None:
        super().on_start(track, vehicle)
        cfg = dict(DEFAULTS)
        cfg.update({k: v for k, v in self.config.items() if k in DEFAULTS})
        self.cfg = cfg
        self.lon = lon_constants(vehicle, cfg)              # cert-2 constants
        self.steer_c = lateral_constants(vehicle, cfg, self.lon)   # steer-2 constants
        self.steer_prev = 0.0
        # --- planner state -------------------------------------------------------------------
        self.mem: list[dict] = []       # remembered obstacles, course coordinates + chosen side
        self.prev_line = None           # last racing-line plan (station array, offset array)
        # --- robustness state (all inactive in nominal conditions) ----------------------------
        self.cmd_hist: list[float] = []  # steering commands we sent, newest last
        self.bias = 0.0                 # measured steering bias
        self.mu_hat = float(vehicle.mu)  # measured grip (scaled down only)
        self.lon_base = dict(self.lon)   # the declared budgets, before grip adaptation
        self._sat_n = 0
        self._obs_steer_prev = 0.0

    # ------------------------------------------------------------------ step
    def step(self, obs: Observation) -> Command:
        try:
            return self._step(obs)
        except Exception as exc:      # a thrown step is a controller error; 20 in a row is a DNF
            return Command(brake=1.0, debug={"phase": -1.0, "status": -1.0,
                                             "exc": float(abs(hash(str(exc)[:48])) % 997)})

    def _step(self, obs: Observation) -> Command:
        cfg, vp = self.cfg, self.vehicle
        road = obs.road
        q = quantise(obs)
        if road is None or not road.valid or len(road.preview_s) < 3 or q is None:
            return Command(brake=1.0, debug={"phase": -2.0, "kind": float(KIND_INVALID),
                                             "status": float(KIND_PLAN_INVALID),
                                             "cert_clear": 0.0, "uncertified_stations": 0.0,
                                             "line": 4.0})
        n = len(q["s"])
        v = max(obs.speed, 0.0)
        s = np.asarray(road.preview_s, float)
        cl = np.asarray(road.preview_centerline, float)
        kap = np.asarray(road.preview_curvature, float)

        # ---------------------------------------------------------- robustness (nominal no-ops)
        if cfg["grip_adapt"] > 0.5:
            self._adapt_grip(obs, v)

        # ---------------------------------------------------------- lateral plan
        # steer-2 is decided every step: it is the certified reference, the fallback, and the
        # safety interval of the racing line below. The racing line is used only *inside* that
        # interval, so it satisfies steer-2's clearance predicate station by station -- the
        # hypothesis of the certified terminal horizon bound -- and the longitudinal decision keeps
        # its meaning on this path. Where the interval is empty or too tight to shrink, the
        # certified offsets are kept verbatim.
        ldec = decide_lateral(self.steer_c, self.lon, q)
        off_cert = np.asarray(ldec["off"] or ([0] * n), float) / 1000.0
        try:
            _remember(self.lon, self.mem, q["obstacles"], q, qfix(obs.progress or 0.0, MM),
                      int(round(cfg["obs_pre"] * 1000.0)), int(round(cfg["obs_post"] * 1000.0)))
        except (ValueError, TypeError):
            pass                                          # a bad progress sample only skips memory
        hd = np.arctan2(np.gradient(cl[:, 1]), np.gradient(cl[:, 0]))
        nrm = np.column_stack((-np.sin(hd), np.cos(hd)))
        lim = float(cfg["max_offset"])
        lo_mm, hi_mm, uncert = _plan_boxes(self.steer_c, q, self.mem, cfg, True)
        has_empty = bool(np.any(uncert))
        # Empty boxes are not silently replaced by the raw steer-2 answer.  They
        # force an explicitly flagged recovery mode, while nonempty stations
        # still receive the same final interval clamp in every branch.
        prog = obs.progress if obs.progress is not None and math.isfinite(obs.progress) else None
        s_abs = s + prog if prog is not None else None
        line = (_racing_line(self.steer_c, q, cl, nrm, lo_mm, hi_mm,
                             self.prev_line if s_abs is not None else None, cfg, s_abs)
                if cfg["line_on"] > 0.5 and not has_empty else None)

        def publish_candidate(candidate_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            published = _publish_offsets(candidate_mm, lo_mm, hi_mm, lim)
            return published, cl + nrm * published[:, None]

        if line is not None:
            off, path = publish_candidate(line * 1000.0)
            line_used = 1.0 if _plausible(path) else 0.0
        else:
            off, path, line_used = np.zeros(n), None, 0.0
        if line_used < 1.0:
            # Rate-limit and re-project the candidate, then publish it through
            # the same cap-first/interval-last operation.
            seed = line * 1000.0 if line is not None else off_cert * 1000.0
            cand_mm = _project_plan(seed, lo_mm, hi_mm, float(cfg["line_ramp"]) * 1000.0)
            off, path = publish_candidate(cand_mm)
            if _plausible(path):
                line_used = 2.0
            else:
                # Keep steer-2's offsets for fallback 3, but clamp the
                # published result into each nonempty certified interval.
                off, path = publish_candidate(off_cert * 1000.0)
                line_used = 3.0
                if not _plausible(path):
                    # Centerline recovery is intentionally uncertified.  The
                    # final publish operation still protects every nonempty
                    # interval; contradictory boxes remain recovery values.
                    off, path = publish_candidate(np.zeros(n))
                    line_used = 4.0
        self.prev_line = (s_abs.copy(), off.copy()) if line_used <= 2.0 and s_abs is not None else None

        # ---------------------------------------------------------- curvature bound + speed plan
        ds = np.diff(s, prepend=s[0] - (s[1] - s[0]))
        ds[0] = ds[1] if n > 1 else 1.0
        # Curvature of the published path, analytically: the perceived centerline curvature plus the
        # second derivative of the offsets about it. Exact for the polyline this controller builds and
        # free of Menger estimation noise on 1 m stations. The speed cap uses the max over
        # +-kappa_window stations, so it bounds the curvature *between* stations too -- never a
        # smoothed average, which would understate a tight peak (practice_loop has kappa 0.174,
        # R ~ 5.7 m) and make the plan arrive over grip. Speed is bought with the declared budgets
        # and the path, not with a flattering curvature estimate.
        d2 = np.gradient(np.gradient(off, s, edge_order=1), s, edge_order=1)
        k_signed = kap + d2
        k_cap = np.abs(k_signed)
        for j in range(1, max(int(cfg["kappa_window"]), 0) + 1):
            k_cap = np.maximum(k_cap, np.abs(np.concatenate((np.full(j, k_cap[0]), k_cap[:-j]))))
            k_cap = np.maximum(k_cap, np.abs(np.concatenate((k_cap[j:], np.full(j, k_cap[-1])))))
        k_cap = np.maximum(k_cap * float(cfg["kappa_safety"]), float(cfg["kappa_floor"]))
        # Sidestep-feasibility gate.  For lateral displacement dy over entry
        # distance D, bang-bang motion takes 2*sqrt(dy/a_lat); requiring D/v
        # to cover it yields v² <= a_lat*D²/(4*dy).
        extra_caps = _sidestep_caps(self.lon, q, self.mem, off)
        k_micro = [int(round(x * 1e6)) for x in k_cap]
        dec = decide_long(self.lon, q, k_micro, float(cfg["cap_margin"]), extra_caps)

        # ---------------------------------------------------------- steering (tracking layer)
        # Actuator delay compensation: the next `nd` commands are already queued in the actuator, so
        # steer from the pose the car will have when *this* command starts to act (kinematic
        # roll-out with the measured yaw-response ratio). Inactive when the declared delay is 0.
        nd = max(int(getattr(vp, "actuation_delay_steps", 0)), 0)
        track = path
        if cfg["delay_comp"] > 0.5 and nd > 0 and len(self.cmd_hist) >= nd:
            px, py, pyaw = self._predict_pose(obs, v, self.cmd_hist[-nd:])
            c_, s_ = math.cos(pyaw), math.sin(pyaw)
            track = np.column_stack([(path[:, 0] - px) * c_ + (path[:, 1] - py) * s_,
                                     -(path[:, 0] - px) * s_ + (path[:, 1] - py) * c_])
        ld = float(np.clip(cfg["ld_base"] + cfg["ld_gain"] * v, cfg["ld_min"], cfg["ld_max"]))
        xs, ys = track[:, 0], track[:, 1]
        x_t, y_t = float(np.interp(ld, s, xs)), float(np.interp(ld, s, ys))
        k_ref = float(np.interp(min(ld, float(cfg["ld_ff"])), s, k_signed))  # curvature here, not far ahead
        dist = max(math.hypot(x_t, y_t), 1.0)
        delta_ff = math.atan(vp.wheelbase * k_ref)                     # blow-up of the certificate
        delta_pp = math.atan2(2.0 * vp.wheelbase * (y_t / dist), dist)  # pure pursuit on the plan
        j = int(np.clip(np.searchsorted(s, ld), 1, n - 1))
        psi = math.atan2(ys[j] - ys[j - 1], xs[j] - xs[j - 1])          # path tangent at look-ahead
        t_ff = cfg["k_ff"] * delta_ff
        t_pp = cfg["k_pp"] * delta_pp
        t_psi = cfg["k_psi"] * psi
        # Pure body-rate damping. The earlier form, -k_r*(yaw_rate - v*k_ref), also *drove* the car
        # toward a reference yaw rate, i.e. it re-commanded the curvature the feed-forward already
        # provides; measured at an obstacle sidestep the three curvature terms (t_ff + t_psi +
        # t_damp) summed to +0.219 and cancelled the +-0.206 cross-track correction, so the car sat
        # 0.84 m off its plan with ~0 steer. The reference rate belongs to the feed-forward.
        t_damp = -cfg["k_r"] * obs.yaw_rate
        # The published interval is certified, but the bicycle tracker can
        # arrive at a hold window on the wrong side of it.  While a remembered
        # obstacle is in/near hold, feed back the signed plan error so the
        # tracker turns toward the published offset instead of following road
        # curvature through the obstacle.
        obs_track = 0.0
        if any(-self.lon["hold"] - 1000 <= o["d"] <= self.lon["hold"] + 1000 for o in self.mem):
            obs_track = cfg["k_obs_track"] * (float(off[0]) - float(road.lateral_offset))
        delta = t_ff + t_pp + t_psi + t_damp + obs_track
        e_y = e_psi = 0.0
        if cfg["trk_mode"] > 0.5:
            # Path-error state feedback. The errors are taken at the pose the car will have once
            # this command acts (queued commands, then the steering lag at the current yaw rate);
            # the feed-forward is the published path's own curvature at that projection, and the
            # feedback places the linearised error dynamics e'' = (v^2/L) u at (omega, zeta).
            px = py = pyaw = 0.0
            if cfg["delay_comp"] > 0.5 and nd > 0 and len(self.cmd_hist) >= nd:
                px, py, pyaw = self._predict_pose(obs, v, self.cmd_hist[-nd:])
            lag = float(cfg["trk_lag"])
            r = float(obs.yaw_rate)
            px += v * lag * math.cos(pyaw + 0.5 * r * lag)
            py += v * lag * math.sin(pyaw + 0.5 * r * lag)
            pyaw += r * lag
            e_y, e_psi, s_p = _path_errors(path, s, px, py, pyaw)
            k_ref = float(np.interp(s_p + v * float(cfg["trk_lead"]), s, k_signed))
            v_e = max(v, float(cfg["trk_vmin"]))
            w_n, zeta = float(cfg["trk_omega"]), float(cfg["trk_zeta"])
            u = -(vp.wheelbase / (v_e * v_e)) * (w_n * w_n * e_y + 2.0 * zeta * w_n * v_e * e_psi)
            delta = math.atan(vp.wheelbase * k_ref + u)
        # Recovery guard. If the car is outside the cone corridor or badly misaligned, the plan is
        # not the priority: track the centerline directly (Stanley-style) and shed speed. This is
        # what stops a wide corner exit from becoming a 'lost' DNF. lateral_offset is + left of the
        # centerline and heading_error is + left of the road heading, so both need right steer.
        level = 0.0
        if road.lateral_offset > cfg["recover_lat"] * max(road.width_left, 1e-6) or \
                -road.lateral_offset > cfg["recover_lat"] * max(road.width_right, 1e-6) or \
                abs(road.heading_error) > cfg["recover_head"]:
            level = 1.0
            delta = float(np.clip(-(cfg["k_rec_head"] * road.heading_error
                                    + cfg["k_rec_lat"] * road.lateral_offset), -0.5, 0.5))
        v_eff = max(v, 4.0)
        d_max = math.atan(cfg["grip_clamp"] * self.mu_hat * G * vp.wheelbase / (v_eff * v_eff))
        d_pre = delta
        delta = float(np.clip(delta, -d_max, d_max))
        d_clip = delta
        steer = float(np.clip(delta / vp.max_steer, -1.0, 1.0))
        # Steering bias: the applied steering should equal what we sent `nd + 1` steps ago. Only
        # estimated while the actuator has settled (consecutive commands nearly equal) and the wheel
        # is not saturated; nominal bias is 0, so this is a no-op on an unbiased car.
        if cfg["bias_est"] > 0.5 and len(self.cmd_hist) >= nd + 2:
            sent, before = self.cmd_hist[-1 - nd], self.cmd_hist[-2 - nd]
            if abs(obs.steer) < 0.95 and abs(sent - before) < 0.02:
                self.bias += float(cfg["bias_gain"]) * ((obs.steer - sent) - self.bias)
                bl = float(cfg["bias_lim"])
                self.bias = float(np.clip(self.bias, -bl, bl))
        steer = float(np.clip(steer - self.bias, -1.0, 1.0))
        rate = vp.max_steer_rate * vp.control_dt
        steer = float(np.clip(steer, self.steer_prev - rate, self.steer_prev + rate))
        self.steer_prev = steer
        self.cmd_hist.append(steer)
        if len(self.cmd_hist) > 24:
            self.cmd_hist.pop(0)

        # ---------------------------------------------------------- pedals (tracking layer)
        if dec["kind"] == KIND_GREATEST_SAFE:
            v_tgt = math.sqrt(max(dec["b1"], 0)) / 1000.0
            a_des = float(np.clip(dec["a"] / 1000.0 + cfg["k_v"] * (v_tgt - v),
                                  -vp.max_decel, vp.max_accel))
            throttle, brake = self._pedals(obs, a_des)
            if level > 0.5:                                  # recovering: hold a steerable speed
                throttle = cfg["recover_throttle"] if v < cfg["recover_vmin"] else 0.0
                brake = cfg["recover_brake"]
        else:
            # over budget (or no feasible gap / invalid input): recover toward the feasible set
            # at the plan's own braking budget rather than the full pedal, so the tyres keep the
            # lateral capability the recovery steer needs in the same step.
            v_tgt = math.sqrt(max(dec["B0"], 0.0)) / 1000.0 if dec["B0"] is not None else 0.0
            a_des = -self.lon["a_brk"] / 1000.0
            throttle = 0.0
            brake = float(np.clip((self.lon["a_brk"] / 1000.0 - cfg["brake_deadband"]) / vp.max_decel, 0.0, 1.0))
            if dec["kind"] == KIND_INVALID:
                throttle, brake = 0.0, 1.0

        dbg = {"phase": 1.0, "kind": float(dec["kind"]), "status": float(ldec["kind"]),
               "a_cert": dec["a"] / 1000.0, "a_des": a_des, "v_tgt": v_tgt, "ld": ld,
               "n_obs": float(len(q["obstacles"])), "off_lo": float(np.min(off)),
               "off_hi": float(np.max(off)), "k_max": float(np.max(k_cap)), "v": v,
               "steer_ff": delta_ff, "steer_pp": delta_pp, "off0": float(off[0]),
               "lat": float(road.lateral_offset), "head": float(road.heading_error),
               "ey": float(off[0]) - float(road.lateral_offset),
               "plan_at3": float(np.interp(min(3.0, s[-1]), s, off)),
               "plan_at10": float(np.interp(min(10.0, s[-1]), s, off)),
               "obst_d": float(min([o.distance for o in road.obstacles], default=-1.0)),
               "t_ff": t_ff, "t_pp": t_pp, "t_psi": t_psi, "t_damp": t_damp, "obs_track": obs_track,
               "d_pre": d_pre, "d_max": d_max, "yt": y_t, "kref": k_ref, "psi": psi,
               "side": float(ldec["side"][0]) if ldec["side"] else 0.0,
               "line": line_used, "cert_clear": float(line_used in (1.0, 2.0, 3.0) and not has_empty),
               "uncertified_stations": float(np.count_nonzero(uncert)),
               "n_mem": float(len(self.mem)), "mu_hat": float(self.mu_hat),
               "bias": float(self.bias), "e_y": e_y, "e_psi": e_psi}
        return Command(steer=steer, throttle=throttle, brake=brake, path=path, path_frame="vehicle",
                       target_speed=v_tgt, debug=dbg)

    # ------------------------------------------------------------------ helpers
    def _pedals(self, obs: Observation, a_des: float) -> tuple[float, float]:
        """Demanded acceleration -> non-overlapping pedals, grip left for cornering.

        Measured side-slip (``obs.vy``/``obs.vx``) eases the pedals: whatever the model believes, a
        car that is actually sliding sideways needs its friction for the tyres, not the throttle.
        """
        vp, cfg = self.vehicle, self.cfg
        vy, vx = float(getattr(obs, "vy", 0.0) or 0.0), float(getattr(obs, "vx", 0.0) or 0.0)
        ease = cfg["beta_ease"] > 0.5 and abs(math.atan2(vy, max(vx, 1.0))) > float(cfg["beta_ease_lim"])
        if a_des >= 0.0:
            lat_use = min(float(cfg["thr_reserve"]) * abs(obs.ay) / max(self.mu_hat * G, 1e-6), 1.0)
            cap = max(1.0 - lat_use ** 2, 0.15)           # no big throttle mid-corner
            thr = float(min(a_des / vp.max_accel, cap, 1.0))
            return (0.4 * thr if ease else thr), 0.0
        brk = float(np.clip((-a_des - cfg["brake_deadband"]) / vp.max_decel, 0.0, 1.0))
        return 0.0, (0.4 * brk if ease else brk)

    def _adapt_grip(self, obs: Observation, v: float) -> None:
        """Scale the grip budgets down to the measured grip when the yaw response proves saturation.

        The certified budgets are *declared*, not measured: if the tyres cannot deliver them (low
        grip, or CARLA's different dynamics), a plan that is feasible in the model is not feasible
        in fact. Saturation = a steady yaw rate well below the kinematic one while cornering hard;
        the budgets then scale down and creep back up only slowly. Nominal no-op.
        """
        vp, cfg = self.vehicle, self.cfg
        if vp.wheelbase <= 0:
            return
        k_kin = v * math.tan(obs.steer * vp.max_steer) / vp.wheelbase
        steady = abs(obs.steer - self._obs_steer_prev) < 0.02
        self._obs_steer_prev = obs.steer
        sat = (v > 5.0 and abs(obs.ay) > float(cfg["sat_ay"]) and abs(k_kin) > 0.1 and steady
               and obs.yaw_rate * k_kin > 0 and obs.yaw_rate / k_kin < float(cfg["sat_ratio"]))
        self._sat_n = self._sat_n + 1 if sat else 0
        if self._sat_n >= 3:
            self.mu_hat = max(min(self.mu_hat, 1.02 * math.hypot(obs.ax, obs.ay) / G), 0.4 * vp.mu)
        else:
            self.mu_hat = min(float(vp.mu), self.mu_hat + 0.003 * vp.control_dt)
        sc = self.mu_hat / max(float(vp.mu), 1e-9)
        for k in ("a_lat", "a_brk", "a_acc"):
            self.lon[k] = int(round(self.lon_base[k] * sc))

    def _predict_pose(self, obs: Observation, v: float, queued: list[float]):
        """Kinematic roll-out of the commands already queued in the actuator (body frame).

        The yaw response is scaled by the observed yaw rate over the kinematic one, which carries
        the understeer the bicycle model does not have.
        """
        vp = self.vehicle
        dt = vp.control_dt
        rate = vp.max_steer_rate * dt
        x = y = yaw = 0.0
        st = float(obs.steer)
        k0 = v * math.tan(st * vp.max_steer) / vp.wheelbase
        ratio = float(np.clip(obs.yaw_rate / k0, 0.5, 1.2)) if abs(k0) > 0.05 else 1.0
        for cmd in queued:
            st = st + float(np.clip(cmd + self.bias - st, -rate, rate))
            r = ratio * v * math.tan(st * vp.max_steer) / vp.wheelbase
            yaw_mid = yaw + 0.5 * r * dt
            x += v * dt * math.cos(yaw_mid)
            y += v * dt * math.sin(yaw_mid)
            yaw += r * dt
        return x, y, yaw
