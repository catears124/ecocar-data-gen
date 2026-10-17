"""Deterministic, non-learned expert that produces imitation labels for ``brake_turn_obstacle``.

Runs as an ordinary harness controller (sandboxed process, ``track_knowledge: perception``), so
it receives exactly the ``Observation`` a team controller receives. After the hand-over point
its command is a pure function of that observation and the published ``VehicleParams``:

    command = expert_action(obs, vehicle, config=None)

It keeps no state between steps, so every label can be recomputed offline from the recorded
observation (``data-validate`` does this for every decision-window sample). It does not read
``obs.x/y/yaw`` (world pose), ``obs.progress`` or ``obs.cones``.

Longitudinal decision (CERTIFIED, see ``cert.py`` and ``lean/Corpus/Cert.lean``): the
acceleration toward the greatest speed profile that stays feasible under model M for every
continuation beyond the preview (or full braking when the car is already over budget). The
Lean checker recomputes this decision from the recorded observation for every clean sample.

Lateral plan and steering (NOT certified): the perceived centerline shifted into the centre of
the wider free gap beside each visible obstacle (smoothstep ramps, clamped inside the
corridor), tracked by pure pursuit. Pedals map the certified acceleration linearly onto
throttle/brake; that mapping is a tracking layer, not part of the certificate.

Before the hand-over point (``config["approach"]``) a scripted speed hold brings the car to the
scenario's declared approach speed. Those steps use privileged scenario information and are
recorded as phase ``approach``, which is never a clean demonstration."""
from __future__ import annotations

import math

import numpy as np

from autocross.api import Command, Controller, Observation, TrackInfo, VehicleParams
from autocross.datagen import cert

EXPERT_NAME = "bto_expert"
EXPERT_VERSION = "2.0.0"

G = 9.81
A_LAT_FRAC = 0.70        # grip budget of model M: lateral accel as a fraction of mu * g
A_BRAKE_PLAN = 6.5       # m/s^2, braking budget of model M (max_decel is 9)
V_MAX = 25.0             # m/s
HORIZON_RADIUS = 9.0     # m, tightest corner assumed possible just beyond the preview
GAP_MARGIN = 0.25        # m, free space required each side of the car for a gap to count
CORRIDOR_MARGIN = 0.30   # m, planned path keeps the car body this far inside the boundaries
RAMP_IN, RAMP_OUT, HOLD = 18.0, 15.0, 3.0   # m, sidestep ramps / full-shift half-length
R_SW = 15                # m, shortest ramp: bounds sidestep curvature by 6|y|/R_SW^2 in model M
GAP_SPEED_BASE, GAP_SPEED_PER_M = 5.0, 6.0  # m/s, + per metre of slack beside an obstacle
LD_BASE, LD_GAIN, LD_MIN, LD_MAX = 1.0, 0.45, 4.0, 14.0  # pure-pursuit look-ahead [m], [s]
BRAKE_DEADBAND = 0.2     # m/s^2 of demanded deceleration left to drag before braking

PHASE_APPROACH, PHASE_EXPERT = 0, 1
STATUS_OK, STATUS_APPROACH, STATUS_NO_GAP, STATUS_ROAD_INVALID, STATUS_OVER_BUDGET = 0, 1, 2, 3, 4
STATUS_NAMES = {0: "ok", 1: "approach", 2: "no_feasible_gap", 3: "road_invalid", 4: "over_budget"}
LD_BASE, LD_GAIN, LD_MIN, LD_MAX = 0.0, 0.45, 3.0, 14.0  # pure-pursuit look-ahead [m], [s]
PARAMS = {k: v for k, v in dict(globals()).items() if k.isupper() and isinstance(v, (int, float))
          and not k.startswith(("PHASE_", "STATUS_"))}


def model_constants(vp: VehicleParams) -> dict:
    return cert.model_constants(vp, A_LAT_FRAC, A_BRAKE_PLAN, V_MAX, HORIZON_RADIUS, HOLD, RAMP_IN, RAMP_OUT, R_SW,
                                GAP_SPEED_BASE, GAP_SPEED_PER_M, GAP_MARGIN)


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _pursuit(path_xy: np.ndarray, s: np.ndarray, v: float, vp: VehicleParams) -> tuple[float, float]:
    ld = float(np.clip(LD_BASE + LD_GAIN * v, LD_MIN, LD_MAX))
    tx, ty = float(np.interp(ld, s, path_xy[:, 0])), float(np.interp(ld, s, path_xy[:, 1]))
    alpha = math.atan2(ty, max(tx, 0.1))
    delta = math.atan2(2.0 * vp.wheelbase * math.sin(alpha), max(math.hypot(tx, ty), 1.0))
    return float(np.clip(delta / vp.max_steer, -1.0, 1.0)), ld


def _pedals(a_des: float, vp: VehicleParams, ay: float) -> tuple[float, float]:
    if a_des >= 0.0:
        lat_use = min(abs(ay) / (vp.mu * G), 1.0)   # leave grip for cornering
        return float(min(a_des / vp.max_accel, max(1.0 - lat_use ** 2, 0.15), 1.0)), 0.0
    return 0.0, float(np.clip((-a_des - BRAKE_DEADBAND) / vp.max_decel, 0.0, 1.0))


def _approach(obs: Observation, vp: VehicleParams, target_speed: float) -> Command:
    road = obs.road
    v = max(obs.speed, 0.0)
    steer, ld = _pursuit(np.asarray(road.preview_centerline, float), np.asarray(road.preview_s, float), v, vp)
    a_des = float(np.clip(1.5 * (target_speed - v), -4.0, vp.max_accel))
    throttle, brake = _pedals(a_des, vp, obs.ay)
    return Command(steer=steer, throttle=throttle, brake=brake, path=road.preview_centerline, path_frame="vehicle",
                   target_speed=target_speed,
                   debug={"phase": PHASE_APPROACH, "status": STATUS_APPROACH, "a_des": a_des, "side": 0.0,
                          "plan_offset_ld": 0.0, "lookahead_m": ld, "n_obstacles": float(len(road.obstacles))})


def lateral_plan(road, vp: VehicleParams, q: dict, c: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Planned path (vehicle frame), its offset from the centerline, and the nearest obstacle's
    chosen side. Side and target use the same station widths as model M."""
    s = np.asarray(road.preview_s, float)
    cl = np.asarray(road.preview_centerline, float)
    wl, wr = np.asarray(road.preview_width_left, float), np.asarray(road.preview_width_right, float)
    offset, weight = np.zeros_like(s), np.zeros_like(s)
    side_sel, d_sel = 0.0, math.inf
    for ob, obq in zip(road.obstacles, q["obstacles"]):
        side, _, _ = cert.obstacle_plan(c, q, obq)
        i = cert._nearest_index(q["s"], obq[0])
        if side == 0:  # equal gaps: keep to the side the car is already on (left on an exact tie)
            side = 1 if road.lateral_offset >= 0.0 else -1
        target = 0.5 * (ob.lateral_max + wl[i]) if side > 0 else 0.5 * (ob.lateral_min - wr[i])
        d0 = ob.distance
        w = np.where(s < d0, _smoothstep((s - (d0 - HOLD - RAMP_IN)) / RAMP_IN),
                     1.0 - _smoothstep((s - (d0 + HOLD)) / RAMP_OUT))
        offset = np.where(w > weight, target * w, offset)
        weight = np.maximum(weight, w)
        if d0 < d_sel:
            side_sel, d_sel = float(side), d0
    hw = 0.5 * vp.width
    lo, hi = -wr + hw + CORRIDOR_MARGIN, wl - hw - CORRIDOR_MARGIN
    offset = np.where(lo <= hi, np.clip(offset, lo, hi), 0.5 * (lo + hi))
    hd = np.arctan2(np.gradient(cl[:, 1]), np.gradient(cl[:, 0]))
    path = cl + np.column_stack((-np.sin(hd), np.cos(hd))) * offset[:, None]
    return path, offset, side_sel


def expert_action(obs: Observation, vp: VehicleParams, config: dict | None = None) -> Command:
    """The expert's command for one observation. Pure function: no hidden state."""
    road = obs.road
    if road is None or not road.valid or len(road.preview_s) < 3:
        return Command(brake=1.0, debug={"phase": PHASE_EXPERT, "status": STATUS_ROAD_INVALID,
                                         "cert_kind": cert.KIND_INVALID})
    appr = (config or {}).get("approach")
    if appr and obs.progress < float(appr["handover_progress"]):
        return _approach(obs, vp, float(appr["speed"]))

    v = max(obs.speed, 0.0)
    c = model_constants(vp)
    q = cert.quantise(obs)
    dec = cert.decide(c, q)
    path, offset, side = lateral_plan(road, vp, q, c)
    s = np.asarray(road.preview_s, float)
    steer, ld = _pursuit(path, s, v, vp)
    a_des = dec["a"] / 1000.0
    status = {cert.KIND_GREATEST_SAFE: STATUS_OK, cert.KIND_MIN_VIOLATION: STATUS_OVER_BUDGET,
              cert.KIND_NO_GAP: STATUS_NO_GAP}.get(dec["kind"], STATUS_ROAD_INVALID)
    if dec["kind"] == cert.KIND_GREATEST_SAFE:
        throttle, brake = _pedals(a_des, vp, obs.ay)
        v_tgt = math.sqrt(dec["b1"]) / 1000.0
    else:
        throttle, brake, v_tgt = 0.0, 1.0, 0.0
    dbg = {"phase": PHASE_EXPERT, "status": status, "a_des": a_des, "side": side,
           "plan_offset_ld": float(np.interp(ld, s, offset)), "lookahead_m": ld,
           "n_obstacles": float(len(road.obstacles)), "cert_kind": dec["kind"], "cert_a": dec["a"]}
    for k in ("b1", "b0", "B0"):
        if dec[k] is not None:
            dbg[f"cert_{k}"] = dec[k]
    return Command(steer=steer, throttle=throttle, brake=brake, path=path, path_frame="vehicle", target_speed=v_tgt,
                   debug=dbg)


class BrakeTurnObstacleExpert(Controller):
    """Harness wrapper; the controller config may only carry the scripted ``approach``."""

    def on_start(self, track: TrackInfo, vehicle: VehicleParams) -> None:
        super().on_start(track, vehicle)
        extra = set(self.config) - {"approach"}
        if extra:
            raise ValueError(f"expert config accepts only 'approach', got {sorted(extra)}")

    def step(self, obs: Observation) -> Command:
        return expert_action(obs, self.vehicle, self.config)
