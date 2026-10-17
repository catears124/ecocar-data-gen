"""Deterministic, non-learned expert that produces imitation labels for ``brake_turn_obstacle``.

Runs as an ordinary harness controller (sandboxed process, ``track_knowledge: perception``), so
it receives exactly the ``Observation`` a team controller receives. After the hand-over point
its command is a pure function of that observation and the published ``VehicleParams``:

    command = expert_action(obs, vehicle, config=None)

It keeps no state between steps, so every label can be recomputed offline from the recorded
observation (``data-validate`` re-runs it on the canonical policy observation of every expert
sample, with every privileged field removed). It does not read ``obs.x/y/yaw`` (world pose),
``obs.progress``, ``obs.cones``, ``road.corners`` or ``road.finish_distance`` after hand-over.

Longitudinal decision (CERTIFIED, see ``cert.py`` and ``lean/Corpus/Cert.lean``): the greatest
integer acceleration (mm/s^2) whose end point over the first preview segment still admits a
feasible speed profile under model M and its declared terminal constraint (or full braking when
the car is already over budget). The Lean checker recomputes this decision from the recorded
observation for every expert sample and requires it for every clean sample.

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
from autocross.datagen import cert, steer
from autocross.datagen.spec import get_family

EXPERT_NAME = "bto_expert"
EXPERT_VERSION = "3.1.0"

G = cert.G
CORRIDOR_MARGIN = 0.30   # m, (uncertified) lateral planner keeps the car body this far inside the boundaries
RAMP_IN, RAMP_OUT, HOLD = 18.0, 15.0, 3.0   # m, sidestep ramps / full-shift half-length of the lateral planner
R_SW = 15                # m, = min(RAMP_IN, RAMP_OUT): bounds sidestep curvature by 6|y|/R_SW^2 in model M
BRAKE_DEADBAND = 0.2     # m/s^2 of demanded deceleration left to drag before braking (pedal mapping)

# Declared scenario domain (the only knowledge about the road beyond the preview): tightest turn
# radius and widest road of the family spec; the largest sidestep offset is half the widest road.
_FAMILY = get_family("brake_turn_obstacle")
DOMAIN_R_MIN_M = min(b.lo for b in _FAMILY.factor("turn_radius_m").bins)
DOMAIN_Y_MAX_M = max(b.hi for b in _FAMILY.factor("road_width_m").bins) / 2.0
HORIZON_RADIUS_MM = cert.domain_horizon_mm(DOMAIN_R_MIN_M, DOMAIN_Y_MAX_M, R_SW)

PHASE_APPROACH, PHASE_EXPERT = 0, 1
STATUS_OK, STATUS_APPROACH, STATUS_NO_GAP, STATUS_ROAD_INVALID, STATUS_OVER_BUDGET = 0, 1, 2, 3, 4
STATUS_NAMES = {0: "ok", 1: "approach", 2: "no_feasible_gap", 3: "road_invalid", 4: "over_budget"}
LD_BASE, LD_GAIN, LD_MIN, LD_MAX = 0.0, 0.45, 3.0, 14.0  # pure-pursuit look-ahead [m], [s]
# Scripted approach: near the target speed an incremental PI on the applied throttle (obs.throttle,
# damped by obs.ax) removes the steady-state offset a P law leaves against drag (CARLA).
APPROACH_PI_BAND = (-0.5, 3.0)   # m/s of speed error where the PI law is used
APPROACH_KI, APPROACH_KD = 0.05, 0.03   # throttle per (m/s) and per (m/s^2), per control step
PARAMS = {k: v for k, v in dict(globals()).items() if k.isupper() and isinstance(v, (int, float))
          and not k.startswith(("PHASE_", "STATUS_"))}


def model_constants(vp: VehicleParams) -> dict:
    """Combined cert-2 and steer-1 constants used by the expert."""
    return steer.model_constants(vp)


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
    err = target_speed - v
    a_des = float(np.clip(1.5 * err, -4.0, vp.max_accel))
    if APPROACH_PI_BAND[0] <= err <= APPROACH_PI_BAND[1] and obs.brake <= 0.0:
        throttle = float(np.clip(obs.throttle + APPROACH_KI * err - APPROACH_KD * obs.ax, 0.0, 1.0))
        brake = 0.0
    else:
        throttle, brake = _pedals(a_des, vp, obs.ay)
    return Command(steer=steer, throttle=throttle, brake=brake, path=road.preview_centerline, path_frame="vehicle",
                   target_speed=target_speed,
                   debug={"phase": PHASE_APPROACH, "status": STATUS_APPROACH, "a_des": a_des, "side": 0.0,
                          "plan_offset_ld": 0.0, "lookahead_m": ld, "n_obstacles": float(len(road.obstacles))})


def _steer_domain_q(q: dict | None) -> dict | None:
    """Map the observed lateral offset into steer-1's declared non-negative domain."""
    if q is None:
        return None
    return {**q, "lat": abs(int(q["lat"]))}


def lateral_plan(road, vp: VehicleParams, q: dict, c: dict, dec: dict | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Build the vehicle-frame path from the certified integer steer-1 offsets."""
    del vp
    if dec is None:
        dec = steer.decide(c, q)
    s = np.asarray(road.preview_s, float)
    cl = np.asarray(road.preview_centerline, float)
    offset = np.asarray(dec["off"], dtype=float) / 1000.0
    hd = np.arctan2(np.gradient(cl[:, 1]), np.gradient(cl[:, 0]))
    path = cl + np.column_stack((-np.sin(hd), np.cos(hd))) * offset[:, None]
    side = float(dec["side"][0]) if dec["side"] else 0.0
    return path, offset, side


def _invalid(dec: dict) -> Command:
    """Road model unusable (or a non-finite input): full brake; the (uncertified) decision is
    still recorded so the certificate fields can be recomputed by the checkers."""
    dbg = {"phase": PHASE_EXPERT, "status": STATUS_ROAD_INVALID, "cert_kind": dec["kind"], "cert_a": dec["a"]}
    for k in ("b1", "b0", "B0"):
        if dec[k] is not None:
            dbg[f"cert_{k}"] = dec[k]
    return Command(brake=1.0, debug=dbg)


def expert_action(obs: Observation, vp: VehicleParams, config: dict | None = None,
                  constants: dict | None = None) -> Command:
    """The expert's command for one observation. Pure function: no hidden state."""
    road = obs.road
    c = model_constants(vp) if constants is None else constants
    base = cert.quantise(obs)
    q = steer.quantise_from_base(base, None if road is None else road.lateral_offset)
    lonq = None if q is None else q["base"]
    if road is None or not road.valid or len(road.preview_s) < 3:
        return _invalid(cert.decide(c["lon"], lonq))
    appr = (config or {}).get("approach")
    if appr and obs.progress < float(appr["handover_progress"]):
        return _approach(obs, vp, float(appr["speed"]))
    v = max(obs.speed, 0.0)
    dec = cert.decide(c["lon"], lonq)
    if q is None:
        return _invalid(dec)
    sq = _steer_domain_q(q)
    ldec = steer.decide(c, sq)
    path, offset, side = lateral_plan(road, vp, sq, c, ldec)
    s = np.asarray(road.preview_s, float)
    steer_cmd, ld = _pursuit(path, s, v, vp)
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
           "n_obstacles": float(len(road.obstacles)), "cert_kind": dec["kind"], "cert_a": dec["a"],
           "lateral_kind": ldec["kind"], "lateral_side": side}
    for k in ("b1", "b0", "B0"):
        if dec[k] is not None:
            dbg[f"cert_{k}"] = dec[k]
    return Command(steer=steer_cmd, throttle=throttle, brake=brake, path=path, path_frame="vehicle", target_speed=v_tgt,
                   debug=dbg)


class BrakeTurnObstacleExpert(Controller):
    """Harness wrapper; the controller config may only carry the scripted ``approach``."""

    def on_start(self, track: TrackInfo, vehicle: VehicleParams) -> None:
        super().on_start(track, vehicle)
        extra = set(self.config) - {"approach"}
        if extra:
            raise ValueError(f"expert config accepts only 'approach', got {sorted(extra)}")
        self._constants = model_constants(vehicle)
        warm = {"speed": 0.0, "road": {"valid": True, "lateral_offset": 0.0,
                "preview_s": [0.0, 1.0], "preview_curvature": [0.0, 0.0],
                "preview_width_left": [3.0, 3.0], "preview_width_right": [3.0, 3.0],
                "obstacles": []}}
        warm_base = cert.quantise_dict(warm)
        warm_q = steer.quantise_from_base(warm_base, 0.0)
        cert.decide(self._constants["lon"], warm_base)
        steer.decide(self._constants, warm_q)

    def step(self, obs: Observation) -> Command:
        return expert_action(obs, self.vehicle, self.config, self._constants)
