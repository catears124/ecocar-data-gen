"""Build the actuator model from a policy observation and map its first action."""
from __future__ import annotations

import math
from typing import Iterable

from autocross.api import Command, Observation, VehicleParams
from autocross.datagen import cert, steer
from autocross.datagen.expert import _pedals

from .model import (DEFAULT_ACC_INPUTS, DEFAULT_DELTAS, DEFAULT_HORIZON,
                    DEFAULT_OFFSETS, DEFAULT_PSIS, DEFAULT_STEER_INPUTS,
                    DEFAULT_TERMINAL_DISTANCE_MM, ActModel, ActState, snap)


def constants(vp: VehicleParams, *, steer_tau_s: float = 0.06) -> dict:
    c = steer.model_constants(vp)
    c = dict(c)
    c["max_steer"] = int(round(vp.max_steer * 1_000_000))
    c["steer_rate_urad_s"] = int(round(vp.max_steer_rate * vp.max_steer * 1_000_000))
    c["steer_tau_us"] = int(round(steer_tau_s * 1_000_000))
    c["wheelbase_mm"] = int(round(vp.wheelbase * 1000.0))
    c["steer_tau_s"] = float(steer_tau_s)
    return c


def _grid(values: Iterable[int] | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if values is None:
        return default
    out = tuple(sorted({int(v) for v in values}))
    if len(out) < 2:
        raise ValueError("grid needs at least two values")
    return out


def model_from_observation(obs: Observation, vp: VehicleParams, *,
                           horizon: int = DEFAULT_HORIZON,
                           offsets: Iterable[int] | None = None,
                           psis: Iterable[int] | None = None,
                           speeds: Iterable[int] | None = None,
                           deltas: Iterable[int] | None = None,
                           steer_inputs: Iterable[int] | None = None,
                           acc_inputs: Iterable[int] | None = None,
                           terminal_distance: int = DEFAULT_TERMINAL_DISTANCE_MM,
                           steer_tau_s: float = 0.06,
                           ref_d: Iterable[int] | None = None,
                           ref_psi: Iterable[int] | None = None) -> ActModel | None:
    road = getattr(obs, "road", None)
    if road is None or not getattr(road, "valid", False):
        return None
    base = cert.quantise(obs)
    if base is None:
        return None
    c = constants(vp, steer_tau_s=steer_tau_s)
    if not steer.well_formed(c, {"base": base, "lat": 0}):
        return None
    n = len(base["s"])
    h = min(int(horizon), n - 1)
    if h < 1:
        return None
    offs = _grid(offsets, DEFAULT_OFFSETS)
    ps = _grid(psis, DEFAULT_PSIS)
    sp = _grid(speeds, tuple((v * 1000) ** 2 for v in range(3, 16)))
    dl = _grid(deltas, DEFAULT_DELTAS)
    ui = _grid(steer_inputs, DEFAULT_STEER_INPUTS)
    ai = _grid(acc_inputs, DEFAULT_ACC_INPUTS)
    try:
        d0 = cert.qfix(road.lateral_offset, 1000)
        psi0 = cert.qfix(road.heading_error, 1_000_000)
    except (TypeError, ValueError, AttributeError):
        return None
    d0 = snap(d0, offs)
    psi0 = snap(psi0, ps)
    b0 = int(base["v0"]) ** 2
    # Road-wheel angle estimate for the initial state.  At speed the kinematic
    # yaw-rate relation is used; at rest the last normalised steering command.
    speed = max(float(getattr(obs, "speed", 0.0)), 0.0)
    try:
        if speed > 1.0:
            drad = math.atan(vp.wheelbase * float(getattr(obs, "yaw_rate", 0.0)) / speed)
        else:
            drad = float(getattr(obs, "steer", 0.0)) * vp.max_steer
    except (TypeError, ValueError):
        drad = 0.0
    delta0 = snap(int(round(max(-vp.max_steer, min(vp.max_steer, drad)) * 1_000_000)), dl)
    if b0 < min(sp):
        # A stopped car is represented on the lowest declared speed bucket;
        # the launch dynamic is outside this model (documented).
        b0 = min(sp)
    if b0 > max(sp):
        b0 = max(sp)
    return ActModel(consts=c, q={"base": base, "lat": abs(d0)}, offsets=offs,
                    psis=ps, speeds=sp, deltas=dl, steer_inputs=ui, acc_inputs=ai,
                    horizon=h, start=ActState(0, d0, psi0, b0, delta0),
                    terminal_distance=int(terminal_distance),
                    ref_d=tuple(int(x) for x in (ref_d or ())),
                    ref_psi=tuple(int(x) for x in (ref_psi or ())))


def command_from_first(obs: Observation, vp: VehicleParams, model: ActModel,
                       u: int, a: int) -> Command:
    """Publish the certified first actuator request.

    ``u`` is the normalised steering command; ``a`` is requested acceleration in
    mm/s^2.  The pedal map is the same deterministic map used by the expert and
    is the only actuation layer after the certified action.
    """
    steer_cmd = max(-1.0, min(1.0, u / 1000.0))
    a_mps2 = a / 1000.0
    throttle, brake = _pedals(a_mps2, vp, float(getattr(obs, "ay", 0.0)))
    return Command(steer=steer_cmd, throttle=throttle, brake=brake,
                   target_speed=math.sqrt(max(model.start.b, 0)) / 1000.0,
                   debug={"planner": "actuator-exact", "certified": True,
                          "u": float(u), "a_mm_s2": float(a),
                          "horizon": float(model.horizon),
                          "grid_offsets": float(len(model.offsets)),
                          "grid_psis": float(len(model.psis)),
                          "grid_speeds": float(len(model.speeds)),
                          "grid_deltas": float(len(model.deltas)),
                          "steer_tau_s": float(model.consts.get("steer_tau_s", 0.0))})


__all__ = ["constants", "model_from_observation", "command_from_first"]
