"""Controller-facing helpers for the exact joint teacher.

The finite model and dynamic program deliberately live in :mod:`model` and
:mod:`solver`.  This module is only the (uncertified) tracking layer: it builds
one declared model from a policy observation, runs the exact DP, publishes the
offset plan in the vehicle frame, and maps the first transition to pedals.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from autocross.api import Command, Observation, VehicleParams
from autocross.datagen.expert import _pedals, _pursuit
from autocross.optimal.certificate import (
    certificate_from_solution,
    check_certificate,
    write_certificate,
)
from autocross.optimal.model import (
    DEFAULT_OFFSETS,
    Model,
    model_from_observation,
)
from autocross.optimal.solver import Solution, solve


@dataclass(frozen=True)
class TeacherResult:
    """The result of one planning attempt, including its proof status."""

    model: Model | None
    solution: Solution | None
    certificate: dict | None
    certificate_ok: bool
    solver_ms: float
    realtime_exact: bool
    fallback_reason: str | None = None
    certificate_written: bool = False

    @property
    def claimed(self) -> int | None:
        return None if self.solution is None else self.solution.cost

    @property
    def feasible(self) -> bool:
        return bool(self.solution is not None and self.solution.feasible)

    @property
    def certified(self) -> bool:
        # A reduced realtime model has a valid certificate for that reduced
        # declaration, but it is not the configured full-horizon optimum.
        return bool(self.certificate_ok and self.feasible and self.realtime_exact)


def _grid(values: Iterable[int] | None) -> tuple[int, ...] | None:
    if values is None:
        return None
    # Materialise once so generators supplied by a config object cannot be
    # consumed while constructing a retry.
    out = tuple(sorted(set(int(v) for v in values)))
    if not out:
        raise ValueError("teacher grid must not be empty")
    return out


def _reduced_grid(values: tuple[int, ...] | None, limit: int, *, include: int | None = None) -> tuple[int, ...] | None:
    """Choose a deterministic small grid for the realtime attempt."""
    if values is None:
        values = tuple(DEFAULT_OFFSETS) if limit <= 7 else None
    if values is None or len(values) <= limit:
        return values
    idx = np.linspace(0, len(values) - 1, limit, dtype=int).tolist()
    chosen = {values[i] for i in idx}
    if include is not None:
        chosen.add(int(include))
    return tuple(sorted(chosen))


def _model(obs: Observation, vp: VehicleParams, *, horizon: int,
           offsets: tuple[int, ...] | None, speeds: tuple[int, ...] | None) -> Model | None:
    road = getattr(obs, "road", None)
    try:
        if road is None or not road.valid or len(road.preview_s) < 3:
            return None
    except (AttributeError, TypeError):
        return None
    return model_from_observation(obs, vp, offsets=DEFAULT_OFFSETS if offsets is None else offsets,
                                  speeds=speeds, horizon=int(horizon))


def _certify(solution: Solution) -> tuple[dict | None, bool]:
    try:
        cert = certificate_from_solution(solution)
        report = check_certificate(cert)
        return cert, bool(report.get("ok", False))
    except (KeyError, TypeError, ValueError, IndexError, OverflowError, RuntimeError):
        return None, False


def _attempt(obs: Observation, vp: VehicleParams, *, horizon: int,
             offsets: tuple[int, ...] | None, speeds: tuple[int, ...] | None,
             cert_dir: str | Path | None, realtime_exact: bool,
             fallback_reason: str | None) -> TeacherResult:
    started = time.perf_counter()
    try:
        model = _model(obs, vp, horizon=horizon, offsets=offsets, speeds=speeds)
    except (AttributeError, TypeError, ValueError, IndexError, OverflowError):
        model = None
    if model is None:
        return TeacherResult(None, None, None, False, (time.perf_counter() - started) * 1000.0,
                             realtime_exact, fallback_reason)
    try:
        solution = solve(model)
    except (AttributeError, TypeError, ValueError, IndexError, OverflowError, RuntimeError):
        return TeacherResult(model, None, None, False, (time.perf_counter() - started) * 1000.0,
                             False, fallback_reason or "exact solver failed")
    cert, cert_ok = _certify(solution)
    wrote = False
    if cert_dir is not None and cert is not None:
        try:
            step = int(getattr(obs, "step", 0))
            write_certificate(cert, Path(cert_dir) / f"teacher_{step:06d}.json")
            wrote = True
        except PermissionError:
            # Event-day sandboxing intentionally denies writes.  The in-memory
            # certificate remains available and its proof status is unchanged.
            wrote = False
    return TeacherResult(model, solution, cert, cert_ok,
                         (time.perf_counter() - started) * 1000.0,
                         bool(realtime_exact), fallback_reason, wrote)


def solve_observation(obs: Observation, vp: VehicleParams, *, horizon: int = 8,
                      offsets: Iterable[int] | None = None,
                      speeds: Iterable[int] | None = None,
                      realtime: bool = False,
                      cert_dir: str | Path | None = None) -> TeacherResult:
    """Build and solve the teacher model for ``obs``.

    Offline (the default) always runs the configured exact model.  Realtime
    mode starts with a bounded reduced declaration and retries once with an
    even smaller declaration if that attempt fails.  Unless no reduction was
    needed, a reduced result is explicitly marked uncertified as a solution
    of the configured model.
    """
    try:
        off = _grid(offsets)
        spd = _grid(speeds)
        h = max(int(horizon), 1)
    except (TypeError, ValueError):
        return TeacherResult(None, None, None, False, 0.0, False, "invalid teacher configuration")

    if not realtime:
        return _attempt(obs, vp, horizon=h, offsets=off, speeds=spd, cert_dir=cert_dir,
                        realtime_exact=True, fallback_reason=None)

    # The reduced model is exact for its realtime declaration.  It is also
    # the configured model only when no reduction was necessary.
    requested_offsets = tuple(DEFAULT_OFFSETS) if off is None else off
    roff = _reduced_grid(off, 7)
    rspd = _reduced_grid(spd, 9)
    full_declaration = (min(h, 4) == h and roff == requested_offsets
                        and (spd is None or rspd == spd))
    first = _attempt(obs, vp, horizon=min(h, 4), offsets=roff, speeds=rspd,
                     cert_dir=cert_dir, realtime_exact=full_declaration,
                     fallback_reason=None if full_declaration else
                     "realtime reduced horizon/grid (not full optimum)")
    if first.solution is not None:
        return first

    # A malformed/no-solution reduced grid should not turn into a controller
    # exception.  The second declaration retains the observed speed grid when
    # possible but further limits lateral choices and horizon.
    second = _attempt(obs, vp, horizon=1, offsets=_reduced_grid(roff, 3),
                      speeds=_reduced_grid(rspd, 3), cert_dir=cert_dir,
                      realtime_exact=False, fallback_reason="realtime exact solve unavailable")
    if second.solution is not None:
        return second
    return second


def _road_path(obs: Observation, solution: Solution) -> tuple[np.ndarray, np.ndarray] | None:
    road = getattr(obs, "road", None)
    if road is None or not solution.plan:
        return None
    try:
        n = len(solution.plan)
        s = np.asarray(road.preview_s, dtype=float)[:n]
        cl = np.asarray(road.preview_centerline, dtype=float)[:n]
        if n < 2 or len(s) != n or cl.shape != (n, 2) or not np.all(np.isfinite(cl)):
            return None
        if not np.all(np.isfinite(s)) or np.any(np.diff(s) <= 0):
            return None
        offset = np.asarray([state.current for state in solution.plan], dtype=float) / 1000.0
        tangent = np.gradient(cl, axis=0)
        norm = np.linalg.norm(tangent, axis=1)
        if np.any(norm <= 1e-9) or not np.all(np.isfinite(norm)):
            return None
        normal = np.column_stack((-tangent[:, 1] / norm, tangent[:, 0] / norm))
        return cl + normal * offset[:, None], s
    except (AttributeError, TypeError, ValueError, IndexError):
        return None


def _common_debug(result: TeacherResult, *, realtime: bool) -> dict:
    model = result.model
    return {
        "planner": "exact-joint",
        "certified": bool(result.certified),
        "claimed": result.claimed,
        "certificate_ok": bool(result.certificate_ok),
        "solver_ms": float(result.solver_ms),
        "horizon": float(model.horizon if model is not None else 0),
        "grid_offsets": float(len(model.offsets) if model is not None else 0),
        "grid_speeds": float(len(model.speeds) if model is not None else 0),
        "realtime": bool(realtime),
        "realtime_exact": bool(result.realtime_exact),
        "fallback_reason": result.fallback_reason or "",
    }


def command_from_result(obs: Observation, vp: VehicleParams, result: TeacherResult,
                        *, realtime: bool = False) -> Command:
    """Turn a planning result into a Command using the expert tracking arithmetic."""
    dbg = _common_debug(result, realtime=realtime)
    road = getattr(obs, "road", None)
    # An infeasible certificate is a real model result, not malformed input.
    # Stopping on it can deadlock the simulator at a tight/edge state, so use
    # an explicitly uncertified centerline recovery rather than pretending a
    # different model was optimal.
    if result.solution is not None and not result.feasible:
        path = None
        try:
            path = np.asarray(road.preview_centerline, dtype=float)
            s = np.asarray(road.preview_s, dtype=float)
            if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2 or len(s) != len(path):
                raise ValueError("invalid recovery preview")
            target_speed = min(3.0, max(float(vp.max_speed), 0.0))
            a_des = float(np.clip(1.5 * (target_speed - float(obs.speed)),
                                  -float(vp.max_decel), float(vp.max_accel)))
            steer_cmd, lookahead = _pursuit(path, s, max(float(obs.speed), 0.0), vp)
            throttle, brake = _pedals(a_des, vp, float(getattr(obs, "ay", 0.0)))
        except (AttributeError, TypeError, ValueError, IndexError, FloatingPointError):
            path = None
        else:
            dbg.update({
                "planner": "exact-joint-recovery",
                "certified": False,
                "fallback_reason": "exact model infeasible: uncertified centerline recovery",
                "lookahead_m": float(lookahead),
                "a_des": float(a_des),
                "target_speed": float(target_speed),
            })
            return Command(steer=steer_cmd, throttle=throttle, brake=brake,
                           path=path, path_frame="vehicle", target_speed=target_speed, debug=dbg)

    if (not result.certificate_ok) or result.solution is None:
        path = None
        if road is not None:
            try:
                path = np.asarray(road.preview_centerline, dtype=float)
            except (AttributeError, TypeError, ValueError):
                path = None
        dbg["certified"] = False
        return Command(brake=1.0, path=path, path_frame="vehicle", target_speed=0.0, debug=dbg)

    path_data = _road_path(obs, result.solution)
    if path_data is None:
        dbg["certified"] = False
        return Command(brake=1.0, target_speed=0.0, debug=dbg)
    path, s = path_data
    plan = result.solution.plan
    first = plan[1]
    transition_d = result.model.gap(0) if result.model is not None else 1
    b0, b1 = int(plan[0].b), int(first.b)
    target_speed = math.sqrt(max(b1, 0)) / 1000.0
    a_des = (b1 - b0) / max(2.0 * float(transition_d) * 1000.0, 1.0)
    try:
        steer_cmd, lookahead = _pursuit(path, s, max(float(obs.speed), 0.0), vp)
        throttle, brake = _pedals(float(a_des), vp, float(getattr(obs, "ay", 0.0)))
    except (AttributeError, TypeError, ValueError, IndexError, FloatingPointError):
        dbg["certified"] = False
        return Command(brake=1.0, path=path, path_frame="vehicle", target_speed=0.0, debug=dbg)
    dbg.update({"lookahead_m": float(lookahead), "a_des": float(a_des),
                "target_speed": float(target_speed),
                "offset_start": float(plan[0].current) / 1000.0,
                "offset_first": float(first.current) / 1000.0})
    return Command(steer=steer_cmd, throttle=throttle, brake=brake,
                   path=path, path_frame="vehicle", target_speed=target_speed, debug=dbg)


def teacher_action(obs: Observation, vp: VehicleParams, *, horizon: int = 8,
                   offsets: Iterable[int] | None = None,
                   speeds: Iterable[int] | None = None,
                   realtime: bool = False,
                   cert_dir: str | Path | None = None) -> Command:
    """Convenience pure function combining planning and command generation."""
    result = solve_observation(obs, vp, horizon=horizon, offsets=offsets, speeds=speeds,
                               realtime=realtime, cert_dir=cert_dir)
    return command_from_result(obs, vp, result, realtime=realtime)


__all__ = ["TeacherResult", "solve_observation", "command_from_result", "teacher_action"]
