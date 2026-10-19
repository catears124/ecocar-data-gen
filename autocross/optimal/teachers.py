"""Scenario-level labels produced by the exact teacher and comparison policies."""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from autocross.api import Observation
from autocross.datagen.expert import expert_action
from .certificate import certificate_from_solution, check_certificate
from .model import State, model_from_observation
from .solver import solve


def _empty(*, method: str, optimal: bool, solver_ms: float, certificate=None) -> dict:
    return {
        "feasible": False,
        "offset_mm": None,
        "accel_mm_s2": None,
        "speed_sq": None,
        "claimed": None,
        "plan": [],
        "certificate": certificate,
        "certificate_path": None,
        "certificate_ok": None,
        "method": method,
        "optimal": bool(optimal),
        "solver_ms": float(solver_ms),
    }


def _state_dict(s: State) -> dict:
    return s.as_dict()


def _numeric_label(sol, cert, *, method: str, optimal: bool, solver_ms: float,
                   plan: list[State] | None = None) -> dict:
    if sol.cost is None:
        return _empty(method=method, optimal=optimal, solver_ms=solver_ms, certificate=cert)
    m = sol.model
    chosen = list(sol.plan if plan is None else plan)
    if len(chosen) < 2:
        return _empty(method=method, optimal=optimal, solver_ms=solver_ms, certificate=cert)
    # The teacher's lateral target is the first state at or beyond 5 m.  The
    # fallback is the final state when the preview is shorter than 5 m.
    station = len(chosen) - 1
    for j, state in enumerate(chosen[1:], 1):
        if state.station < len(m.s) and m.s[state.station] >= 5000:
            station = j
            break
    first = chosen[1]
    return {
        "feasible": True,
        "offset_mm": int(chosen[station].current),
        "accel_mm_s2": int((first.b - chosen[0].b) // (2 * m.gap(0))),
        "speed_sq": int(first.b),
        "claimed": int(sol.cost),
        "plan": [_state_dict(s) for s in chosen],
        "certificate": cert,
        "certificate_path": None,
        "certificate_ok": True if cert is not None else None,
        "method": method,
        "optimal": bool(optimal),
        "solver_ms": float(solver_ms),
    }


def optimal_label(obs, vp, horizon: int = 10) -> dict:
    """Solve the declared finite-horizon joint model and return a serialisable label."""
    started = time.perf_counter()
    model = model_from_observation(obs, vp, horizon=int(horizon))
    if model is None:
        return _empty(method="exact-dp", optimal=True,
                      solver_ms=(time.perf_counter() - started) * 1000.0)
    sol = solve(model)
    cert = certificate_from_solution(sol)
    # Keep this assertion local: callers can rely on every exact teacher
    # certificate being independently checked before it is persisted.
    report = check_certificate(cert)
    if not report.get("ok"):
        raise ValueError(f"teacher generated an invalid certificate: {report.get('errors')}")
    return _numeric_label(sol, cert, method="exact-dp", optimal=True,
                          solver_ms=(time.perf_counter() - started) * 1000.0)


def approx_label(obs, vp, horizon: int = 10, short_horizon: int = 2) -> dict:
    """Use exact DP for a short prefix, then hold its final action to the horizon."""
    started = time.perf_counter()
    full = model_from_observation(obs, vp, horizon=int(horizon))
    if full is None:
        return _empty(method="truncated-horizon", optimal=False,
                      solver_ms=(time.perf_counter() - started) * 1000.0)
    short = min(max(int(short_horizon), 1), full.horizon)
    prefix_model = replace(full, horizon=short)
    sol = solve(prefix_model)
    cert = certificate_from_solution(sol)
    report = check_certificate(cert)
    if not report.get("ok"):
        raise ValueError(f"truncated teacher generated an invalid certificate: {report.get('errors')}")
    if sol.cost is None or not sol.plan:
        return _empty(method="truncated-horizon", optimal=False,
                      solver_ms=(time.perf_counter() - started) * 1000.0, certificate=cert)

    plan = list(sol.plan)
    # Extend with a constant offset and speed.  Validation uses the original
    # model, so the extension cannot accidentally relax the full-horizon rules.
    while len(plan) < full.horizon + 1:
        last = plan[-1]
        nxt = State(last.station + 1, last.current, last.current, last.b)
        if not full.step_ok(last, nxt):
            return _empty(method="truncated-horizon", optimal=False,
                          solver_ms=(time.perf_counter() - started) * 1000.0,
                          certificate=cert)
        plan.append(nxt)
    if full.terminal_value(plan[-1]) is None:
        return _empty(method="truncated-horizon", optimal=False,
                      solver_ms=(time.perf_counter() - started) * 1000.0,
                      certificate=cert)
    # The prefix certificate's claimed cost remains the deliberately
    # truncated objective; the returned plan is the physically checked hold.
    return _numeric_label(sol, cert, method="truncated-horizon", optimal=False,
                          solver_ms=(time.perf_counter() - started) * 1000.0, plan=plan)


def _heuristic_offset(obs, command) -> int | None:
    road = obs.road
    path = getattr(command, "path", None)
    if road is not None and path is not None and len(path):
        import numpy as np
        cl = np.asarray(road.preview_centerline, dtype=float)
        p = np.asarray(path, dtype=float)
        if len(cl) and len(p):
            if len(cl) > 1:
                tangent = np.asarray(cl[1]) - np.asarray(cl[0])
            else:
                tangent = np.array([1.0, 0.0], dtype=float)
            norm = np.array([-tangent[1], tangent[0]], dtype=float)
            den = float(np.linalg.norm(norm))
            if den > 1e-12:
                return int(round(float(np.dot(p[0] - cl[0], norm / den)) * 1000.0))
    if road is not None:
        try:
            return int(round(float(road.lateral_offset) * 1000.0))
        except (TypeError, ValueError):
            pass
    return None


def heuristic_label(obs, vp) -> dict:
    """Return the uncertified Veridrive heuristic action as a common label shape."""
    started = time.perf_counter()
    # expert_action is pure; use a shallow dataclass replacement so this label
    # cannot accidentally depend on a scenario's privileged progress value.
    try:
        decision_obs = replace(obs, progress=0.0)
    except TypeError:
        decision_obs = obs
        decision_obs.progress = 0.0
    command = expert_action(decision_obs, vp, config=None)
    dbg = command.debug or {}
    accel = dbg.get("a_des")
    if accel is None:
        accel_mm = dbg.get("cert_a")
    else:
        accel_mm = int(round(float(accel) * 1000.0))
    speed = command.target_speed if command.target_speed is not None else getattr(obs, "speed", 0.0)
    speed_sq = int(round(float(speed) * 1000.0)) ** 2 if speed is not None else None
    offset = _heuristic_offset(obs, command)
    plan = []
    if speed_sq is not None:
        plan = [{"station": 0, "previous": int(offset or 0), "current": int(offset or 0), "b": speed_sq}]
    return {
        "feasible": bool(obs.road is not None and obs.road.valid),
        "offset_mm": offset,
        "accel_mm_s2": None if accel_mm is None else int(accel_mm),
        "speed_sq": speed_sq,
        "claimed": None,
        "plan": plan,
        "certificate": None,
        "certificate_path": None,
        "certificate_ok": None,
        "method": "veridrive-heuristic",
        "optimal": False,
        "solver_ms": float((time.perf_counter() - started) * 1000.0),
    }


def cert_control_label(obs, vp, horizon: int = 10) -> dict:
    """Exact numeric control label without a persisted certificate artifact."""
    label = optimal_label(obs, vp, horizon=horizon)
    label["certificate_path"] = None
    label["certificate_ok"] = None
    return label


__all__ = ["optimal_label", "approx_label", "heuristic_label", "cert_control_label"]
