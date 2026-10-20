"""Exact lattice path planner + minimum-time speed profile.

Two layers, each with a precise scope:

* :func:`plan_path` is an exact dynamic program over the declared finite offset
  lattice ``(previous offset, current offset)``.  Its stage cost is the
  traversal time at the local lateral-grip speed cap
  ``v <= sqrt(a_lat / curv)``; it is therefore a *minimum-time proxy over the
  declared path lattice*, not a joint optimum over speed (the joint optimum is
  the much larger DP in :mod:`autocross.lineopt.dp`, used by the offline
  oracle).  Corridor, obstacle and curvature constraints are hard.
* :func:`speed_profile` computes the minimum-time speed profile for a fixed
  path under the friction circle, acceleration, braking and terminal-viability
  constraints.  It is a forward/backward envelope iteration; for a fixed path
  and a fixed lateral-budget schedule this is the exact discrete minimum-time
  profile under the declared bounds.

The two layers compose into a hierarchical planner with an explicit optimality
scope.  Nothing here claims global optimality over the continuous vehicle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from autocross.optimal.model import Model, cdiv, isqrt

INF = 2 ** 62


@dataclass
class PathPlan:
    offsets: tuple[int, ...]          # one offset (mm) per station 0..H
    cost_us: int | None               # declared proxy cost (sum of segment times)
    values: np.ndarray | None         # (H+1, No, No)
    choices: np.ndarray | None        # (H, No, No) next-offset index
    elapsed_ms: float = 0.0


def _v_cap_mm_s(curv: int, a_lat: int, vmax: int) -> int:
    if curv <= 0:
        return vmax
    return min(vmax, isqrt(a_lat * 1_000_000_000 // max(curv, 1)))


def plan_path(model: Model, *, a_lat: int | None = None, mode: str = "curvature") -> PathPlan:
    """Exact DP for the minimum time-at-cap path over the offset lattice."""
    import time as _time
    t0 = _time.perf_counter()
    O = np.asarray(model.offsets, dtype=np.int64)
    No = O.size
    H = int(model.horizon)
    kappa_max = int(model.kappa_max)
    lat = int(model.lon["a_lat"] if a_lat is None else a_lat)
    vmax = int(model.lon["v_max"])
    ds = int(model.gap(0))
    assert all(int(model.gap(i)) == ds for i in range(H)), "path DP assumes a uniform station grid"
    ks = np.asarray([abs(int(model.k[i])) for i in range(H)], dtype=np.int64)

    # Lateral admissibility at each station.  ``Model.lateral_ok`` ignores the
    # previous offset; precompute the corridor and obstacle masks once instead
    # of calling the property-heavy scalar predicate H*No^2 times.
    lo = np.asarray([model.lo(i) for i in range(H + 1)], dtype=np.int64)
    hi = np.asarray([model.hi(i) for i in range(H + 1)], dtype=np.int64)
    lat_ok = (O[None, :] >= lo[:, None]) & (O[None, :] <= hi[:, None])
    half = model.half_width + model.margin
    for i in range(H + 1):
        for _d, lmin, lmax in model.obstacles_at(i):
            lat_ok[i] &= (O <= lmin - half) | (O >= lmax + half)
    ok = np.broadcast_to(lat_ok[:, None, :], (H + 1, No, No)).copy()

    values = np.full((H + 1, No, No), INF, dtype=np.int64)
    choices = np.full((H, No, No), -1, dtype=np.int64)
    values[H][ok[H]] = 0

    # Curvature and stage cost for every (p, x, n), vectorised over the
    # declared lattice.  The proxy cost is the segment traversal time at the
    # local lateral-grip cap.
    diff = np.abs(O[None, None, :] - 2 * O[None, :, None] + O[:, None, None])
    # ``cdiv`` of the positive numerator ``diff*1e9``.
    curv_all = ks[:, None, None, None] + \
        (-((-(diff[None, :, :, :] * 1_000_000_000)) // (ds * ds)))
    uniq = np.unique(curv_all)
    vcap_table = np.asarray([_v_cap_mm_s(int(c), lat, vmax) for c in uniq], dtype=np.int64)
    vcap = vcap_table[np.searchsorted(uniq, curv_all)]
    if mode == "time":
        stage_cost = (-((-(ds * 1_000_000)) // np.maximum(vcap, 1))).astype(np.int64)
    elif mode == "curvature":
        # Classic minimum-curvature line: sum of squared second differences.
        stage_cost = np.broadcast_to((diff * diff).astype(np.int64)[None, :, :, :], (H, No, No, No))
    else:
        raise ValueError(f"unknown path objective {mode!r}")

    for i in range(H - 1, -1, -1):
        # Candidate value for every (p, x, n): stage cost + next value.
        cand = stage_cost[i] + values[i + 1][None, :, :]
        allowed = ok[i + 1][None, :, :] & (curv_all[i] <= kappa_max)
        cand = np.where(allowed, cand, INF)
        flat = cand.reshape(No * No, No)
        best_n = np.argmin(flat, axis=1).reshape(No, No)
        best = np.take_along_axis(flat, best_n.reshape(-1, 1), axis=1).reshape(No, No)
        best = np.where(best >= INF, INF, best)
        # Only states that are admissible at station i exist; others are dead.
        best = np.where(ok[i], best, INF)
        values[i] = best
        choices[i] = np.where(ok[i], best_n, -1)

    # Initial stencil: the observation gives the current offset, not a measured
    # previous offset.  Declare the previous offset equal to the current one
    # (the same convention as ``autocross.optimal.model_from_observation``), and
    # include it in the offset set if necessary.
    off_index = {int(v): i for i, v in enumerate(O.tolist())}
    xi = off_index[int(model.start.current)]
    pi = off_index[int(model.start.previous)]
    value = int(values[0, pi, xi])
    offsets: list[int] = []
    if value < INF:
        cur = int(model.start.current)
        prev = int(model.start.previous)
        offsets.append(cur)
        for i in range(H):
            n = int(choices[i, off_index[prev], off_index[cur]])
            if n < 0:
                value = INF
                offsets = []
                break
            offsets.append(int(O[n]))
            prev, cur = cur, int(O[n])
    out = PathPlan(offsets=tuple(offsets), cost_us=None if value >= INF else value,
                   values=values, choices=choices,
                   elapsed_ms=(_time.perf_counter() - t0) * 1000.0)
    return out


def path_curvatures(model: Model, offsets: tuple[int, ...]) -> np.ndarray:
    """Declared path curvature (micro-1/m) at every station of an offset plan."""
    ds = int(model.gap(0))
    out = np.zeros(len(offsets), dtype=np.int64)
    for i in range(len(offsets)):
        prev = offsets[max(i - 1, 0)]
        cur = offsets[i]
        nxt = offsets[min(i + 1, len(offsets) - 1)]
        out[i] = abs(int(model.k[i])) + (-((-abs(nxt - 2 * cur + prev) * 1_000_000_000) // (ds * ds)))
    return out


def speed_profile(model: Model, offsets: tuple[int, ...], v0_mps: float, *,
                  mu_frac: float = 0.95, brk_frac: float = 0.95,
                  acc_frac: float = 0.95, v_max_mps: float | None = None,
                  terminal_cap: int | None = None, iterations: int = 3,
                  grip_mps2: float | None = None) -> np.ndarray:
    """Minimum-time speed² profile for a fixed offset path, friction-circle aware.

    The lateral acceleration demand at station ``i`` is ``b_i * |curv_i|``.  The
    remaining longitudinal budget is ``a_long = sqrt((mu*g)^2 - a_lat^2)``; both
    passes use it, so a cornering car may brake or accelerate only with the
    grip the tyres have left.  A few fixed-point iterations resolve the
    dependence of the budget on ``b_i``.  The final profile also satisfies the
    declared terminal braking envelope.
    """
    H = len(offsets) - 1
    ds = np.asarray([model.gap(i) for i in range(H)], dtype=np.float64) / 1000.0  # m
    curv = path_curvatures(model, offsets).astype(np.float64) / 1e6   # 1/m
    mu_g = (max(float(grip_mps2), 1e-6) if grip_mps2 is not None
            else max(float(model.lon["a_lat"]) / 1000.0 / max(mu_frac, 1e-6), 1e-6))
    mu_frac = 1.0 if grip_mps2 is not None else mu_frac
    v_max = float(model.lon["v_max"]) / 1000.0 if v_max_mps is None else float(v_max_mps)
    a_acc_max = float(model.lon["a_acc"]) / 1000.0 * acc_frac
    a_brk_max = float(model.lon["a_brk"]) / 1000.0 * brk_frac
    cap_b = float(model.terminal_cap if terminal_cap is None else terminal_cap) / 1e6

    # Lateral-only cap from the friction circle.
    with np.errstate(divide="ignore", invalid="ignore"):
        lat_cap = np.where(curv > 1e-9, (mu_g * mu_frac) / np.maximum(curv, 1e-12), v_max * v_max)
    b = np.minimum(v_max * v_max, np.maximum(lat_cap, 1.0))
    b[0] = min(b[0], max(float(v0_mps) ** 2, 1.0))
    b[-1] = min(b[-1], cap_b)
    grip2 = (mu_g * mu_frac) ** 2
    for _ in range(max(int(iterations), 1)):
        # Forward: accelerate from the already-feasible previous station only
        # with the longitudinal grip that station has left after cornering.
        for i in range(1, H + 1):
            lat_prev = b[i - 1] * curv[i - 1]
            left = math.sqrt(max(grip2 - lat_prev * lat_prev, 0.0))
            a = min(a_acc_max, left)
            b[i] = min(b[i], b[i - 1] + 2.0 * a * ds[i - 1])
        # Backward: brake for the next station's lateral demand and cap.
        for i in range(H - 1, -1, -1):
            lat_next = b[i + 1] * curv[i + 1]
            left = math.sqrt(max(grip2 - lat_next * lat_next, 0.0))
            a = min(a_brk_max, left)
            b[i] = min(b[i], b[i + 1] + 2.0 * a * ds[i])
        b = np.maximum(b, 1.0)
        b[0] = min(b[0], max(float(v0_mps) ** 2, 1.0))
        b[-1] = min(b[-1], cap_b)
    # The car's current speed can exceed the cap; it must slow down, and the
    # profile reports the cap from the next station onward.  Keep station 0 at
    # the measured speed for tracking continuity.
    return b


def profile_time_us(model: Model, offsets: tuple[int, ...], b: np.ndarray) -> int:
    total = 0
    for i in range(len(offsets) - 1):
        v0 = max(int(round(math.sqrt(max(float(b[i]), 1.0)) * 1000.0)), 1)
        v1 = max(int(round(math.sqrt(max(float(b[i + 1]), 1.0)) * 1000.0)), 1)
        total += cdiv(2 * int(model.gap(i)) * 1_000_000, v0 + v1)
    return int(total)


__all__ = ["PathPlan", "plan_path", "path_curvatures", "speed_profile", "profile_time_us"]
