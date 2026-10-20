"""Vectorised exact dynamic program for the declared joint offset/speed lattice.

This is the same optimization problem as :mod:`autocross.optimal.solver`
(the ``teacher-joint-1`` model): every state of the declared Cartesian universe
``(station, previous offset, current offset, speed^2)`` and every admissible
successor is visited; no heuristic prunes the state space.  Only the arithmetic
is vectorised, so the returned value is bit-identical to the scalar reference
and the same certificate applies.

The DP is exact for the *declared finite lattice*: offset grid, speed grid,
station step, curvature cap, acceleration/braking budgets and terminal speed
cap.  It is not a claim about the continuous vehicle; that gap is measured
separately.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autocross.optimal.model import Model, State, cdiv, isqrt, time_cost

INF = 2 ** 62


@dataclass
class FastSolution:
    model: Model
    value: int | None
    plan: list[State]
    values: list[np.ndarray]            # [i] shape (No, No, Nb), indexed (prev, cur, b)
    choices: list[np.ndarray]           # [i] shape (No, No, Nb), index n*Nb + w
    elapsed_ms: float = 0.0

    @property
    def feasible(self) -> bool:
        return self.value is not None


def _offset_indices(model: Model) -> dict[int, int]:
    return {int(v): i for i, v in enumerate(model.offsets)}


def _speed_indices(model: Model, start_b: int | None = None) -> tuple[tuple[int, ...], dict[int, int]]:
    speeds = tuple(int(v) for v in model.speeds)
    if start_b is not None and start_b not in speeds:
        speeds = tuple(sorted(set(speeds) | {int(start_b)}))
    return speeds, {v: i for i, v in enumerate(speeds)}


def solve_fast(model: Model) -> FastSolution:
    """Exact backward induction over the full declared state universe.

    ``values[i][p, x, b]`` is the minimum declared remaining cost from station
    ``i`` with previous/current offsets ``O[p]``/``O[x]`` and speed squared
    ``B[b]``.  ``choices[i]`` stores the first minimising action in the model's
    declared order (next offset outer, next speed inner), so ties break exactly
    as in :func:`autocross.optimal.solver.solve`.
    """
    import time as _time
    t0 = _time.perf_counter()

    H = int(model.horizon)
    O = np.asarray(model.offsets, dtype=np.int64)
    speeds, _ = _speed_indices(model, int(model.start.b))
    B = np.asarray(speeds, dtype=np.int64)
    No, Nb = O.size, B.size

    off_index = {int(v): i for i, v in enumerate(O.tolist())}
    spd_index = {int(v): i for i, v in enumerate(B.tolist())}
    roots = np.asarray([isqrt(int(v)) for v in B.tolist()], dtype=np.int64)

    # Cost matrix for one station segment.  ``time_cost`` is a ceiling
    # division; the vectorised form below is the same function.
    ds_by_stage = [int(model.gap(i)) for i in range(max(H, 1))]
    cost_cache: dict[int, np.ndarray] = {}
    for ds in set(ds_by_stage):
        denom = roots[:, None] + roots[None, :]
        cost_cache[ds] = -((-(2 * ds * 1_000_000)) // np.maximum(denom, 1))

    kappa_max = int(model.kappa_max)
    lon = model.lon
    a_acc = int(lon["a_acc"])
    a_brk = int(lon["a_brk"])
    vmax2 = int(lon["v_max"]) ** 2
    a_lat = int(lon["a_lat"])

    values: list[np.ndarray] = [None] * (H + 1)  # type: ignore[list-item]
    choices: list[np.ndarray] = [None] * H       # type: ignore[list-item]

    Vnext = np.where(B[None, None, :] <= int(model.terminal_cap), 0, INF).astype(np.int64)
    Vnext = np.broadcast_to(Vnext, (No, No, Nb)).copy()
    values[H] = Vnext

    # offset index arrays
    p = O[:, None, None]
    x = O[None, :, None]
    n = O[None, None, :]
    b = B[None, :, None]

    for i in range(H - 1, -1, -1):
        ds = ds_by_stage[i]
        cost = cost_cache[ds]
        # Curvature of the path through (p, x, n) at departure station i.
        diff = np.abs(n - 2 * x + p)
        curv = abs(int(model.k[i])) + (-((-diff * 1_000_000_000) // (ds * ds)))
        curv_ok = curv <= kappa_max
        # Departure speed cap implied by the path curvature.
        cap = np.where(curv <= 0, vmax2, np.minimum(vmax2, a_lat * 1_000_000_000 // np.maximum(curv, 1)))
        cap_ok = B[None, None, :, None, None] <= cap[:, :, None, :, None]
        # Corridor and obstacle clearance at the arrival station.
        # ``lateral_ok`` depends only on the arrival offset (the previous
        # offset is carried for curvature only), so this is an (x, n) mask.
        lat_ok = np.zeros((No, No), dtype=bool)
        for xi in range(No):
            for ni in range(No):
                lat_ok[xi, ni] = model.lateral_ok(i + 1, int(O[xi]), int(O[ni]))
        # Speed transition feasibility.
        bs = B[:, None]
        ws = B[None, :]
        speed_ok = (ws <= bs + 2 * a_acc * ds) & (bs <= ws + 2 * a_brk * ds)

        # Candidate tensor: [p, x, b, n, w]
        vals = (cost[None, None, :, None, :] + Vnext[None, :, None, :, :])
        ok = (lat_ok[None, :, None, :, None] & curv_ok[:, :, None, :, None]
              & cap_ok & speed_ok[None, None, :, None, :])
        masked = np.where(ok, vals, INF)
        flat = masked.reshape(No, No, Nb, No * Nb)
        idx = np.argmin(flat, axis=-1)
        best = np.take_along_axis(flat, idx[..., None], axis=-1)[..., 0]
        # A state with no admissible successor is ``none`` (INF) in the model.
        best = np.where(best >= INF, INF, best)
        values[i] = best
        choices[i] = idx
        Vnext = best

    # Extract the plan from the observed start state.
    start = model.start
    pi = off_index[int(start.previous)]
    xi = off_index[int(start.current)]
    bi = spd_index[int(start.b)]
    value = int(values[0][pi, xi, bi])
    plan: list[int] = []
    if value < INF:
        cur_state = start
        for i in range(H):
            pii = off_index[int(cur_state.previous)]
            xii = off_index[int(cur_state.current)]
            bii = spd_index[int(cur_state.b)]
            a = int(choices[i][pii, xii, bii])
            ni, wi = divmod(a, Nb)
            cur_state = State(i + 1, int(O[xii]), int(O[ni]), int(B[wi]))
            plan.append(cur_state.station)
    else:
        value = None

    out = FastSolution(model=model, value=value, plan=[], values=values, choices=choices,
                       elapsed_ms=(_time.perf_counter() - t0) * 1000.0)
    if value is not None:
        # Rebuild the full state plan (station 0 first) for consumers.
        states = [start]
        cur_state = start
        for i in range(H):
            pii = off_index[int(cur_state.previous)]
            xii = off_index[int(cur_state.current)]
            bii = spd_index[int(cur_state.b)]
            a = int(choices[i][pii, xii, bii])
            ni, wi = divmod(a, Nb)
            cur_state = State(i + 1, int(O[xii]), int(O[ni]), int(B[wi]))
            states.append(cur_state)
        out.plan = states
    return out


def evaluate_plan(model: Model, plan: list[State]) -> int | None:
    """Declared cost of a state plan, or ``None`` if any transition is inadmissible."""
    if not plan or plan[0] != model.start:
        return None
    total = 0
    for a, b in zip(plan, plan[1:]):
        if not model.step_ok(a, b):
            return None
        total += time_cost(model.gap(a.station), a.b, b.b)
    if model.terminal_value(plan[-1]) is None:
        return None
    return int(total)


def plan_cost_breakdown(model: Model, plan: list[State]) -> dict:
    """Objective decomposition of a plan (for reports; not used by the solver)."""
    if not plan:
        return {"segments": 0, "time_us": None, "mean_speed_mps": None}
    total = 0
    dist = 0
    for a, b in zip(plan, plan[1:]):
        if not model.step_ok(a, b):
            return {"segments": 0, "time_us": None, "mean_speed_mps": None}
        total += time_cost(model.gap(a.station), a.b, b.b)
        dist += model.gap(a.station)
    return {"segments": len(plan) - 1, "time_us": total,
            "mean_speed_mps": (dist / 1000.0) / (total / 1e6) if total else None}


__all__ = ["FastSolution", "solve_fast", "evaluate_plan", "plan_cost_breakdown"]
