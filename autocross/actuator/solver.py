"""Vectorised exact DP for the finite actuator model.

The dense state universe is the Cartesian product of the declared offset,
heading-error, speed-squared and steering-angle grids.  The DP visits every
state and every declared actuator pair; invalid transitions are masked, never
pruned.  The scalar model in :mod:`autocross.actuator.model` is the reference;
``assert_vector_parity`` checks the vectorised transition against it on random
states.

The first action is extracted with the deterministic order (steering input
outer, acceleration inner).  A plan is returned only for tests/diagnostics; the
controller executes its first action in receding-horizon fashion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import ActModel, ActState, COST_TIME_WEIGHT, D_WEIGHT, PSI_WEIGHT, isqrt, snap_np, time_cost

INF = 2 ** 62


@dataclass
class Dense:
    d: np.ndarray
    psi: np.ndarray
    b: np.ndarray
    delta: np.ndarray
    root: np.ndarray
    root_grid: np.ndarray
    shape: tuple[int, int, int, int]

    @property
    def n(self) -> int:
        return self.d.size


def dense_grid(model: ActModel) -> Dense:
    D = np.asarray(model.offsets, dtype=np.int64)
    P = np.asarray(model.psis, dtype=np.int64)
    B = np.asarray(model.speeds, dtype=np.int64)
    Del = np.asarray(model.deltas, dtype=np.int64)
    shape = (D.size, P.size, B.size, Del.size)
    Dg, Pg, Bg, Dg2 = np.meshgrid(D, P, B, Del, indexing="ij")
    d = Dg.reshape(-1)
    psi = Pg.reshape(-1)
    b = Bg.reshape(-1)
    delta = Dg2.reshape(-1)
    root_grid = np.array([isqrt(int(x)) for x in B], dtype=np.int64)
    root = np.array([isqrt(int(x)) for x in b], dtype=np.int64)
    return Dense(d=d, psi=psi, b=b, delta=delta, root=root, root_grid=root_grid, shape=shape)


def _index(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    # snapped values are exactly grid members; clip for safety.
    idx = np.searchsorted(grid, values)
    return np.clip(idx, 0, grid.size - 1)


def transition_arrays(model: ActModel, state: Dense, stage: int):
    """Return next index, cost, and valid mask for every dense state/action."""
    U0 = np.asarray(model.steer_inputs, dtype=np.int64)
    A0 = np.asarray(model.acc_inputs, dtype=np.int64)
    U = np.repeat(U0, A0.size)
    A = np.tile(A0, U0.size)
    D = np.asarray(model.offsets, dtype=np.int64)
    P = np.asarray(model.psis, dtype=np.int64)
    B = np.asarray(model.speeds, dtype=np.int64)
    Del = np.asarray(model.deltas, dtype=np.int64)
    d, psi, b, delta, root = state.d, state.psi, state.b, state.delta, state.root
    ds = model.gap(stage)
    kroad = model.k[stage]

    v = root
    dt = ds * 1_000_000 // np.maximum(v, 1000)
    target = U * model.max_steer // 1000
    lag = (target[None, :] - delta[:, None]) * np.minimum(dt, model.tau)[:, None] // model.tau
    lim = model.steer_rate * dt // 1_000_000
    change = np.clip(lag, -lim[:, None], lim[:, None])
    delta_n = snap_np(delta[:, None] + change, Del)
    den = model.wheelbase + model.understeer_num * b // model.understeer_den
    kappa = delta_n * 1000 // den[:, None]
    psi_n = snap_np(psi[:, None] + (kappa - kroad) * ds // 1000, P)
    d_n = snap_np(d[:, None] + ((psi[:, None] + psi_n) // 2) * ds // 1_000_000, D)
    b_cont = b[:, None] + 2 * A[None, :] * ds
    b_n = snap_np(b_cont, B)

    di = _index(d_n, D)
    pi = _index(psi_n, P)
    bi = _index(b_n, B)
    deli = _index(delta_n, Del)
    next_idx = (((di * P.size + pi) * B.size + bi) * Del.size + deli)
    # cost vector from roots: cdiv(2*ds*1e6, root_b + root_w)
    denom = root[:, None] + state.root_grid[bi]
    denom = np.maximum(denom, 1)
    cost = (COST_TIME_WEIGHT * (-((-(2 * ds * 1_000_000)) // denom))
            + np.abs(delta_n - delta[:, None]) // 1000
            + D_WEIGHT * np.abs(d_n) + PSI_WEIGHT * np.abs(psi_n) // 100)

    # constraints
    lo = model.lo(stage + 1)
    hi = model.hi(stage + 1)
    valid = (b_cont >= 0)
    valid &= (d_n >= lo) & (d_n <= hi)
    valid &= (d_n + d[:, None]) // 2 >= lo
    valid &= (d_n + d[:, None]) // 2 <= hi
    # obstacle intervals at arrival and midpoint
    half = model.half_width + model.margin
    for _, lmin, lmax in model.obstacles_at(stage + 1):
        for x in (d_n, (d_n + d[:, None]) // 2):
            valid &= (x <= lmin - half) | (x >= lmax + half)
    # lateral acceleration and friction ellipse
    kappa_path = kroad + (psi_n - psi[:, None]) * 1000 // ds
    lat = b_n * np.abs(kappa_path) // 1_000_000_000
    a_lat = int(model.lon["a_lat"])
    valid &= lat <= a_lat
    acc = A[None, :]
    a_long = np.where(acc >= 0, int(model.lon["a_acc"]), int(model.lon["a_brk"]))
    valid &= (lat * a_long) ** 2 + (acc * a_lat) ** 2 <= (a_lat * a_long) ** 2
    valid &= (acc >= -int(model.lon["a_brk"])) & (acc <= int(model.lon["a_acc"]))
    # terminal
    if stage + 1 == model.horizon:
        valid &= np.abs(psi_n) <= model.terminal_max_psi
        valid &= np.abs(delta_n) <= model.terminal_max_delta
        valid &= (b_n >= model.terminal_min_b) & (b_n <= model.terminal_cap())
    return next_idx, cost, valid


def terminal_values(model: ActModel, state: Dense) -> np.ndarray:
    v = np.full(state.n, INF, dtype=np.int64)
    for k in range(state.n):
        s = ActState(model.horizon, int(state.d[k]), int(state.psi[k]),
                     int(state.b[k]), int(state.delta[k]))
        c = model.terminal_cost(s)
        if c is not None:
            v[k] = COST_TIME_WEIGHT * c
    return v


def _dense_index(model: ActModel, s: ActState) -> int:
    di = int(np.searchsorted(np.asarray(model.offsets), s.d))
    pi = int(np.searchsorted(np.asarray(model.psis), s.psi))
    bi = int(np.searchsorted(np.asarray(model.speeds), s.b))
    deli = int(np.searchsorted(np.asarray(model.deltas), s.delta))
    return (((di * len(model.psis) + pi) * len(model.speeds) + bi) * len(model.deltas) + deli)


def _decode_index(model: ActModel, idx: int) -> ActState:
    n_del = len(model.deltas)
    n_b = len(model.speeds)
    n_p = len(model.psis)
    deli = idx % n_del; idx //= n_del
    bi = idx % n_b; idx //= n_b
    pi = idx % n_p; idx //= n_p
    di = idx
    return ActState(0, model.offsets[di], model.psis[pi], model.speeds[bi], model.deltas[deli])


def solve(model: ActModel) -> tuple[ActState | None, int | None, tuple[int, int] | None, dict]:
    """Return (first successor, optimal cost, first (steer,accel) action, diagnostics).

    ``diag`` also carries the complete per-stage value/argmin tables and the
    deterministic optimal plan.  They are the raw material for a proof-carrying
    certificate; the controller itself only uses the first action.
    """
    state = dense_grid(model)
    V = terminal_values(model, state)
    values: list[np.ndarray | None] = [None] * (model.horizon + 1)
    args: list[np.ndarray | None] = [None] * (model.horizon + 1)
    nxts: list[np.ndarray | None] = [None] * (model.horizon + 1)
    values[model.horizon] = V.copy()
    for stage in range(model.horizon - 1, 0, -1):
        nxt, cost, valid = transition_arrays(model, state, stage)
        cand = cost + V[nxt]
        cand = np.where(valid, cand, INF)
        best = cand.min(axis=1)
        arg = cand.argmin(axis=1)
        V = best.astype(np.int64)
        values[stage] = V.copy()
        args[stage] = arg.copy()
        nxts[stage] = nxt
    # Stage 0: scalar start state, all actions.
    s0 = model.start
    best_cost = None
    best_action = None
    best_t = None
    best_idx = None
    for u in model.steer_inputs:
        for a in model.acc_inputs:
            t = model.action_ok(s0, u, a)
            if t is None:
                continue
            idx = _dense_index(model, t)
            v = values[1][idx]  # type: ignore[index]
            if v >= INF // 2:
                continue
            c = model.action_cost(s0, t) + int(v)
            if best_cost is None or c < best_cost:
                best_cost = c
                best_action = (u, a)
                best_t = t
                best_idx = idx
    # Reconstruct the deterministic optimal plan (states at stations 0..H).
    plan: list[ActState] = [s0]
    if best_t is not None and best_idx is not None:
        plan.append(best_t)
        cur = best_t
        for stage in range(1, model.horizon):
            arg = args[stage]
            nxt = nxts[stage]
            if arg is None or nxt is None:
                plan = []
                break
            idx = _dense_index(model, cur)
            ai = int(arg[idx])
            nxt_idx = int(nxt[idx, ai])
            cur = _decode_index(model, nxt_idx)
            cur = ActState(stage + 1, cur.d, cur.psi, cur.b, cur.delta)
            plan.append(cur)
    diag = {"states": state.n, "actions": len(model.steer_inputs) * len(model.acc_inputs),
            "feasible": best_cost is not None, "first": best_t,
            "values": values, "args": args, "nxts": nxts,
            "grid_shape": state.shape, "plan": plan,
            "first_index": best_idx}
    return best_t, best_cost, best_action, diag


def assert_vector_parity(model: ActModel, samples: int = 200, seed: int = 0) -> int:
    """Check the vectorised transition against the scalar reference."""
    rng = np.random.default_rng(seed)
    state = dense_grid(model)
    nxt, cost, valid = transition_arrays(model, state, 0)
    checked = 0
    for _ in range(samples):
        k = int(rng.integers(state.n))
        ai = int(rng.integers(len(model.steer_inputs) * len(model.acc_inputs)))
        u = model.steer_inputs[ai // len(model.acc_inputs)]
        a = model.acc_inputs[ai % len(model.acc_inputs)]
        s = ActState(0, int(state.d[k]), int(state.psi[k]), int(state.b[k]), int(state.delta[k]))
        t = model.action_ok(s, u, a)
        if t is None:
            assert not valid[k, ai], (s, u, a)
            continue
        exp_idx = (((int(np.searchsorted(model.offsets, t.d)) * len(model.psis) +
                     int(np.searchsorted(model.psis, t.psi))) * len(model.speeds) +
                    int(np.searchsorted(model.speeds, t.b))) * len(model.deltas) +
                   int(np.searchsorted(model.deltas, t.delta)))
        assert valid[k, ai], (s, u, a, t)
        assert int(nxt[k, ai]) == exp_idx, (s, u, a, t, int(nxt[k, ai]), exp_idx)
        assert int(cost[k, ai]) == model.action_cost(s, t), (s, u, a)
        checked += 1
    return checked


def forward_universe(model: ActModel) -> list[list[ActState]]:
    univ = [[model.start]]
    for i in range(model.horizon):
        nxt = []
        seen = set()
        for s in univ[i]:
            for t, _c, _u, _a in model.actions(s):
                if t not in seen:
                    seen.add(t); nxt.append(t)
        nxt.sort(key=lambda s: (s.d, s.psi, s.b, s.delta))
        univ.append(nxt)
    return univ


def solve_sparse(model: ActModel):
    """Scalar sparse exact DP; useful for realtime-size declarations and tests."""
    univ = forward_universe(model)
    V: list[dict[ActState, int | None]] = [dict() for _ in range(model.horizon + 1)]
    choice: list[dict[ActState, tuple[int, int, int]]] = [dict() for _ in range(model.horizon + 1)]
    for s in univ[model.horizon]:
        c = model.terminal_cost(s)
        V[model.horizon][s] = None if c is None else COST_TIME_WEIGHT * c
    for i in range(model.horizon - 1, 0, -1):
        for s in univ[i]:
            best = None; besta = None
            for t, c, u, a in model.actions(s):
                nv = V[i + 1].get(t)
                if nv is None:
                    continue
                cand = c + nv
                if best is None or cand < best:
                    best = cand; besta = (u, a, c)
            V[i][s] = best
            if besta is not None:
                choice[i][s] = besta
    s0 = model.start
    best = None; besta = None; bestt = None
    for t, c, u, a in model.actions(s0):
        nv = V[1].get(t)
        if nv is None:
            continue
        cand = c + nv
        if best is None or cand < best:
            best = cand; besta = (u, a); bestt = t
    plan: list[ActState] = [s0]
    plan_actions: list[tuple[int, int]] = []
    if bestt is not None and besta is not None:
        plan.append(bestt); plan_actions.append(besta)
        cur = bestt
        for i in range(1, model.horizon):
            ch = choice[i].get(cur)
            if ch is None:
                plan = []; plan_actions = []
                break
            u, a, _c = ch
            nxt = model.action_ok(cur, u, a)
            if nxt is None:
                plan = []; plan_actions = []
                break
            plan.append(nxt); plan_actions.append((u, a)); cur = nxt
    V[0][s0] = best
    diag = {"states": sum(len(x) for x in univ), "univ": univ,
            "univ_sizes": [len(x) for x in univ], "values": V,
            "choice": choice, "feasible": best is not None, "first": bestt,
            "plan": plan, "plan_actions": plan_actions}
    return bestt, best, besta, diag
