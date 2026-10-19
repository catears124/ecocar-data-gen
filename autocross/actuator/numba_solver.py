"""Compiled exact DP kernel for the actuator model.

The kernel is a line-by-line translation of ``ActModel.next_state``,
``ActModel.action_ok`` and ``ActModel.action_cost`` (the scalar reference):

* the corridor/obstacle test is evaluated on the exact successor offset and on
  the exact, unsnapped midpoint ``(d' + d) // 2`` (never a grid-snapped one);
* the lateral-acceleration bound is ``b' * |kappa_path| <= a_lat * 1e9`` before
  flooring, and the friction ellipse uses the floored lateral acceleration;
* the stage cost tracks ``ref_d[i+1]`` / ``ref_psi[i+1]`` (0 when the declared
  reference is shorter), exactly like ``action_cost``;
* ties are broken by the declared order (steering outer, acceleration inner,
  first minimum wins), as in ``solve`` and ``solve_sparse``.

Feasibility is an explicit flag (the best action index is ``-1`` when no
admissible action exists); no action value is ever used as a sentinel.  All
arithmetic is int64; :func:`autocross.actuator.solver.dense_violations` bounds
every intermediate product for the instance before the kernel runs.  Integer
``//`` in numba follows Python's floor semantics for negative operands.

The returned plan is replayed through the scalar model; any disagreement with
the kernel's value raises, so a kernel defect can never be reported as an
optimal action.

numba needs ``ctypes.dlopen``.  The controller sandbox denies it, so inside the
sandbox ``NUMBA_AVAILABLE`` is False and the numpy solver runs instead.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
    NUMBA_IMPORT_ERROR = ""
except Exception as _exc:  # pragma: no cover - depends on the environment
    njit = None
    NUMBA_AVAILABLE = False
    NUMBA_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

INF = 2 ** 62


def _kernel_source():
    @njit(cache=False)
    def _index_of(vals, x):
        """Index of ``model.snap(x, vals)`` (nearest, ties to the larger value, clamped)."""
        lo = 0
        hi = vals.size - 1
        if x <= vals[0]:
            return 0
        if x >= vals[hi]:
            return hi
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if vals[mid] <= x:
                lo = mid
            else:
                hi = mid
        return hi if (x - vals[lo]) >= (vals[hi] - x) else lo

    @njit(cache=False)
    def _lateral_ok(lo, hi, obs, nobs, i, x):
        if x < lo[i] or x > hi[i]:
            return False
        for j in range(nobs[i]):
            if not (x <= obs[i, j, 0] or x >= obs[i, j, 1]):
                return False
        return True

    @njit(cache=False)
    def _succ(D, P, B, Del, roots, lo, hi, obs, nobs, stage, ds, kr, rd, rp,
              d0, psi0, b0, del0, root0, u, acc,
              max_steer, steer_rate, tau, wheelbase, kus_num, kus_den,
              a_lat, a_acc, a_brk, v_min):
        """Successor grid indices and stage cost; ok=False when inadmissible."""
        dt = ds * 1000000 // max(root0, v_min)
        target = u * max_steer // 1000
        lag = (target - del0) * min(dt, tau) // tau
        lim = steer_rate * dt // 1000000
        change = min(max(lag, -lim), lim)
        qi1 = _index_of(Del, del0 + change)
        del1 = Del[qi1]
        den = wheelbase + kus_num * b0 // kus_den
        if den <= 0:
            return False, 0, 0, 0, 0, 0
        kappa = del1 * 1000 // den
        pi1 = _index_of(P, psi0 + (kappa - kr) * ds // 1000)
        psi1 = P[pi1]
        di1 = _index_of(D, d0 + ((psi0 + psi1) // 2) * ds // 1000000)
        d1 = D[di1]
        bc = b0 + 2 * acc * ds
        if bc < 0:
            return False, 0, 0, 0, 0, 0
        bi1 = _index_of(B, bc)
        b1 = B[bi1]
        if not _lateral_ok(lo, hi, obs, nobs, stage + 1, d1):
            return False, 0, 0, 0, 0, 0
        if not _lateral_ok(lo, hi, obs, nobs, stage + 1, (d1 + d0) // 2):
            return False, 0, 0, 0, 0, 0
        if acc < -a_brk or acc > a_acc:
            return False, 0, 0, 0, 0, 0
        kp = kr + (psi1 - psi0) * 1000 // ds
        if b1 * abs(kp) > a_lat * 1000000000:
            return False, 0, 0, 0, 0, 0
        lng = a_acc if acc >= 0 else a_brk
        lat = b1 * abs(kp) // 1000000000
        if (lat * lng) * (lat * lng) + (acc * a_lat) * (acc * a_lat) > (a_lat * lng) * (a_lat * lng):
            return False, 0, 0, 0, 0, 0
        denom = root0 + roots[bi1]
        if denom <= 0:
            raise ValueError("zero-speed segment has no finite time")
        cost = (1000 * (-((-(2 * ds * 1000000)) // denom)) + abs(del1 - del0) // 1000
                + 5000 * abs(d1 - rd) + 5000 * abs(psi1 - rp) // 100)
        return True, di1, pi1, bi1, qi1, cost

    @njit(cache=False)
    def dp_kernel(D, P, B, Del, U, A, roots, Vterm, ds_arr, kroad, rd_arr, rp_arr,
                  lo, hi, obs, nobs, H, start_d, start_psi, start_b, start_delta, root_start,
                  max_steer, steer_rate, tau, wheelbase, kus_num, kus_den,
                  a_lat, a_acc, a_brk, v_min):
        Nd = D.size; Np = P.size; Nb = B.size; Nq = Del.size; Na = A.size
        N = Nd * Np * Nb * Nq
        V = Vterm.copy()
        Vn = np.full(N, INF, dtype=np.int64)
        args = np.full((H, N), -1, dtype=np.int64)
        for stage in range(H - 1, 0, -1):
            ds = ds_arr[stage]; kr = kroad[stage]; rd = rd_arr[stage]; rp = rp_arr[stage]
            for di in range(Nd):
                for pi in range(Np):
                    for bi in range(Nb):
                        for qi in range(Nq):
                            idx = ((di * Np + pi) * Nb + bi) * Nq + qi
                            best = INF
                            besti = -1
                            for ui in range(U.size):
                                for ai in range(Na):
                                    ok, di1, pi1, bi1, qi1, cost = _succ(
                                        D, P, B, Del, roots, lo, hi, obs, nobs, stage, ds, kr, rd, rp,
                                        D[di], P[pi], B[bi], Del[qi], roots[bi], U[ui], A[ai],
                                        max_steer, steer_rate, tau, wheelbase, kus_num, kus_den,
                                        a_lat, a_acc, a_brk, v_min)
                                    if not ok:
                                        continue
                                    nv = V[((di1 * Np + pi1) * Nb + bi1) * Nq + qi1]
                                    if nv >= INF:
                                        continue
                                    cand = cost + nv
                                    if cand < best:
                                        best = cand
                                        besti = ui * Na + ai
                            Vn[idx] = best
                            args[stage, idx] = besti
            tmp = V; V = Vn; Vn = tmp
            for k in range(N):
                Vn[k] = INF
        best = INF
        besti = -1
        for ui in range(U.size):
            for ai in range(Na):
                ok, di1, pi1, bi1, qi1, cost = _succ(
                    D, P, B, Del, roots, lo, hi, obs, nobs, 0, ds_arr[0], kroad[0], rd_arr[0], rp_arr[0],
                    start_d, start_psi, start_b, start_delta, root_start, U[ui], A[ai],
                    max_steer, steer_rate, tau, wheelbase, kus_num, kus_den,
                    a_lat, a_acc, a_brk, v_min)
                if not ok:
                    continue
                nv = V[((di1 * Np + pi1) * Nb + bi1) * Nq + qi1]
                if nv >= INF:
                    continue
                cand = cost + nv
                if cand < best:
                    best = cand
                    besti = ui * Na + ai
        return besti, best, args

    return dp_kernel


dp_kernel = _kernel_source() if NUMBA_AVAILABLE else None


def kernel_inputs(model):
    """Python-side arrays for :func:`dp_kernel` (all int64 / bool)."""
    from .model import COST_TIME_WEIGHT, V_MIN_MM_S, isqrt
    from .solver import _dense_index, dense_grid, dense_violations
    bad = dense_violations(model)
    if bad:
        raise OverflowError("instance exceeds int64-safe bounds: " + ", ".join(bad))
    state = dense_grid(model)
    D = np.asarray(model.offsets, dtype=np.int64)
    P = np.asarray(model.psis, dtype=np.int64)
    B = np.asarray(model.speeds, dtype=np.int64)
    Del = np.asarray(model.deltas, dtype=np.int64)
    U = np.asarray(model.steer_inputs, dtype=np.int64)
    A = np.asarray(model.acc_inputs, dtype=np.int64)
    H = model.horizon
    lo = np.array([model.lo(i) for i in range(H + 1)], dtype=np.int64)
    hi = np.array([model.hi(i) for i in range(H + 1)], dtype=np.int64)
    half = model.half_width + model.margin
    obs_rows = [model.obstacles_at(i) for i in range(H + 1)]
    nobs = np.array([len(r) for r in obs_rows], dtype=np.int64)
    obs = np.zeros((H + 1, max(1, int(nobs.max(initial=0))), 2), dtype=np.int64)
    for i, rows in enumerate(obs_rows):
        for j, (_s, lmin, lmax) in enumerate(rows):
            obs[i, j, 0] = lmin - half
            obs[i, j, 1] = lmax + half
    # Terminal values: model.terminal_cost on every dense state.
    lat_term = np.array([model.lateral_ok(H, int(d)) for d in D], dtype=np.bool_)
    di = np.searchsorted(D, state.d)
    term = (lat_term[di] & (np.abs(state.psi) <= model.terminal_max_psi)
            & (np.abs(state.delta) <= model.terminal_max_delta)
            & (state.b >= model.terminal_min_b) & (state.b <= model.terminal_cap()))
    Vterm = np.full(state.n, INF, dtype=np.int64)
    if model.terminal_distance <= 0:
        Vterm[term] = 0
    else:
        pos = term & (state.root > 0)
        num = np.int64(model.terminal_distance) * np.int64(1_000_000)
        Vterm[pos] = COST_TIME_WEIGHT * (-((-num) // state.root[pos]))
    ref = lambda xs, i: int(xs[i + 1]) if len(xs) > i + 1 else 0  # noqa: E731
    args = (D, P, B, Del, U, A, state.root_grid, Vterm,
            np.array([model.gap(i) for i in range(H)], dtype=np.int64),
            np.array([model.k[i] for i in range(H)], dtype=np.int64),
            np.array([ref(model.ref_d, i) for i in range(H)], dtype=np.int64),
            np.array([ref(model.ref_psi, i) for i in range(H)], dtype=np.int64),
            lo, hi, obs, nobs, np.int64(H),
            np.int64(model.start.d), np.int64(model.start.psi),
            np.int64(model.start.b), np.int64(model.start.delta),
            np.int64(isqrt(model.start.b)),
            np.int64(model.max_steer), np.int64(model.steer_rate),
            np.int64(model.tau), np.int64(model.wheelbase),
            np.int64(model.understeer_num), np.int64(model.understeer_den),
            np.int64(model.lon["a_lat"]), np.int64(model.lon["a_acc"]),
            np.int64(model.lon["a_brk"]), np.int64(V_MIN_MM_S))
    return state, args, _dense_index


def solve_numba(model):
    """Compiled exact DP.  Returns the same tuple as ``solve()``.

    ``(first successor, optimal cost, (u, a), diag)``; ``diag`` carries
    ``feasible``, ``plan`` (states 0..H) and ``plan_actions``.
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError(f"numba unavailable: {NUMBA_IMPORT_ERROR}")
    from .model import COST_TIME_WEIGHT
    state, args, dense_index = kernel_inputs(model)
    besti, cost, table = dp_kernel(*args)
    besti = int(besti)
    cost = int(cost)
    nact = len(model.steer_inputs) * len(model.acc_inputs)
    diag = {"states": state.n, "actions": nact, "feasible": besti >= 0}
    if besti < 0:
        if cost < INF:
            raise RuntimeError("numba kernel returned a value without an action")
        diag.update(first=None, plan=[], plan_actions=[])
        return None, None, None, diag
    na = len(model.acc_inputs)
    # Replay the deterministic optimal plan through the scalar reference model.
    plan = [model.start]
    plan_actions = []
    total = 0
    cur = model.start
    for stage in range(model.horizon):
        ai = besti if stage == 0 else int(table[stage, dense_index(model, cur)])
        if ai < 0:
            raise RuntimeError(f"numba plan has no action at stage {stage}")
        u, a = model.steer_inputs[ai // na], model.acc_inputs[ai % na]
        nxt = model.action_ok(cur, u, a)
        if nxt is None:
            raise RuntimeError(f"numba plan action {(u, a)} inadmissible at stage {stage}")
        total += model.action_cost(cur, nxt)
        plan.append(nxt)
        plan_actions.append((u, a))
        cur = nxt
    term = model.terminal_cost(cur)
    if term is None or total + COST_TIME_WEIGHT * term != cost:
        raise RuntimeError(f"numba value {cost} disagrees with scalar plan replay")
    first = plan[1]
    diag.update(first=first, plan=plan, plan_actions=plan_actions)
    return first, cost, plan_actions[0], diag
