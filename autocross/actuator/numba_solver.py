"""Compiled exact DP kernel for the actuator model.

Same integer transition and constraints as ``ActModel.next_state``/``actions``.
No state or action is pruned.  Terminal values and per-station lateral-validity
masks are precomputed by the Python caller, so the kernel contains only the
backward induction and the start-state action scan.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None

INF = 2 ** 62


@njit(cache=False)
def _index_of(vals, x):
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
def dp_kernel(D, P, B, Del, U, A, roots, root_start, Vterm,
              ds_arr, kroad, ok_arr,
              H, start_d, start_psi, start_b, start_delta,
              max_steer, steer_rate, tau, wheelbase,
              kus_num, kus_den, a_lat, a_acc, a_brk,
              v_min=1000):
    Nd = D.size; Np = P.size; Nb = B.size; Nq = Del.size
    N = Nd * Np * Nb * Nq
    V = Vterm.copy()
    Vn = np.full(N, INF, dtype=np.int64)

    for stage in range(H - 1, 0, -1):
        ds = ds_arr[stage]
        kr = kroad[stage]
        ok1 = ok_arr[stage + 1]
        for di in range(Nd):
            d0 = D[di]
            for pi in range(Np):
                psi0 = P[pi]
                for bi in range(Nb):
                    b0 = B[bi]
                    root0 = roots[bi]
                    for qi in range(Nq):
                        del0 = Del[qi]
                        idx = ((di * Np + pi) * Nb + bi) * Nq + qi
                        best = INF
                        v = root0
                        dt = ds * 1000000 // max(v, v_min)
                        lim = steer_rate * dt // 1000000
                        den = wheelbase + kus_num * b0 // kus_den
                        for ui in range(U.size):
                            target = U[ui] * max_steer // 1000
                            lag = (target - del0) * min(dt, tau) // tau
                            change = min(max(lag, -lim), lim)
                            del1 = Del[_index_of(Del, del0 + change)]
                            kappa = del1 * 1000 // den
                            psi1 = P[_index_of(P, psi0 + (kappa - kr) * ds // 1000)]
                            d1 = D[_index_of(D, d0 + ((psi0 + psi1) // 2) * ds // 1000000)]
                            dmid = (d0 + d1) // 2
                            di1 = _index_of(D, d1)
                            dimid = _index_of(D, dmid)
                            if not ok1[di1] or not ok1[dimid]:
                                continue
                            kp = kr + (psi1 - psi0) * 1000 // ds
                            for ai in range(A.size):
                                acc = A[ai]
                                if acc < -a_brk or acc > a_acc:
                                    continue
                                bc = b0 + 2 * acc * ds
                                if bc < 0:
                                    continue
                                b1 = B[_index_of(B, bc)]
                                lng = a_acc if acc >= 0 else a_brk
                                lat = b1 * abs(kp) // 1000000000
                                if lat > a_lat:
                                    continue
                                if (lat * lng) ** 2 + (acc * a_lat) ** 2 > (a_lat * lng) ** 2:
                                    continue
                                bi1 = _index_of(B, b1)
                                pi1 = _index_of(P, psi1)
                                qi1 = _index_of(Del, del1)
                                ni = ((di1 * Np + pi1) * Nb + bi1) * Nq + qi1
                                nv = V[ni]
                                if nv >= INF:
                                    continue
                                denom = root0 + roots[bi1]
                                if denom <= 0:
                                    continue
                                cost = (1000 * (-((-(2 * ds * 1000000)) // denom)) + abs(del1 - del0) // 1000
                                        + 5000 * abs(d1) + 5000 * abs(psi1) // 100)
                                cand = cost + nv
                                if cand < best:
                                    best = cand
                        Vn[idx] = best
        tmp = V; V = Vn; Vn = tmp
        for k in range(N):
            Vn[k] = INF

    ds0 = ds_arr[0]
    kr0 = kroad[0]
    v0 = root_start
    dt0 = ds0 * 1000000 // max(v0, v_min)
    lim0 = steer_rate * dt0 // 1000000
    den0 = wheelbase + kus_num * start_b // kus_den
    ok1 = ok_arr[1]
    best = INF; best_u = -1; best_a = -1
    best_t = (-1, -1, -1, -1, -1)
    for ui in range(U.size):
        target = U[ui] * max_steer // 1000
        lag = (target - start_delta) * min(dt0, tau) // tau
        change = min(max(lag, -lim0), lim0)
        del1 = Del[_index_of(Del, start_delta + change)]
        kappa = del1 * 1000 // den0
        psi1 = P[_index_of(P, start_psi + (kappa - kr0) * ds0 // 1000)]
        d1 = D[_index_of(D, start_d + ((start_psi + psi1) // 2) * ds0 // 1000000)]
        dmid = (start_d + d1) // 2
        di1 = _index_of(D, d1); dimid = _index_of(D, dmid)
        if not ok1[di1] or not ok1[dimid]:
            continue
        kp = kr0 + (psi1 - start_psi) * 1000 // ds0
        for ai in range(A.size):
            acc = A[ai]
            if acc < -a_brk or acc > a_acc:
                continue
            bc = start_b + 2 * acc * ds0
            if bc < 0:
                continue
            b1 = B[_index_of(B, bc)]
            lng = a_acc if acc >= 0 else a_brk
            lat = b1 * abs(kp) // 1000000000
            if lat > a_lat:
                continue
            if (lat * lng) ** 2 + (acc * a_lat) ** 2 > (a_lat * lng) ** 2:
                continue
            bi1 = _index_of(B, b1)
            qi1 = _index_of(Del, del1)
            ni = ((di1 * Np + _index_of(P, psi1)) * Nb + bi1) * Nq + qi1
            nv = V[ni]
            if nv >= INF:
                continue
            denom = root_start + roots[bi1]
            if denom <= 0:
                continue
            cost = (1000 * (-((-(2 * ds0 * 1000000)) // denom)) + abs(del1 - start_delta) // 1000
                    + 5000 * abs(d1) + 5000 * abs(psi1) // 100)
            cand = cost + nv
            if cand < best:
                best = cand; best_u = U[ui]; best_a = A[ai]
                best_t = (1, d1, psi1, b1, del1)
    return best_u, best_a, best, best_t


def solve_numba(model):
    """Build arrays and run the compiled kernel.  Returns the same tuple as solve()."""
    from .model import ActState, isqrt
    from .solver import dense_grid
    state = dense_grid(model)
    D = np.asarray(model.offsets, dtype=np.int64)
    P = np.asarray(model.psis, dtype=np.int64)
    B = np.asarray(model.speeds, dtype=np.int64)
    Del = np.asarray(model.deltas, dtype=np.int64)
    U = np.asarray(model.steer_inputs, dtype=np.int64)
    A = np.asarray(model.acc_inputs, dtype=np.int64)
    H = model.horizon
    ok = np.zeros((H + 1, D.size), dtype=np.bool_)
    for i in range(H + 1):
        for dj, d in enumerate(D):
            ok[i, dj] = model.lateral_ok(i, int(d))
    # Terminal values, vectorised over the dense state universe.
    di = np.repeat(np.arange(D.size), P.size * B.size * Del.size)
    pi = np.tile(np.repeat(np.arange(P.size), B.size * Del.size), D.size)
    bi = np.tile(np.repeat(np.arange(B.size), Del.size), D.size * P.size)
    qi = np.tile(np.arange(Del.size), D.size * P.size * B.size)
    term = (ok[H, di] & (np.abs(state.psi) <= model.terminal_max_psi)
            & (np.abs(state.delta) <= model.terminal_max_delta)
            & (state.b >= model.terminal_min_b) & (state.b <= model.terminal_cap()))
    Vterm = np.full(state.n, INF, dtype=np.int64)
    if model.terminal_distance <= 0:
        Vterm[term] = 0
    else:
        v = state.root
        pos = term & (v > 0)
        Vterm[pos] = 1000 * (-((-(np.int64(2) * model.terminal_distance * 1000000)) // (2 * v[pos])))
    ds_arr = np.array([model.gap(i) for i in range(H)], dtype=np.int64)
    kroad = np.array([model.k[i] for i in range(H)], dtype=np.int64)
    res = dp_kernel(D, P, B, Del, U, A, state.root_grid, np.int64(isqrt(model.start.b)),
                    Vterm, ds_arr, kroad, ok, H,
                    np.int64(model.start.d), np.int64(model.start.psi),
                    np.int64(model.start.b), np.int64(model.start.delta),
                    np.int64(model.max_steer), np.int64(model.steer_rate),
                    np.int64(model.tau), np.int64(model.wheelbase),
                    np.int64(model.understeer_num), np.int64(model.understeer_den),
                    np.int64(model.lon["a_lat"]), np.int64(model.lon["a_acc"]),
                    np.int64(model.lon["a_brk"]))
    u, a, cost, t = int(res[0]), int(res[1]), int(res[2]), res[3]
    if u < 0 or cost >= INF:
        return None, None, None, {"states": state.n, "feasible": False}
    first = ActState(int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4]))
    return first, cost, (u, a), {"states": state.n, "feasible": True, "first": first,
                                 "actions": int(U.size * A.size)}
