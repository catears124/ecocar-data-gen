"""Batch rollout of candidate command sequences through the bicycle simulator.

The transition function is the harness's own
:func:`autocross.backends.bicycle.substep`, so a certificate produced from this
module is a statement about *that* simulator, not about a separate surrogate.
The implementation is vectorised over candidates: every candidate is a row of
the state/command arrays, and ``substep`` is evaluated once per physics
substep for all rows.

Scope: one control period is two 25 ms physics ticks (the harness setting
``physics_dt=0.025``, ``control_hz=20``).  The candidate set is finite and
declared in the certificate.  This module does not discretise the continuous
control set; see ``docs/simulation_optimality.md`` for the bounded-error
qualification.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autocross.backends.bicycle import model_params, substep

PHYSICS_DT = 0.025
TICKS_PER_CONTROL = 2
SUBSTEPS = 10


@dataclass(frozen=True)
class SimInstance:
    """A fully declared simulation decision problem at one observation time."""

    vehicle: dict
    state0: tuple[float, float, float, float, float, float, float]
    app0: tuple[float, float, float]
    acc0: tuple[float, float]
    candidates: tuple[tuple[tuple[float, float, float], ...], ...]
    preview_cl: tuple[tuple[float, float], ...]
    preview_s: tuple[float, ...]
    preview_wl: tuple[float, ...]
    preview_wr: tuple[float, ...]
    preview_k: tuple[float, ...]
    obstacles: tuple[tuple[float, float, float], ...]
    half_width: float
    half_length: float
    margin: float
    t_tail: float = 0.5
    max_steer_rate: float = 2.0
    baseline_index: int = -1

    @property
    def horizon(self) -> int:
        return len(self.candidates[0])


def _normals(cl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.gradient(cl, axis=0)
    n = np.linalg.norm(t, axis=1)
    if np.any(n <= 1e-12):
        t = np.tile(np.array([1.0, 0.0]), (cl.shape[0], 1))
        n[:] = 1.0
    t = t / n[:, None]
    return t, np.column_stack((-t[:, 1], t[:, 0]))


def project_batch(cl: np.ndarray, s_grid: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-station projection of a batch of positions.

    Returns ``(s, d, j)`` in the preview frame (metres).  The nearest vertex is
    used, then the chord tangent gives the along/lateral decomposition.  This is
    the same convention as the controller's road model and is deterministic.
    """
    tangent, normal = _normals(cl)
    rel = pos[:, None, :] - cl[None, :, :]
    d2 = np.einsum("nmk,nmk->nm", rel, rel)
    j = np.argmin(d2, axis=1)
    r = rel[np.arange(pos.shape[0]), j]
    tn = tangent[j]
    nn = normal[j]
    along = np.einsum("nk,nk->n", r, tn)
    lat = np.einsum("nk,nk->n", r, nn)
    s = s_grid[j] + along
    return s, lat, j


def rollout(inst: SimInstance) -> dict:
    """Roll every declared candidate through the simulator.

    Returns final states, control-period boundary states, the per-boundary
    ``(s, d, j)`` projection, and the integer objective tuples
    ``(violation_mm, -progress_mm, effort_milli)``.
    """
    p = model_params(inst.vehicle, {})
    cl = np.asarray(inst.preview_cl, dtype=float)
    s_grid = np.asarray(inst.preview_s, dtype=float)
    wl = np.asarray(inst.preview_wl, dtype=float)
    wr = np.asarray(inst.preview_wr, dtype=float)
    kk = np.asarray(inst.preview_k, dtype=float)
    obs = np.asarray(inst.obstacles, dtype=float).reshape(-1, 3)
    C = np.asarray(inst.candidates, dtype=float)       # (N, H, 3)
    N, H, _ = C.shape
    state = [np.full(N, float(v)) for v in inst.state0]
    app = np.tile(np.asarray(inst.app0, dtype=float), (N, 1))
    acc = np.full(N, float(inst.acc0[0]))
    rate = float(inst.max_steer_rate) * PHYSICS_DT
    sub_h = PHYSICS_DT / SUBSTEPS
    viol = np.zeros(N, dtype=np.int64)
    effort = np.zeros(N, dtype=np.int64)
    prev = np.tile(np.asarray(inst.app0, dtype=float), (N, 1))
    boundaries: list[dict] = []
    for k in range(H):
        app[:, 0] = np.clip(app[:, 0] + np.clip(C[:, k, 0] - app[:, 0], -rate, rate), -1.0, 1.0)
        app[:, 1] = C[:, k, 1]
        app[:, 2] = C[:, k, 2]
        # Declared tie-break: command change from the previous control period.
        effort += np.rint((np.abs(app[:, 0] - prev[:, 0])
                           + np.abs(app[:, 1] - prev[:, 1])
                           + np.abs(app[:, 2] - prev[:, 2])) * 1000.0).astype(np.int64)
        prev = app.copy()
        for _tick in range(TICKS_PER_CONTROL):
            v0x, v0y = state[3].copy(), state[4].copy()
            cmd = (app[:, 0], app[:, 1], app[:, 2])
            for _ in range(SUBSTEPS):
                state = list(substep(tuple(state), cmd, acc, p, sub_h))
            dvx = (state[3] - v0x) / PHYSICS_DT
            dvy = (state[4] - v0y) / PHYSICS_DT
            acc = dvx - state[5] * state[4]
        # The preview is in the observation's vehicle frame; the simulator
        # state is in world coordinates.  Rotate before projecting.
        x0, y0, yaw0 = inst.state0[0], inst.state0[1], inst.state0[2]
        relx = state[0] - x0
        rely = state[1] - y0
        cy, sy = np.cos(yaw0), np.sin(yaw0)
        pos = np.column_stack((relx * cy + rely * sy, -relx * sy + rely * cy))
        s, d, j = project_batch(cl, s_grid, pos)
        viol += _body_violation(inst, state, cl, s_grid, wl, wr, obs)
        boundaries.append({"s": s.copy(), "d": d.copy(), "j": j.copy(),
                           "world": np.column_stack((state[0], state[1]))})
    vx = state[3]
    s_final = boundaries[-1]["s"]
    j_final = boundaries[-1]["j"]
    # Terminal viability: the final speed must be compatible with the tightest
    # visible curvature ahead (a braking envelope over the preview).
    a_lat = 0.80 * float(p["mu"]) * 9.81
    v_cap = np.empty(N, dtype=float)
    for i in range(N):
        j = int(j_final[i])
        ahead = np.abs(kk[j: min(j + 16, kk.size)])
        kmax = float(ahead.max()) if ahead.size else 0.0
        v_cap[i] = float(np.sqrt(a_lat / kmax)) if kmax > 1e-6 else float(np.sqrt(a_lat * 1e3))
    over = np.maximum(0.0, vx * vx - v_cap * v_cap)
    viol += np.rint(over * 100.0).astype(np.int64)
    progress = np.rint((s_final + vx * inst.t_tail) * 1000.0).astype(np.int64)
    costs = [(int(viol[i]), -int(progress[i]), int(round(effort[i]))) for i in range(N)]
    return {"costs": costs, "final": state, "boundaries": boundaries,
            "boundary_world": [b["world"] for b in boundaries],
            "violation": viol, "progress": progress}


@dataclass(frozen=True)
class Policy:
    """A declared closed-loop candidate: pure pursuit + curvature speed cap."""
    lookahead: float
    speed_frac: float
    gain: float = 1.0


def _body_violation(inst: SimInstance, state, cl, s_grid, wl, wr, obs, margin: float = 0.10) -> np.ndarray:
    """Corridor/obstacle violation of the four body corners (mm), vectorised.

    Checking only the CG lateral offset lets a yawed car put a front corner
    over the cone line while the CG is still inside.  The harness scores wheel
    points and the body OBB, so the certificate model checks the body corners.
    """
    x0, y0, yaw0 = inst.state0[0], inst.state0[1], inst.state0[2]
    relx = state[0] - x0
    rely = state[1] - y0
    cy0, sy0 = np.cos(yaw0), np.sin(yaw0)
    px = relx * cy0 + rely * sy0
    py = -relx * sy0 + rely * cy0
    yaw = state[2] - yaw0
    c, sn = np.cos(yaw), np.sin(yaw)
    pts = []
    for lx, ly in ((inst.half_length, inst.half_width),
                   (inst.half_length, -inst.half_width),
                   (-inst.half_length, inst.half_width),
                   (-inst.half_length, -inst.half_width)):
        pts.append(np.column_stack((px + c * lx - sn * ly, py + sn * lx + c * ly)))
    pos = np.concatenate(pts, axis=0)
    s_c, d_c, j_c = project_batch(cl, s_grid, pos)
    n = state[0].size
    s_c = s_c.reshape(4, n).T
    d_c = d_c.reshape(4, n).T
    j_c = j_c.reshape(4, n).T
    lo = -wr[j_c] + margin
    hi = wl[j_c] - margin
    v = (np.maximum(0.0, d_c - hi) + np.maximum(0.0, lo - d_c)).sum(axis=1) * 1000.0
    for dist, lmin, lmax in obs:
        inside_s = np.abs(s_c - dist) <= (inst.half_length + 0.15)
        inside_l = (d_c >= lmin) & (d_c <= lmax)
        v += (inside_s & inside_l).sum(axis=1) * 100.0
    return np.rint(v).astype(np.int64)


def _project_step(inst: SimInstance, state, cl, s_grid):
    x0, y0, yaw0 = inst.state0[0], inst.state0[1], inst.state0[2]
    relx = state[0] - x0
    rely = state[1] - y0
    cy, sy = np.cos(yaw0), np.sin(yaw0)
    pos = np.column_stack((relx * cy + rely * sy, -relx * sy + rely * cy))
    return project_batch(cl, s_grid, pos) + (pos,)


def rollout_policies(inst: SimInstance, policies: tuple[Policy, ...], *,
                     max_accel: float = 5.2, max_decel: float = 9.0) -> dict:
    """Closed-loop rollout of a finite declared policy set in the simulator.

    Each policy is a pure-pursuit steering law on the preview centreline plus a
    curvature speed cap.  The certificate declares the policy list and the
    realized command sequence; the simulator transition is the harness's own.
    """
    p = model_params(inst.vehicle, {})
    cl = np.asarray(inst.preview_cl, dtype=float)
    s_grid = np.asarray(inst.preview_s, dtype=float)
    wl = np.asarray(inst.preview_wl, dtype=float)
    wr = np.asarray(inst.preview_wr, dtype=float)
    kk = np.asarray(inst.preview_k, dtype=float)
    obs = np.asarray(inst.obstacles, dtype=float).reshape(-1, 3)
    N, H = len(policies), inst.horizon
    state = [np.full(N, float(v)) for v in inst.state0]
    app = np.tile(np.asarray(inst.app0, dtype=float), (N, 1))
    acc = np.full(N, float(inst.acc0[0]))
    rate = float(inst.max_steer_rate) * PHYSICS_DT
    sub_h = PHYSICS_DT / SUBSTEPS
    max_steer = float(p["max_steer"])
    a_lat = 0.75 * float(p["mu"]) * 9.81
    L = np.asarray([q.lookahead for q in policies], dtype=float)
    frac = np.asarray([q.speed_frac for q in policies], dtype=float)
    gain = np.asarray([q.gain for q in policies], dtype=float)
    viol = np.zeros(N, dtype=np.int64)
    effort = np.zeros(N, dtype=np.int64)
    prev = np.tile(np.asarray(inst.app0, dtype=float), (N, 1))
    realized = np.zeros((N, H, 3), dtype=float)
    boundaries: list[dict] = []
    for k in range(H):
        s_cur, d_cur, j_cur, pos = _project_step(inst, state, cl, s_grid)
        yaw_v = np.arctan2(np.sin(state[2] - inst.state0[2]), np.cos(state[2] - inst.state0[2]))
        s_t = np.clip(s_cur + L, s_grid[0], s_grid[-1])
        tx = np.interp(s_t, s_grid, cl[:, 0])
        ty = np.interp(s_t, s_grid, cl[:, 1])
        dx, dy = tx - pos[:, 0], ty - pos[:, 1]
        cy, sy = np.cos(yaw_v), np.sin(yaw_v)
        dxb = dx * cy + dy * sy
        dyb = -dx * sy + dy * cy
        den = np.maximum(dxb * dxb + dyb * dyb, 1e-6)
        delta = np.arctan2(2.0 * L * dyb, den)
        steer = np.clip(delta / max(max_steer, 1e-6) * gain, -1.0, 1.0)
        kt = np.interp(s_t, s_grid, kk)
        v_cap = np.sqrt(a_lat / np.maximum(np.abs(kt), 1e-4)) * frac
        a_des = np.clip(0.6 * (v_cap - state[3]), -max_decel, max_accel)
        thr = np.maximum(a_des, 0.0) / max(max_accel, 1e-6)
        brk = np.maximum(-a_des, 0.0) / max(max_decel, 1e-6)
        # Non-overlapping pedals.
        thr = np.where(brk > 0.0, 0.0, thr)
        cmd_now = np.column_stack((steer, thr, brk))
        realized[:, k, :] = cmd_now
        app[:, 0] = np.clip(app[:, 0] + np.clip(cmd_now[:, 0] - app[:, 0], -rate, rate), -1.0, 1.0)
        app[:, 1] = cmd_now[:, 1]
        app[:, 2] = cmd_now[:, 2]
        effort += np.rint((np.abs(app[:, 0] - prev[:, 0]) + np.abs(app[:, 1] - prev[:, 1])
                           + np.abs(app[:, 2] - prev[:, 2])) * 1000.0).astype(np.int64)
        prev = app.copy()
        for _tick in range(TICKS_PER_CONTROL):
            v0x, v0y = state[3].copy(), state[4].copy()
            cmd = (app[:, 0], app[:, 1], app[:, 2])
            for _ in range(SUBSTEPS):
                state = list(substep(tuple(state), cmd, acc, p, sub_h))
            dvx = (state[3] - v0x) / PHYSICS_DT
            dvy = (state[4] - v0y) / PHYSICS_DT
            acc = dvx - state[5] * state[4]
        s, d, j, pos = _project_step(inst, state, cl, s_grid)
        viol += _body_violation(inst, state, cl, s_grid, wl, wr, obs)
        boundaries.append({"s": s.copy(), "d": d.copy(), "j": j.copy(),
                           "world": np.column_stack((state[0], state[1]))})
    vx = state[3]
    s_final = boundaries[-1]["s"]
    j_final = boundaries[-1]["j"]
    v_cap = np.empty(N, dtype=float)
    for i in range(N):
        j = int(j_final[i])
        ahead = np.abs(kk[j: min(j + 16, kk.size)])
        kmax = float(ahead.max()) if ahead.size else 0.0
        v_cap[i] = float(np.sqrt(a_lat / kmax)) if kmax > 1e-6 else float(np.sqrt(a_lat * 1e3))
    over = np.maximum(0.0, vx * vx - v_cap * v_cap)
    viol += np.rint(over * 100.0).astype(np.int64)
    progress = np.rint((s_final + vx * inst.t_tail) * 1000.0).astype(np.int64)
    costs = [(int(viol[i]), -int(progress[i]), int(round(effort[i]))) for i in range(N)]
    return {"costs": costs, "final": state, "boundaries": boundaries,
            "boundary_world": [b["world"] for b in boundaries],
            "violation": viol, "progress": progress, "realized": realized,
            "policies": policies}


def first_command(inst: SimInstance, index: int) -> tuple[float, float, float]:
    cmd = inst.candidates[int(index)][0]
    return float(cmd[0]), float(cmd[1]), float(cmd[2])


__all__ = ["SimInstance", "rollout", "first_command", "project_batch", "PHYSICS_DT",
           "TICKS_PER_CONTROL", "SUBSTEPS"]
