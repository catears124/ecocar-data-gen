"""Open-loop manoeuvres that measure the published handling limits of the configured vehicle
(max accel, max decel, lateral grip) on a given backend. Use it after changing the vehicle
blueprint or physics overrides, then copy the numbers into config/sim.yaml."""
from __future__ import annotations

import math

import numpy as np

from .track import build

PAD = {"name": "calibration_pad", "closed": False, "width": 120.0, "cone_spacing": 400.0, "start_offset": 8.0,
       "finish_runoff": 20.0, "segments": [{"type": "straight", "length": 260.0}]}


def _drive(be, n, steer, throttle, brake):
    out = []
    for _ in range(n):
        be.apply(steer, throttle, brake)
        out.append(be.tick())
    return out


def calibrate(cfg: dict, backend_name: str) -> dict:
    from .runner import make_backend
    dt = float(cfg["sim"]["physics_dt"])
    track = build(PAD)
    rng = np.random.default_rng(0)
    res = {}

    def fresh(y_offset=0.0):
        be = make_backend(backend_name, cfg)
        if hasattr(be, "cfg"):
            be.cfg = dict(be.cfg, spawn_cones=False)
        pp = be.setup(track, cfg["vehicle"], {}, rng)
        x, y, yaw = track.spawn_pose()
        be.reset(x, y + y_offset, yaw)
        return be, pp

    # 1) straight-line acceleration and braking
    be, pp = fresh()
    try:
        acc = []
        while len(acc) < int(9.0 / dt) and (not acc or acc[-1].speed < 22.0):  # stay on the pad
            acc += _drive(be, 1, 0.0, 1.0, 0.0)
        v = np.array([s.speed for s in acc])
        t = np.array([s.t for s in acc])
        i0, i1 = np.searchsorted(v, 2.0), np.searchsorted(v, 15.0)
        res["max_accel"] = float((v[i1] - v[i0]) / (t[i1] - t[i0])) if i1 < len(v) and i1 > i0 else float("nan")
        brk = _drive(be, int(5.0 / dt), 0.0, 0.0, 1.0)
        v = np.array([s.speed for s in brk])
        t = np.array([s.t for s in brk])
        j0 = int(np.argmax(v < v[0] - 1.0))
        j1 = int(np.argmax(v < 3.0)) if (v < 3.0).any() else len(v) - 1
        res["max_decel"] = float((v[j0] - v[j1]) / max(t[j1] - t[j0], 1e-6))
    finally:
        be.close()

    # 2) steady-state circle with slowly increasing speed -> peak lateral acceleration
    be, pp = fresh(y_offset=-50.0)
    try:
        steer = 0.35
        v_tgt, i_err, ay_hist = 4.0, 0.0, []
        n = int(80.0 / dt)
        for _ in range(n):
            st = be.tick()
            v_tgt += 0.3 * dt
            err = v_tgt - st.speed
            i_err = float(np.clip(i_err + err * dt, -3, 3))
            u = 0.6 * err + 0.3 * i_err
            be.apply(steer, max(min(u, 1.0), 0.0), max(min(-u, 1.0), 0.0))
            ay_hist.append((st.t, st.speed, abs(st.ay), abs(math.degrees(math.atan2(st.vy, max(st.vx, 0.1))))))
        a = np.array(ay_hist)
        win = max(int(1.0 / dt), 1)
        ay_f = np.convolve(a[:, 2], np.ones(win) / win, mode="same")
        # steady-state limit: highest smoothed lateral accel before the vehicle first slides
        slide = np.flatnonzero(a[:, 3] > 8.0)
        end = int(slide[0]) if len(slide) else len(a) - win
        k = int(np.argmax(ay_f[win:max(end, win + 1)])) + win
        res["max_lateral_accel"] = float(ay_f[k])
        res["mu_estimate"] = float(ay_f[k] / 9.81)
        res["speed_at_limit"] = float(a[k, 1])
        res["sideslip_at_limit_deg"] = float(a[k, 3])
        res["turn_radius_at_limit_m"] = float(a[k, 1] ** 2 / max(ay_f[k], 1e-6))
    finally:
        be.close()
    res.update({"mass": pp.mass, "wheelbase": pp.wheelbase, "track_width": pp.track_width, "length": pp.length,
                "width": pp.width, "cg_to_front": pp.cg_to_front, "max_steer_deg": math.degrees(pp.max_steer)})
    return res
