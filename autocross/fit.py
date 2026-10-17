"""Fit the bicycle backend to CARLA telemetry, and compare the two backends lap for lap.

The fit replays logged CARLA runs in short windows: each window starts the bicycle model from
the logged state, feeds it the logged (actuator-model output) commands, and scores how far
its body velocities, yaw rate and wheel angle drift from CARLA's. Nelder-Mead in log-parameter
space minimises the total error. Short windows keep the comparison local, so a mismatch in
one corner cannot snowball into a meaningless trajectory difference."""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import numpy as np

from .backends.bicycle import DEFAULTS, FIT_KEYS, derive, model_params, substep

GEOMETRY_KEYS = ("mass", "wheelbase", "cg_to_front", "track_width", "length", "width")
# error scale per channel: vx [m/s], vy [m/s], yaw rate [rad/s], wheel angle [rad]
SCALES = np.array([0.25, 0.1, 0.03, 0.01])


def _telemetry(run_dir: Path) -> dict:
    with open(run_dir / "telemetry.csv", newline="") as f:
        rd = csv.reader(f)
        head = next(rd)
        cols = ("t", "vx", "vy", "yaw_rate", "ax", "steer_angle", "app_steer", "app_throttle", "app_brake")
        idx = [head.index(c) for c in cols]
        rows = [[float(r[i]) if r[i] != "" else math.nan for i in idx] for r in rd]
    a = np.array(rows, dtype=float).reshape(-1, len(cols))
    return {c: a[:, i] for i, c in enumerate(cols)}


def find_runs(paths, backend="carla") -> list[Path]:
    """Run folders (with telemetry.csv and summary.json) under ``paths`` recorded on ``backend``."""
    out = []
    for p in map(Path, paths):
        for s in sorted([p / "summary.json"] if (p / "summary.json").exists() else p.rglob("summary.json")):
            if (s.parent / "telemetry.csv").exists() and json.loads(s.read_text()).get("backend") == backend:
                out.append(s.parent)
    return out


class Windows:
    """Telemetry cut into windows of ``n`` physics ticks, stacked for batched simulation."""

    def __init__(self, run_dirs, window_s=1.5, stride_s=0.75, min_speed=0.5):
        x0, cmd, tgt, self.sources = [], [], [], []
        self.dt = self.max_steer = None
        self.geometry = {}
        for rd in map(Path, run_dirs):
            s = json.loads((rd / "summary.json").read_text())
            dt = float(s["config"]["sim"]["physics_dt"])
            if self.dt is not None and abs(dt - self.dt) > 1e-9:
                raise ValueError(f"{rd}: physics_dt {dt} differs from the other runs ({self.dt})")
            self.dt = dt
            self.max_steer = math.radians(float(s["config"]["vehicle"]["max_steer_deg"]))
            self.geometry = {k: float(s["vehicle"][k]) for k in GEOMETRY_KEYS}
            tl = _telemetry(rd)
            delta = tl["steer_angle"]
            if not np.isfinite(delta).all():  # wheel angle not measured: no steering target
                delta = np.full_like(delta, np.nan)
            n, stride = int(round(window_s / dt)), max(int(round(stride_s / dt)), 1)
            self.n = n
            for k0 in range(1, len(tl["t"]) - n, stride):
                sl = slice(k0, k0 + n)
                if np.nanmax(np.hypot(tl["vx"][sl], tl["vy"][sl])) < min_speed:
                    continue
                d0 = delta[k0 - 1] if np.isfinite(delta[k0 - 1]) else tl["app_steer"][k0 - 1] * self.max_steer
                x0.append((tl["vx"][k0 - 1], tl["vy"][k0 - 1], tl["yaw_rate"][k0 - 1], d0, tl["ax"][k0 - 1]))
                cmd.append(np.stack([tl["app_steer"][sl], tl["app_throttle"][sl], tl["app_brake"][sl]]))
                tgt.append(np.stack([tl["vx"][sl], tl["vy"][sl], tl["yaw_rate"][sl], delta[sl]]))
                self.sources.append((rd.name, float(tl["t"][k0])))
        if not x0:
            raise ValueError("no usable telemetry windows (are these CARLA runs with the car moving?)")
        self.x0 = np.array(x0).T          # (5, N)
        self.cmd = np.array(cmd)          # (N, 3, n)
        self.tgt = np.array(tgt)          # (N, 4, n)
        self.sub = max(int(round(self.dt / 0.0025)), 1)

    def __len__(self):
        return self.x0.shape[1]

    def simulate(self, p: dict) -> np.ndarray:
        """Bicycle response for every window, shape (N, 4, n): vx, vy, yaw rate, wheel angle."""
        vx, vy, r, delta, ax = (v.copy() for v in self.x0)
        z = np.zeros_like(vx)
        s = (z, z, z, vx, vy, r, delta)
        h = self.dt / self.sub
        out = np.empty((len(self), 4, self.n))
        for j in range(self.n):
            c = (self.cmd[:, 0, j], self.cmd[:, 1, j], self.cmd[:, 2, j])
            v0x, v0y = s[3], s[4]
            for _ in range(self.sub):
                s = substep(s, c, ax, p, h)
            ax = (s[3] - v0x) / self.dt - s[5] * s[4]
            out[:, 0, j], out[:, 1, j], out[:, 2, j], out[:, 3, j] = s[3], s[4], s[5], s[6]
        return out

    def errors(self, p: dict) -> np.ndarray:
        """RMS error per channel, in physical units (nan channels ignored)."""
        e = self.simulate(p) - self.tgt
        return np.sqrt(np.nanmean(e ** 2, axis=(0, 2)))

    def cost(self, p: dict) -> float:
        e = self.errors(p)
        e = np.where(np.isfinite(e), e, 0.0)
        return float(np.sum((e / SCALES) ** 2))


def _nelder_mead(f, x0, step=0.15, max_evals=1200, tol=1e-6):
    n = len(x0)
    pts = [np.asarray(x0, float)] + [np.asarray(x0, float) + step * np.eye(n)[i] for i in range(n)]
    vals = [f(p) for p in pts]
    evals = n + 1
    while evals < max_evals:
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) <= tol * max(abs(vals[0]), 1e-12):
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1])
        fr = f(xr)
        evals += 1
        if fr < vals[0]:
            xe = c + 2.0 * (c - pts[-1])
            fe = f(xe)
            evals += 1
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c) if fr >= vals[-1] else c + 0.5 * (xr - c)
            fc = f(xc)
            evals += 1
            if fc < min(fr, vals[-1]):
                pts[-1], vals[-1] = xc, fc
            else:  # shrink towards the best point
                for i in range(1, n + 1):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0])
                    vals[i] = f(pts[i])
                evals += n
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def fit(windows: Windows, vehicle_cfg: dict, keys=FIT_KEYS, max_evals=1500, restarts=2, verbose=True) -> dict:
    """Fit ``keys`` of ``vehicle.bicycle`` to the telemetry. Geometry (mass, wheelbase, CG) is
    taken from the CARLA vehicle as recorded in the runs. Returns the new bicycle section."""
    base = model_params(vehicle_cfg)
    base.update(windows.geometry)
    base["max_steer"] = windows.max_steer
    base = derive(base)
    start = np.array([base[k] for k in keys], dtype=float)
    floor = 1e-4

    def params(z):
        p = dict(base)
        p.update({k: max(s, floor) * math.exp(v) for k, s, v in zip(keys, start, z)})
        return derive(p)

    def f(z):
        c = windows.cost(params(z))
        return c if math.isfinite(c) else 1e12

    before = windows.errors(base)
    z, best = np.zeros(len(keys)), f(np.zeros(len(keys)))
    if verbose:
        print(f"[fit] {len(windows)} windows of {windows.n * windows.dt:.2f} s, fitting {len(keys)} parameters")
        print(f"[fit] start cost {best:.3f}")
    for i in range(restarts + 1):  # restarting Nelder-Mead from its own optimum escapes collapsed simplexes
        z_new, c_new = _nelder_mead(f, z, step=0.15 if i == 0 else 0.05, max_evals=max_evals)
        if verbose:
            print(f"[fit] pass {i + 1}: cost {c_new:.3f}")
        if c_new >= best * (1 - 1e-4):
            if c_new < best:
                z, best = z_new, c_new
            break
        z, best = z_new, c_new
    p = params(z)
    after = windows.errors(p)
    if verbose:
        names = ("vx m/s", "vy m/s", "yaw rate rad/s", "wheel angle rad")
        for nm, b, a in zip(names, before, after):
            print(f"[fit]   RMS {nm:<16} {b:8.4f} -> {a:8.4f}")
    out = {k: float(p[k]) for k in DEFAULTS}
    return out


def write_bicycle_section(path, values: dict) -> None:
    """Update ``vehicle.bicycle`` values in a sim.yaml in place, keeping comments and layout."""
    lines = Path(path).read_text().splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if re.match(r"^\s+bicycle:", ln))
    indent = len(lines[start]) - len(lines[start].lstrip()) + 2
    end = start + 1
    while end < len(lines) and (not lines[end].strip() or len(lines[end]) - len(lines[end].lstrip()) >= indent):
        end += 1
    while lines[end - 1].strip() == "":
        end -= 1
    todo = dict(values)
    for i in range(start + 1, end):
        m = re.match(r"^(\s+)(\w+):(\s*)([^#\n]*?)(\s*)(#.*)?$", lines[i].rstrip("\n"))
        if m and m.group(2) in todo:
            v = _fmt(todo.pop(m.group(2)))
            pad = m.group(5) or ""
            if m.group(6):  # keep the comment column where possible
                pad = " " * max(len(m.group(4)) + len(pad) - len(v), 1)
            lines[i] = f"{m.group(1)}{m.group(2)}:{m.group(3) or ' '}{v}{pad}{m.group(6) or ''}\n"
    lines[end:end] = [f"{' ' * indent}{k}: {_fmt(v)}\n" for k, v in todo.items()]
    Path(path).write_text("".join(lines))


def _fmt(v: float) -> str:
    return f"{v:.1f}" if abs(v) >= 100 else f"{v:.4g}"


def compare_summaries(ref: dict, dev: dict) -> dict:
    """Lap-time and speed-profile agreement between a reference (CARLA) and a dev (bicycle) run."""
    def info(s):
        c = s["metrics"]["completion"]
        return c["finished"], c["time_s"]
    fr, tr = info(ref)
    fd, td = info(dev)
    res = {"ref_time_s": tr, "dev_time_s": td, "ref_finished": fr, "dev_finished": fd,
           "diff_pct": 100.0 * (td - tr) / tr if fr and fd else None}
    try:  # speed vs progress, so a time offset does not show up as a speed error
        a, b = (_telemetry_cols(Path(s["out_dir"]), ("progress", "speed")) for s in (ref, dev))
        lo, hi = max(a[0].min(), b[0].min()), min(a[0].max(), b[0].max())
        grid = np.linspace(lo, hi, 500)
        va = np.interp(grid, *_monotone(*a))
        vb = np.interp(grid, *_monotone(*b))
        res["speed_rms_mps"] = float(np.sqrt(np.mean((va - vb) ** 2)))
    except (KeyError, ValueError, OSError):
        res["speed_rms_mps"] = None
    return res


def _telemetry_cols(run_dir: Path, cols):
    with open(run_dir / "telemetry.csv", newline="") as f:
        rd = csv.reader(f)
        head = next(rd)
        idx = [head.index(c) for c in cols]
        a = np.array([[float(r[i]) for i in idx] for r in rd])
    return tuple(a[:, i] for i in range(len(cols)))


def _monotone(x, y):
    keep = np.concatenate([[True], np.diff(np.maximum.accumulate(x)) > 1e-6])
    return np.maximum.accumulate(x)[keep], y[keep]
