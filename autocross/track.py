"""Autocross course definition: YAML <-> centerline/width corridor, cones, gates, obstacles.

Course YAML (all lengths in metres, angles in degrees, + = left turn)::

    name: practice_basic
    closed: false          # open = point-to-point (classic autocross); closed = lap course
    laps: 1                # closed courses only
    width: 5.0             # default corridor width
    cone_spacing: 5.0      # nominal cone spacing on straights (denser in corners)
    start_offset: 8.0      # open: start line distance from spawn; closed: spawn distance behind start/finish
    finish_runoff: 35.0    # open only: braking zone after the finish line
    origin: [0, 0, 0]      # x, y, yaw of the first segment
    segments:              # either `segments` ...
      - {type: straight, length: 40}
      - {type: arc, radius: 18, angle: 90, width: 6}
      - {type: slalom, cycles: 3, wavelength: 24, swing: 3}
    centerline: [[x, y], [x, y, width], ...]   # ... or explicit points
    obstacles:             # unmapped orange cones, only visible through perception
      - {s: 120, offset: 1.0, width: 2.0}
    extra_cones: [{x: 10, y: 2, color: orange}]
    reference_time: 42.0   # optional organiser override of the scoring reference time
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from . import geometry as g
from .api import CONE_LEFT, CONE_ORANGE, CONE_RIGHT, TrackInfo, VehicleParams

DS = 0.5  # centerline sample spacing [m]
MARKER_OFFSETS = (0.6, 1.4)  # start/finish marker distances outside the line ends [m]
MIN_CONE_GAP = 0.5  # closest two physical cones may stand [m]; closer props collide at spawn
COLOR_NAMES = {"left": CONE_LEFT, "blue": CONE_LEFT, "right": CONE_RIGHT, "yellow": CONE_RIGHT,
               "orange": CONE_ORANGE}


@dataclass
class Track:
    name: str
    closed: bool
    center: np.ndarray
    wl: np.ndarray
    wr: np.ndarray
    laps: int = 1
    cone_spacing: float = 5.0
    start_offset: float = 8.0
    finish_runoff: float = 35.0
    obstacles: list = field(default_factory=list)
    extra_cones: list = field(default_factory=list)
    reference_time: float | None = None
    source: dict = field(default_factory=dict)

    def __post_init__(self):
        self.line = g.Polyline(self.center, self.closed)
        self.length = self.line.length
        self.normals = _normals(self.center, self.closed)
        self.left = self.center + self.normals * self.wl[:, None]
        self.right = self.center - self.normals * self.wr[:, None]
        self.station = self.line.s[: len(self.center)]
        self.heading = g.headings(self.center, self.closed)
        self.kappa = self._smoothed_curvature()
        self.cones, self.cone_colors = self._place_cones()
        self.obstacle_cones = self._place_obstacles()

    # ----------------------------------------------------------------- ground-truth road geometry
    def _smoothed_curvature(self, window_m: float = 4.0) -> np.ndarray:
        c2 = g.resample(self.center, 2.0, self.closed)
        s2 = g.arclength(c2, self.closed)[: len(c2)]
        k2 = g.curvature(c2, self.closed)
        k = np.interp(self.station, s2, k2, period=self.length if self.closed else None)
        n = max(int(window_m / DS) | 1, 1)
        pad = n // 2
        k = np.convolve(np.pad(k, pad, mode="wrap" if self.closed else "edge"), np.ones(n) / n, "valid")
        return k

    def sample(self, s):
        """Vectorised centerline lookup at stations ``s``: points, heading, curvature, widths, valid."""
        s = np.asarray(s, dtype=float)
        if self.closed:
            sm, valid, per = s % self.length, np.ones(s.shape, bool), self.length
        else:
            sm, valid, per = np.clip(s, 0.0, self.length), (s >= 0.0) & (s <= self.length), None
        st = self.station
        x = np.interp(sm, st, self.center[:, 0], period=per)
        y = np.interp(sm, st, self.center[:, 1], period=per)
        hc = np.interp(sm, st, np.cos(self.heading), period=per)
        hs = np.interp(sm, st, np.sin(self.heading), period=per)
        return (np.column_stack((x, y)), np.arctan2(hs, hc), np.interp(sm, st, self.kappa, period=per),
                np.interp(sm, st, self.wl, period=per), np.interp(sm, st, self.wr, period=per), valid)

    def find_corners(self, min_radius: float = 80.0, min_angle_deg: float = 15.0, merge_gap: float = 4.0) -> list[dict]:
        """Corners = runs of same-sign curvature tighter than ``min_radius`` turning at least
        ``min_angle_deg``. Stations are track s; for closed courses ``exit`` may exceed length."""
        k, s = self.kappa, self.station
        if self.closed:
            k, s = np.concatenate((k, k)), np.concatenate((s, s + self.length))
        sign = np.where(np.abs(k) >= 1.0 / min_radius, np.sign(k), 0.0)
        runs, i, n = [], 0, len(k)
        while i < n:
            if sign[i] == 0:
                i += 1
                continue
            j = i
            while j + 1 < n and sign[j + 1] == sign[i]:
                j += 1
            if runs and runs[-1][2] == sign[i] and s[i] - s[runs[-1][1]] <= merge_gap:
                runs[-1][1] = j
            else:
                runs.append([i, j, sign[i]])
            i = j + 1
        out = []
        for a, b, sg in runs:
            if self.closed and (a == 0 or s[a] >= self.length):
                continue  # wrap-around duplicates: each corner is kept once, from its true entry
            seg = slice(a, b + 1)
            ks, ss = k[seg], s[seg]
            angle = float(np.sum(0.5 * (ks[1:] + ks[:-1]) * np.diff(ss))) if b > a else 0.0
            if abs(math.degrees(angle)) < min_angle_deg:
                continue
            apex = a + int(np.argmax(np.abs(k[seg])))
            out.append({"direction": int(sg), "entry": float(s[a]), "apex": float(s[apex]), "exit": float(s[b]),
                        "min_radius": float(1.0 / max(abs(k[apex]), 1e-9)), "angle": angle})
        return out

    # ----------------------------------------------------------------- race layout
    @property
    def spawn_s(self) -> float:
        return (self.length - self.start_offset) if self.closed else 0.0

    @property
    def start_progress(self) -> float:
        """Progress (m from spawn along centerline) of the start line."""
        return self.start_offset

    @property
    def finish_progress(self) -> float:
        if self.closed:
            return self.start_offset + self.laps * self.length
        return self.length - self.finish_runoff

    def s_of_progress(self, progress: float) -> float:
        s = self.spawn_s + progress
        return s % self.length if self.closed else s

    def spawn_pose(self) -> tuple[float, float, float]:
        s = self.spawn_s + (1.0 if not self.closed else 0.0)
        p = self.line.point_at(s)
        return float(p[0]), float(p[1]), self.line.heading_at(s)

    def width_at(self, s: float) -> tuple[float, float]:
        i = int(np.clip(round((s % self.length if self.closed else s) / self.length * (len(self.center) - (0 if self.closed else 1))),
                        0, len(self.center) - 1))
        return float(self.wl[i]), float(self.wr[i])

    def cross_line(self, s: float) -> np.ndarray:
        """(2,2) segment (left point, right point) across the corridor at station s."""
        p, h = self.line.point_at(s), self.line.heading_at(s)
        n = np.array([-math.sin(h), math.cos(h)])
        wl, wr = self.width_at(s)
        return np.array([p + n * wl, p - n * wr])

    def gate_progress(self, spacing: float) -> np.ndarray:
        """Progress values of scored gates between start and finish (exclusive of both)."""
        n = int((self.finish_progress - self.start_progress) // spacing)
        gp = self.start_progress + spacing * np.arange(1, n + 1)
        return gp[gp < self.finish_progress - 0.5 * spacing]

    def all_cones(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All physical cones: positions, colours, mapped-flag."""
        n_obs = len(self.obstacle_cones)
        pos = np.vstack((self.cones, self.obstacle_cones)) if n_obs else self.cones
        col = np.concatenate((self.cone_colors, np.full(n_obs, CONE_ORANGE)))
        mapped = np.concatenate((np.ones(len(self.cones), bool), np.zeros(n_obs, bool)))
        return pos, col, mapped

    def bounds(self, margin: float = 0.0) -> tuple[float, float, float, float]:
        pts = np.vstack((self.left, self.right))
        lo, hi = pts.min(0) - margin, pts.max(0) + margin
        return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])

    # ----------------------------------------------------------------- cones
    def _place_cones(self):
        pos, col = [], []
        kappa = np.abs(g.curvature(g.resample(self.center, 2.0, self.closed), self.closed))
        k_center = np.interp(np.linspace(0, 1, len(self.center)), np.linspace(0, 1, len(kappa)), kappa)
        for boundary, color in ((self.left, CONE_LEFT), (self.right, CONE_RIGHT)):
            s = g.arclength(boundary, self.closed)
            p = np.vstack((boundary, boundary[:1])) if self.closed else boundary
            kk = np.concatenate((k_center, k_center[:1])) if self.closed else k_center
            t, end = 0.0, s[-1] - (0.5 * self.cone_spacing if self.closed else 0.0)
            while t <= end + 1e-9:
                pos.append([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])
                col.append(color)
                k = np.interp(t, s, kk)
                t += self.cone_spacing * float(np.clip(1.0 / (1.0 + 12.0 * k), 0.35, 1.0))
        # doubled orange markers just outside each end of the start and finish lines, spaced so
        # the physical props do not touch
        lines = [self.cross_line(self.s_of_progress(self.start_progress))]
        if not self.closed:
            lines.append(self.cross_line(self.s_of_progress(self.finish_progress)))
        for ln in lines:
            d = (ln[0] - ln[1]) / max(np.linalg.norm(ln[0] - ln[1]), 1e-9)
            for end, sgn in ((ln[0], 1), (ln[1], -1)):
                for k in MARKER_OFFSETS:
                    pos.append(end + sgn * d * k)
                    col.append(CONE_ORANGE)
        for c in self.extra_cones:
            pos.append([c["x"], c["y"]])
            col.append(COLOR_NAMES.get(str(c.get("color", "orange")), CONE_ORANGE))
        return np.asarray(pos, float), np.asarray(col, int)

    def _place_obstacles(self):
        """Obstacle cone positions; also records each cone's station, lateral offset and group.
        A cone that would stand on top of a mapped cone (e.g. a group reaching the boundary) is
        left out: that spot is already blocked, and overlapping props collide at spawn."""
        pts, meta = [], []
        for gid, ob in enumerate(self.obstacles):
            s = float(ob["s"])
            p, h = self.line.point_at(s), self.line.heading_at(s)
            n = np.array([-math.sin(h), math.cos(h)])
            off, w, sp = float(ob.get("offset", 0.0)), float(ob.get("width", 1.5)), float(ob.get("spacing", 0.6))
            k = max(int(round(w / sp)), 0)
            for i in range(k + 1):
                lat = off - w / 2 + (w * i / k if k else w / 2)
                q = p + n * lat
                if len(self.cones) and np.min(np.hypot(*(self.cones - q).T)) < MIN_CONE_GAP:
                    continue
                pts.append(q)
                meta.append((s, lat, gid))
        m = np.asarray(meta, float).reshape(-1, 3)
        self.obstacle_s, self.obstacle_lat, self.obstacle_gid = m[:, 0], m[:, 1], m[:, 2].astype(int)
        return np.asarray(pts, float).reshape(-1, 2)

    # ----------------------------------------------------------------- checks
    def validate(self, min_gap: float = 6.0) -> list[str]:
        problems = []
        kappa = np.abs(g.curvature(g.resample(self.center, 2.0, self.closed), self.closed))
        r_min = 1.0 / max(kappa.max(), 1e-9)
        if r_min < self.wl.max() + 1.5 or r_min < self.wr.max() + 1.5:
            problems.append(f"minimum centerline radius {r_min:.1f} m is too tight for the corridor width")
        if r_min < 5.5:
            problems.append(f"minimum radius {r_min:.1f} m is below the vehicle's turning radius")
        c = g.resample(self.center, 2.0, self.closed)
        wl = np.interp(np.linspace(0, 1, len(c)), np.linspace(0, 1, len(self.wl)), self.wl)
        wr = np.interp(np.linspace(0, 1, len(c)), np.linspace(0, 1, len(self.wr)), self.wr)
        s = g.arclength(c, self.closed)[: len(c)]
        d = np.hypot(*(c[:, None, :] - c[None, :, :]).transpose(2, 0, 1))
        ds = np.abs(s[:, None] - s[None, :])
        if self.closed:
            ds = np.minimum(ds, self.length - ds)
        wmax = np.maximum(wl, wr)
        need = wmax[:, None] + wmax[None, :] + min_gap
        clash = (ds > 3.0 * need) & (d < need)
        if clash.any():
            i, j = np.argwhere(clash)[0]
            problems.append(f"corridor overlaps itself near s={s[i]:.0f} m and s={s[j]:.0f} m")
        pos, _, _ = self.all_cones()
        if len(pos) > 1:
            d = np.hypot(*(pos[:, None, :] - pos[None, :, :]).transpose(2, 0, 1))
            np.fill_diagonal(d, np.inf)
            i, j = np.unravel_index(int(np.argmin(d)), d.shape)
            if d[i, j] < MIN_CONE_GAP:
                problems.append(f"cones at ({pos[i][0]:.1f}, {pos[i][1]:.1f}) and ({pos[j][0]:.1f}, {pos[j][1]:.1f}) "
                                f"are {d[i, j]:.2f} m apart (min {MIN_CONE_GAP} m); check obstacle spacing/extra_cones")
        if not self.closed and self.finish_progress <= self.start_progress + 20:
            problems.append("open course is too short for start offset + finish runoff")
        return problems

    # ----------------------------------------------------------------- views
    def info(self, full_knowledge: bool, gate_spacing: float) -> TrackInfo:
        gates = np.array([self.cross_line(self.s_of_progress(p)) for p in self.gate_progress(gate_spacing)]).reshape(-1, 2, 2)
        start = self.cross_line(self.s_of_progress(self.start_progress))
        finish = self.cross_line(self.s_of_progress(self.finish_progress))
        if full_knowledge:
            return TrackInfo(self.name, self.closed, self.laps, self.center.copy(), self.wl.copy(), self.wr.copy(),
                             self.cones.copy(), self.cone_colors.copy(), start, finish, gates, True)
        e2, e1 = np.zeros((0, 2)), np.zeros(0)  # no map; the name is withheld too (it may encode a seed)
        return TrackInfo("course", self.closed, self.laps, e2, e1, e1, e2, np.zeros(0, int), start, finish,
                         np.zeros((0, 2, 2)), False)

    def reference_time_estimate(self, vp: VehicleParams, grip_fraction: float = 0.9) -> float:
        """Centerline lap time with a friction-limited speed profile. Used as the scoring
        reference when the course file does not provide ``reference_time``."""
        c = g.resample(self.center, 1.0, self.closed)
        kappa = np.abs(g.curvature(c, self.closed))
        kappa = np.convolve(np.pad(kappa, 3, mode="wrap" if self.closed else "edge"), np.ones(7) / 7, "valid")
        seg = np.hypot(*np.diff(np.vstack((c, c[:1])) if self.closed else c, axis=0).T)
        a_lat = grip_fraction * vp.mu * 9.81
        vcap = np.minimum(np.sqrt(a_lat / np.maximum(kappa, 1e-6)), vp.max_speed)
        # unroll over laps so start/finish transitions are handled like an open course
        laps = self.laps if self.closed else 1
        reps = laps + 1 if self.closed else 1
        vcap = np.tile(vcap, reps)
        seg = np.tile(seg if self.closed else np.append(seg, seg[-1]), reps)[: len(vcap)]
        s = np.concatenate(([0.0], np.cumsum(seg)))[: len(vcap)]
        s0 = self.spawn_s
        s = s - (s0 if self.closed else 0.0)
        v = vcap.copy()
        i0 = int(np.searchsorted(s, 0.0))
        v[: i0 + 1] = 0.0
        for i in range(i0 + 1, len(v)):
            v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * vp.max_accel * 0.8 * seg[i - 1]))
        for i in range(len(v) - 2, -1, -1):
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * vp.max_decel * 0.8 * seg[i]))
        mask = (s >= self.start_progress) & (s <= self.finish_progress)
        vm = np.maximum(v[mask], 0.5)
        return float(np.sum(seg[mask] / vm))

    def to_yaml(self) -> dict:
        d = dict(self.source)
        d.setdefault("name", self.name)
        return d


# --------------------------------------------------------------------- building
def _normals(pts: np.ndarray, closed: bool) -> np.ndarray:
    h = g.headings(pts, closed)
    return np.column_stack((-np.sin(h), np.cos(h)))


def _integrate_segments(segments, origin, default_width):
    x, y, yaw = float(origin[0]), float(origin[1]), math.radians(float(origin[2]))
    pts, widths = [[x, y]], [default_width]
    h = 0.05  # integration step
    for seg in segments:
        typ = seg["type"]
        w = float(seg.get("width", default_width))
        if typ == "straight":
            length, kappa = float(seg["length"]), (lambda s: 0.0)
        elif typ == "arc":
            r, ang = float(seg["radius"]), math.radians(float(seg["angle"]))
            length, k0 = r * abs(ang), math.copysign(1.0 / r, ang)
            kappa = (lambda s, k0=k0: k0)
        elif typ == "slalom":
            lam, cyc = float(seg.get("wavelength", 24.0)), int(seg.get("cycles", 2))
            amp_h = math.pi * float(seg.get("swing", 3.0)) / lam
            sign = -1.0 if seg.get("first", "left") == "right" else 1.0
            length = lam * cyc
            kappa = (lambda s, a=amp_h, l=lam, sg=sign: sg * a * 2 * math.pi / l * math.cos(2 * math.pi * s / l))
        else:
            raise ValueError(f"unknown segment type '{typ}'")
        n = max(int(math.ceil(length / h)), 1)
        dh = length / n
        for i in range(n):
            s_mid = (i + 0.5) * dh
            yaw_mid = yaw + 0.5 * kappa(s_mid) * dh
            x += dh * math.cos(yaw_mid)
            y += dh * math.sin(yaw_mid)
            yaw += kappa(s_mid) * dh
            pts.append([x, y])
            widths.append(w)
    return np.asarray(pts), np.asarray(widths)


def build(spec: dict) -> Track:
    closed = bool(spec.get("closed", False))
    width = float(spec.get("width", 5.0))
    if "segments" in spec:
        raw, w = _integrate_segments(spec["segments"], spec.get("origin", [0, 0, 0]), width)
        if closed:
            gap = float(np.hypot(*(raw[-1] - raw[0])))
            if gap > 3.0:
                raise ValueError(f"closed course segments end {gap:.1f} m from the start; adjust segments")
            raw, w = raw[:-1], w[:-1]
    else:
        arr = np.asarray(spec["centerline"], dtype=float)
        raw = arr[:, :2]
        w = arr[:, 2] if arr.shape[1] > 2 else np.full(len(arr), width)
    center = g.resample(raw, DS, closed)
    s_raw = g.arclength(raw, closed)
    s_new = g.arclength(center, closed)[: len(center)]
    s_raw_pts = s_raw[: len(raw)]
    wf = np.interp(s_new * (s_raw[-1] / s_new[-1] if not closed else 1.0), s_raw_pts, w)
    # smooth width transitions over ~6 m
    k = int(6.0 / DS)
    wf = np.convolve(np.pad(wf, k, mode="wrap" if closed else "edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid")
    return Track(
        name=str(spec.get("name", "course")), closed=closed, center=center, wl=wf / 2, wr=wf / 2,
        laps=int(spec.get("laps", 1)), cone_spacing=float(spec.get("cone_spacing", 5.0)),
        start_offset=float(spec.get("start_offset", 8.0 if not closed else 12.0)),
        finish_runoff=float(spec.get("finish_runoff", 35.0)),
        obstacles=list(spec.get("obstacles", []) or []), extra_cones=list(spec.get("extra_cones", []) or []),
        reference_time=spec.get("reference_time"), source=spec)


def load(path: str | Path) -> Track:
    with open(path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    spec.setdefault("name", Path(path).stem)
    return build(spec)


def save(spec: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=None)
