"""Race rules, evaluated every physics tick from ground truth (backend-independent).

Events (each deducts a fixed number of points in scoring.yaml):
  cone_hit      a cone's surveyed position falls inside the vehicle footprint (each cone once)
  off_course    the vehicle leaves the coned corridor (rule: center | any_wheel | all_wheels);
                one event per excursion, with hysteresis
  missed_gate   the vehicle passes a scored gate station while outside the corridor
                (i.e. it went around the gate / cut or overshot the turn)
DNF reasons: timeout, off_course_too_long, lost (too far from the course), wrong_way,
stuck, rollover, controller_error.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import geometry as g


@dataclass
class Event:
    t: float
    kind: str
    x: float
    y: float
    progress: float
    detail: dict = field(default_factory=dict)


class RaceMonitor:
    def __init__(self, track, rules: dict, vehicle, max_time: float):
        self.track = track
        self.rules = rules
        self.half_l = vehicle.length / 2
        self.half_w = vehicle.width / 2
        self.axle_f, self.axle_r = vehicle.cg_to_front, -vehicle.cg_to_rear
        self.half_track = vehicle.track_width / 2
        self.max_time = max_time
        self.rule = str(rules.get("off_course_rule", "all_wheels"))
        self.off_dnf = float(rules.get("off_course_dnf_s", 5.0))
        self.lost = float(rules.get("lost_distance_m", 15.0))
        self.stuck_s = float(rules.get("stuck_s", 8.0))
        self.wrong_way = float(rules.get("wrong_way_m", 15.0))
        self.cone_margin = float(rules.get("cone_hit_margin_m", 0.1))
        self.hyst = float(rules.get("off_course_reentry_s", 0.3))
        self.cones, self.cone_colors, self.cone_mapped = track.all_cones()
        self.cone_hit = np.zeros(len(self.cones), bool)
        self.gates = track.gate_progress(float(rules.get("gate_spacing_m", 20.0)))
        self.next_gate = 0
        self.events: list[Event] = []
        self.s_hint = track.spawn_s
        self.progress = 0.0
        self.max_progress = 0.0
        self.lateral = 0.0
        self.boundary_margin = 0.0
        self.off = False
        self.off_since = None
        self.on_since = None
        self.off_time = 0.0
        self.slow_since = None
        self.start_time = None
        self.finish_time = None
        self.lap_times: list[float] = []
        self._lap_mark = None
        self.dnf: str | None = None
        self._last_t = 0.0
        self._prev_lat = 0.0
        self._prev_prog = 0.0

    @property
    def started(self):
        return self.start_time is not None

    @property
    def done(self):
        return self.finish_time is not None or self.dnf is not None

    @property
    def lap(self):
        if not self.track.closed:
            return 0
        return max(int((self.progress - self.track.start_progress) // self.track.length), 0)

    def elapsed(self, t):
        if self.start_time is None:
            return 0.0
        return (self.finish_time if self.finish_time is not None else t) - self.start_time

    def _event(self, st, kind, **detail):
        self.events.append(Event(st.t, kind, st.x, st.y, self.progress, detail))

    def _lateral_at(self, pts):
        """Lateral offsets and local widths of several world points, using the current station."""
        out = []
        for p in pts:
            s, lat, _ = self.track.line.project(p, self.s_hint, 12.0)
            wl, wr = self.track.width_at(s)
            out.append((lat, wl, wr))
        return out

    def update(self, st, dt):
        tr = self.track
        p = np.array([st.x, st.y])
        s, lat, _ = tr.line.project(p, self.s_hint, 25.0)
        ds = s - self.s_hint
        if tr.closed:
            ds = (ds + tr.length / 2) % tr.length - tr.length / 2
        self.s_hint = s
        prev_prog = self.progress
        self.progress += ds
        self.max_progress = max(self.max_progress, self.progress)
        self.lateral = lat
        wl, wr = tr.width_at(s)

        # wheel contact points (vehicle frame -> world)
        c, sn = math.cos(st.yaw), math.sin(st.yaw)
        wheels_b = [(self.axle_f, self.half_track), (self.axle_f, -self.half_track),
                    (self.axle_r, self.half_track), (self.axle_r, -self.half_track)]
        wheels = [(st.x + c * a - sn * b, st.y + sn * a + c * b) for a, b in wheels_b]
        wl_lat = self._lateral_at(wheels)
        outside = [(la > l) or (-la > r) for la, l, r in wl_lat]
        if self.rule == "center":
            is_off = lat > wl or -lat > wr
        elif self.rule == "any_wheel":
            is_off = any(outside)
        else:
            is_off = all(outside)
        # signed margin of the wheel contact patches to the nearest boundary (negative = over the line)
        corner_lat = [la for la, _, _ in wl_lat]
        self.boundary_margin = min(wl - max(corner_lat), wr + min(corner_lat))

        # timing
        if self.start_time is None and prev_prog < tr.start_progress <= self.progress:
            frac = (tr.start_progress - prev_prog) / max(self.progress - prev_prog, 1e-9)
            self.start_time = self._last_t + frac * dt
            self._lap_mark = self.start_time
            self._event(st, "start")
        if self.started and tr.closed:
            lap_line = tr.start_progress + (len(self.lap_times) + 1) * tr.length
            if prev_prog < lap_line <= self.progress:
                self.lap_times.append(st.t - self._lap_mark)
                self._lap_mark = st.t
        if self.started and self.finish_time is None and prev_prog < tr.finish_progress <= self.progress:
            frac = (tr.finish_progress - prev_prog) / max(self.progress - prev_prog, 1e-9)
            self.finish_time = self._last_t + frac * dt
            self._event(st, "finish", time=self.finish_time - self.start_time)

        if self.started:
            # gates: passed station while the centre was inside the corridor?
            while self.next_gate < len(self.gates) and self.progress >= self.gates[self.next_gate]:
                gp = self.gates[self.next_gate]
                w = (gp - prev_prog) / max(self.progress - prev_prog, 1e-9)
                lat_g = self._prev_lat + (lat - self._prev_lat) * float(np.clip(w, 0, 1))
                gwl, gwr = tr.width_at(tr.s_of_progress(gp))
                if lat_g > gwl or -lat_g > gwr:
                    self._event(st, "missed_gate", gate=int(self.next_gate), lateral=round(lat_g, 2))
                self.next_gate += 1

            # off-course excursions (debounced re-entry)
            if is_off:
                self.off_time += dt
                self.on_since = None
                if not self.off:
                    self.off = True
                    self.off_since = st.t
                    self._event(st, "off_course", lateral=round(lat, 2))
                elif st.t - self.off_since > self.off_dnf:
                    self.dnf = "off_course_too_long"
            elif self.off:
                self.on_since = self.on_since if self.on_since is not None else st.t
                if st.t - self.on_since >= self.hyst:
                    self.off = False

            # cone strikes
            near = np.flatnonzero(~self.cone_hit & (np.abs(self.cones[:, 0] - st.x) < 4) & (np.abs(self.cones[:, 1] - st.y) < 4))
            if len(near):
                inside = g.points_in_obb(self.cones[near], st.x, st.y, st.yaw, self.half_l, self.half_w, self.cone_margin)
                for i in near[inside]:
                    self.cone_hit[i] = True
                    self._event(st, "cone_hit", cone=int(i), mapped=bool(self.cone_mapped[i]))

        # DNF checks
        if abs(lat) > max(wl, wr) + self.lost:
            self.dnf = "lost"
        if self.progress < self.max_progress - self.wrong_way:
            self.dnf = "wrong_way"
        if abs(st.roll) > 1.0 or abs(st.pitch) > 1.0:
            self.dnf = "rollover"
        if st.speed < 0.3 and st.t > 3.0:
            self.slow_since = self.slow_since if self.slow_since is not None else st.t
            if st.t - self.slow_since > self.stuck_s:
                self.dnf = "stuck"
        else:
            self.slow_since = None
        if st.t > self.max_time:
            self.dnf = "timeout"
        if self.dnf and not any(e.kind == "dnf" for e in self.events):
            self._event(st, "dnf", reason=self.dnf)
        self._prev_lat = lat
        self._last_t = st.t
        return is_off
