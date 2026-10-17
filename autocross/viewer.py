"""Optional live top-down view (pygame). Works with every backend, including CARLA in
no-rendering mode, and costs far less than the 3D view. Close the window or press Esc to abort."""
from __future__ import annotations

import math

import numpy as np

from .api import CONE_LEFT, CONE_RIGHT
from .geometry import body_to_world

COLORS = {CONE_LEFT: (40, 100, 230), CONE_RIGHT: (235, 190, 0)}
GROUND, GRID, SURFACE, EDGE = (46, 50, 46), (58, 63, 58), (84, 86, 92), (110, 112, 118)
GRID_M = 10.0


class TopDownViewer:
    def __init__(self, track, vehicle_len=4.8, vehicle_wid=2.1, size=(900, 750), zoom=6.0):
        import pygame
        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption(f"autocross - {track.name}")
        self.font = pygame.font.SysFont("consolas", 16)
        self.track = track
        self.size = np.array(size, float)
        self.zoom = zoom
        self.vl, self.vw = vehicle_len, vehicle_wid
        self.trail: list = []
        self.cones, self.colors, self.mapped = track.all_cones()
        # course surface as quads between the boundaries (~2 m long), drawn over the ground grid
        k = 4
        idx = np.arange(0, len(track.center), k)
        if track.closed:
            idx = np.append(idx, 0)
        elif idx[-1] != len(track.center) - 1:
            idx = np.append(idx, len(track.center) - 1)
        L, R = track.left[idx], track.right[idx]
        self.quads = np.stack((L[:-1], L[1:], R[1:], R[:-1]), axis=1)  # (Q, 4, 2)
        marks = [(track.start_progress, (245, 245, 245))] if not track.closed else []
        marks.append((track.finish_progress if not track.closed else track.start_progress, (245, 245, 245)))
        self.lines = [(track.cross_line(track.s_of_progress(p)), col) for p, col in marks]

    def _px(self, pts, cx, cy):
        p = (np.atleast_2d(pts) - (cx, cy)) * self.zoom
        return np.column_stack((self.size[0] / 2 + p[:, 0], self.size[1] / 2 - p[:, 1]))

    def update(self, st, monitor, path, obs) -> bool:
        """obs: the latest Observation on control steps, else None (previous one is reused)."""
        pg = self.pg
        if obs is not None:
            self._obs = obs
        obs = getattr(self, "_obs", None)
        detections = obs.cones if obs is not None else None
        road = obs.road if obs is not None else None
        for ev in pg.event.get():
            if ev.type == pg.QUIT or (ev.type == pg.KEYDOWN and ev.key == pg.K_ESCAPE):
                return False
            if ev.type == pg.MOUSEWHEEL:
                self.zoom = float(np.clip(self.zoom * (1.15 ** ev.y), 1.0, 30.0))
        self.trail.append((st.x, st.y))
        s = self.screen
        cx, cy = st.x, st.y
        self._draw_floor(cx, cy)
        cp = self._px(self.cones, cx, cy)
        vis = (cp[:, 0] > -10) & (cp[:, 0] < self.size[0] + 10) & (cp[:, 1] > -10) & (cp[:, 1] < self.size[1] + 10)
        for (px, py), c, hit, mp in zip(cp[vis], self.colors[vis], monitor.cone_hit[vis], self.mapped[vis]):
            col = (255, 40, 40) if hit else COLORS.get(int(c), (255, 130, 0))
            pg.draw.circle(s, col, (int(px), int(py)), 4 if mp else 6)
        if len(self.trail) > 1:
            pg.draw.lines(s, (120, 220, 120), False, self._px(np.array(self.trail[-3000:]), cx, cy).tolist(), 2)
        if road is not None and road.valid and obs is not None:  # perceived road model (as the team sees it)
            for poly, col in ((road.preview_left_boundary, (90, 140, 255)), (road.preview_right_boundary, (255, 210, 60)),
                              (road.preview_centerline, (200, 200, 200))):
                if len(poly) > 1:
                    pg.draw.lines(s, col, False, self._px(body_to_world(poly, obs.x, obs.y, obs.yaw), cx, cy).tolist(), 1)
        if path is not None and len(path) > 1:
            pg.draw.lines(s, (0, 200, 255), False, self._px(path, cx, cy).tolist(), 2)
        if detections is not None and len(detections.positions):
            for px, py in self._px(body_to_world(detections.positions, st.x, st.y, st.yaw), cx, cy):
                pg.draw.circle(s, (255, 255, 255), (int(px), int(py)), 7, 1)
        body = np.array([[self.vl / 2, self.vw / 2], [self.vl / 2, -self.vw / 2],
                         [-self.vl / 2, -self.vw / 2], [-self.vl / 2, self.vw / 2]])
        pg.draw.polygon(s, (220, 40, 50), self._px(body_to_world(body, st.x, st.y, st.yaw), cx, cy).tolist())
        t = monitor.elapsed(st.t)
        lines = [f"t {t:6.2f} s   v {st.speed * 3.6:5.1f} km/h   progress {monitor.progress:6.1f} m",
                 f"cones {int(monitor.cone_hit.sum())}   off-course {sum(e.kind == 'off_course' for e in monitor.events)}"
                 f"   missed gates {sum(e.kind == 'missed_gate' for e in monitor.events)}"
                 + ("   OFF COURSE" if monitor.off else "") + (f"   DNF: {monitor.dnf}" if monitor.dnf else ""),
                 f"yaw rate {math.degrees(st.yaw_rate):6.1f} deg/s   ay {st.ay:5.2f} m/s^2   (mouse wheel = zoom, Esc = abort)",
                 _road_text(road)]
        hud = pg.Surface((int(self.size[0]), 14 + 20 * len(lines)), pg.SRCALPHA)
        hud.fill((20, 22, 20, 170))
        s.blit(hud, (0, 0))
        for i, ln in enumerate(lines):
            s.blit(self.font.render(ln, True, (240, 240, 240)), (10, 8 + 20 * i))
        pg.display.flip()
        return True

    def _draw_floor(self, cx, cy):
        """Ground with a 10 m grid (so motion is visible anywhere), the course surface, its
        edges, and the start/finish lines."""
        pg, s, w, h = self.pg, self.screen, self.size[0], self.size[1]
        s.fill(GROUND)
        half = self.size / (2 * self.zoom)
        if GRID_M * self.zoom >= 6:
            for gx in np.arange(np.floor((cx - half[0]) / GRID_M), np.ceil((cx + half[0]) / GRID_M) + 1) * GRID_M:
                px = w / 2 + (gx - cx) * self.zoom
                pg.draw.line(s, GRID, (px, 0), (px, h), 1)
            for gy in np.arange(np.floor((cy - half[1]) / GRID_M), np.ceil((cy + half[1]) / GRID_M) + 1) * GRID_M:
                py = h / 2 - (gy - cy) * self.zoom
                pg.draw.line(s, GRID, (0, py), (w, py), 1)
        q = self._px(self.quads.reshape(-1, 2), cx, cy).reshape(-1, 4, 2)
        vis = ((q[:, :, 0].max(1) > 0) & (q[:, :, 0].min(1) < w) & (q[:, :, 1].max(1) > 0) & (q[:, :, 1].min(1) < h))
        for quad in q[vis]:
            pg.draw.polygon(s, SURFACE, quad.tolist())
        for b in (self.track.left, self.track.right):
            pg.draw.lines(s, EDGE, self.track.closed, self._px(b, cx, cy).tolist(), 1)
        for ln, col in self.lines:
            pg.draw.line(s, col, *self._px(ln, cx, cy).tolist(), max(int(0.4 * self.zoom), 2))

    def close(self):
        self.pg.quit()


def _road_text(road) -> str:
    if road is None or not road.valid:
        return "road model: n/a"
    nxt = next((c for c in road.corners if c.exit_distance != c.exit_distance or c.exit_distance > 0), None)
    corner = (f"next corner {'L' if nxt.direction > 0 else 'R'} in {nxt.entry_distance:5.1f} m, R {nxt.min_radius:5.1f} m"
              if nxt else "no corner in range")
    return (f"road: offset {road.lateral_offset:+5.2f} m  heading err {math.degrees(road.heading_error):+5.1f} deg  "
            f"curv {road.curvature:+.3f}  {corner}  obstacles {len(road.obstacles)}")
