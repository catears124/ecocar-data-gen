"""Seeded procedural course generator. Output is a normal course YAML spec (segments for
open courses, explicit points for closed ones) so organisers can hand-edit the result."""
from __future__ import annotations

import math

import numpy as np

from . import geometry as g
from .track import build


def _random_segment(rng: np.random.Generator, width: float) -> dict:
    kind = rng.choice(["straight", "sweeper", "turn", "hairpin", "slalom", "chicane"],
                      p=[0.22, 0.2, 0.25, 0.1, 0.13, 0.1])
    side = float(rng.choice([-1.0, 1.0]))
    if kind == "straight":
        return {"type": "straight", "length": round(float(rng.uniform(12, 45)), 1)}
    if kind == "sweeper":
        return {"type": "arc", "radius": round(float(rng.uniform(25, 45)), 1),
                "angle": round(side * float(rng.uniform(30, 100)), 1)}
    if kind == "turn":
        return {"type": "arc", "radius": round(float(rng.uniform(12, 22)), 1),
                "angle": round(side * float(rng.uniform(60, 120)), 1)}
    if kind == "hairpin":
        return {"type": "arc", "radius": round(float(rng.uniform(9, 11.5)), 1),
                "angle": round(side * float(rng.uniform(150, 185)), 1), "width": round(width + 1.0, 1)}
    if kind == "slalom":
        return {"type": "slalom", "cycles": int(rng.integers(2, 4)), "wavelength": round(float(rng.uniform(22, 30)), 1),
                "swing": round(float(rng.uniform(2.0, 3.0)), 1), "first": "left" if side > 0 else "right"}
    return {"type": "slalom", "cycles": 1, "wavelength": round(float(rng.uniform(26, 34)), 1),
            "swing": round(float(rng.uniform(2.5, 3.5)), 1), "first": "left" if side > 0 else "right"}


def generate_open(seed: int, target_length: float = 700.0, width: float = 5.5,
                  max_extent: float = 320.0, n_obstacles: int = 0, attempts: int = 400) -> dict:
    rng = np.random.default_rng(seed)
    for _ in range(attempts):
        segs = [{"type": "straight", "length": 30.0}]
        length, failures = 30.0, 0
        while length < target_length - 60 and failures < 60:
            failures += 1
            s = _random_segment(rng, width)
            trial = segs + [s]
            spec = _spec(f"gen_open_{seed}", False, width, trial)
            try:
                t = build(spec)
            except ValueError:
                continue
            if t.validate() or _extent(t) > max_extent:
                continue
            segs, length = trial, t.length
        segs.append({"type": "straight", "length": 55.0})
        spec = _spec(f"gen_open_{seed}", False, width, segs)
        t = build(spec)
        if t.validate() or _extent(t) > max_extent or t.length < 0.7 * target_length:
            continue
        spec["obstacles"] = _obstacles(rng, t, n_obstacles)
        return spec
    raise RuntimeError("could not generate a valid course; try another seed or relax limits")


def generate_closed(seed: int, target_length: float = 600.0, width: float = 6.0,
                    laps: int = 2, n_obstacles: int = 0, attempts: int = 400) -> dict:
    rng = np.random.default_rng(seed)
    th = np.linspace(0, 2 * math.pi, 720, endpoint=False)
    for _ in range(attempts):
        r = np.ones_like(th)
        for k in range(2, 8):
            r += rng.uniform(0.0, 0.45 / k ** 0.8) * np.cos(k * th + rng.uniform(0, 2 * math.pi))
        if r.min() < 0.25:
            continue
        pts = np.column_stack((r * np.cos(th), r * np.sin(th)))
        pts = g.smooth_closed(pts, 9)
        pts *= target_length / g.arclength(pts, True)[-1]
        if g.segments_intersect_any(pts[::6], closed=True):
            continue
        pts = g.resample(pts, 2.0, True)
        spec = {"name": f"gen_closed_{seed}", "closed": True, "laps": laps, "width": width, "cone_spacing": 5.0,
                "start_offset": 12.0, "centerline": [[round(float(x), 2), round(float(y), 2)] for x, y in pts]}
        t = build(spec)
        if t.validate():
            continue
        spec["obstacles"] = _obstacles(rng, t, n_obstacles)
        return spec
    raise RuntimeError("could not generate a valid closed course; try another seed")


def _spec(name, closed, width, segs):
    return {"name": name, "closed": closed, "width": width, "cone_spacing": 5.0, "start_offset": 8.0,
            "finish_runoff": 40.0, "origin": [0.0, 0.0, 0.0], "segments": segs}


def _extent(t) -> float:
    x0, y0, x1, y1 = t.bounds()
    return max(x1 - x0, y1 - y0)


def _obstacles(rng, t, n) -> list:
    if n <= 0:
        return []
    lo, hi = t.start_progress + 60.0, t.finish_progress - 40.0
    if t.closed:
        lo, hi = t.start_offset + 60.0, t.start_offset + t.length - 40.0
    out = []
    for s in np.sort(rng.uniform(lo, hi, n)):
        wl, wr = t.width_at(t.s_of_progress(float(s)))
        side = float(rng.choice([-1.0, 1.0]))
        w = float(rng.uniform(1.2, 2.0))
        off = side * ((wl if side > 0 else wr) - w / 2 - 0.3)  # outer cone just inside the boundary line
        out.append({"s": round(float(t.s_of_progress(float(s))), 1), "offset": round(off, 2), "width": round(w, 2)})
    return out
