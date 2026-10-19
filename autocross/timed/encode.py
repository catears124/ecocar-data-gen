"""Observation encoding contract for ``timed-1``.

``Encoder.encode(obs)`` is a deterministic function of the observation sequence
(the current observation plus the road-wheel estimate carried from the previous
one).  The offline audit replays the same encoder over the recorded observation
Quantisation (round-half-even of the float observation).  The road frame is the
preview centreline polyline itself, so that curvature, the start pose and the
corridor/cone geometry are mutually consistent (the road model's own
``heading_error``/``preview_curvature`` disagree with its centreline geometry by
~0.01 rad per control step on S-bends; measured in experiments/timed_calibration):
    h_j  = station heading of the preview polyline (mean of adjacent chord headings,
           linear extrapolation at the ends), k_j = central difference of h_j * 1e6,
    psi  = -h_0 * 1e6 (the car's own heading is 0 in the vehicle frame),
    d    = lateral offset of the car's origin from preview point 0 along the
           station normal, * 1e3,
    v = max(vx, 0) * 1e3, vy = vy * 1e3, r = yaw_rate * 1e6, sigma = steer * 1e3,
    wl_j / wr_j = preview widths * 1e3 (j = 0..n-1, 1 m apart),
    p_prev = (throttle - brake) * 1e3.
Road-wheel angle: replayed from the applied-steer samples with the model's own
slew/lag arithmetic (exact for the harness when the previous step was observed);
after a gap or at the first step it is the steady value ``sigma * MAX_STEER / 1000``.
Obstacles: each ``ObstacleInfo`` becomes the box ``[s_min - pad, s_max + pad] x
[lateral_min - pad, lateral_max + pad]`` (mm), where ``s_min/s_max`` are the
stations of its cones projected on the preview centreline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import DEFAULT_BLOCKS, DEFAULT_CONSTS, Block, Instance, State, clamp, well_formed

OBS_PAD_MM = 150


class EncodeError(ValueError):
    pass


def lag(consts: dict, delta: int, sigma: int) -> int:
    tgt = sigma * consts["MAX_STEER"] // 1000
    return tgt + (delta - tgt) * consts["LAG_NUM"] // consts["LAG_DEN"]


def delta_update(consts: dict, delta_prev: int, sigma_prev: int, sigma_now: int) -> int:
    """Replay the two physics ticks of one control period (constant command)."""
    slew = consts["SLEW"]
    mid = clamp(sigma_prev + clamp(sigma_now - sigma_prev, -slew, slew), -1000, 1000)
    return lag(consts, lag(consts, delta_prev, mid), sigma_now)


def _q(x: float, scale: float) -> int:
    x = float(x)
    if not np.isfinite(x):
        raise EncodeError("non-finite observation")
    return int(round(x * scale))


def _obstacle_boxes(road, pad: int) -> tuple[tuple[int, int, int, int], ...]:
    cl = np.asarray(road.preview_centerline, dtype=float)
    s = np.asarray(road.preview_s, dtype=float)
    out = []
    for o in getattr(road, "obstacles", []) or []:
        pts = np.asarray(getattr(o, "points", np.zeros((0, 2))), dtype=float).reshape(-1, 2)
        st = []
        for p in pts:
            j = int(np.argmin(np.hypot(cl[:, 0] - p[0], cl[:, 1] - p[1])))
            j2 = min(j + 1, len(cl) - 1)
            j1 = j2 - 1
            t = cl[j2] - cl[j1]
            nt = float(np.hypot(*t)) or 1.0
            st.append(s[j] + float(np.dot(p - cl[j], t / nt)))
        if not st:
            st = [float(o.distance)]
        out.append((_q(min(st), 1000) - pad, _q(max(st), 1000) + pad,
                    _q(o.lateral_min, 1000) - pad, _q(o.lateral_max, 1000) + pad))
    return tuple(out)


def station_headings(cl: np.ndarray) -> np.ndarray:
    """Heading at each preview station: mean of the adjacent chord headings."""
    seg = np.diff(cl, axis=0)
    th = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
    n = cl.shape[0]
    h = np.empty(n)
    if th.size == 1:
        h[:] = th[0]
        return h
    h[1:-1] = 0.5 * (th[:-1] + th[1:])
    h[0] = th[0] - 0.5 * (th[1] - th[0])
    h[-1] = th[-1] + 0.5 * (th[-1] - th[-2])
    return h


@dataclass
class Encoder:
    consts: dict = field(default_factory=lambda: dict(DEFAULT_CONSTS))
    blocks: tuple[Block, ...] = DEFAULT_BLOCKS
    obs_pad: int = OBS_PAD_MM
    _delta: int | None = None
    _sigma: int | None = None
    _step: int | None = None

    def road_wheel(self, step: int, sigma: int) -> int:
        if self._step is not None and self._delta is not None and self._sigma is not None \
                and step == self._step + 1:
            d = delta_update(self.consts, self._delta, self._sigma, sigma)
        else:
            d = sigma * self.consts["MAX_STEER"] // 1000
        self._delta, self._sigma, self._step = d, sigma, step
        return d

    def encode(self, obs) -> Instance:
        road = getattr(obs, "road", None)
        sigma = clamp(_q(obs.steer, 1000), -1000, 1000)
        delta = self.road_wheel(int(obs.step), sigma)
        if road is None or not getattr(road, "valid", False):
            raise EncodeError("invalid road model")
        s = np.asarray(road.preview_s, dtype=float)
        n = s.size
        if n < 2 or np.any(np.abs(np.diff(s) - self.consts["STEP"] / 1000.0) > 1e-6) or abs(s[0]) > 1e-6:
            raise EncodeError("preview spacing is not the declared 1 m grid")
        cl = np.asarray(road.preview_centerline, dtype=float).reshape(-1, 2)
        if cl.shape[0] != n or not np.all(np.isfinite(cl)):
            raise EncodeError("preview centreline does not match preview_s")
        h = station_headings(cl)
        kap = np.gradient(h) if n >= 3 else np.zeros(n)
        k = tuple(_q(x, 1e6) for x in kap)
        wl = tuple(_q(x, 1000) for x in road.preview_width_left)
        wr = tuple(_q(x, 1000) for x in road.preview_width_right)
        h0 = float(h[0])
        d0 = -(-np.sin(h0) * cl[0, 0] + np.cos(h0) * cl[0, 1])
        start = State(0, _q(d0, 1000), _q(-h0, 1e6),
                      max(_q(obs.vx, 1000), 0), _q(obs.vy, 1000), _q(obs.yaw_rate, 1e6),
                      sigma, delta)
        p_prev = clamp(_q(float(obs.throttle) - float(obs.brake), 1000), -1000, 1000)
        inst = Instance(dict(self.consts), k, wl, wr, _obstacle_boxes(road, self.obs_pad),
                        start, p_prev, self.blocks)
        err = well_formed(inst)
        if err:
            raise EncodeError("; ".join(err))
        return inst


__all__ = ["Encoder", "EncodeError", "delta_update", "lag", "OBS_PAD_MM"]
