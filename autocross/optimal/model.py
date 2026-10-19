"""Exact finite-horizon joint driving model used by the proof-carrying teacher.

Everything here is integer arithmetic and mirrors, line by line, the Lean module
``Corpus.Teacher``.  Units: millimetres, mm/s, mm^2/s^2, mm/s^2, micro-1/m, and
microseconds for the declared objective.

State
-----
``(station i, previous offset p, current offset x, speed^2 b)``.  ``p`` and
``x`` are lateral offsets (mm, + left) from the preview centreline.  A step
``s -> t`` advances exactly one preview station, carries ``s.current`` into
``t.previous``, lands on the declared offset and speed grids, satisfies

* corridor:  ``lo_i <= x <= hi_i`` with ``lo/hi`` the same certified interval
  used by the controller (vehicle half-width and margin inside the cone rows);
* obstacles: for every obstacle whose inflated longitudinal footprint contains
  station ``i``, the offset is either left of the inflated left edge or right
  of the inflated right edge -- no obstacle is silently ignored;
* curvature: ``|k_i| + ceil(1e9 * |p - 2x + n| / ds^2) <= kappa_max`` where
  ``k_i`` is the road curvature at the departure station and ``n`` the chosen
  next offset;
* speed: departure speed^2 is under the curvature cap
  ``min(v_max^2, floor(a_lat*1e9 / curv))``, the arrival speed^2 is on the
  declared grid and non-negative, and
  ``w <= b + 2 a_acc ds``, ``b <= w + 2 a_brk ds``.

The terminal condition is ``b <= min(v_max^2, a_lat*r_h)``: the declared
horizon radius is the only information about the road beyond the preview.

Objective
---------
For a segment of length ``d`` and endpoint speed^2 values ``b, w``, charge

    time_cost(d,b,w) = ceil(2 * d * 1e6 / (isqrt(b) + isqrt(w)))

microseconds.  This is the constant-acceleration segment traversal time
``2d/(v_i+v_{i+1})`` with both speeds replaced by their integer floors, rounded
up.  The declared objective is the sum of these integer charges.  It is
therefore exactly computable, never smaller than the constant-acceleration time
computed from the floored speeds, and within one microsecond per segment of the
same formula at full precision.  ``Corpus.Teacher`` proves the quantised
objective's optimality; the relation to continuous traversal time is stated as
a bound (and measured in the paper), not as an identity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, NamedTuple

from autocross.datagen import cert, steer

MODEL_ID = "teacher-joint-1"
OBJECTIVE_ID = "sum_ceiling_constant_accel_microseconds"
TIME_SCALE = 1_000_000
DEFAULT_OFFSETS = tuple(range(-1500, 1501, 250))
DEFAULT_HORIZON = 10


class State(NamedTuple):
    station: int
    previous: int
    current: int
    b: int

    def as_dict(self) -> dict:
        return {"station": self.station, "previous": self.previous,
                "current": self.current, "b": self.b}

    @staticmethod
    def from_any(x) -> "State":
        if isinstance(x, State):
            return x
        if isinstance(x, dict):
            return State(int(x["station"]), int(x["previous"]), int(x["current"]), int(x["b"]))
        return State(int(x[0]), int(x[1]), int(x[2]), int(x[3]))


def isqrt(n: int) -> int:
    if n < 0:
        raise ValueError("isqrt of a negative integer")
    return math.isqrt(int(n))


def cdiv(a: int, b: int) -> int:
    """Ceiling division for a positive divisor (Lean ``Cert.cdiv``)."""
    if b <= 0:
        raise ValueError("cdiv divisor must be positive")
    return -((-int(a)) // int(b))


def time_cost(d: int, b: int, w: int) -> int:
    """Declared traversal-time charge (microseconds) of one segment."""
    if d <= 0:
        raise ValueError("segment length must be positive")
    if b < 0 or w < 0:
        raise ValueError("speed^2 must be non-negative")
    den = isqrt(b) + isqrt(w)
    if den == 0:
        raise ValueError("a segment with zero speed at both ends has no finite traversal time")
    return cdiv(2 * d * TIME_SCALE, den)


def lat_curv(step: int, p: int, x: int, n: int) -> int:
    """Integer lateral curvature contribution (micro-1/m), matching Lean ``Joint.latCurv``."""
    return cdiv(1_000_000_000 * abs(n - 2 * x + p), step * step)


@dataclass(frozen=True)
class Model:
    """A fully declared finite teacher model instance."""

    consts: dict
    q: dict
    offsets: tuple[int, ...]
    speeds: tuple[int, ...]
    horizon: int
    start: State
    terminal_cap: int
    model_id: str = MODEL_ID
    objective: str = OBJECTIVE_ID
    time_scale: int = TIME_SCALE

    # -- basic derived quantities -------------------------------------------------
    @property
    def s(self) -> list[int]:
        return [int(v) for v in self.q["base"]["s"]]

    @property
    def k(self) -> list[int]:
        return [int(v) for v in self.q["base"]["k"]]

    @property
    def lon(self) -> dict:
        return self.consts["lon"]

    @property
    def step(self) -> int:
        return int(self.consts["step"])

    @property
    def margin(self) -> int:
        return int(self.consts["margin"])

    @property
    def kappa_max(self) -> int:
        return int(self.consts["kappa_max"])

    @property
    def half_width(self) -> int:
        return int(self.lon["veh_width"]) // 2

    def gap(self, i: int) -> int:
        return self.s[i + 1] - self.s[i]

    def lo(self, i: int) -> int:
        return -(int(self.q["base"]["wr"][i]) - self.half_width - self.margin)

    def hi(self, i: int) -> int:
        return int(self.q["base"]["wl"][i]) - self.half_width - self.margin

    def corridor_ok(self, i: int, x: int) -> bool:
        return self.lo(i) <= x <= self.hi(i)

    def obstacles_at(self, i: int) -> list[tuple[int, int, int]]:
        si = self.s[i]
        hold = int(self.lon["hold"])
        out = []
        for ob in self.q["base"]["obstacles"]:
            if not isinstance(ob, (list, tuple)) or len(ob) != 3:
                continue
            d, lmin, lmax = int(ob[0]), int(ob[1]), int(ob[2])
            if d - hold <= si <= d + hold:
                out.append((d, lmin, lmax))
        return out

    def obstacle_ok(self, i: int, x: int) -> bool:
        half = self.half_width + self.margin
        for _, lmin, lmax in self.obstacles_at(i):
            # The inflated obstacle body is [lmin-half, lmax+half]; the car
            # reference point must be outside it on one side.
            if not (x <= lmin - half or x >= lmax + half):
                return False
        return True

    def lateral_ok(self, i: int, p: int, x: int) -> bool:
        return (p in self.offsets and x in self.offsets
                and self.corridor_ok(i, x) and self.obstacle_ok(i, x))

    def path_curv(self, i: int, p: int, x: int, n: int) -> int:
        return abs(self.k[i]) + lat_curv(self.step, p, x, n)

    def path_cap(self, curv: int) -> int:
        vmax2 = int(self.lon["v_max"]) ** 2
        if curv <= 0:
            return vmax2
        return min(vmax2, int(self.lon["a_lat"]) * 1_000_000_000 // curv)

    def terminal_value(self, s: State) -> int | None:
        return 0 if 0 <= s.b <= self.terminal_cap else None

    def speed_ok(self, i: int, b: int, w: int) -> bool:
        d = self.gap(i)
        return (w in self.speeds and 0 <= b and 0 <= w and (0 < b or 0 < w)
                and w <= b + 2 * int(self.lon["a_acc"]) * d
                and b <= w + 2 * int(self.lon["a_brk"]) * d)

    def step_ok(self, s: State, t: State) -> bool:
        if t.station != s.station + 1 or not (0 <= s.station < self.horizon):
            return False
        if t.previous != s.current:
            return False
        if t.current not in self.offsets or t.b not in self.speeds:
            return False
        if not self.lateral_ok(t.station, t.previous, t.current):
            return False
        curv = self.path_curv(s.station, s.previous, s.current, t.current)
        if curv > self.kappa_max:
            return False
        if s.b > self.path_cap(curv):
            return False
        return self.speed_ok(s.station, s.b, t.b)

    def actions(self, s: State) -> list[tuple[State, int]]:
        """All admissible (successor, declared cost) pairs, in deterministic order."""
        if s.station >= self.horizon or s.station + 1 >= len(self.s):
            return []
        out: list[tuple[State, int]] = []
        d = self.gap(s.station)
        for n in self.offsets:
            t0 = State(s.station + 1, s.current, n, 0)
            if not self.lateral_ok(t0.station, t0.previous, n):
                continue
            curv = self.path_curv(s.station, s.previous, s.current, n)
            if curv > self.kappa_max or s.b > self.path_cap(curv):
                continue
            for w in self.speeds:
                if not self.speed_ok(s.station, s.b, w):
                    continue
                t = State(t0.station, t0.previous, n, w)
                out.append((t, time_cost(d, s.b, w)))
        return out

    def full_states(self, i: int) -> list[State]:
        if i == 0:
            return [self.start]
        return [State(i, p, x, b) for p in self.offsets for x in self.offsets for b in self.speeds]

    def univ(self, i: int) -> list[State]:
        return self.full_states(i)

    def as_instance(self) -> dict:
        return {"model": self.model_id, "objective": self.objective, "time_scale": self.time_scale,
                "consts": self.consts, "q": self.q, "offsets": list(self.offsets),
                "speeds": list(self.speeds), "horizon": self.horizon,
                "start": self.start.as_dict(), "terminal_cap": self.terminal_cap}


def _snap(values: Iterable[int], x: int) -> int:
    vals = sorted(set(int(v) for v in values))
    return min(vals, key=lambda v: (abs(v - x), v))


def model_from_observation(obs, vp, *, offsets: Iterable[int] = DEFAULT_OFFSETS,
                           speeds: Iterable[int] | None = None, horizon: int = DEFAULT_HORIZON,
                           terminal_cap: int | None = None) -> Model | None:
    """Quantise a policy observation and build the exact teacher model.

    Returns ``None`` when the observation is outside the model domain (the
    controller must then report invalid input, never a fabricated optimum).
    """
    base = cert.quantise(obs)
    if base is None:
        return None
    consts = steer.model_constants(vp)
    if not steer.well_formed(consts, {"base": base, "lat": 0}):
        return None
    try:
        lat = cert.qfix(obs.road.lateral_offset, 1000)
        heading = cert.qfix(obs.road.heading_error, 1_000_000)
    except (AttributeError, TypeError, ValueError):
        return None
    n = len(base["s"])
    m = min(int(horizon), n - 1)
    if m < 1:
        return None
    step = int(consts["step"])
    # Current offset: the declared grid point nearest the observed offset.
    # The initial stencil history is deliberately declared straight: the
    # perception contract gives a lateral offset and a heading error, not a
    # measured previous-station offset.  Using the heading to invent one makes
    # the finite model infeasible whenever tracking leaves a small heading
    # error (the first-step curvature demand then exceeds kappa_max), which is
    # a modelling artifact rather than a property of the vehicle.  The first
    # action's curvature is therefore the curvature from `cur` to the chosen
    # next offset; tracking is outside the certified model.
    cur = _snap(offsets, lat)
    prev = cur
    offs = tuple(sorted(set(int(v) for v in offsets) | {prev, cur}))
    if speeds is None:
        speeds = default_speed_grid(consts)
    sp = tuple(sorted(set(int(v) for v in speeds)))
    b0 = int(base["v0"]) ** 2
    lon = consts["lon"]
    cap = int(lon["a_lat"]) * int(lon["r_h"])
    cap = min(cap, int(lon["v_max"]) ** 2)
    if terminal_cap is not None:
        cap = min(cap, int(terminal_cap))
    return Model(consts=consts, q={"base": base, "lat": abs(lat)}, offsets=offs,
                 speeds=sp, horizon=m, start=State(0, prev, cur, b0), terminal_cap=cap)


def default_speed_grid(consts: dict, v_min_mps: float = 2.0, v_max_mps: float | None = 18.0,
                       dv_mps: float = 1.0) -> tuple[int, ...]:
    """Speed^2 grid from a physical speed grid (exact integer squares, mm^2/s^2)."""
    vmax = float(consts["lon"]["v_max"]) / 1000.0 if v_max_mps is None else float(v_max_mps)
    out = []
    v = v_min_mps
    while v <= vmax + 1e-9:
        mm = int(round(v * 1000.0))
        out.append(mm * mm)
        v += dv_mps
    return tuple(sorted(set(out)))


def model_from_instance(d: dict) -> Model:
    """Rebuild a model from its serialised instance (certificate checking)."""
    return Model(consts=d["consts"], q=d["q"], offsets=tuple(int(v) for v in d["offsets"]),
                 speeds=tuple(int(v) for v in d["speeds"]), horizon=int(d["horizon"]),
                 start=State.from_any(d["start"]), terminal_cap=int(d["terminal_cap"]),
                 model_id=str(d.get("model", MODEL_ID)), objective=str(d.get("objective", OBJECTIVE_ID)),
                 time_scale=int(d.get("time_scale", TIME_SCALE)))
