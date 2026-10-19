"""Integer, path-relative actuator model for the direct optimal controller.

The model's state is the vehicle's relation to the preview centreline at a
station, its squared speed, and its road-wheel steering angle:

    (station i, lateral offset d, heading error psi, speed^2 b, steering delta)

A control is an actual actuator request ``(u, a)``: ``u`` is the normalised
steering command in [-1000, 1000] and ``a`` is the requested longitudinal
acceleration in mm/s^2.  The first control of an optimal sequence is published
as a ``Command``; the pedal map is the only remaining actuation layer.

All quantities are integers: mm, mm/s, mm^2/s^2, mm/s^2, micro-radians,
micro-1/m.  The transition is a rate- and lag-limited kinematic bicycle with a
declared understeer correction.  It is a finite model: every successor is
snapped to the declared grids.  Global optimality is claimed for this finite
transition, not for the simulator's continuous dynamics.

The understeer term uses the steady-state relation ``kappa = delta /
(L + K_us * v^2)``.  ``K_US_MM_NUM / K_US_B_DEN`` is the declared K_us in
mm per (mm^2/s^2); it is calibrated against simulator telemetry and is a model
constant, not a theorem.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, NamedTuple

from autocross.datagen import cert, steer

MODEL_ID = "actuator-1"
OBJECTIVE_ID = "sum_ceiling_constant_accel_plus_terminal"
TIME_SCALE = 1_000_000
COST_TIME_WEIGHT = 1000  # lexicographic: 1000*time + effort + tracking
D_WEIGHT = 5000
PSI_WEIGHT = 5000

# Declared finite grids.  They are model choices, deliberately exposed.
DEFAULT_OFFSETS = tuple(range(-750, 751, 250))            # 7
DEFAULT_PSIS = tuple(range(-80_000, 80_001, 40_000))     # 5 (0.04 rad)
DEFAULT_DELTAS = (-300_000, 0, 300_000)                    # 3
DEFAULT_STEER_INPUTS = (-1000, 0, 1000)
DEFAULT_ACC_INPUTS = (-6000, -3000, 0, 3000, 5200)
DEFAULT_HORIZON = 4
DEFAULT_TERMINAL_DISTANCE_MM = 8_000
K_US_MM_NUM = 25
K_US_B_DEN = 10_000_000  # den = L + 25*b/1e7 mm; b=1e8 -> +250 mm
V_MIN_MM_S = 1000


class ActState(NamedTuple):
    station: int
    d: int
    psi: int
    b: int
    delta: int

    def as_dict(self) -> dict:
        return {"station": self.station, "d": self.d, "psi": self.psi,
                "b": self.b, "delta": self.delta}

    @staticmethod
    def from_any(x) -> "ActState":
        if isinstance(x, ActState):
            return x
        if isinstance(x, dict):
            return ActState(int(x["station"]), int(x["d"]), int(x["psi"]),
                            int(x["b"]), int(x["delta"]))
        return ActState(int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(x[4]))


def isqrt(n: int) -> int:
    return math.isqrt(max(0, int(n)))


def cdiv(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("cdiv divisor must be positive")
    return -((-int(a)) // int(b))


def snap(x: int, vals: tuple[int, ...]) -> int:
    """Nearest declared grid value (ties to the larger value), then clamp.

    Works for non-uniform grids (the speed-squared grid is not uniform).
    """
    import bisect
    x = int(x)
    if x <= vals[0]:
        return vals[0]
    if x >= vals[-1]:
        return vals[-1]
    i = bisect.bisect_left(vals, x)
    left, right = vals[i - 1], vals[i]
    return right if x - left >= right - x else left


def snap_np(x, vals):
    import numpy as np
    vals = np.asarray(vals, dtype=np.int64)
    x = np.asarray(x)
    i = np.clip(np.searchsorted(vals, x, side="left"), 1, vals.size - 1)
    left = vals[i - 1]
    right = vals[i]
    y = np.where(x - left >= right - x, right, left)
    return np.where(x <= vals[0], vals[0], np.where(x >= vals[-1], vals[-1], y))


def time_cost(d: int, b: int, w: int) -> int:
    if b < 0 or w < 0:
        raise ValueError("negative speed^2")
    den = isqrt(b) + isqrt(w)
    if den == 0:
        raise ValueError("zero-speed segment has no finite time")
    return cdiv(2 * d * TIME_SCALE, den)


@dataclass(frozen=True)
class ActModel:
    consts: dict
    q: dict
    offsets: tuple[int, ...]
    psis: tuple[int, ...]
    speeds: tuple[int, ...]
    deltas: tuple[int, ...]
    steer_inputs: tuple[int, ...]
    acc_inputs: tuple[int, ...]
    horizon: int
    start: ActState
    terminal_distance: int
    ref_d: tuple[int, ...] = ()
    ref_psi: tuple[int, ...] = ()
    understeer_num: int = K_US_MM_NUM
    understeer_den: int = K_US_B_DEN
    terminal_min_b: int = 4_000_000
    terminal_max_psi: int = 120_000
    terminal_max_delta: int = 600_000
    model_id: str = MODEL_ID
    objective: str = OBJECTIVE_ID

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
    def ds(self) -> int:
        return int(self.consts["step"])

    @property
    def max_steer(self) -> int:
        return int(self.consts["max_steer"])

    @property
    def steer_rate(self) -> int:
        return int(self.consts["steer_rate_urad_s"])

    @property
    def tau(self) -> int:
        return int(self.consts["steer_tau_us"])

    @property
    def wheelbase(self) -> int:
        return int(self.consts["wheelbase_mm"])

    @property
    def half_width(self) -> int:
        return int(self.lon["veh_width"]) // 2

    @property
    def margin(self) -> int:
        return int(self.consts["margin"])

    def gap(self, i: int) -> int:
        return self.s[i + 1] - self.s[i]

    def lo(self, i: int) -> int:
        return -(int(self.q["base"]["wr"][i]) - self.half_width - self.margin)

    def hi(self, i: int) -> int:
        return int(self.q["base"]["wl"][i]) - self.half_width - self.margin

    def corridor_ok(self, i: int, d: int) -> bool:
        return self.lo(i) <= d <= self.hi(i)

    def obstacles_at(self, i: int) -> list[tuple[int, int, int]]:
        si = self.s[i]
        hold = int(self.lon["hold"])
        return [(int(ob[0]), int(ob[1]), int(ob[2])) for ob in self.q["base"]["obstacles"]
                if isinstance(ob, (list, tuple)) and len(ob) == 3
                and int(ob[0]) - hold <= si <= int(ob[0]) + hold]

    def obstacle_ok(self, i: int, d: int) -> bool:
        half = self.half_width + self.margin
        for _, lmin, lmax in self.obstacles_at(i):
            if not (d <= lmin - half or d >= lmax + half):
                return False
        return True

    def lateral_ok(self, i: int, d: int) -> bool:
        return self.corridor_ok(i, d) and self.obstacle_ok(i, d)

    def b_max(self) -> int:
        return max(self.speeds)

    def b_min(self) -> int:
        return min(self.speeds)

    def terminal_cap(self) -> int:
        """Largest terminal speed^2 that can still brake for the visible road.

        The local grip cap alone is unsafe when a corner follows the horizon;
        a fixed radius cap is over-conservative and forces every plan to
        ~3.4 m/s.  The braking envelope below is the recursive-feasibility
        condition: from the horizon speed, every later station j must satisfy
        ``b_H <= cap_j + 2*a_brk*(s_j-s_H)``.
        """
        vmax2 = int(self.lon["v_max"]) ** 2
        h = min(self.horizon, len(self.k) - 1)
        k = abs(self.k[h])
        cap = vmax2 if k == 0 else min(vmax2, int(self.lon["a_lat"]) * 1_000_000_000 // k)
        for j in range(h + 1, len(self.k)):
            kj = abs(self.k[j])
            cj = vmax2 if kj == 0 else min(vmax2, int(self.lon["a_lat"]) * 1_000_000_000 // kj)
            cap = min(cap, cj + 2 * int(self.lon["a_brk"]) * (self.s[j] - self.s[h]))
        return max(cap, 0)

    def terminal_ok(self, s: ActState, *, index: int | None = None,
                    cap: int | None = None) -> bool:
        i = s.station if index is None else index
        if i != self.horizon:
            return False
        if not self.lateral_ok(i, s.d):
            return False
        if abs(s.psi) > self.terminal_max_psi or abs(s.delta) > self.terminal_max_delta:
            return False
        return self.terminal_min_b <= s.b <= (self.terminal_cap() if cap is None else cap)

    def terminal_cost(self, s: ActState, *, cap: int | None = None) -> int | None:
        if not self.terminal_ok(s, cap=cap):
            return None
        if self.terminal_distance <= 0:
            return 0
        v = isqrt(s.b)
        if v <= 0:
            return None
        return cdiv(self.terminal_distance * TIME_SCALE, v)

    # -- scalar transition (also the Lean reference) -------------------------
    def next_state(self, s: ActState, u: int, a: int) -> ActState | None:
        if s.station >= self.horizon:
            return None
        if u not in self.steer_inputs or a not in self.acc_inputs:
            return None
        ds = self.gap(s.station)
        v = isqrt(s.b)
        dt = ds * 1_000_000 // max(v, V_MIN_MM_S)
        target = u * self.max_steer // 1000
        lag = (target - s.delta) * min(dt, self.tau) // self.tau
        lim = self.steer_rate * dt // 1_000_000
        change = min(max(lag, -lim), lim)
        delta = snap(s.delta + change, self.deltas)
        den = self.wheelbase + self.understeer_num * s.b // self.understeer_den
        if den <= 0:
            return None
        kappa = delta * 1000 // den
        kroad = self.k[s.station]
        psi = snap(s.psi + (kappa - kroad) * ds // 1000, self.psis)
        d = snap(s.d + ((s.psi + psi) // 2) * ds // 1_000_000, self.offsets)
        b_cont = s.b + 2 * a * ds
        if b_cont < 0:
            return None
        b = snap(b_cont, self.speeds)
        t = ActState(s.station + 1, d, psi, b, delta)
        if not self.lateral_ok(t.station, t.d):
            return None
        if not self.lateral_ok(t.station, (t.d + s.d) // 2):
            return None
        return t

    def _lateral_accel(self, s: ActState, t: ActState) -> int:
        """b * |path curvature| in the model's integer scale (a_lat bound is 1e9*a)."""
        ds = self.gap(s.station)
        kappa_path = self.k[s.station] + (t.psi - s.psi) * 1000 // ds
        return t.b * abs(kappa_path)

    def action_ok(self, s: ActState, u: int, a: int) -> ActState | None:
        t = self.next_state(s, u, a)
        if t is None:
            return None
        if not (-int(self.lon["a_brk"]) <= a <= int(self.lon["a_acc"])):
            return None
        if self._lateral_accel(s, t) > int(self.lon["a_lat"]) * 1_000_000_000:
            return None
        # Friction ellipse in physical units: (lat/a_lat)^2 + (a/a_long)^2 <= 1.
        # b*kappa/1e9 is lateral acceleration in mm/s^2.
        a_lat = int(self.lon["a_lat"])
        a_long = int(self.lon["a_acc"]) if a >= 0 else int(self.lon["a_brk"])
        kappa_path = self.k[s.station] + (t.psi - s.psi) * 1000 // self.gap(s.station)
        lat = t.b * abs(kappa_path) // 1_000_000_000
        if (lat * a_long) ** 2 + (a * a_lat) ** 2 > (a_lat * a_long) ** 2:
            return None
        return t

    def action_cost(self, s: ActState, t: ActState) -> int:
        # Lexicographic (time, steering effort): effort is bounded by 1000,
        # so a one-microsecond time difference dominates any effort difference.
        rd = self.ref_d[s.station + 1] if len(self.ref_d) > s.station + 1 else 0
        rp = self.ref_psi[s.station + 1] if len(self.ref_psi) > s.station + 1 else 0
        return (COST_TIME_WEIGHT * time_cost(self.gap(s.station), s.b, t.b)
                + abs(t.delta - s.delta) // 1000
                + D_WEIGHT * abs(t.d - rd) + PSI_WEIGHT * abs(t.psi - rp) // 100)

    def actions(self, s: ActState) -> list[tuple[ActState, int, int, int]]:
        out = []
        for u in self.steer_inputs:
            for a in self.acc_inputs:
                t = self.action_ok(s, u, a)
                if t is not None:
                    out.append((t, self.action_cost(s, t), u, a))
        return out

    def as_instance(self) -> dict:
        return {
            "model": self.model_id, "objective": self.objective,
            "consts": self.consts, "q": self.q,
            "offsets": list(self.offsets), "psis": list(self.psis),
            "speeds": list(self.speeds), "deltas": list(self.deltas),
            "steer_inputs": list(self.steer_inputs), "acc_inputs": list(self.acc_inputs),
            "horizon": self.horizon, "start": self.start.as_dict(),
            "terminal_distance": self.terminal_distance,
            "ref_d": list(self.ref_d), "ref_psi": list(self.ref_psi),
            "understeer_num": self.understeer_num, "understeer_den": self.understeer_den,
            "terminal_min_b": self.terminal_min_b,
            "terminal_max_psi": self.terminal_max_psi,
            "terminal_max_delta": self.terminal_max_delta,
        }


def model_from_instance(instance: dict) -> ActModel:
    """Re-encode the declared model.  No field is taken from a solution table."""
    return ActModel(
        consts={k: dict(v) if isinstance(v, dict) else v for k, v in instance["consts"].items()},
        q=instance["q"],
        offsets=tuple(int(x) for x in instance["offsets"]),
        psis=tuple(int(x) for x in instance["psis"]),
        speeds=tuple(int(x) for x in instance["speeds"]),
        deltas=tuple(int(x) for x in instance["deltas"]),
        steer_inputs=tuple(int(x) for x in instance["steer_inputs"]),
        acc_inputs=tuple(int(x) for x in instance["acc_inputs"]),
        horizon=int(instance["horizon"]),
        start=ActState.from_any(instance["start"]),
        terminal_distance=int(instance["terminal_distance"]),
        ref_d=tuple(int(x) for x in instance.get("ref_d", [])),
        ref_psi=tuple(int(x) for x in instance.get("ref_psi", [])),
        understeer_num=int(instance.get("understeer_num", K_US_MM_NUM)),
        understeer_den=int(instance.get("understeer_den", K_US_B_DEN)),
        terminal_min_b=int(instance.get("terminal_min_b", 4_000_000)),
        terminal_max_psi=int(instance.get("terminal_max_psi", 120_000)),
        terminal_max_delta=int(instance.get("terminal_max_delta", 600_000)),
        model_id=str(instance.get("model", MODEL_ID)),
        objective=str(instance.get("objective", OBJECTIVE_ID)),
    )
