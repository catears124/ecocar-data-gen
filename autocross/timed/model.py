"""Time-indexed integer vehicle model ``timed-1`` (scalar reference).

This is the mathematical reference for the controller in ``teams/timed_exact.py``.
The numpy runtime solver (:mod:`autocross.timed.vector`), the compiled offline
oracle (:mod:`autocross.timed.oracle`) and the Lean checker
(``lean/Corpus/Timed.lean``) implement exactly these integer functions; parity
tests compare all of them.

Time base.  One model tick is one simulator physics tick (``H_US`` = 25 ms);
one control period (50 ms) is ``TICKS`` = 2 ticks.  The first plan block is one
control period, so the model predicts the consequence of exactly the command
that is published, for exactly the time it is applied.

State (all integers)::

    s      progress along the preview centreline from the observation point, mm
    d      lateral offset from the centreline, mm (+ left)
    psi    heading error w.r.t. the centreline, micro-rad (+ left)
    v      body longitudinal speed, mm/s (>= 0)
    vy     body lateral speed, mm/s (+ left)
    r      yaw rate, micro-rad/s (+ CCW)
    sigma  applied normalised steering (after the harness slew limiter), milli
    delta  road-wheel angle, micro-rad

Command (the published command, not an intermediate request)::

    u      steering command, milli (Command.steer = u/1000)
    p      pedal, milli: p >= 0 -> Command.throttle = p/1000, brake = 0
                          p <  0 -> Command.brake = -p/1000, throttle = 0

Tick dynamics: harness slew limiter on sigma (exact), first-order road-wheel lag
(the backend's per-substep lag folded into one tick factor), longitudinal force
from pedal (drive force/power limit, brake force with axle-limit cap, drag,
rolling resistance), linear single-track lateral dynamics (explicit Euler) with
the backend's kinematic blend below 3 m/s, and Frenet kinematics on the preview
centreline with polynomial trig.  Constraint violations are accumulated every
tick: box footprint vs. corridor (three stations along the body), footprint vs.
obstacle boxes, and per-axle friction circles with longitudinal load transfer.
After the last tick a terminal viability violation is added: braking-envelope
speed cap for every visible station (plus the declared domain radius past the
preview) and heading recoverability with the available curvature.

Objective (lexicographic, compared as tuples, no scalarisation)::

    J(plan) = ( V(plan),  -floor(P(plan) / PQ),  E(plan) )

``V`` total violation (0 iff every hard constraint holds), ``P`` = terminal
progress + terminal speed * T_TAIL (mm), ``E`` = steering/pedal change effort.
Ties are broken by the plan's index tuple (first in enumeration order).

Plans.  A plan is a tuple of action indices, one per block.  Block ``k`` lasts
``m_k`` control periods; its action index ``a`` selects steering offset
``C_k[a // |P_k|]`` (relative to sigma at the block start, then clamped to
[-1000, 1000]) and pedal ``P_k[a % |P_k|]``.  Every index tuple is admissible:
constraints enter through ``V``, so the plan set is the full Cartesian product.

All divisions are floor divisions by strictly positive divisors (identical in
Python, numpy, numba and Lean's Euclidean ``Int`` division).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

MODEL_ID = "timed-1"
SCHEMA = "autocross.timed.instance"
SCHEMA_VERSION = 1

E6 = 1_000_000
E9 = 1_000_000_000
BIG = 1_000_000_000          # violation used when a recovery requirement is unbounded
PSI_CLAMP = 1_500_000        # trig polynomial argument clamp (|psi| <= 1.5 rad)

# Declared constants.  Values are calibrated against the bicycle backend (see
# experiments/timed_calibration); they are model choices and part of every instance.
DEFAULT_CONSTS = {
    "H_US": 25_000,          # tick length, us
    "TICKS": 2,              # ticks per control period
    "SLEW": 50,              # harness slew limit per tick, milli (2.0/s * 25 ms)
    "MAX_STEER": 610_865,    # road-wheel angle at |steer| = 1, urad
    "LAG_NUM": 65_334,       # road-wheel lag factor per tick, (1 - 2.5/60)^10
    "LAG_DEN": 100_000,
    "L": 3000,               # wheelbase, mm
    "LF": 1450,              # CG to front axle, mm
    "LR": 1550,              # CG to rear axle, mm
    "HCG": 500,              # CG height, mm
    "MASS": 1845,            # kg
    "IZ": 4147,              # yaw inertia, kg m^2
    "CF": 130_919,           # front cornering stiffness, N/rad
    "CR": 122_473,           # rear cornering stiffness, N/rad
    "FD": 9900,              # max drive force, N
    "PW": 200_000,           # max drive power, W
    "FB": 17_500,            # brake force at full pedal, N
    "FBMAX": 16_000,         # axle-limited total brake force, N
    "FDF": 400,              # front share of drive force, milli
    "FBF": 600,              # front share of brake force, milli
    "CD": 330,               # aero drag, mN per (m/s)^2
    "FRR": 217,              # rolling resistance, N
    "G": 9810,               # mm/s^2
    "MU": 870,               # usable friction coefficient per axle (constraint budget), milli
    "MU_T": 970,             # tyre friction coefficient (force saturation), milli
    "V_KIN": 3000,           # kinematic blend upper speed, mm/s
    "V_FLOOR": 1000,         # slip-angle speed floor, mm/s
    "HL": 2395,              # half body length (box centred on the CG), mm
    "HW": 1330,              # half body width (1080) + cone margin (100) + clearance (150), mm
    "A_LT": 8000,            # terminal lateral-acceleration budget, mm/s^2
    "A_BT": 8000,            # terminal braking budget, mm/s^2
    "VMAX": 40_000,          # mm/s
    "RMIN": 6500,            # declared minimum road radius beyond the preview, mm
    "KSTEER": 233_000,       # steering-limited curvature, micro-1/m
    "STEP": 1000,            # preview spacing, mm
    "T_TAIL": 500,           # terminal speed tail, ms
    "PQ": 1,                 # progress quantum, mm
}


def cdiv(a: int, b: int) -> int:
    """Ceiling division for b > 0."""
    return -((-a) // b)


def isqrt(n: int) -> int:
    return math.isqrt(n) if n > 0 else 0


def clamp(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x


def trig(psi: int) -> tuple[int, int]:
    """(sin, cos) * 1e6 of a micro-radian angle; 3-term Taylor polynomials."""
    p = clamp(psi, -PSI_CLAMP, PSI_CLAMP)
    q = p * p // E6
    sn = p - p * q // (6 * E6)
    cs = E6 - q // 2 + q * q // (24 * E6)
    return sn, cs


def tan_u(delta: int) -> int:
    """tan(delta) * 1e6 for a micro-radian angle; 3-term Taylor polynomial."""
    q = delta * delta // E6
    return delta + delta * q // (3 * E6) + 2 * delta * q * q // (15 * E6 * E6)


@dataclass(frozen=True)
class Block:
    steps: int                   # control periods
    steer: tuple[int, ...]       # steering offsets, milli
    pedal: tuple[int, ...]       # pedal values, milli

    @property
    def n(self) -> int:
        return len(self.steer) * len(self.pedal)

    def action(self, a: int) -> tuple[int, int]:
        return self.steer[a // len(self.pedal)], self.pedal[a % len(self.pedal)]


DEFAULT_BLOCKS = (
    Block(1, (-100, -50, 0, 50, 100), (-1000, -600, -300, 0, 300, 600, 1000)),
    Block(3, (-200, -100, 0, 100, 200), (-1000, 0, 1000)),
    Block(6, (-300, 0, 300), (-1000, 0, 1000)),
)


@dataclass(frozen=True)
class State:
    s: int
    d: int
    psi: int
    v: int
    vy: int
    r: int
    sigma: int
    delta: int

    def as_list(self) -> list[int]:
        return [self.s, self.d, self.psi, self.v, self.vy, self.r, self.sigma, self.delta]


@dataclass(frozen=True)
class Instance:
    consts: dict
    k: tuple[int, ...]           # preview curvature, micro-1/m, station j*STEP
    wl: tuple[int, ...]          # left width, mm
    wr: tuple[int, ...]          # right width, mm
    obstacles: tuple[tuple[int, int, int, int], ...]   # (s_lo, s_hi, l_lo, l_hi), mm
    start: State
    p_prev: int                  # pedal applied before the observation, milli
    blocks: tuple[Block, ...] = DEFAULT_BLOCKS
    model: str = MODEL_ID

    def c(self, key: str) -> int:
        return int(self.consts[key])

    @property
    def n(self) -> int:
        return len(self.k)

    def to_json(self) -> dict:
        return {
            "schema": SCHEMA, "schema_version": SCHEMA_VERSION, "model": self.model,
            "consts": {k: int(v) for k, v in sorted(self.consts.items())},
            "k": list(self.k), "wl": list(self.wl), "wr": list(self.wr),
            "obstacles": [list(o) for o in self.obstacles],
            "start": self.start.as_list(), "p_prev": self.p_prev,
            "blocks": [{"steps": b.steps, "steer": list(b.steer), "pedal": list(b.pedal)}
                       for b in self.blocks],
        }

    @staticmethod
    def from_json(j: dict) -> "Instance":
        if j.get("schema") != SCHEMA or int(j.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("not a timed-1 instance")
        if j.get("model") != MODEL_ID:
            raise ValueError(f"unknown model {j.get('model')!r}")
        missing = set(DEFAULT_CONSTS) - set(j["consts"])
        if missing:
            raise ValueError(f"missing constants {sorted(missing)}")
        return Instance(
            consts={k: int(v) for k, v in j["consts"].items()},
            k=tuple(int(x) for x in j["k"]), wl=tuple(int(x) for x in j["wl"]),
            wr=tuple(int(x) for x in j["wr"]),
            obstacles=tuple(tuple(int(x) for x in o) for o in j["obstacles"]),  # type: ignore[misc]
            start=State(*(int(x) for x in j["start"])), p_prev=int(j["p_prev"]),
            blocks=tuple(Block(int(b["steps"]), tuple(int(x) for x in b["steer"]),
                               tuple(int(x) for x in b["pedal"])) for b in j["blocks"]),
        )

    def sha256(self) -> str:
        return hashlib.sha256(canonical(self.to_json()).encode()).hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def well_formed(inst: Instance) -> list[str]:
    """Domain guard (also checked by Lean): positive divisors and int64-safe magnitudes."""
    err = []
    c = inst.consts
    if set(c) != set(DEFAULT_CONSTS):
        err.append("constant set differs from the declared set")
        return err
    for key in ("H_US", "TICKS", "LAG_DEN", "L", "MASS", "IZ", "STEP", "PQ", "A_BT", "V_FLOOR", "MU_T"):
        if int(c[key]) <= 0:
            err.append(f"{key} must be positive")
    if not (0 <= c["LAG_NUM"] <= c["LAG_DEN"]):
        err.append("LAG_NUM outside [0, LAG_DEN]")
    if c["V_KIN"] <= c["V_FLOOR"] or c["V_KIN"] - c["V_FLOOR"] <= 0:
        err.append("V_KIN must exceed V_FLOOR")
    n = inst.n
    if n < 2 or len(inst.wl) != n or len(inst.wr) != n:
        err.append("preview arrays must have equal length >= 2")
    if any(abs(x) > 1_000_000 for x in inst.k):
        err.append("|curvature| > 1 /m")
    if any(not (0 <= x <= 50_000) for x in inst.wl + inst.wr):
        err.append("width outside [0, 50 m]")
    st = inst.start
    if not (0 <= st.v <= 60_000 and abs(st.vy) <= 30_000 and abs(st.r) <= 5_000_000
            and abs(st.d) <= 60_000 and abs(st.psi) <= 4_000_000 and abs(st.sigma) <= 1000
            and abs(st.delta) <= 1_000_000 and st.s == 0):
        err.append("start state outside the declared domain")
    if not inst.blocks or inst.blocks[0].steps != 1:
        err.append("the first block must last exactly one control period")
    for b in inst.blocks:
        if b.steps <= 0 or not b.steer or not b.pedal:
            err.append("empty block")
        if any(abs(x) > 2000 for x in b.steer) or any(abs(x) > 1000 for x in b.pedal):
            err.append("block action outside range")
    for o in inst.obstacles:
        if len(o) != 4 or o[0] > o[1] or o[2] > o[3]:
            err.append("malformed obstacle box")
    return err


def idx(inst: Instance, x: int) -> int:
    return clamp(x // inst.c("STEP"), 0, inst.n - 1)


def kappa_at(inst: Instance, x: int) -> int:
    """Curvature linearly interpolated between preview stations (clamped at the ends)."""
    step = inst.c("STEP")
    j = idx(inst, x)
    j1 = min(j + 1, inst.n - 1)
    fr = clamp(x - j * step, 0, step)
    return inst.k[j] + (inst.k[j1] - inst.k[j]) * fr // step


def tyre_force(fy_lin: int, fx: int, cap: int) -> int:
    """Saturating lateral tyre force, N.

    The backend uses ``Fy = -avail * tanh(C * alpha / avail)`` with
    ``avail = sqrt((mu*Fz)^2 - Fx^2)``; this is the same law with the (3,2) Pade
    approximation ``tanh(x) ~ x(27 + x^2)/(27 + 9x^2)`` evaluated in milli-units,
    so a linear tyre is used below the friction limit and the force saturates
    smoothly towards ``avail`` above it.
    """
    av2 = cap * cap - fx * fx
    avail = isqrt(av2) if av2 > 0 else 1
    x = clamp(fy_lin * 1000 // avail, -3000, 3000)
    num = 27_000_000 + x * x
    return avail * (x * num // (27_000_000 + 9 * x * x)) // 1000


def tick(inst: Instance, st: State, u: int, p: int) -> tuple[State, int]:
    """Advance one physics tick under command (u, p); return (state, violation)."""
    c = inst.c
    H = c("H_US")
    # 1. harness slew limiter (per physics tick), then clamp to the actuator range
    sigma = clamp(st.sigma + clamp(u - st.sigma, -c("SLEW"), c("SLEW")), -1000, 1000)
    # 2. road-wheel lag toward the applied target
    tgt = sigma * c("MAX_STEER") // 1000
    delta = tgt + (st.delta - tgt) * c("LAG_NUM") // c("LAG_DEN")
    v = st.v
    # 3. longitudinal force, N
    if p >= 0:
        fdrv = min(c("FD"), c("PW") * 1000 // max(v, 1000))
        fx = p * fdrv // 1000
        fxf = fx * c("FDF") // 1000
    else:
        fx = max(p * c("FB") // 1000, -c("FBMAX"))
        fxf = fx * c("FBF") // 1000
    fxr = fx - fxf
    fres = c("CD") * v * v // E9 + c("FRR")
    ax = (fx - fres) * 1000 // c("MASS")
    v1 = max(v + ax * H // E6, 0)
    # 4. lateral dynamics: linear single track with per-axle load transfer and a
    # friction-limited tyre force (the backend saturates at mu_tyre * Fz, so a
    # linear tyre would keep predicting grip that does not exist)
    fzf = c("MASS") * (c("G") * c("LR") - ax * c("HCG")) // (c("L") * 1000)
    fzr = c("MASS") * c("G") // 1000 - fzf
    vs = max(v, c("V_FLOOR"))
    af = (st.vy + c("LF") * st.r // E6) * E6 // vs - delta
    ar = (st.vy - c("LR") * st.r // E6) * E6 // vs
    demand_f = -(c("CF") * af // E6)
    demand_r = -(c("CR") * ar // E6)
    fyf = tyre_force(demand_f, fxf, c("MU_T") * fzf // 1000)
    fyr = tyre_force(demand_r, fxr, c("MU_T") * fzr // 1000)
    vy_dot = (fyf + fyr) * 1000 // c("MASS") - st.r * v // E6
    r_dot = (c("LF") * fyf - c("LR") * fyr) * 1000 // c("IZ")
    vy_d = st.vy + vy_dot * H // E6
    r_d = st.r + r_dot * H // E6
    w = clamp((c("V_KIN") - v1) * 1000 // (c("V_KIN") - c("V_FLOOR")), 0, 1000)
    r_k = v1 * tan_u(delta) // c("L")
    vy_k = c("LR") * r_k // E6
    r1 = (w * r_k + (1000 - w) * r_d) // 1000
    vy1 = (w * vy_k + (1000 - w) * vy_d) // 1000
    # 5. Frenet kinematics on the preview centreline; curvature at the tick midpoint
    sn, cs = trig(st.psi)
    vt = (v1 * cs - vy1 * sn) // E6
    kap = kappa_at(inst, st.s + (vt * H // E6) // 2)
    den = max(E9 - kap * st.d, E9 // 10)
    sdot = vt * E9 // den
    ddot = (v1 * sn + vy1 * cs) // E6
    psidot = r1 - kap * sdot // 1000
    s1 = st.s + sdot * H // E6
    d1 = st.d + ddot * H // E6
    psi1 = st.psi + psidot * H // E6
    out = State(s1, d1, psi1, v1, vy1, r1, sigma, delta)
    # 6. violations at the end of the tick
    # The grip budget is enforced on the *demanded* lateral force (what the plan
    # asks the tyres for), not on the saturated force: a demand above
    # MU * Fz is a plan the tyres cannot deliver with reserve, even though the
    # saturating tyre law would cap the force actually applied.
    viol = (footprint_violation(inst, out)
            + friction_violation(inst, demand_f, demand_r, fxf, fxr, ax))
    return out, viol


def body_points(inst: Instance, st: State) -> list[tuple[int, int]]:
    """(station, lateral centre) at the rear, middle and front of the body box."""
    hl = inst.c("HL")
    kap = kappa_at(inst, st.s)
    sn, _ = trig(st.psi)
    out = []
    for ell in (-hl, 0, hl):
        lat = st.d + ell * sn // E6 - kap * ell * ell // (2 * E9)
        out.append((st.s + ell, lat))
    return out


def footprint_violation(inst: Instance, st: State) -> int:
    hw = inst.c("HW")
    hl = inst.c("HL")
    pts = body_points(inst, st)
    viol = 0
    for s_, lat in pts:
        j = idx(inst, s_)
        viol += max(0, lat + hw - inst.wl[j]) + max(0, -inst.wr[j] - (lat - hw))
    lo = min(lat for _, lat in pts) - hw
    hi = max(lat for _, lat in pts) + hw
    for s_lo, s_hi, l_lo, l_hi in inst.obstacles:
        if st.s + hl >= s_lo and st.s - hl <= s_hi:
            viol += max(0, min(hi - l_lo, l_hi - lo))
    return viol


def friction_violation(inst: Instance, fyf: int, fyr: int, fxf: int, fxr: int, ax: int) -> int:
    c = inst.c
    fzf = c("MASS") * (c("G") * c("LR") - ax * c("HCG")) // (c("L") * 1000)
    fzr = c("MASS") * c("G") // 1000 - fzf
    viol = 0
    for fy, fx, fz in ((fyf, fxf, fzf), (fyr, fxr, fzr)):
        cap = max(c("MU") * fz // 1000, 1)
        ex = fy * fy + fx * fx - cap * cap
        if ex > 0:
            viol += cdiv(ex, cap)
    return viol


def terminal_violation(inst: Instance, st: State) -> int:
    c = inst.c
    a_lt, a_bt, step = c("A_LT"), c("A_BT"), c("STEP")
    v2 = st.v * st.v
    vmax2 = c("VMAX") * c("VMAX")
    worst = v2 - (a_lt * c("RMIN") + 2 * a_bt * ((inst.n - 1) * step - st.s))
    for j in range(1, inst.n):
        sj = j * step
        if sj <= st.s:
            continue
        kj = abs(inst.k[j])
        cap = vmax2 if kj == 0 else min(vmax2, a_lt * E9 // kj)
        worst = max(worst, v2 - (cap + 2 * a_bt * (sj - st.s)))
    viol = cdiv(worst, 2 * a_bt) if worst > 0 else 0
    # heading recoverability with the available curvature margin
    kcap = min(a_lt * E9 // max(v2, 1), c("KSTEER"))
    kap = kappa_at(inst, st.s)
    j = idx(inst, st.s)
    psi = clamp(st.psi, -PSI_CLAMP, PSI_CLAMP)
    hw = c("HW")
    if psi > 0:
        dk = kcap + kap
        room = inst.wl[j] - hw - st.d
    elif psi < 0:
        dk = kcap - kap
        room = st.d + inst.wr[j] - hw
    else:
        dk, room = 1, 0
    if psi != 0:
        req = BIG if dk <= 0 else cdiv(psi * psi, 2 * dk * 1000)
        viol += max(0, req - room)
    return viol


@dataclass
class Rollout:
    states: list[State] = field(default_factory=list)   # state after every tick
    viol: int = 0
    effort: int = 0
    commands: list[tuple[int, int]] = field(default_factory=list)  # (u, p) per block


def rollout(inst: Instance, plan: Sequence[int]) -> Rollout:
    if len(plan) != len(inst.blocks):
        raise ValueError("plan length differs from the number of blocks")
    st = inst.start
    ro = Rollout()
    p_last = inst.p_prev
    for b, a in zip(inst.blocks, plan):
        if not (0 <= a < b.n):
            raise ValueError("action index out of range")
        off, p = b.action(a)
        u = clamp(st.sigma + off, -1000, 1000)
        ro.effort += abs(u - st.sigma) + abs(p - p_last)
        p_last = p
        ro.commands.append((u, p))
        for _ in range(b.steps * inst.c("TICKS")):
            st, vt = tick(inst, st, u, p)
            ro.viol += vt
            ro.states.append(st)
    ro.viol += terminal_violation(inst, st)
    return ro


def objective(inst: Instance, ro: Rollout) -> tuple[int, int, int]:
    st = ro.states[-1]
    prog = st.s + st.v * inst.c("T_TAIL") // 1000
    return (ro.viol, -(prog // inst.c("PQ")), ro.effort)


def evaluate(inst: Instance, plan: Sequence[int]) -> tuple[int, int, int]:
    return objective(inst, rollout(inst, plan))


def all_plans(inst: Instance) -> Iterable[tuple[int, ...]]:
    return itertools.product(*(range(b.n) for b in inst.blocks))


def solve_exhaustive(inst: Instance) -> tuple[tuple[int, ...], tuple[int, int, int]]:
    """Independent reference: evaluate every plan from scratch; first minimum wins."""
    best = None
    best_j = None
    for plan in all_plans(inst):
        j = evaluate(inst, plan)
        if best_j is None or j < best_j:
            best, best_j = plan, j
    assert best is not None and best_j is not None
    return best, best_j


def first_command(inst: Instance, plan: Sequence[int]) -> tuple[int, int]:
    off, p = inst.blocks[0].action(plan[0])
    return clamp(inst.start.sigma + off, -1000, 1000), p


def command_floats(u: int, p: int) -> tuple[float, float, float]:
    """The exact published command (steer, throttle, brake) for model command (u, p)."""
    return u / 1000.0, (p / 1000.0 if p > 0 else 0.0), (-p / 1000.0 if p < 0 else 0.0)


__all__ = [
    "MODEL_ID", "DEFAULT_CONSTS", "DEFAULT_BLOCKS", "Block", "State", "Instance", "Rollout",
    "tick", "rollout", "objective", "evaluate", "all_plans", "solve_exhaustive",
    "first_command", "command_floats", "well_formed", "terminal_violation",
    "footprint_violation", "friction_violation", "trig", "tan_u", "cdiv", "clamp", "canonical",
]
