"""Vectorised exact solver for ``timed-1`` (numpy only: runs inside the sandbox).

Breadth-first over plan blocks: every node of block ``k`` is expanded by every
action of block ``k`` and simulated with the same integer tick as
:func:`autocross.timed.model.tick`, vectorised over nodes.  Leaves are compared
with the lexicographic objective ``(V, -floor(P/PQ), E)`` and ties go to the
smallest plan index, which is the enumeration order of
:func:`autocross.timed.model.all_plans`.

Exact pruning (optional, ``incumbent``): violation is a sum of non-negative
terms, so a node whose accumulated violation already exceeds the violation of a
complete incumbent plan cannot reach a leaf with ``J <= J(incumbent)``; such
nodes are dropped at block boundaries.  Nothing else is pruned.  The incumbent
itself is a complete plan evaluated by the same arithmetic, so the optimum (and
its tie-break) is unchanged; the offline oracles enumerate without pruning and
the parity tests compare both.

Every intermediate magnitude is checked against int64-safe bounds at block
boundaries; if any bound is exceeded the solve raises :class:`DomainOverflow`
instead of returning a wrapped value.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import BIG, E6, E9, PSI_CLAMP, Instance

I64 = np.int64
# Bounds that keep every product in tick() below 2**62 (see model.py docstring).
SAFE = {"v": 80_000, "vy": 60_000, "r": 8_000_000, "d": 100_000, "psi": 6_000_000,
        "delta": 1_000_000, "s": 200_000}


class DomainOverflow(ArithmeticError):
    pass


def _trig(psi):
    p = np.clip(psi, -PSI_CLAMP, PSI_CLAMP)
    q = p * p // E6
    return p - p * q // (6 * E6), E6 - q // 2 + q * q // (24 * E6)


def _tan(delta):
    q = delta * delta // E6
    return delta + delta * q // (3 * E6) + 2 * delta * q * q // (15 * E6 * E6)


def _cdiv(a, b):
    return -((-a) // b)


def _tyre(fy_lin, fx, cap):
    """Vectorised `model.tyre_force` (Padé tanh, fractional power via sqrt)."""
    av2 = cap * cap - fx * fx
    avail = np.maximum(np.sqrt(np.maximum(av2, 0).astype(np.float64)).astype(np.int64), 1)
    x = np.clip(fy_lin * 1000 // avail, -3000, 3000)
    num = 27_000_000 + x * x
    return avail * (x * num // (27_000_000 + 9 * x * x)) // 1000


@dataclass
class Nodes:
    s: np.ndarray
    d: np.ndarray
    psi: np.ndarray
    v: np.ndarray
    vy: np.ndarray
    r: np.ndarray
    sigma: np.ndarray
    delta: np.ndarray
    viol: np.ndarray
    effort: np.ndarray
    p_last: np.ndarray
    plan: np.ndarray          # (N, depth) action indices

    FIELDS = ("s", "d", "psi", "v", "vy", "r", "sigma", "delta", "viol", "effort", "p_last")

    def take(self, sel) -> "Nodes":
        return Nodes(*(getattr(self, f)[sel] for f in self.FIELDS), plan=self.plan[sel])

    def repeat(self, n: int) -> "Nodes":
        return Nodes(*(np.repeat(getattr(self, f), n) for f in self.FIELDS),
                     plan=np.repeat(self.plan, n, axis=0))


class Solver:
    """Pre-binds the instance constants and preview arrays once per solve."""

    def __init__(self, inst: Instance):
        self.inst = inst
        c = {k: int(v) for k, v in inst.consts.items()}
        self.c = c
        self.K = np.asarray(inst.k, dtype=I64)
        self.WL = np.asarray(inst.wl, dtype=I64)
        self.WR = np.asarray(inst.wr, dtype=I64)
        self.n = len(inst.k)
        self.step = c["STEP"]
        self.obs = [tuple(int(x) for x in o) for o in inst.obstacles]
        self.ticks = 0

    def _idx(self, x):
        return np.clip(x // self.step, 0, self.n - 1)

    def _kap(self, x):
        j = self._idx(x)
        j1 = np.minimum(j + 1, self.n - 1)
        fr = np.minimum(np.maximum(x - j * self.step, 0), self.step)
        k0 = self.K[j]
        return k0 + (self.K[j1] - k0) * fr // self.step

    def _check(self, nd: Nodes) -> None:
        for f, lim in SAFE.items():
            a = getattr(nd, f)
            if a.size and int(np.abs(a).max()) > lim:
                raise DomainOverflow(f"{f} exceeds {lim}")

    def tick(self, nd: Nodes, u, p, pos, brake_tab=None, share=None, sn=None, cs=None):
        """Advance every node one tick.  ``sn``/``cs`` are sin/cos of ``nd.psi``
        (passed in to avoid recomputing them twice per tick); returns the trig
        values of the new heading for the next tick."""
        c = self.c
        H = c["H_US"]
        mx, mn = np.maximum, np.minimum
        slew = c["SLEW"]
        sigma = mn(mx(nd.sigma + mn(mx(u - nd.sigma, -slew), slew), -1000), 1000)
        tgt = sigma * c["MAX_STEER"] // 1000
        delta = tgt + (nd.delta - tgt) * c["LAG_NUM"] // c["LAG_DEN"]
        v = nd.v
        fdrv = mn(c["FD"], c["PW"] * 1000 // mx(v, 1000))
        if brake_tab is None:
            brake_tab = mx(p * c["FB"] // 1000, -c["FBMAX"])
            share = np.where(pos, c["FDF"], c["FBF"])
        fx = np.where(pos, p * fdrv // 1000, brake_tab)
        fxf = fx * share // 1000
        fxr = fx - fxf
        ax = (fx - (v * v * c["CD"] // E9 + c["FRR"])) * 1000 // c["MASS"]
        v1 = mx(v + ax * H // E6, 0)
        fzf = c["MASS"] * (c["G"] * c["LR"] - ax * c["HCG"]) // (c["L"] * 1000)
        fzr = c["MASS"] * c["G"] // 1000 - fzf
        vs = mx(v, c["V_FLOOR"])
        r = nd.r
        vy = nd.vy
        af = (vy + c["LF"] * r // E6) * E6 // vs - delta
        ar = (vy - c["LR"] * r // E6) * E6 // vs
        demand_f = -(c["CF"] * af // E6)
        demand_r = -(c["CR"] * ar // E6)
        fyf = _tyre(demand_f, fxf, c["MU_T"] * fzf // 1000)
        fyr = _tyre(demand_r, fxr, c["MU_T"] * fzr // 1000)
        vy_d = vy + ((fyf + fyr) * 1000 // c["MASS"] - r * v // E6) * H // E6
        r_d = r + ((c["LF"] * fyf - c["LR"] * fyr) * 1000 // c["IZ"]) * H // E6
        if bool((v1 < c["V_KIN"]).any()):
            w = mn(mx((c["V_KIN"] - v1) * 1000 // (c["V_KIN"] - c["V_FLOOR"]), 0), 1000)
            r_k = v1 * _tan(delta) // c["L"]
            vy_k = c["LR"] * r_k // E6
            r1 = (w * r_k + (1000 - w) * r_d) // 1000
            vy1 = (w * vy_k + (1000 - w) * vy_d) // 1000
        else:
            # every node has w == 0, and (0*x + 1000*y)//1000 == y exactly
            r1, vy1 = r_d, vy_d
        if sn is None:
            sn, cs = _trig(nd.psi)
        vt = (v1 * cs - vy1 * sn) // E6
        kap = self._kap(nd.s + (vt * H // E6) // 2)
        den = mx(E9 - kap * nd.d, E9 // 10)
        sdot = vt * E9 // den
        ddot = (v1 * sn + vy1 * cs) // E6
        psidot = r1 - kap * sdot // 1000
        s1 = nd.s + sdot * H // E6
        d1 = nd.d + ddot * H // E6
        psi1 = nd.psi + psidot * H // E6
        nd.s, nd.d, nd.psi, nd.v, nd.vy, nd.r, nd.sigma, nd.delta = s1, d1, psi1, v1, vy1, r1, sigma, delta
        sn1, cs1 = _trig(psi1)
        nd.viol = nd.viol + self.footprint(s1, d1, psi1, sn1) + self.friction(demand_f, demand_r, fxf, fxr, fzf, fzr)
        self.ticks += 1
        return sn1, cs1

    def footprint(self, s, d, psi, sn):
        """Corridor and obstacle violation of the body box at (s, d, psi)."""
        c = self.c
        hw, hl = c["HW"], c["HL"]
        kap = self._kap(s)
        mx = np.maximum
        viol = None
        lats = []
        kterm = kap * (hl * hl) // (2 * E9)
        lat_front = d + hl * sn // E6 - kterm
        lat_rear = d - hl * sn // E6 - kterm
        for ell, lat in ((-hl, lat_rear), (0, d), (hl, lat_front)):
            j = self._idx(s + ell)
            term = mx(0, lat + hw - self.WL[j]) + mx(0, -self.WR[j] - (lat - hw))
            viol = term if viol is None else viol + term
            lats.append(lat)
        if self.obs:
            lo = np.minimum(np.minimum(lats[0], lats[1]), lats[2]) - hw
            hi = np.maximum(np.maximum(lats[0], lats[1]), lats[2]) + hw
            for s_lo, s_hi, l_lo, l_hi in self.obs:
                hit = (s + hl >= s_lo) & (s - hl <= s_hi)
                viol = viol + np.where(hit, np.maximum(0, np.minimum(hi - l_lo, l_hi - lo)), 0)
        return viol

    def friction(self, fyf, fyr, fxf, fxr, fzf, fzr):
        c = self.c
        viol = None
        for fy, fx, fz in ((fyf, fxf, fzf), (fyr, fxr, fzr)):
            cap = np.maximum(c["MU"] * fz // 1000, 1)
            ex = fy * fy + fx * fx - cap * cap
            term = np.where(ex > 0, _cdiv(ex, cap), 0)
            viol = term if viol is None else viol + term
        return viol

    def terminal(self, nd: Nodes):
        c = self.c
        a_lt, a_bt, step = c["A_LT"], c["A_BT"], self.step
        v2 = nd.v * nd.v
        vmax2 = c["VMAX"] * c["VMAX"]
        worst = v2 - (a_lt * c["RMIN"] + 2 * a_bt * ((self.n - 1) * step - nd.s))
        s = nd.s
        for j in range(1, self.n):
            sj = j * step
            kj = abs(int(self.K[j]))
            cap = vmax2 if kj == 0 else min(vmax2, a_lt * E9 // kj)
            cand = v2 - (cap + 2 * a_bt * (sj - s))
            worst = np.where(sj > s, np.maximum(worst, cand), worst)
        viol = np.where(worst > 0, _cdiv(worst, 2 * a_bt), 0)
        kcap = np.minimum(a_lt * E9 // np.maximum(v2, 1), c["KSTEER"])
        j = self._idx(s)
        kap = self._kap(s)
        psi = np.clip(nd.psi, -PSI_CLAMP, PSI_CLAMP)
        hw = c["HW"]
        dk = np.where(psi > 0, kcap + kap, kcap - kap)
        room = np.where(psi > 0, self.WL[j] - hw - nd.d, nd.d + self.WR[j] - hw)
        req = np.where(dk <= 0, BIG, _cdiv(psi * psi, 2 * np.maximum(dk, 1) * 1000))
        viol = viol + np.where(psi != 0, np.maximum(0, req - room), 0)
        return viol

    def objective(self, nd: Nodes):
        prog = nd.s + nd.v * self.c["T_TAIL"] // 1000
        return -(prog // self.c["PQ"])

    def start_nodes(self) -> Nodes:
        st = self.inst.start
        one = lambda x: np.array([x], dtype=I64)  # noqa: E731
        return Nodes(one(st.s), one(st.d), one(st.psi), one(st.v), one(st.vy), one(st.r),
                     one(st.sigma), one(st.delta), one(0), one(0), one(self.inst.p_prev),
                     plan=np.zeros((1, 0), dtype=I64))

    def run_block(self, nd: Nodes, k: int) -> Nodes:
        b = self.inst.blocks[k]
        na = b.n
        n0 = nd.s.size
        nd = nd.repeat(na)
        a = np.tile(np.arange(na, dtype=I64), n0)
        steer = np.asarray(b.steer, dtype=I64)[a // len(b.pedal)]
        ped = np.asarray(b.pedal, dtype=I64)[a % len(b.pedal)]
        u = np.clip(nd.sigma + steer, -1000, 1000)
        nd.effort = nd.effort + np.abs(u - nd.sigma) + np.abs(ped - nd.p_last)
        nd.p_last = ped
        nd.plan = np.concatenate([nd.plan, a[:, None]], axis=1)
        pos = ped >= 0
        brake_tab = np.maximum(ped * self.c["FB"] // 1000, -self.c["FBMAX"])
        share = np.where(pos, self.c["FDF"], self.c["FBF"])
        sn, cs = _trig(nd.psi)
        for _ in range(b.steps * self.c["TICKS"]):
            sn, cs = self.tick(nd, u, ped, pos, brake_tab, share, sn, cs)
        self._check(nd)
        return nd

    def solve(self, incumbent: tuple[int, ...] | None = None):
        """Return (plan, J, stats).

        ``incumbent`` enables exact pruning on violation only: every per-tick
        violation term is non-negative, so a node whose accumulated violation
        already exceeds the incumbent's total violation cannot reach a leaf with
        a better lexicographic cost.  Ties are kept, so the optimum and its
        tie-break are unchanged (``tests/test_timed_model.py`` compares pruned
        and unpruned solves, including from outside the corridor).

        Progress cannot be bounded cheaply: outside the corridor the progress
        rate ``v_t/(1-κd)`` is only bounded by its clamp (10x), so a
        progress-based prune is *not* admissible.  An earlier version had one;
        it silently changed the optimum from outside the corridor.
        """
        inc_v = None
        if incumbent is not None and len(incumbent) == len(self.inst.blocks):
            inc_v = self.evaluate(incumbent)[0]
        nd = self.start_nodes()
        nodes = []
        last = len(self.inst.blocks) - 1
        for k in range(len(self.inst.blocks)):
            nd = self.run_block(nd, k)
            if inc_v is not None and k < last:
                keep = nd.viol <= inc_v
                if not keep.all():
                    nd = nd.take(keep)
            nodes.append(int(nd.s.size))
        V = nd.viol + self.terminal(nd)
        P = self.objective(nd)
        E = nd.effort
        m = V == V.min()
        m &= P == P[m].min()
        m &= E == E[m].min()
        i = int(np.flatnonzero(m)[0])     # nodes are in plan-index order
        plan = tuple(int(x) for x in nd.plan[i])
        return plan, (int(V[i]), int(P[i]), int(E[i])), {"leaves": nodes, "ticks": self.ticks}

    def evaluate(self, plan) -> tuple[int, int, int]:
        nd = self.start_nodes()
        for k, a in enumerate(plan):
            nd = self.run_block(nd, k).take(np.array([int(a)]))
        V = nd.viol + self.terminal(nd)
        return int(V[0]), int(self.objective(nd)[0]), int(nd.effort[0])

    def trajectory(self, plan):
        """States after every tick for one plan (scalar reference rollout)."""
        from .model import rollout
        return [(st.s, st.d, st.psi, st.v) for st in rollout(self.inst, plan).states]


def solve(inst: Instance, incumbent=None):
    return Solver(inst).solve(incumbent)


__all__ = ["Solver", "solve", "DomainOverflow"]
