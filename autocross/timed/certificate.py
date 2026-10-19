"""Proof-carrying certificates for the time-indexed model ``timed-1``.

A certificate names the declared model (``Instance.to_json()``), a plan of
action indices, the objective tuple that plan attains, and the command the plan
publishes for the first control period.  It is accepted only if the plan is
admissible, its objective is exactly the claimed one, no plan of the declared
model has a lexicographically smaller objective, the claimed plan is the FIRST
minimiser in enumeration order, and the published command is the one the model
derives from block 0.

The checker below is an INDEPENDENT re-implementation of the integer model: it
never calls ``model.rollout``/``model.evaluate``/``model.solve_exhaustive`` or
the vector solver.  Only the data classes (``Instance``/``Block``/``State``)
are shared; every arithmetic step is recomputed here from the JSON instance in
exactly the order of ``model.py`` (all divisions are floor divisions with
strictly positive divisors, so Python ``//`` and Lean's Euclidean ``Int``
division agree).  The Lean checker ``lean/Corpus/Timed.lean`` is a third
implementation of the same arithmetic; both must accept a certificate.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from typing import Iterable, Sequence

from .model import BIG, E6, E9, PSI_CLAMP, DEFAULT_CONSTS, Block, Instance, State

SCHEMA = "autocross.timed.certificate"
SCHEMA_VERSION = 1
OBJECTIVE_LEX = ["violation", "neg_progress", "effort"]
OBJECTIVE_TIE_BREAK = "first plan in enumeration order"

MODEL_ID = "timed-1"
INSTANCE_SCHEMA = "autocross.timed.instance"

Cost = tuple[int, int, int]


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def instance_sha256(instance_json: dict) -> str:
    return hashlib.sha256(canonical_json(instance_json).encode()).hexdigest()


# ----------------------------------------------------------------- primitives


def _clamp(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x


def _cdiv(a: int, b: int) -> int:
    """Ceiling division for b > 0 (b is guarded positive at every call site)."""
    return -((-a) // b)


def _isqrt(n: int) -> int:
    return math.isqrt(n) if n > 0 else 0


def _tyre_force(fy_lin: int, fx: int, cap: int) -> int:
    """Saturating lateral tyre force (the model's (3,2) Padé tanh law)."""
    av2 = cap * cap - fx * fx
    avail = _isqrt(av2) if av2 > 0 else 1
    x = _clamp(fy_lin * 1000 // avail, -3000, 3000)
    num = 27_000_000 + x * x
    return avail * (x * num // (27_000_000 + 9 * x * x)) // 1000


def _trig(psi: int) -> tuple[int, int]:
    p = _clamp(psi, -PSI_CLAMP, PSI_CLAMP)
    q = p * p // E6
    return p - p * q // (6 * E6), E6 - q // 2 + q * q // (24 * E6)


def _tan_u(delta: int) -> int:
    q = delta * delta // E6
    return delta + delta * q // (3 * E6) + 2 * delta * q * q // (15 * E6 * E6)


def _idx(inst: Instance, x: int) -> int:
    return _clamp(x // inst.consts["STEP"], 0, len(inst.k) - 1)


def _kappa_at(inst: Instance, x: int) -> int:
    step = inst.consts["STEP"]
    j = _idx(inst, x)
    j1 = min(j + 1, len(inst.k) - 1)
    fr = _clamp(x - j * step, 0, step)
    return inst.k[j] + (inst.k[j1] - inst.k[j]) * fr // step


def _action(b: Block, a: int) -> tuple[int, int]:
    return b.steer[a // len(b.pedal)], b.pedal[a % len(b.pedal)]


def _body_points(inst: Instance, st: State) -> list[tuple[int, int]]:
    hl = inst.consts["HL"]
    kap = _kappa_at(inst, st.s)
    sn, _ = _trig(st.psi)
    out = []
    for ell in (-hl, 0, hl):
        lat = st.d + ell * sn // E6 - kap * ell * ell // (2 * E9)
        out.append((st.s + ell, lat))
    return out


def _footprint_violation(inst: Instance, st: State) -> int:
    hw = inst.consts["HW"]
    hl = inst.consts["HL"]
    pts = _body_points(inst, st)
    viol = 0
    for s_, lat in pts:
        j = _idx(inst, s_)
        viol += max(0, lat + hw - inst.wl[j]) + max(0, -inst.wr[j] - (lat - hw))
    lo = min(lat for _, lat in pts) - hw
    hi = max(lat for _, lat in pts) + hw
    for s_lo, s_hi, l_lo, l_hi in inst.obstacles:
        if st.s + hl >= s_lo and st.s - hl <= s_hi:
            viol += max(0, min(hi - l_lo, l_hi - lo))
    return viol


def _friction_violation(inst: Instance, fyf: int, fyr: int, fxf: int, fxr: int, ax: int) -> int:
    c = inst.consts
    fzf = c["MASS"] * (c["G"] * c["LR"] - ax * c["HCG"]) // (c["L"] * 1000)
    fzr = c["MASS"] * c["G"] // 1000 - fzf
    viol = 0
    for fy, fx, fz in ((fyf, fxf, fzf), (fyr, fxr, fzr)):
        cap = max(c["MU"] * fz // 1000, 1)
        ex = fy * fy + fx * fx - cap * cap
        if ex > 0:
            viol += _cdiv(ex, cap)
    return viol


def _terminal_violation(inst: Instance, st: State) -> int:
    c = inst.consts
    a_lt, a_bt, step = c["A_LT"], c["A_BT"], c["STEP"]
    n = len(inst.k)
    v2 = st.v * st.v
    vmax2 = c["VMAX"] * c["VMAX"]
    worst = v2 - (a_lt * c["RMIN"] + 2 * a_bt * ((n - 1) * step - st.s))
    for j in range(1, n):
        sj = j * step
        if sj <= st.s:
            continue
        kj = abs(inst.k[j])
        cap = vmax2 if kj == 0 else min(vmax2, a_lt * E9 // kj)
        worst = max(worst, v2 - (cap + 2 * a_bt * (sj - st.s)))
    viol = _cdiv(worst, 2 * a_bt) if worst > 0 else 0
    kcap = min(a_lt * E9 // max(v2, 1), c["KSTEER"])
    kap = _kappa_at(inst, st.s)
    j = _idx(inst, st.s)
    psi = _clamp(st.psi, -PSI_CLAMP, PSI_CLAMP)
    hw = c["HW"]
    if psi > 0:
        dk, room = kcap + kap, inst.wl[j] - hw - st.d
    elif psi < 0:
        dk, room = kcap - kap, st.d + inst.wr[j] - hw
    else:
        dk, room = 1, 0
    if psi != 0:
        req = BIG if dk <= 0 else _cdiv(psi * psi, 2 * dk * 1000)
        viol += max(0, req - room)
    return viol


def _tick(inst: Instance, st: State, u: int, p: int) -> tuple[State, int]:
    c = inst.consts
    H = c["H_US"]
    sigma = _clamp(st.sigma + _clamp(u - st.sigma, -c["SLEW"], c["SLEW"]), -1000, 1000)
    tgt = sigma * c["MAX_STEER"] // 1000
    delta = tgt + (st.delta - tgt) * c["LAG_NUM"] // c["LAG_DEN"]
    v = st.v
    if p >= 0:
        fdrv = min(c["FD"], c["PW"] * 1000 // max(v, 1000))
        fx = p * fdrv // 1000
        fxf = fx * c["FDF"] // 1000
    else:
        fx = max(p * c["FB"] // 1000, -c["FBMAX"])
        fxf = fx * c["FBF"] // 1000
    fxr = fx - fxf
    fres = c["CD"] * v * v // E9 + c["FRR"]
    ax = (fx - fres) * 1000 // c["MASS"]
    v1 = max(v + ax * H // E6, 0)
    fzf = c["MASS"] * (c["G"] * c["LR"] - ax * c["HCG"]) // (c["L"] * 1000)
    fzr = c["MASS"] * c["G"] // 1000 - fzf
    vs = max(v, c["V_FLOOR"])
    af = (st.vy + c["LF"] * st.r // E6) * E6 // vs - delta
    ar = (st.vy - c["LR"] * st.r // E6) * E6 // vs
    demand_f = -(c["CF"] * af // E6)
    demand_r = -(c["CR"] * ar // E6)
    fyf = _tyre_force(demand_f, fxf, c["MU_T"] * fzf // 1000)
    fyr = _tyre_force(demand_r, fxr, c["MU_T"] * fzr // 1000)
    vy_dot = (fyf + fyr) * 1000 // c["MASS"] - st.r * v // E6
    r_dot = (c["LF"] * fyf - c["LR"] * fyr) * 1000 // c["IZ"]
    vy_d = st.vy + vy_dot * H // E6
    r_d = st.r + r_dot * H // E6
    w = _clamp((c["V_KIN"] - v1) * 1000 // (c["V_KIN"] - c["V_FLOOR"]), 0, 1000)
    r_k = v1 * _tan_u(delta) // c["L"]
    vy_k = c["LR"] * r_k // E6
    r1 = (w * r_k + (1000 - w) * r_d) // 1000
    vy1 = (w * vy_k + (1000 - w) * vy_d) // 1000
    sn, cs = _trig(st.psi)
    vt = (v1 * cs - vy1 * sn) // E6
    kap = _kappa_at(inst, st.s + (vt * H // E6) // 2)
    den = max(E9 - kap * st.d, E9 // 10)
    sdot = vt * E9 // den
    ddot = (v1 * sn + vy1 * cs) // E6
    psidot = r1 - kap * sdot // 1000
    s1 = st.s + sdot * H // E6
    d1 = st.d + ddot * H // E6
    psi1 = st.psi + psidot * H // E6
    out = State(s1, d1, psi1, v1, vy1, r1, sigma, delta)
    viol = _footprint_violation(inst, out) + _friction_violation(inst, demand_f, demand_r, fxf, fxr, ax)
    return out, viol


def _rollout(inst: Instance, plan: Sequence[int]) -> tuple[State, int, int]:
    st = inst.start
    viol = 0
    effort = 0
    p_last = inst.p_prev
    for b, a in zip(inst.blocks, plan):
        off, p = _action(b, a)
        u = _clamp(st.sigma + off, -1000, 1000)
        effort += abs(u - st.sigma) + abs(p - p_last)
        p_last = p
        for _ in range(b.steps * inst.consts["TICKS"]):
            st, vt = _tick(inst, st, u, p)
            viol += vt
    viol += _terminal_violation(inst, st)
    return st, viol, effort


def _cost(inst: Instance, plan: Sequence[int]) -> Cost:
    st, viol, effort = _rollout(inst, plan)
    prog = st.s + st.v * inst.consts["T_TAIL"] // 1000
    return viol, -(prog // inst.consts["PQ"]), effort


def _all_plans(inst: Instance) -> Iterable[tuple[int, ...]]:
    return itertools.product(*(range(b.n) for b in inst.blocks))


def _first_command(inst: Instance, plan: Sequence[int]) -> tuple[int, int]:
    off, p = _action(inst.blocks[0], plan[0])
    return _clamp(inst.start.sigma + off, -1000, 1000), p


# ------------------------------------------------------------ instance encode


def _instance_from_json(j: dict) -> Instance:
    """Re-encode the declared model without calling ``Instance.from_json``."""
    if j.get("schema") != INSTANCE_SCHEMA or int(j.get("schema_version", -1)) != 1:
        raise ValueError("not a timed-1 instance")
    if j.get("model") != MODEL_ID:
        raise ValueError("unknown model")
    consts = {str(k): int(v) for k, v in j["consts"].items()}
    start = j["start"]
    if not isinstance(start, list) or len(start) != 8:
        raise ValueError("start state must have 8 entries")
    blocks = []
    for b in j["blocks"]:
        blocks.append(Block(int(b["steps"]), tuple(int(x) for x in b["steer"]),
                            tuple(int(x) for x in b["pedal"])))
    return Instance(consts, tuple(int(x) for x in j["k"]), tuple(int(x) for x in j["wl"]),
                    tuple(int(x) for x in j["wr"]),
                    tuple(tuple(int(x) for x in o) for o in j["obstacles"]),
                    State(*(int(x) for x in start)), int(j["p_prev"]), tuple(blocks))


def _well_formed(inst: Instance) -> list[str]:
    """Independent copy of ``model.well_formed`` (the Lean domain guard)."""
    err: list[str] = []
    c = inst.consts
    if set(c) != set(DEFAULT_CONSTS):
        return ["constant set differs from the declared set"]
    for key in ("H_US", "TICKS", "LAG_DEN", "L", "MASS", "IZ", "STEP", "PQ", "A_BT", "V_FLOOR", "MU_T"):
        if int(c[key]) <= 0:
            err.append(f"{key} must be positive")
    if not (0 <= c["LAG_NUM"] <= c["LAG_DEN"]):
        err.append("LAG_NUM outside [0, LAG_DEN]")
    if c["V_KIN"] <= c["V_FLOOR"]:
        err.append("V_KIN must exceed V_FLOOR")
    n = len(inst.k)
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


# ------------------------------------------------------------------ builder


def certificate_for_instance(inst: Instance, plan: Sequence[int]) -> dict:
    """Certificate claiming ``plan`` as the first minimiser of ``inst``."""
    plan = [int(a) for a in plan]
    if len(plan) != len(inst.blocks):
        raise ValueError("plan length differs from the number of blocks")
    for a, b in zip(plan, inst.blocks):
        if not (0 <= a < b.n):
            raise ValueError("action index out of range")
    claimed = _cost(inst, plan)
    u, p = _first_command(inst, plan)
    inst_json = inst.to_json()
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "instance": inst_json,
        "instance_sha256": instance_sha256(inst_json),
        "plan": plan,
        "claimed": list(claimed),
        "first": [int(u), int(p)],
        "objective": {"lex": list(OBJECTIVE_LEX), "tie_break": OBJECTIVE_TIE_BREAK},
    }


def write_certificate(cert: dict, path) -> None:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cert, separators=(",", ":")), encoding="utf-8")


# ------------------------------------------------------------------ checker


def _triple(x) -> Cost:
    if not isinstance(x, (list, tuple)) or len(x) != 3:
        raise ValueError("claimed objective must have three entries")
    return int(x[0]), int(x[1]), int(x[2])


def _pair(x) -> tuple[int, int]:
    if not isinstance(x, (list, tuple)) or len(x) != 2:
        raise ValueError("first command must have two entries")
    return int(x[0]), int(x[1])


def check_certificate(cert: dict) -> dict:
    """Independent decision on a certificate; never trusts the solver."""
    try:
        if not isinstance(cert, dict):
            return {"ok": False, "errors": ["malformed: not an object"]}
        if cert.get("schema") != SCHEMA:
            return {"ok": False, "errors": [f"schema is {cert.get('schema')!r}"]}
        if int(cert.get("schema_version", -1)) != SCHEMA_VERSION:
            return {"ok": False, "errors": ["schema_version differs"]}
        obj = cert.get("objective") or {}
        if obj.get("lex") != OBJECTIVE_LEX or obj.get("tie_break") != OBJECTIVE_TIE_BREAK:
            return {"ok": False, "errors": ["objective declaration differs"]}
        inst_json = cert["instance"]
        if cert.get("instance_sha256") != instance_sha256(inst_json):
            return {"ok": False, "errors": ["instance_sha256 does not bind the instance"]}
        inst = _instance_from_json(inst_json)
        err = _well_formed(inst)
        if err:
            return {"ok": False, "errors": ["not well formed: " + "; ".join(err)]}
        plan = [int(a) for a in cert["plan"]]
        if len(plan) != len(inst.blocks):
            return {"ok": False, "errors": ["plan length differs from the blocks"]}
        for a, b in zip(plan, inst.blocks):
            if not (0 <= a < b.n):
                return {"ok": False, "errors": [f"action index {a} out of range"]}
        claimed = _triple(cert["claimed"])
        first = _pair(cert["first"])
        got = _cost(inst, plan)
        if got != claimed:
            return {"ok": False, "errors": [f"plan cost {got} differs from claimed {claimed}"]}
        if _first_command(inst, plan) != first:
            return {"ok": False, "errors": [f"first command {_first_command(inst, plan)} differs"]}
        plans = list(_all_plans(inst))
        try:
            pos = plans.index(tuple(plan))
        except ValueError:
            return {"ok": False, "errors": ["plan is not in the enumeration"]}
        for i, p in enumerate(plans):
            j = _cost(inst, p)
            if i < pos:
                if not (j > claimed):
                    return {"ok": False,
                            "errors": [f"plan {i} {p} costs {j} <= claimed {claimed}"]}
            elif not (j >= claimed):
                return {"ok": False,
                        "errors": [f"plan {i} {p} beats the claimed optimum: {j} < {claimed}"]}
        return {"ok": True, "errors": [], "plans": len(plans), "claimed": list(claimed),
                "first": list(first), "index": pos}
    except Exception as e:  # fail closed
        return {"ok": False, "errors": [f"malformed: {type(e).__name__}: {e}"]}


# --------------------------------------------------------------- corruptions

CORRUPTIONS = ("value", "plan", "first", "first_action_index", "instance_const",
               "instance_start", "instance_preview", "objective", "schema")


def _rehash(cert: dict) -> None:
    cert["instance_sha256"] = instance_sha256(cert["instance"])


def corrupt(cert: dict, kind: str) -> dict:
    """Adversarial mutation; every kind must be rejected by both checkers."""
    c = copy.deepcopy(cert)
    inst = c["instance"]
    if kind == "value":
        c["claimed"][0] += 1
    elif kind == "plan":
        k = len(c["plan"]) - 1
        b = inst["blocks"][k]
        n = len(b["steer"]) * len(b["pedal"])
        if n < 2:
            raise ValueError("cannot mutate a singleton block")
        c["plan"][k] = (c["plan"][k] + 1) % n
    elif kind == "first":
        c["first"][0] += 1
    elif kind == "first_action_index":
        b = inst["blocks"][0]
        n = len(b["steer"]) * len(b["pedal"])
        sigma0 = int(inst["start"][6])
        for cand in range(n):
            if cand == c["plan"][0]:
                continue
            off = b["steer"][cand // len(b["pedal"])]
            ped = b["pedal"][cand % len(b["pedal"])]
            if [_clamp(sigma0 + off, -1000, 1000), ped] != c["first"]:
                c["plan"][0] = cand
                break
        else:
            raise ValueError("every first action publishes the same command")
    elif kind == "instance_const":
        # MASS+1 happens not to change any rounded quantity on the tiny vectors;
        # T_TAIL changes the final progress term of every plan.
        c["instance"]["consts"]["T_TAIL"] += 1
        _rehash(c)
    elif kind == "instance_start":
        # a speed nudge can round away; the applied steer changes the published
        # command and the slew-limited trajectory.
        sigma0 = int(c["instance"]["start"][6])
        c["instance"]["start"][6] = sigma0 + (10 if sigma0 <= 990 else -10)
        _rehash(c)
    elif kind == "instance_preview":
        # zero the first preview widths: the three body points of the first
        # tick all index station 0, so the corridor violation strictly grows.
        c["instance"]["wl"][0] = 0
        c["instance"]["wr"][0] = 0
        _rehash(c)
    elif kind == "objective":
        c["objective"]["lex"] = ["violation", "effort", "neg_progress"]
    elif kind == "schema":
        c["schema"] = SCHEMA + ".v2"
    else:
        raise ValueError(f"unknown corruption {kind!r}")
    return c


__all__ = ["SCHEMA", "SCHEMA_VERSION", "OBJECTIVE_LEX", "OBJECTIVE_TIE_BREAK", "CORRUPTIONS",
           "canonical_json", "instance_sha256", "certificate_for_instance", "check_certificate",
           "corrupt", "write_certificate"]
