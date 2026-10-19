"""Machine-checkable certificates for the finite actuator model.

The certificate is deliberately redundant: it carries the complete declared
model, the reachable state tables, their optimal values, the deterministic plan,
and the first actuator request.  The Python checker below is an independent
translation of the recurrence (not a replay of the solver): it re-encodes the
instance, recomputes every admissible successor, and verifies every row.  The
Lean checker (``Corpus.Actuator``) proves that acceptance implies global
optimality for exactly this declared finite model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .model import ActModel, ActState, COST_TIME_WEIGHT, isqrt, model_from_instance

SCHEMA = "autocross.actuator.certificate"
SCHEMA_VERSION = 1


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def certificate_from_diag(model: ActModel, diag: dict, *, claimed: int | None,
                          first_action: tuple[int, int] | None) -> dict:
    univ = diag["univ"]
    values = diag["values"]
    states: list[list[dict]] = []
    vals: list[list[int | None]] = []
    for i in range(model.horizon + 1):
        states.append([s.as_dict() for s in univ[i]])
        vals.append([values[i].get(s) for s in univ[i]])
    plan = diag["plan"]
    plan_actions = diag["plan_actions"]
    cert = {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        "instance": model.as_instance(),
        "roots": [[b, isqrt(b)] for b in sorted({model.start.b, *model.speeds}) if b >= 0],
        "states": states, "values": vals,
        "claimed": claimed,
        "plan": [s.as_dict() for s in plan],
        "planActions": [[int(u), int(a)] for u, a in plan_actions],
        "firstAction": None if first_action is None else [int(first_action[0]), int(first_action[1])],
        "tie_break": "first (steer,accel) in declared input order",
        "objective": {
            "id": model.objective, "timeScale": 1_000_000, "timeWeight": COST_TIME_WEIGHT,
            "formula": ("1000*traversal_time + |delta'-delta|/1000 + 5000*|d'-ref_d[i+1]|"
                        " + 5000*|psi'-ref_psi[i+1]|/100 (absent reference = 0)"),
            "units": "microseconds + microsteer + millimetres + microradians",
        },
        "information": {
            "uses": "quantised preview observation only",
            "horizon": model.horizon,
            "beyond_preview": "none; terminal condition is declared in the instance",
        },
    }
    return rehash(cert)


def instance_sha256(model: ActModel | dict) -> str:
    """SHA-256 of the canonical declared instance (binds a certificate to a solve)."""
    inst = model.as_instance() if isinstance(model, ActModel) else model
    return hashlib.sha256(canonical_json(inst).encode()).hexdigest()


def sha52(hexdigest: str) -> float:
    """First 52 bits of a hex digest as an exactly representable float (telemetry)."""
    return float(int(hexdigest[:13], 16))


def certificate_for_action(model: ActModel, action: tuple[int, int], *,
                           sparse: tuple | None = None) -> dict | None:
    """Certificate whose ``firstAction`` and plan start with the executed action.

    ``plan`` is the executed first action followed by the deterministic optimal
    continuation from its successor.  Returns ``None`` when the instance is
    infeasible or the executed action is not optimal for ``model`` (it is
    inadmissible, leads to an infeasible state, or its cost-to-go exceeds the
    optimum).  ``sparse`` may pass a precomputed ``solve_sparse(model)`` result.
    """
    from .solver import solve_sparse
    _t, best, _a, diag = sparse if sparse is not None else solve_sparse(model)
    if best is None:
        return None
    u, a = int(action[0]), int(action[1])
    t = model.action_ok(model.start, u, a)
    if t is None:
        return None
    v1 = diag["values"][1].get(t)
    if v1 is None or model.action_cost(model.start, t) + v1 != best:
        return None
    plan = [model.start, t]
    plan_actions = [(u, a)]
    cur = t
    for i in range(1, model.horizon):
        ch = diag["choice"][i].get(cur)
        if ch is None:
            return None
        nxt = model.action_ok(cur, ch[0], ch[1])
        if nxt is None:
            return None
        plan.append(nxt)
        plan_actions.append((ch[0], ch[1]))
        cur = nxt
    return certificate_from_diag(model, {**diag, "plan": plan, "plan_actions": plan_actions},
                                 claimed=best, first_action=(u, a))


def rehash(cert: dict) -> dict:
    """Recompute ``certificate_sha256`` (an adversary can always do this)."""
    body = {k: v for k, v in cert.items() if k != "certificate_sha256"}
    cert["certificate_sha256"] = hashlib.sha256(canonical_json(body).encode()).hexdigest()
    return cert


def write_certificate(cert: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cert, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return p


def _row_state(row) -> ActState:
    return ActState.from_any(row)


def domain_errors(model: ActModel) -> list[str]:
    """The certificate domain (``Corpus.Actuator.domainOK``), checked identically.

    The start offset, heading and steering angle must be declared grid values
    (``model_from_observation`` snaps them); the start speed^2 may be off-grid.
    """
    errs: list[str] = []
    base = model.q["base"]
    H = model.horizon
    if model.start.station != 0:
        errs.append("start station must be 0")
    if H < 1:
        errs.append("horizon must be >= 1")
    if len(base["s"]) < H + 1 or len(base["k"]) < H + 1 or len(base["k"]) > len(base["s"]):
        errs.append("station/curvature lists do not cover the horizon")
    if len(base["wl"]) < H + 1 or len(base["wr"]) < H + 1:
        errs.append("width lists do not cover the horizon")
    if len(base["s"]) >= H + 1 and any(model.gap(i) <= 0 for i in range(H)):
        errs.append("station gaps must be positive")
    if model.start.d not in model.offsets or model.start.psi not in model.psis \
            or model.start.delta not in model.deltas:
        errs.append("start d/psi/delta must be declared grid values")
    if model.start.b < 0:
        errs.append("negative start speed^2")
    for name in ("offsets", "psis", "speeds", "deltas", "steer_inputs", "acc_inputs"):
        g = getattr(model, name)
        if any(x >= y for x, y in zip(g, g[1:])):
            errs.append(f"{name} must be strictly increasing")
    if any(b <= 0 for b in model.speeds):
        errs.append("declared speeds must be positive")
    if model.tau <= 0 or model.understeer_den <= 0:
        errs.append("steer_tau_us and understeer_den must be positive")
    return errs


def _lookup(states: list[dict], values: list[int | None], s: ActState) -> tuple[bool, int | None]:
    for row, v in zip(states, values):
        if _row_state(row) == s:
            return True, v
    return False, None


def _best(actions: list[tuple[ActState, int, int, int]], nxt: dict[ActState, int | None]):
    best: int | None = None
    first: tuple[ActState, int] | None = None
    for t, c, u, a in actions:
        v = nxt.get(t)
        if v is None:
            continue
        cand = c + v
        if best is None or cand < best:
            best = cand
            first = (t, c)
    return best, first


def check_certificate(cert: dict) -> dict:
    errors: list[str] = []
    try:
        if cert.get("schema") != SCHEMA or int(cert.get("schema_version", -1)) != SCHEMA_VERSION:
            return {"ok": False, "errors": ["schema"]}
        body = {k: v for k, v in cert.items() if k != "certificate_sha256"}
        if hashlib.sha256(canonical_json(body).encode()).hexdigest() != cert.get("certificate_sha256"):
            return {"ok": False, "errors": ["certificate hash mismatch"]}
        inst = cert["instance"]
        model: ActModel = model_from_instance(inst)
        dom = domain_errors(model)
        if dom:
            return {"ok": False, "errors": dom}
        expected_roots = [[b, isqrt(b)] for b in sorted({model.start.b, *model.speeds}) if b >= 0]
        if cert.get("roots") != expected_roots:
            errors.append("roots do not match declared speeds")
        states = cert["states"]
        values = cert["values"]
        if len(states) != model.horizon + 1 or len(values) != model.horizon + 1:
            errors.append("table count does not match horizon")
        tables: list[dict[ActState, int | None]] = []
        for i in range(min(len(states), model.horizon + 1)):
            if len(states[i]) != len(values[i]):
                errors.append(f"stage {i}: states/values length mismatch")
            tab: dict[ActState, int | None] = {}
            for row, v in zip(states[i], values[i]):
                s = _row_state(row)
                if s.station != i:
                    errors.append(f"stage {i}: state station {s.station}")
                if (s.d not in model.offsets or s.psi not in model.psis
                        or s.delta not in model.deltas
                        or (s.b not in model.speeds and s.b != model.start.b)):
                    errors.append(f"stage {i}: state off the declared grids {s}")
                if s in tab:
                    errors.append(f"stage {i}: duplicate state {s}")
                tab[s] = None if v is None else int(v)
            tables.append(tab)
        if tables and tables[0] != {model.start: cert.get("claimed")}:
            # Start table may contain exactly the start row; compare explicitly.
            if list(tables[0]) != [model.start]:
                errors.append("stage 0 must be exactly the start row")
        # Terminal rows.
        if len(tables) == model.horizon + 1:
            for s, v in tables[model.horizon].items():
                c = model.terminal_cost(s)
                want = None if c is None else COST_TIME_WEIGHT * c
                if v != want:
                    errors.append(f"terminal value mismatch at {s}: {v} != {want}")
        # Bellman and closure.
        for i in range(model.horizon - 1, -1, -1):
            if i + 1 >= len(tables):
                continue
            for s, v in tables[i].items():
                acts = model.actions(s)
                for t, _c, _u, _a in acts:
                    if t not in tables[i + 1]:
                        errors.append(f"stage {i}: successor outside table {t}")
                want, first = _best(acts, tables[i + 1])
                if v != want:
                    errors.append(f"stage {i}: Bellman mismatch at {s}: {v} != {want}")
                    break
        # Claimed first action.
        fa = cert.get("firstAction")
        plan = [_row_state(x) for x in cert.get("plan", [])]
        acts = cert.get("planActions", [])
        if cert.get("claimed") is None:
            if fa is not None or plan or acts:
                errors.append("infeasible certificate carries an action or plan")
        else:
            if not plan or plan[0] != model.start:
                errors.append("plan must start at the start state")
            if fa is None or len(fa) != 2:
                errors.append("missing first action")
            else:
                u, a = int(fa[0]), int(fa[1])
                t = model.action_ok(model.start, u, a)
                if t is None:
                    errors.append("first action is not admissible")
                elif len(plan) < 2 or plan[1] != t:
                    errors.append("first action does not produce plan[1]")
                elif tables and len(tables) > 1:
                    vt = tables[1].get(t)
                    if vt is None:
                        errors.append("first successor has no value")
                    elif model.action_cost(model.start, t) + vt != cert.get("claimed"):
                        errors.append("first action cost does not reach claimed value")
            # Plan replay and cost.
            if plan and len(plan) == model.horizon + 1 and len(acts) == model.horizon:
                total = 0
                ok = True
                for i, (u, a) in enumerate(acts):
                    t = model.action_ok(plan[i], int(u), int(a))
                    if t is None or t != plan[i + 1]:
                        ok = False; errors.append(f"plan step {i} invalid"); break
                    total += model.action_cost(plan[i], t)
                if ok:
                    c = model.terminal_cost(plan[-1])
                    if c is None:
                        errors.append("plan does not end in the terminal set")
                    elif total + COST_TIME_WEIGHT * c != cert.get("claimed"):
                        errors.append("plan cost does not equal claimed value")
            elif cert.get("claimed") is not None:
                errors.append("plan length does not match horizon")
        return {"ok": not errors, "errors": errors, "claimed": cert.get("claimed")}
    except Exception as e:  # noqa: BLE001 - checker must reject malformed input
        return {"ok": False, "errors": [f"malformed: {type(e).__name__}: {e}"]}


CORRUPTIONS = ("value", "terminal_value", "first_action", "first_action_suboptimal",
               "plan_cost", "plan_state", "plan_actions", "successor",
               "instance_curvature", "instance_width", "instance_start", "sha")


def corrupt(cert: dict, kind: str) -> dict:
    """Adversarial mutation of a feasible certificate.

    Every kind except ``sha`` recomputes ``certificate_sha256``, so rejection
    must come from the semantic checks, not from the hash.
    """
    c = json.loads(json.dumps(cert))
    if kind == "sha":
        c["certificate_sha256"] = "0" * 64
        return c
    model = model_from_instance(c["instance"])
    if kind == "value":
        for vs in c["values"]:
            for j, v in enumerate(vs):
                if v is not None:
                    vs[j] = int(v) + 1
                    return rehash(c)
        raise KeyError(kind)
    elif kind == "terminal_value":
        vs = c["values"][-1]
        j = next(j for j, v in enumerate(vs) if v is not None)
        vs[j] = int(vs[j]) - 1
    elif kind == "first_action":
        # Not a declared input: inadmissible.
        c["firstAction"][0] = max(model.steer_inputs) + 1
    elif kind == "first_action_suboptimal":
        # A self-consistent forgery: an admissible but suboptimal first action
        # with its own continuation, plan, plan cost and stage-0 value.
        from .solver import solve_sparse
        _t, best, _a, diag = solve_sparse(model)
        forged = None
        for t, cost, u, a in model.actions(model.start):
            v1 = diag["values"][1].get(t)
            if v1 is not None and cost + v1 > int(best):
                forged = (u, a, t, cost + v1)
                break
        if forged is None:
            raise KeyError(kind)
        u, a, t, value = forged
        plan = [model.start, t]
        acts = [(u, a)]
        cur = t
        for i in range(1, model.horizon):
            cu, ca, _c = diag["choice"][i][cur]
            cur = model.action_ok(cur, cu, ca)
            plan.append(cur)
            acts.append((cu, ca))
        c["firstAction"] = [int(u), int(a)]
        c["plan"] = [s.as_dict() for s in plan]
        c["planActions"] = [[int(x), int(y)] for x, y in acts]
        c["claimed"] = int(value)
        c["values"][0] = [int(value)]
    elif kind == "plan_cost":
        c["claimed"] = int(c["claimed"]) + 1
    elif kind == "plan_state":
        c["plan"][-1]["d"] = int(c["plan"][-1]["d"]) + 1000
    elif kind == "plan_actions":
        last = ActState.from_any(c["plan"][-2])
        old = tuple(int(x) for x in c["planActions"][-1])
        want = ActState.from_any(c["plan"][-1])
        alt = [(u, a) for t, _c, u, a in model.actions(last) if t != want]
        c["planActions"][-1] = list(alt[0]) if alt else [old[0] + 1000, old[1]]
    elif kind == "successor":
        c["states"][-1].pop()
        c["values"][-1].pop()
    elif kind == "instance_curvature":
        c["instance"]["q"]["base"]["k"][0] = int(c["instance"]["q"]["base"]["k"][0]) + 200_000
    elif kind == "instance_width":
        # Station-1 corridor upper bound drops just below plan[1].d.
        half = model.half_width + model.margin
        t1 = ActState.from_any(c["plan"][1])
        c["instance"]["q"]["base"]["wl"][1] = half + t1.d - 1
    elif kind == "instance_start":
        offs = list(model.offsets)
        i = offs.index(model.start.d) if model.start.d in offs else 0
        c["instance"]["start"]["d"] = offs[i + 1] if i + 1 < len(offs) else offs[i - 1]
    else:
        raise KeyError(kind)
    return rehash(c)
