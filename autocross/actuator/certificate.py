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
            "formula": "1000*traversal_time + |delta'-delta|/1000 + 5000*|d'| + 5000*|psi'|/100",
            "units": "microseconds + microsteer + millimetres + microradians",
        },
        "information": {
            "uses": "quantised preview observation only",
            "horizon": model.horizon,
            "beyond_preview": "none; terminal condition is declared in the instance",
        },
    }
    cert["certificate_sha256"] = hashlib.sha256(canonical_json(cert).encode()).hexdigest()
    return cert


def write_certificate(cert: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cert, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return p


def _row_state(row) -> ActState:
    return ActState.from_any(row)


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


def corrupt(cert: dict, kind: str) -> dict:
    c = json.loads(json.dumps(cert))
    if kind == "value":
        for i, vs in enumerate(c["values"]):
            for j, v in enumerate(vs):
                if v is not None:
                    vs[j] = int(v) + 1
                    return c
        raise KeyError(kind)
    if kind == "first_action":
        c["firstAction"][0] = int(c["firstAction"][0]) + 1000
        return c
    if kind == "plan_cost":
        c["claimed"] = int(c["claimed"]) + 1
        return c
    if kind == "plan_state":
        c["plan"][-1]["d"] = int(c["plan"][-1]["d"]) + 1000
        return c
    if kind == "successor":
        c["states"][-1].pop()
        c["values"][-1].pop()
        return c
    if kind == "sha":
        c["certificate_sha256"] = "0" * 64
        return c
    raise KeyError(kind)
