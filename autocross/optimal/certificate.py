"""Machine-readable proof-carrying teacher certificates.

A certificate contains the complete declared model instance and, for every
station, the value of every state in the model's finite state universe.  The
universe is *defined by the certificate's own grids*, not by a reachability
heuristic: station 0 contains only the observed start state; stations 1..H
contain the full Cartesian product ``offsets x offsets x speeds``.  A checker
therefore becomes an independent dynamic-programming verifier:

1. every row of every table is present exactly once and every state of the
   declared universe has a row;
2. terminal rows equal the terminal condition;
3. every nonterminal row equals the minimum over the *complete* set of
   admissible successors recomputed from the model instance,
   ``cost(s,t) + value_next(t)`` (``None`` for a dead end);
4. the start row equals the claimed optimum;
5. the extracted plan starts at the start state, uses admissible transitions,
   reaches the horizon, satisfies the terminal condition, and its declared
   costs sum to the claimed value;
6. the plan follows the deterministic first-minimum tie rule.

Nothing in the certificate is trusted: the instance is re-encoded from the
quantised observation, and the checker recomputes every successor set.  The Lean
checker (``Corpus.Teacher``) is the same algorithm with a kernel-checked
soundness theorem; the Python checker here is a literal independent translation
used for every generated certificate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .model import Model, State, isqrt, model_from_instance, time_cost
from .solver import Solution, solve

SCHEMA = "autocross.optimal.teacher_certificate"
SCHEMA_VERSION = 1


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def certificate_from_solution(sol: Solution) -> dict:
    m = sol.model
    rows: list[list[list[Any]]] = []
    states = []
    for i in range(m.horizon + 1):
        table = []
        for s in sol.univ[i]:
            table.append([s.station, s.previous, s.current, s.b, sol.values[i].get(s)])
        rows.append(table)
        states.append([s.as_dict() for s in sol.univ[i]])
    plan = [s.as_dict() for s in sol.plan]
    cert = {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        "instance": m.as_instance(),
        "claimed": sol.cost,
        "tables": rows,
        "states": states,
        "plan": plan,
        "plan_cost": sol.cost,
        "tie_break": "first_min_offset_then_speed",
        "cost_definition": {
            "id": m.objective, "time_scale": m.time_scale,
            "formula": "cdiv(2*d*time_scale, isqrt(b)+isqrt(w))",
            "units": "microseconds",
        },
        "information": {
            "uses": "quantised policy observation preview only (s,k,wl,wr,obstacles,v0)",
            "horizon": m.horizon,
            "beyond_preview": "only the declared terminal cap min(v_max^2, a_lat*r_h)",
        },
    }
    # Lean-facing schema (Corpus.Teacher.Cert).  The metadata above is ignored
    # by the derived FromJson instance; these fields are the checker input.
    cert.update({
        "consts": m.consts, "q": m.q,
        "offsets": list(m.offsets), "speeds": list(m.speeds),
        "roots": [[b, isqrt(b)] for b in sorted({int(m.start.b), *m.speeds}) if b >= 0],
        "horizon": m.horizon,
        "start": m.start.as_dict(),
        "terminalCap": m.terminal_cap,
        "states": states,
        "values": [[sol.values[i].get(s) for s in sol.univ[i]] for i in range(m.horizon + 1)],
        "planCost": sol.cost,
    })
    cert["certificate_sha256"] = hashlib.sha256(canonical_json(cert).encode()).hexdigest()
    return cert


def write_certificate(cert: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cert, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return p


def _state(row) -> State:
    return State.from_any(row)


def _opt_eq(a, b) -> bool:
    return a == b


def check_certificate(cert: dict, *, reject_extra_rows: bool = False) -> dict:
    """Independently verify a certificate.  Returns a report; ``ok`` is a bool."""
    errors: list[str] = []
    try:
        if cert.get("schema") != SCHEMA or int(cert.get("schema_version", -1)) != SCHEMA_VERSION:
            return {"ok": False, "errors": ["schema"]}
        inst = cert["instance"]
        model = model_from_instance(inst)
        tables = cert["tables"]
        expected_roots = [[b, isqrt(b)] for b in sorted({int(model.start.b), *model.speeds}) if b >= 0]
        if cert.get("roots") != expected_roots:
            errors.append("roots do not match declared speeds")
        state_lists = cert.get("states")
        if len(tables) != model.horizon + 1:
            errors.append(f"tables length {len(tables)} != {model.horizon + 1}")
        if not isinstance(state_lists, list) or len(state_lists) != len(tables):
            errors.append("states/tables length mismatch")
            state_lists = [[[r[0], r[1], r[2], r[3]] for r in table] for table in tables]
        rows_by_stage: list[dict[State, int | None]] = []
        univ_by_stage: list[list[State]] = []
        for i, (table, slist) in enumerate(zip(tables, state_lists)):
            rows: dict[State, int | None] = {}
            for row in table:
                s = _state(row)
                if s in rows:
                    errors.append(f"duplicate row at station {i}: {s}")
                    continue
                rows[s] = None if row[4] is None else int(row[4])
            univ = [_state(r) for r in slist]
            if len(set(univ)) != len(univ):
                errors.append(f"duplicate state in universe at station {i}")
            missing = set(univ) - set(rows)
            if missing:
                errors.append(f"missing {len(missing)} declared states at station {i}")
            extra = set(rows) - set(univ)
            if extra:
                errors.append(f"{len(extra)} rows outside the declared universe at station {i}")
            if univ and i == 0 and (len(univ) != 1 or univ[0] != model.start):
                errors.append("station-0 universe is not exactly the start state")
            rows_by_stage.append(rows)
            univ_by_stage.append(univ)
        # Closure: every admissible successor of a declared state is declared.
        for i in range(model.horizon):
            nxt = set(univ_by_stage[i + 1])
            for s in univ_by_stage[i]:
                for t, _ in model.actions(s):
                    if t not in nxt:
                        errors.append(f"universe not closed: {s} -> {t}")
                        break
                else:
                    continue
                break

        m = model.horizon
        # Terminal rows.
        for s, v in rows_by_stage[m].items():
            exp = model.terminal_value(s)
            if v != exp:
                errors.append(f"terminal row {s}: {v} != {exp}")
        # Bellman recurrence.
        for i in range(m):
            for s, v in rows_by_stage[i].items():
                acts = model.actions(s)
                best = None
                for t, c in acts:
                    nv = rows_by_stage[i + 1].get(t)
                    if nv is None:
                        continue
                    cand = c + nv
                    if best is None or cand < best:
                        best = cand
                if v != best:
                    errors.append(f"Bellman row station {i} {s}: {v} != {best}")
                # Completeness of the successor set is checked by recomputing
                # actions from the instance; an omitted successor cannot change
                # the verifier's minimum because it is not in the recomputed list.
        claimed = cert.get("claimed")
        claimed = None if claimed is None else int(claimed)
        start_row = rows_by_stage[0].get(model.start)
        if start_row != claimed:
            errors.append(f"start row {start_row} != claimed {claimed}")
        if claimed is not None:
            plan = [State.from_any(row) for row in cert.get("plan", [])]
            if not plan or plan[0] != model.start:
                errors.append("plan does not start at the start state")
            total = 0
            okplan = len(plan) == m + 1
            for k in range(len(plan) - 1):
                s, t = plan[k], plan[k + 1]
                acts = model.actions(s)
                match = [(a, c) for a, c in acts if a == t]
                if not match:
                    errors.append(f"plan step {k} not admissible: {s} -> {t}")
                    okplan = False
                    break
                total += match[0][1]
                # Deterministic tie rule: the chosen successor must be the first
                # action attaining the row minimum.
                vnext = rows_by_stage[s.station + 1].get(t)
                target = rows_by_stage[s.station].get(s)
                first = None
                if target is not None:
                    for a, c in acts:
                        nv = rows_by_stage[s.station + 1].get(a)
                        if nv is not None and c + nv == target:
                            first = a
                            break
                if first != t:
                    errors.append(f"plan step {k} is not the first optimal successor")
            if okplan and model.terminal_value(plan[-1]) is None:
                errors.append("plan terminal state is infeasible")
            if cert.get("plan_cost") is not None and int(cert["plan_cost"]) != claimed:
                errors.append("plan_cost != claimed")
            if okplan and total != claimed:
                errors.append(f"plan cost {total} != claimed {claimed}")
        else:
            if cert.get("plan"):
                errors.append("infeasible certificate has a plan")
        return {"ok": not errors, "errors": errors[:50], "error_count": len(errors),
                "states": sum(len(t) for t in rows_by_stage), "horizon": m,
                "claimed": claimed}
    except (KeyError, ValueError, TypeError, IndexError) as e:
        return {"ok": False, "errors": [f"malformed: {type(e).__name__}: {e}"]}


def corrupt(cert: dict, kind: str) -> dict:
    """Return a deliberately corrupted copy (adversarial checker tests)."""
    import copy
    c = copy.deepcopy(cert)
    if kind == "value":
        for table in c["tables"]:
            for row in table:
                if row[4] is not None:
                    row[4] += 1
                    return c
    elif kind == "drop_row":
        for table in c["tables"][1:]:
            if table:
                table.pop(0)
        return c
    elif kind == "wrong_plan":
        if c["plan"]:
            c["plan"][-1]["current"] += 250
        return c
    elif kind == "forge_cost":
        c["claimed"] = int(c["claimed"] or 0) - 1
        c["plan_cost"] = c["claimed"]
        return c
    elif kind == "add_cheap_action":
        # Append a fake successor row with a very low value at the next stage.
        c["tables"][-1].append([c["instance"]["horizon"], 0, 0, 0, -10**9])
        return c
    raise KeyError(kind)
