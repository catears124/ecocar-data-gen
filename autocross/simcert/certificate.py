"""Certificates for a simulator-evaluated finite-candidate decision.

The certificate is a statement relative to the harness's bicycle transition
function: the selected command sequence is the first minimiser of the declared
lexicographic objective over the declared finite candidate set, and the
baseline candidate is in that set.  It does not assert optimality over all
continuous command sequences; the declared resolution and the measured
Lipschitz gap are separate fields (see ``tools/simulation_certificate.py``).
"""
from __future__ import annotations

import json

from .sim import SimInstance, first_command, rollout

SCHEMA = "autocross.simcert.decision"
SCHEMA_VERSION = 1


def instance_from_cert(cert: dict) -> SimInstance:
    return SimInstance(
        vehicle=cert["vehicle"],
        state0=tuple(float(x) for x in cert["state0"]),
        app0=tuple(float(x) for x in cert["app0"]),
        acc0=tuple(float(x) for x in cert["acc0"]),
        candidates=tuple(tuple(tuple(float(v) for v in c) for c in seq)
                         for seq in cert["candidates"]),
        preview_cl=tuple(tuple(float(v) for v in p) for p in cert["preview_cl"]),
        preview_s=tuple(float(x) for x in cert["preview_s"]),
        preview_wl=tuple(float(x) for x in cert["preview_wl"]),
        preview_wr=tuple(float(x) for x in cert["preview_wr"]),
        preview_k=tuple(float(x) for x in cert["preview_k"]),
        obstacles=tuple(tuple(float(v) for v in o) for o in cert["obstacles"]),
        half_width=float(cert["half_width"]),
        half_length=float(cert["half_length"]),
        margin=float(cert["margin"]),
        t_tail=float(cert.get("t_tail", 0.5)),
        max_steer_rate=float(cert.get("max_steer_rate", 2.0)),
        baseline_index=int(cert.get("baseline_index", -1)),
    )


def check_certificate(cert: dict) -> dict:
    """Independent recomputation of the simulator rollout and selection."""
    errors: list[str] = []
    if cert.get("schema") != SCHEMA or int(cert.get("schema_version", -1)) != SCHEMA_VERSION:
        return {"ok": False, "errors": ["schema"]}
    try:
        inst = instance_from_cert(cert)
        result = rollout(inst)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "errors": [f"rollout: {type(exc).__name__}: {exc}"]}
    costs = [list(c) for c in result["costs"]]
    if costs != [list(c) for c in cert["costs"]]:
        errors.append("recomputed costs differ from the certificate")
    selected = int(cert["selected"])
    if not (0 <= selected < len(costs)):
        errors.append("selected index out of range")
    else:
        target = costs[selected]
        if any(c < target for c in costs):
            errors.append("selected candidate is not the first minimiser")
        if costs.index(target) != selected:
            errors.append("tie-break differs: selected is not the first minimiser")
        first = first_command(inst, selected)
        claimed = tuple(float(v) for v in cert["first_command"])
        if any(abs(a - b) > 1e-12 for a, b in zip(first, claimed)):
            errors.append("first command does not match the selected candidate")
    baseline = int(cert.get("baseline_index", -1))
    if baseline < 0 or baseline >= len(costs):
        errors.append("baseline candidate missing")
    else:
        if baseline in cert.get("baseline_members", []):
            pass
        if costs[baseline] != list(cert["baseline_cost"]):
            errors.append("baseline cost does not match")
        if not (costs[selected] <= costs[baseline]):
            errors.append("selected candidate is worse than the baseline candidate")
    return {"ok": not errors, "errors": errors, "selected_cost": costs[selected] if 0 <= selected < len(costs) else None,
            "baseline_cost": costs[baseline] if 0 <= baseline < len(costs) else None,
            "candidates": len(costs)}


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


__all__ = ["SCHEMA", "SCHEMA_VERSION", "instance_from_cert", "check_certificate", "canonical"]
