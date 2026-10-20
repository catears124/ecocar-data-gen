"""Certification status ladder shared by the exact controllers.

The harness drops non-numeric ``Command.debug`` values, so every label is an
integer code published as a float.  The ladder is strictly ordered:

* ``STATUS_FALLBACK``     heuristic / safety fallback; optimal for nothing.
* ``STATUS_SOLVER``       runtime exact-solver output for the declared instance;
                          no checked certificate.
* ``STATUS_PY_CHECKED``   a certificate for exactly the executed instance and
                          executed first action was accepted by the independent
                          Python checker at runtime.
* ``STATUS_LEAN_CHECKED`` assigned only by an OFFLINE audit after the compiled
                          Lean checker accepted the certificate bound to the
                          executed action.  Runtime code never sets it: the
                          controller sandbox forbids subprocesses and native code.
"""
from __future__ import annotations

STATUS_FALLBACK = 0
STATUS_SOLVER = 1
STATUS_PY_CHECKED = 2
STATUS_LEAN_CHECKED = 3

STATUS_NAMES = {
    STATUS_FALLBACK: "fallback",
    STATUS_SOLVER: "solver",
    STATUS_PY_CHECKED: "python_checked",
    STATUS_LEAN_CHECKED: "lean_checked",
}

FALLBACK_REASONS = {
    "none": 0,
    "invalid_preview": 1,
    "lateral_envelope": 2,
    "heading_envelope": 3,
    "obstacle": 4,
    "tight_curvature": 5,
    "infeasible": 6,
    "performance_guard": 7,
    "solver_error": 8,
    "certificate_mismatch": 9,
    "encoder_domain": 10,
    "recovery_guard": 11,
    "planner_error": 12,
    "model_infeasible": 13,
    "path_build_failed": 14,
    "implausible_path": 15,
    "tracker_error": 16,
}


def status_debug(status: int, reason: str = "none") -> dict[str, float]:
    """Numeric debug fields for one command.

    ``certified`` is 1.0 only for ``STATUS_LEAN_CHECKED``, which runtime code
    never produces.
    """
    if status not in STATUS_NAMES:
        raise ValueError(f"unknown certification status {status!r}")
    if reason not in FALLBACK_REASONS:
        raise ValueError(f"unknown fallback reason {reason!r}")
    return {"status": float(status),
            "certified": 1.0 if status == STATUS_LEAN_CHECKED else 0.0,
            "fallback_reason": float(FALLBACK_REASONS[reason])}


__all__ = ["STATUS_FALLBACK", "STATUS_SOLVER", "STATUS_PY_CHECKED", "STATUS_LEAN_CHECKED",
           "STATUS_NAMES", "FALLBACK_REASONS", "status_debug"]
