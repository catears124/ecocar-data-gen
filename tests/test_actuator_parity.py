"""Parity, execution-binding and status-ladder tests for the actuator certificate."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import lean_exe, numba_or_skip
from autocross.actuator.certificate import (
    certificate_for_action, certificate_from_diag, check_certificate, instance_sha256, sha52,
)
from autocross.actuator.model import model_from_instance
from autocross.actuator.solver import solve, solve_sparse
from autocross.api import TrackInfo
from autocross.certstatus import (
    FALLBACK_REASONS, STATUS_FALLBACK, STATUS_LEAN_CHECKED, STATUS_PY_CHECKED, STATUS_SOLVER,
)
from tests.test_actuator_optimal import make_model

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "teams", ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def test_instance_roundtrip_preserves_reference_cost():
    m = make_model(horizon=2)
    m = m.__class__(**{**m.__dict__, "ref_d": (0, 250, 0), "ref_psi": (0, 1000, 0)})
    r = model_from_instance(m.as_instance())
    assert r.ref_d == (0, 250, 0) and r.ref_psi == (0, 1000, 0)
    b, c, a, d = solve_sparse(r)
    cert = certificate_from_diag(r, d, claimed=c, first_action=a)
    assert check_certificate(cert)["ok"]
    assert b is not None and a is not None
    assert r.action_ok(r.start, *a) == b


def test_dense_and_sparse_plans_have_the_same_cost():
    m = make_model(horizon=2)
    b1, c1, a1, d1 = solve(m)
    b2, c2, a2, d2 = solve_sparse(m)
    assert c1 == c2
    assert d1["plan"][-1].station == m.horizon
    assert d2["plan"][-1].station == m.horizon
    assert a1 == a2
    assert d1["plan_actions"] == d2["plan_actions"]


@pytest.mark.verification
def test_compiled_lean_checker_accepts_generated_certificate(tmp_path):
    exe = lean_exe("actuatorcheck")
    m = make_model(horizon=2)
    _b, c, a, d = solve_sparse(m)
    cert = certificate_from_diag(m, d, claimed=c, first_action=a)
    p = tmp_path / "cert.json"
    p.write_text(json.dumps(cert), encoding="utf-8")
    r = subprocess.run([str(exe), str(p)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert '"ok":true' in r.stdout


# ------------------------------------------------------------------ controller status ladder

def _track_info() -> TrackInfo:
    e = np.zeros((0, 2))
    return TrackInfo(name="course", closed=False, laps=1, centerline=e, width_left=np.zeros(0),
                     width_right=np.zeros(0), cones=e, cone_colors=np.zeros(0, int),
                     start_line=np.zeros((2, 2)), finish_line=np.zeros((2, 2)),
                     gates=np.zeros((0, 2, 2)), full_knowledge=False)


def _controller(**cfg):
    from actuator_certificate import VEHICLE
    from actuator_exact import ActuatorExactController
    c = ActuatorExactController(cfg)
    c.on_start(_track_info(), VEHICLE)
    return c


def _obs(station=20.0, speed=8.0):
    from actuator_certificate import observation
    return observation(ROOT / "tracks" / "practice_basic.yaml", station, speed)


def _numeric(debug: dict):
    for k in ("status", "certified", "fallback_reason"):
        assert isinstance(debug[k], float), k


def test_exact_step_is_solver_status_without_certificates():
    cmd = _controller().step(_obs())
    _numeric(cmd.debug)
    assert cmd.debug["status"] == STATUS_SOLVER
    assert cmd.debug["certified"] == 0.0
    assert cmd.debug["fallback_reason"] == FALLBACK_REASONS["none"]
    assert cmd.debug["certificate_ok"] == 0.0


@pytest.mark.verification
def test_certificate_is_bound_to_executed_action_and_instance(tmp_path):
    ctrl = _controller(emit_certificates=True)
    cmd = ctrl.step(_obs())
    _numeric(cmd.debug)
    assert cmd.debug["status"] == STATUS_PY_CHECKED
    assert cmd.debug["certified"] == 0.0  # only an offline Lean audit may certify
    cert = ctrl._certificate
    assert cert["firstAction"] == [int(cmd.debug["u"]), int(cmd.debug["a_mm_s2"])]
    assert cmd.debug["cert_sha52"] == sha52(cert["certificate_sha256"])
    assert cmd.debug["instance_sha52"] == sha52(instance_sha256(cert["instance"]))
    # Offline audit: the bound certificate is accepted by the compiled Lean checker.
    exe = lean_exe("actuatorcheck")
    p = tmp_path / "bound.json"
    p.write_text(json.dumps(cert), encoding="utf-8")
    r = subprocess.run([str(exe), str(p)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and '"ok":true' in r.stdout, r.stdout


def test_solver_code_reports_the_solver_that_ran(monkeypatch):
    import actuator_exact
    monkeypatch.setattr(actuator_exact, "NUMBA_AVAILABLE", False)
    assert _controller().step(_obs()).debug["solver_code"] == 0.0

    def broken(_m):
        raise RuntimeError("kernel unavailable")
    monkeypatch.setattr(actuator_exact, "NUMBA_AVAILABLE", True)
    monkeypatch.setattr(actuator_exact, "solve_numba", broken)
    cmd = _controller().step(_obs())
    assert cmd.debug["solver_code"] == 2.0 and cmd.debug["status"] == STATUS_SOLVER


@pytest.mark.verification
def test_numba_runs_outside_the_sandbox():
    numba_or_skip()
    assert _controller().step(_obs()).debug["solver_code"] == 1.0


def test_suboptimal_runtime_action_is_never_certified(monkeypatch):
    """A runtime solver that returns a non-optimal action is caught by the
    certificate binding and replaced by an explicit fallback."""
    import actuator_exact

    def wrong(m):
        t, cost, act, diag = solve(m)
        _ts, best, _a, sd = solve_sparse(m)
        for t2, c2, u, a in m.actions(m.start):
            v1 = sd["values"][1].get(t2)
            if v1 is not None and c2 + v1 > best:
                return t2, c2 + v1, (u, a), diag
        pytest.skip("instance has no suboptimal admissible action")
    monkeypatch.setattr(actuator_exact, "NUMBA_AVAILABLE", False)
    monkeypatch.setattr(actuator_exact, "solve", wrong)
    cmd = _controller(emit_certificates=True).step(_obs())
    assert cmd.debug["status"] == STATUS_FALLBACK
    assert cmd.debug["fallback_reason"] == FALLBACK_REASONS["certificate_mismatch"]
    assert cmd.debug["certified"] == 0.0


def test_fallback_commands_carry_reason_codes():
    import dataclasses
    ctrl = _controller()
    obs = _obs()
    bad = dataclasses.replace(obs, road=dataclasses.replace(obs.road, valid=False))
    d = ctrl.step(bad).debug
    assert (d["status"], d["fallback_reason"]) == (STATUS_FALLBACK, FALLBACK_REASONS["invalid_preview"])
    off = dataclasses.replace(obs, road=dataclasses.replace(obs.road, lateral_offset=0.5))
    d = ctrl.step(off).debug
    _numeric(d)
    assert (d["status"], d["fallback_reason"], d["certified"]) == (
        STATUS_FALLBACK, FALLBACK_REASONS["lateral_envelope"], 0.0)
    head = dataclasses.replace(obs, road=dataclasses.replace(obs.road, heading_error=0.05))
    d = ctrl.step(head).debug
    assert d["fallback_reason"] == FALLBACK_REASONS["heading_envelope"]


def test_runtime_commands_cannot_claim_lean_status():
    from actuator_certificate import VEHICLE
    from autocross.actuator.build import command_from_first
    m = make_model(horizon=1)
    with pytest.raises(ValueError):
        command_from_first(_obs(), VEHICLE, m, 0, 0, status=STATUS_LEAN_CHECKED)
    d = command_from_first(_obs(), VEHICLE, m, 0, 0).debug
    assert (d["status"], d["certified"]) == (STATUS_SOLVER, 0.0)


@pytest.mark.verification
def test_offline_audit_tool_requires_lean_acceptance_and_rejection(tmp_path):
    lean_exe("actuatorcheck")
    import actuator_certificate
    rc = actuator_certificate.main(["--track", str(ROOT / "tracks" / "practice_basic.yaml"),
                                    "--station", "20", "--out", str(tmp_path),
                                    "--require-lean", "--corrupt"])
    assert rc == 0


def test_certificate_for_action_rejects_inadmissible_action():
    m = make_model(horizon=2)
    assert certificate_for_action(m, (max(m.steer_inputs) + 1, 0)) is None
