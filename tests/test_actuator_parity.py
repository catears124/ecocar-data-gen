"""Parity and execution-binding tests for the actuator certificate."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from autocross.actuator.certificate import certificate_from_diag, check_certificate
from autocross.actuator.model import model_from_instance
from autocross.actuator.solver import solve, solve_sparse
from tests.test_actuator_optimal import make_model


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


@pytest.mark.skipif(os.name != "nt", reason="compiled checker path is Windows-specific here")
def test_compiled_lean_checker_accepts_generated_certificate():
    root = Path(__file__).resolve().parents[1]
    exe = root / "lean" / ".lake" / "build" / "bin" / "actuatorcheck.exe"
    if not exe.exists():
        pytest.skip("actuatorcheck has not been built")
    m = make_model(horizon=2)
    _b, c, a, d = solve_sparse(m)
    cert = certificate_from_diag(m, d, claimed=c, first_action=a)
    p = Path(__file__).with_name("_actuator_cert.json")
    try:
        p.write_text(json.dumps(cert), encoding="utf-8")
        r = subprocess.run([str(exe), str(p)], capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, (r.stdout, r.stderr)
        assert '"ok":true' in r.stdout
    finally:
        p.unlink(missing_ok=True)
