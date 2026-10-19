"""Independent tests for the ``timed-1`` certificate and its Lean checker.

The conformance oracle is ``lean/timed_vectors.json``: every plan of every
reference instance with the scalar reference's objective tuple, the optimum and
the published first command.  These tests require the independent Python
checker to accept the solver's certificate, every adversarial mutation to be
rejected by Python (and by the compiled Lean checker on the tiny instances), the
compiled Lean checker to accept every real certificate, and the scalar reference
to reproduce every stored plan cost.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from conftest import ROOT, lean_exe
from autocross.timed.certificate import (
    CORRUPTIONS, certificate_for_instance, check_certificate, corrupt, write_certificate,
)
from autocross.timed.model import Instance, evaluate
from autocross.timed.vector import Solver

VECTORS = json.loads((ROOT / "lean" / "timed_vectors.json").read_text(encoding="utf-8"))
CASES = VECTORS["cases"]
CASE_BY_NAME = {c["name"]: c for c in CASES}
TINY = [c for c in CASES if c["name"].startswith("tiny")]
DISAGREEMENT = ROOT / "tests" / "fixtures" / "timed_lean_disagreement_392.json"


@pytest.fixture
def scratch():
    """Scratch directory under the repo (the runner's pytest tmp dir may be locked)."""
    base = ROOT / "tmp" / "timed_lean_tests"
    base.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="run_", dir=base))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def solved(name: str):
    case = CASE_BY_NAME[name]
    inst = Instance.from_json(case["instance"])
    plan, claimed, _stats = Solver(inst).solve()
    assert list(plan) == case["best"]
    assert list(claimed) == case["best_cost"]
    return inst, plan, certificate_for_instance(inst, plan)


@pytest.mark.verification
@pytest.mark.parametrize("name", [c["name"] for c in CASES])
def test_python_checker_accepts_solver_certificates(name):
    _inst, _plan, cert = solved(name)
    result = check_certificate(cert)
    assert result["ok"], result["errors"]


@pytest.mark.verification
@pytest.mark.parametrize("name", ["tiny_straight", "full_default"])
def test_corruptions_rejected(name):
    _inst, _plan, cert = solved(name)
    for kind in CORRUPTIONS:
        bad = corrupt(cert, kind)
        result = check_certificate(bad)
        assert not result["ok"], f"{kind} accepted: {result}"


@pytest.mark.verification
def test_corruptions_rejected_by_lean_tiny(scratch):
    exe = lean_exe("timedcheck")
    for case in TINY:
        _inst, _plan, cert = solved(case["name"])
        for kind in CORRUPTIONS:
            path = scratch / f"{case['name']}.{kind}.json"
            write_certificate(corrupt(cert, kind), path)
            r = subprocess.run([str(exe), str(path)], capture_output=True, text=True, timeout=300)
            assert r.returncode != 0, f"{case['name']}/{kind} accepted: {r.stdout}"
            assert '"ok":false' in r.stdout, f"{case['name']}/{kind}: {r.stdout} {r.stderr}"


@pytest.mark.verification
def test_accepts_audit_disagreement_fixture(scratch):
    """Regression: a recorded closed-loop step (v > V_KIN, non-zero body stations)
    that the checker once rejected because body points omitted the station."""
    cert = json.loads(DISAGREEMENT.read_text(encoding="utf-8"))
    result = check_certificate(cert)
    assert result["ok"], result["errors"]
    exe = lean_exe("timedcheck")
    path = scratch / "timed_lean_disagreement_392.json"
    write_certificate(cert, path)
    r = subprocess.run([str(exe), str(path)], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"Lean rejected the audit fixture: {r.stdout} {r.stderr}"
    assert '"ok":true' in r.stdout


@pytest.mark.verification
@pytest.mark.parametrize("name", [c["name"] for c in CASES])
def test_lean_accepts_real_certificates(name, scratch):
    exe = lean_exe("timedcheck")
    case = CASE_BY_NAME[name]
    inst = Instance.from_json(case["instance"])
    plan, _claimed, _stats = Solver(inst).solve()
    path = scratch / f"{name}.json"
    write_certificate(certificate_for_instance(inst, plan), path)
    r = subprocess.run([str(exe), str(path)], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"Lean rejected {name}: {r.stdout} {r.stderr}"
    assert '"ok":true' in r.stdout
    assert f'"plans":{len(case["plans"])}' in r.stdout


@pytest.mark.parametrize("case", TINY, ids=[c["name"] for c in TINY])
def test_plan_costs_match_vectors(case):
    inst = Instance.from_json(case["instance"])
    for plan, cost in zip(case["plans"], case["costs"]):
        assert list(evaluate(inst, plan)) == cost, (case["name"], plan)
