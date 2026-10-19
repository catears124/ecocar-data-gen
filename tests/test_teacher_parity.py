"""Parity checks against the compiled Lean teacher certificate checker."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from autocross.optimal.certificate import certificate_from_solution, corrupt, write_certificate
from autocross.optimal.solver import solve
from test_teacher_optimal import make_model


@pytest.fixture
def workdir(request):
    safe_name = "".join(character if character.isalnum() or character in "._-" else "_" for character in request.node.name)
    path = Path.cwd() / "runs" / "pytest-teacher-parity" / safe_name
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)

@pytest.fixture
def teachercheck(root):
    exe = root / "lean" / ".lake" / "build" / "bin" / "teachercheck.exe"
    if not exe.exists():
        pytest.skip("Lean teachercheck executable is absent")
    return exe


def run_checker(exe: Path, path: Path):
    return subprocess.run([str(exe), str(path)], capture_output=True, text=True, check=False)


def test_python_certificates_are_accepted_by_lean(teachercheck, workdir):
    models = [
        make_model(horizon=2, b0=4, terminal_cap=10_000_000_000),
        make_model(horizon=2, b0=4, terminal_cap=10_000_000_000,
                   obstacles=((1000, -3000, -2500),)),
        make_model(horizon=2, b0=4, terminal_cap=0, a_lat=0),
    ]
    for i, model in enumerate(models):
        cert = certificate_from_solution(solve(model))
        path = write_certificate(cert, workdir / f"teacher-{i}.json")
        result = run_checker(teachercheck, path)
        assert result.returncode == 0, result.stdout + result.stderr
        payload = json.loads(result.stdout)
        assert payload.get("ok") is True


def test_lean_rejects_corrupted_certificate(teachercheck, workdir):
    cert = certificate_from_solution(solve(make_model(horizon=2, b0=4, terminal_cap=10_000_000_000)))
    bad = corrupt(cert, "value")
    bad["values"][0][0] = (bad["values"][0][0] or 0) + 1
    path = write_certificate(bad, workdir / "corrupt.json")
    result = run_checker(teachercheck, path)
    assert result.returncode != 0
