import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def cfg():
    from autocross.config import load_sim_config
    c = load_sim_config()
    c["backend"] = "bicycle"
    c["output"]["plot"] = False
    return c


@pytest.fixture
def root():
    return ROOT


@pytest.fixture(scope="session")
def design():
    import yaml
    return yaml.safe_load(open(ROOT / "config" / "datagen.yaml"))


@pytest.fixture(scope="session")
def smoke(design):
    from autocross.datagen.design import build_manifest
    return build_manifest(design, "smoke")


@pytest.fixture(scope="session")
def corpus(tmp_path_factory, smoke):
    """A 2-scenario bicycle corpus (both runs clean) shared by the corpus tests (read-only)."""
    from autocross.datagen.generate import generate_corpus
    base = tmp_path_factory.mktemp("corpus")
    mp = base / "manifest.json"
    mp.write_text(json.dumps(smoke))
    out = base / "c"
    generate_corpus(str(mp), str(out), "bicycle", limit=2, verbose=False)
    return out


def lake_or_skip():
    """Lean is mandatory when REQUIRE_LEAN=1 (CI and release): a missing toolchain FAILS the test."""
    from autocross.datagen import DatagenError
    from autocross.datagen.lean_bridge import lake_path
    try:
        return lake_path()
    except DatagenError as e:
        if os.environ.get("REQUIRE_LEAN") == "1":
            pytest.fail(f"REQUIRE_LEAN=1 but Lean is unavailable: {e}")
        pytest.skip("Lean (lake) not installed")


# ------------------------------------------------------------------ verification no-skip guard
# Tests marked ``@pytest.mark.verification`` carry the solver/checker parity and
# Lean-acceptance evidence.  With REQUIRE_VERIFICATION=1 (CI) any such test
# that ends up skipped fails the session, so missing tooling can never turn
# into a silently green run.

_VERIFICATION_SKIPS: list[str] = []


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "verification: solver/certificate verification evidence; a skip fails the session "
        "when REQUIRE_VERIFICATION=1")


def pytest_runtest_logreport(report):
    if report.skipped and "verification" in report.keywords:
        _VERIFICATION_SKIPS.append(report.nodeid)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    if os.environ.get("REQUIRE_VERIFICATION") == "1" and _VERIFICATION_SKIPS:
        tr = session.config.pluginmanager.get_plugin("terminalreporter")
        msg = (f"REQUIRE_VERIFICATION=1: {len(_VERIFICATION_SKIPS)} verification test(s) skipped: "
               + ", ".join(_VERIFICATION_SKIPS))
        if tr is not None:
            tr.write_line(msg, red=True, bold=True)
        else:
            print(msg, file=sys.stderr)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def lean_exe(name: str) -> Path:
    """Compiled Lean checker ``lean/.lake/build/bin/<name>[.exe]``.

    With REQUIRE_LEAN=1 a missing executable FAILS the test; otherwise it skips.
    """
    bindir = ROOT / "lean" / ".lake" / "build" / "bin"
    for cand in (bindir / f"{name}.exe", bindir / name):
        if cand.is_file():
            return cand
    msg = f"{name} is not built (cd lean && lake build {name})"
    if os.environ.get("REQUIRE_LEAN") == "1":
        pytest.fail(f"REQUIRE_LEAN=1 but {msg}")
    pytest.skip(msg)


def numba_or_skip():
    """The compiled solver; with REQUIRE_VERIFICATION=1 a missing numba FAILS the test."""
    from autocross.actuator.numba_solver import NUMBA_AVAILABLE, NUMBA_IMPORT_ERROR, solve_numba
    if not NUMBA_AVAILABLE:
        if os.environ.get("REQUIRE_VERIFICATION") == "1":
            pytest.fail(f"REQUIRE_VERIFICATION=1 but numba is unavailable: {NUMBA_IMPORT_ERROR}")
        pytest.skip(f"numba unavailable: {NUMBA_IMPORT_ERROR}")
    return solve_numba
