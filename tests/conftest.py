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
