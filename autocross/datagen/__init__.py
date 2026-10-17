"""Proof-carrying contrastive imitation corpus: scenario manifests, expert labels from simulator
runs, corpus validation and the Lean artifact checker. See docs/datagen.md."""
from __future__ import annotations

import subprocess

from ..config import ROOT

GENERATOR_NAME = "autocross.datagen"
GENERATOR_VERSION = "1.0.1"
MANIFEST_SCHEMA = "autocross.datagen.manifest"
INDEX_SCHEMA = "autocross.datagen.index"
SAMPLE_SCHEMA = "autocross.datagen.sample"
INTERCHANGE_SCHEMA = "autocross.datagen.lean_interchange"
SCHEMA_VERSION = 2


class DatagenError(RuntimeError):
    """A data-generation precondition failed (bad manifest, backend unavailable, ...)."""


def git_state() -> dict:
    """Commit and dirty flag of the harness checkout (None outside a git checkout)."""
    def _git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    commit = _git("rev-parse", "HEAD") or None
    status = _git("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}
