"""Pure-source checks for longitudinal Lean proof inventory."""
import re

from autocross.datagen.lean_bridge import LEAN_DIR, _declared_theorems


def test_audit_lists_every_declared_corpus_theorem_without_running_lean():
    src = (LEAN_DIR / "Corpus" / "Audit.lean").read_text(encoding="utf-8")
    listed = set(re.findall(r"^#print axioms (\S+)", src, re.M))
    assert sorted(set(_declared_theorems()) - listed) == []
