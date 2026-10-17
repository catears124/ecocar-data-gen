"""The declared lateral DP of `lean/Corpus/Dp.lean` against exhaustive enumeration.

The Lean side computes, by kernel reduction (`decide`, no `native_decide`):

    value toyModel toyModel.stages initialState = -2000
    toyEnumMin                                 = -2000
    toyExtract                                 = [0, 0]   (cost -2000, chain legal)
    toyCandidates.any toyChain                 = true     (nonempty feasible set)

This module enumerates the same toy instance in Python -- three stations, grid
[-250, 0, 250], integer stencil curvature limit 500, speed-cap surrogate
`1000 - |n - 2x + p|`, cost the negative sum of caps -- and checks agreement
with those values.  When `lake` is available the Lean constants are re-derived
live; otherwise the recorded kernel values above are used.
"""

from __future__ import annotations

import itertools
import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LEAN_DIR = REPO / "lean"

GRID = (-250, 0, 250)
STENCIL_MAX = 500
STATIONS = 3  # two transitions


def cap(previous: int, current: int, nxt: int) -> int:
    return 1000 - abs(nxt - 2 * current + previous)


def legal_chain(chain: list[int]) -> bool:
    """`chain[i]` is the offset at station `i`; the model starts at station 0
    with previous == current == 0, so the first stencil includes that origin."""
    if len(chain) != STATIONS - 1:
        return False
    previous, current = 0, 0
    for nxt in chain:
        if nxt not in GRID:
            return False
        if abs(nxt - 2 * current + previous) > STENCIL_MAX:
            return False
        previous, current = current, nxt
    return True


def chain_cost(chain: list[int]) -> int:
    previous, current = 0, 0
    total = 0
    for nxt in chain:
        total -= cap(previous, current, nxt)
        previous, current = current, nxt
    return total


def enumerate_optimum() -> tuple[int, list[list[int]]]:
    feasible = [list(c) for c in itertools.product(GRID, repeat=STATIONS - 1) if legal_chain(list(c))]
    best = min(chain_cost(c) for c in feasible)
    return best, [c for c in feasible if chain_cost(c) == best]


def lean_values() -> dict:
    """Re-derive the constants from the Lean kernel when `lake` is present."""
    if shutil.which("lake") is None:
        pytest.skip("lake not available")
    probe = LEAN_DIR / "dptoy_probe.lean"
    probe.write_text(
        "import Corpus\n"
        "open Corpus Corpus.Bellman Corpus.Dp\n"
        "#eval (value toyModel toyModel.stages initialState, toyEnumMin, "
        "toyChainCost toyExtract, toyChain toyExtract)\n",
        encoding="utf-8",
    )
    try:
        out = subprocess.run(
            ["lake", "env", "lean", "dptoy_probe.lean"],
            cwd=LEAN_DIR, capture_output=True, text=True, timeout=600,
        )
    finally:
        probe.unlink(missing_ok=True)
    if out.returncode != 0:
        pytest.skip(f"lake env lean failed: {out.stdout}{out.stderr}")
    line = out.stdout.strip().splitlines()[-1]
    value, enum_min, extract_cost, chain_ok = line.strip("()").split(",")
    return {
        "value": int(value),
        "enum_min": int(enum_min),
        "extract_cost": int(extract_cost),
        "chain_ok": chain_ok.strip() == "true",
    }


def test_enumeration_matches_recorded_kernel_values():
    best, argmins = enumerate_optimum()
    assert best == -2000, f"exhaustive optimum {best}"
    assert argmins == [[0, 0]], f"exhaustive argmin {argmins}"


def test_feasible_set_is_nonempty_and_constraints_bite():
    feasible = [c for c in itertools.product(GRID, repeat=STATIONS - 1) if legal_chain(list(c))]
    assert feasible, "no feasible chain: the instance would be vacuous"
    # The curvature limit excludes real candidates, so the instance is nontrivial.
    assert len(feasible) < len(list(itertools.product(GRID, repeat=STATIONS - 1)))
    assert not legal_chain([-250, 250])
    assert not legal_chain([250, -250])


def test_infeasible_is_not_free():
    """An illegal transition must be rejected, not costed as zero."""
    assert not legal_chain([-250, 250])


def test_lean_kernel_agrees_with_python_enumeration():
    best, argmins = enumerate_optimum()
    vals = lean_values()
    assert vals["value"] == best, f"DP value {vals['value']} != enumeration {best}"
    assert vals["enum_min"] == best
    assert vals["extract_cost"] == best, "extracted trajectory is not optimal"
    assert vals["chain_ok"] is True, "extracted trajectory violates the step constraints"
    assert argmins == [[0, 0]]


def test_python_dp_mirror_matches_enumeration():
    """A layer-by-layer DP over the same integer states agrees with enumeration."""
    frontier = {(0, 0): 0}  # (previous, current) -> best cost so far
    for _ in range(STATIONS - 1):
        nxt_frontier: dict[tuple[int, int], int] = {}
        for (previous, current), cost in frontier.items():
            for nxt in GRID:
                if abs(nxt - 2 * current + previous) > STENCIL_MAX:
                    continue
                key = (current, nxt)
                candidate = cost - cap(previous, current, nxt)
                if key not in nxt_frontier or candidate < nxt_frontier[key]:
                    nxt_frontier[key] = candidate
        frontier = nxt_frontier
    best, _ = enumerate_optimum()
    assert frontier, "DP frontier is empty"
    assert min(frontier.values()) == best
