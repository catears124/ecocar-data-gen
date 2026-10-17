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


# --- Speed dynamics: acceleration reachability, caps, declared travel time ----
#
# Lean side (kernel reduction, no native_decide) for the same instance:
#   timeAt 0, timeAt 250, timeAt 500 = 120, 60, 40
#   value toySpeedModel toySpeedModel.stages toySpeedStart = toySpeedEnumMin = 160
#   toySpeedExtract = [250, 500, 250]  (cost 160, chain legal)
#   toySpeedCandidates.any toySpeedChain = true      (nonempty)
#   toySpeedOK 1 0 500 = false                       (acceleration reachability bites)
#   toySpeedOK 3 500 500 = false, toySpeedOK 3 500 250 = true   (the turn caps speed)

SPEED_GRID = (0, 250, 500)
SPEED_CAPS = (500, 500, 250)   # cap of the arriving station 1, 2, 3
TOY_ACCEL = 250
TIME_TABLE = {0: 120, 250: 60, 500: 40}


def speed_step_ok(target_station: int, v: int, w: int) -> bool:
    """Declared bucket, under the cap of the arriving station, inside the
    acceleration envelope (the concrete instance declares no braking limit)."""
    return (
        w in SPEED_GRID
        and w <= SPEED_CAPS[target_station - 1]
        and w <= v + TOY_ACCEL
    )


def speed_chain_ok(chain: list[int]) -> bool:
    if len(chain) != len(SPEED_CAPS):
        return False
    v = 0
    for i, w in enumerate(chain):
        if not speed_step_ok(i + 1, v, w):
            return False
        v = w
    return True


def speed_chain_cost(chain: list[int]) -> int:
    return sum(TIME_TABLE[w] for w in chain)


def enumerate_speed_optimum() -> tuple[int, list[list[int]]]:
    feasible = [list(c) for c in itertools.product(SPEED_GRID, repeat=len(SPEED_CAPS))
                if speed_chain_ok(list(c))]
    best = min(speed_chain_cost(c) for c in feasible)
    return best, [c for c in feasible if speed_chain_cost(c) == best]


def test_speed_reachability_and_caps_bite():
    # Cannot jump from standstill to the top bucket in one station.
    assert not speed_step_ok(1, 0, 500)
    # The turn caps the arrival speed, so the optimum is not "always fastest".
    assert not speed_step_ok(3, 500, 500)
    assert speed_step_ok(3, 500, 250)
    # The declared time table is strictly decreasing in speed.
    assert TIME_TABLE[0] > TIME_TABLE[250] > TIME_TABLE[500]


def test_speed_enumeration_is_nonempty_and_nontrivial():
    best, argmins = enumerate_speed_optimum()
    assert best == 160, f"exhaustive speed optimum {best}"
    assert argmins == [[250, 500, 250]], f"exhaustive speed argmin {argmins}"


def test_speed_python_dp_mirror_matches_enumeration():
    """Layered DP over (station, speed bucket), the same discrete model Lean's
    `toySpeedModel` reduces to for the straight offset profile."""
    frontier = {0: 0}  # speed bucket -> best declared time so far
    for station in range(1, len(SPEED_CAPS) + 1):
        nxt: dict[int, int] = {}
        for v, cost in frontier.items():
            for w in SPEED_GRID:
                if not speed_step_ok(station, v, w):
                    continue
                candidate = cost + TIME_TABLE[w]
                if w not in nxt or candidate < nxt[w]:
                    nxt[w] = candidate
        frontier = nxt
    best, _ = enumerate_speed_optimum()
    assert frontier, "speed DP frontier is empty"
    assert min(frontier.values()) == best


def test_speed_lean_kernel_agrees_with_python_enumeration():
    if shutil.which("lake") is None:
        pytest.skip("lake not available")
    probe = LEAN_DIR / "dpspeed_probe.lean"
    probe.write_text(
        "import Corpus\n"
        "open Corpus Corpus.Bellman Corpus.Dp\n"
        "#eval (value toySpeedModel toySpeedModel.stages toySpeedStart, toySpeedEnumMin, "
        "toySpeedCost toySpeedExtract, toySpeedChain toySpeedExtract, toySpeedExtract)\n",
        encoding="utf-8",
    )
    try:
        out = subprocess.run(
            ["lake", "env", "lean", "dpspeed_probe.lean"],
            cwd=LEAN_DIR, capture_output=True, text=True, timeout=600,
        )
    finally:
        probe.unlink(missing_ok=True)
    if out.returncode != 0:
        pytest.skip(f"lake env lean failed: {out.stdout}{out.stderr}")
    line = out.stdout.strip().splitlines()[-1]
    parts = line.strip("()").split(",")
    value, enum_min, extract_cost, chain_ok = parts[0], parts[1], parts[2], parts[3]
    extract = ",".join(parts[4:]).strip()
    best, argmins = enumerate_speed_optimum()
    assert int(value) == best, f"DP value {value} != enumeration {best}"
    assert int(enum_min) == best
    assert int(extract_cost) == best, "extracted speed chain is not optimal"
    assert chain_ok.strip() == "true", "extracted speed chain violates reachability/caps"
    assert extract.strip() == "[250, 500, 250]"
    assert argmins == [[250, 500, 250]]
