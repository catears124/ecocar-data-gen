"""Imitation-data pipeline: manifest design, coverage, contrasts, splits (corpus and certificate
tests: test_certification.py)."""
import copy
import itertools
from pathlib import Path

import pytest

from autocross.datagen import cert
from autocross.datagen.design import build_manifest, check_manifest, coverage_report, required_cells
from autocross.datagen.spec import get_family

ROOT = Path(__file__).resolve().parent.parent
FAM = get_family("brake_turn_obstacle")



def _base_params():
    return {"turn_radius_m": 15.0, "turn_direction": "left", "preview_m": 30.0, "approach_speed_mps": 12.0,
            "obstacle_side": "inside", "obstacle_clearance_m": 3.5, "road_width_m": 6.0, "turn_angle_deg": 90.0,
            "obstacle_arc_frac": 0.5}


# ------------------------------------------------------------------ scenario bounds and bins
def test_bins_respect_half_open_edges():
    r = FAM.factor("turn_radius_m")
    assert r.bin_of(13.999) == "tight" and r.bin_of(14.0) == "medium" and r.bin_of(35.0) == "open"
    assert r.bin_of(8.999) is None and r.bin_of(35.001) is None


@pytest.mark.parametrize("change,reason", [
    ({"turn_radius_m": 8.5}, "formal:out_of_bounds:turn_radius_m"),
    ({"preview_m": 61.0}, "formal:out_of_bounds:preview_m"),
    ({"turn_direction": "up"}, "formal:unknown_level:turn_direction"),
    ({"obstacle_arc_frac": 0.9}, "formal:out_of_bounds:obstacle_arc_frac"),
    ({"road_width_m": 5.0, "obstacle_clearance_m": 4.0}, "formal:clearance_exceeds_width"),
    ({"preview_m": 30.0001}, "schema:bad_value:preview_m"),
])
def test_invalid_scenarios_are_rejected_with_reason(change, reason):
    p = {**_base_params(), **change}
    assert reason in FAM.violations(p)


def test_valid_scenario_builds_the_declared_course():
    p = _base_params()
    assert FAM.violations(p) == []
    lay = FAM.layout(p)
    # clearance is realised exactly: obstacle inner edge to the opposite boundary
    assert lay["obstacle_lateral_min_m"] + p["road_width_m"] / 2 == pytest.approx(p["obstacle_clearance_m"])
    assert lay["corner_entry_progress"] - lay["handover_progress"] > p["preview_m"]  # turn hidden at hand-over


# ------------------------------------------------------------------ manifest determinism, coverage, contrasts, splits
def test_manifest_is_deterministic_and_seed_sensitive(design, smoke):
    again = build_manifest(design, "smoke")
    assert again["content_sha256"] == smoke["content_sha256"]
    assert [e["scenario_id"] for e in again["scenarios"]] == [e["scenario_id"] for e in smoke["scenarios"]]
    other = build_manifest({**design, "seed": design["seed"] + 1}, "smoke")
    assert {e["scenario_id"] for e in other["scenarios"]} != {e["scenario_id"] for e in smoke["scenarios"]}
    assert check_manifest(smoke) == []


def test_rejected_draws_are_recorded_not_dropped(smoke):
    rej = [e for e in smoke["scenarios"] if e["status"] == "rejected"]
    assert rej and all(e["rejection_reasons"] for e in rej)


def test_pairwise_cells_and_coverage_accounting(design):
    cells = required_cells(FAM, 2)
    levels = [len(f.labels) for f in FAM.factors]
    assert len(cells) == sum(a * b for a, b in itertools.combinations(levels, 2))
    full = build_manifest(design, "full")
    assert full["coverage"]["missing"] == []
    # drop every scenario covering one cell: the report must name exactly that cell as missing
    acc = [e for e in full["scenarios"] if e["status"] == "accepted"]
    cell = {"turn_radius_m": "tight", "preview_m": "short"}
    kept = [e for e in acc if not (e["bins"]["turn_radius_m"] == "tight" and e["bins"]["preview_m"] == "short")]
    rep = coverage_report(cells, kept)
    assert rep["missing"] == ["turn_radius_m=tight|preview_m=short"]


def test_contrast_pairs_change_exactly_one_factor(smoke):
    ids = {e["scenario_id"]: e for e in smoke["scenarios"] if e["status"] == "accepted"}
    assert smoke["contrast_pairs"]
    for p in smoke["contrast_pairs"]:
        a, b = ids[p["base_id"]]["params"], ids[p["variant_id"]]["params"]
        assert [k for k in a if a[k] != b[k]] == [p["factor"]]


def test_check_manifest_catches_a_second_changed_factor(smoke):
    m = copy.deepcopy(smoke)
    p = m["contrast_pairs"][0]
    var = next(e for e in m["scenarios"] if e["scenario_id"] == p["variant_id"])
    other = "turn_angle_deg"
    var["params"][other] = round(var["params"][other] + 1.0, 3)
    errs = check_manifest(m)
    assert any("scenario_id does not match" in e for e in errs)
    assert any(f"differs in" in e and other in e for e in errs)


def test_check_manifest_catches_split_leakage(smoke):
    m = copy.deepcopy(smoke)
    p = m["contrast_pairs"][0]
    var = next(e for e in m["scenarios"] if e["scenario_id"] == p["variant_id"])
    var["split"] = "test" if var["split"] != "test" else "train"
    errs = check_manifest(m)
    assert any("spans splits" in e for e in errs) and any("split of its region" in e for e in errs)


def test_held_out_regions_never_reach_train(design):
    full = build_manifest(design, "full")
    held = {("open", "nominal"): "val", ("medium", "short"): "test"}
    for e in full["scenarios"]:
        if e["status"] == "accepted":
            key = (e["bins"]["turn_radius_m"], e["bins"]["preview_m"])
            assert e["split"] == held.get(key, "train")


# ------------------------------------------------------------------ certified decision vs brute force
def _brute_max_b1(caps, s, b0, aA, aB):
    best = None
    for tail in itertools.product(range(0, max(caps) + 1), repeat=len(caps) - 1):
        b = (b0,) + tail
        ok = all(b[i] <= caps[i] for i in range(len(b)))
        ok = ok and all(b[i + 1] <= b[i] + 2 * aA * (s[i + 1] - s[i]) and b[i] <= b[i + 1] + 2 * aB * (s[i + 1] - s[i])
                        for i in range(len(b) - 1))
        if ok:
            best = b[1] if best is None else max(best, b[1])
    return best


@pytest.mark.parametrize("caps,b0", [([30, 25, 4, 30], 20), ([30, 30, 30, 2], 9), ([12, 30, 1, 1], 12),
                                     ([30, 8, 30, 30], 0), ([5, 30, 30, 3], 6)])
def test_certified_next_speed_is_the_greatest_feasible(caps, b0):
    c = {"a_acc": 2, "a_brk": 3}
    s = [0, 1, 2, 3]
    bw = cert.backward(c, caps, s)
    brute = _brute_max_b1(caps, s, b0, c["a_acc"], c["a_brk"])
    if b0 > bw[0]:
        assert brute is None  # over budget: no feasible profile at all -> min_violation branch
    else:
        assert cert.forward(c, bw, s, b0)[1] == brute



