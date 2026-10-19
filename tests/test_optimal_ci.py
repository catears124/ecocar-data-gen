"""Fast, Lean-independent release gates for the optimal teacher."""
from __future__ import annotations

from pathlib import Path

import autocross.optimal as optimal


def _tiny_model() -> optimal.Model:
    """Return the smallest feasible model that still exercises the certificate schema."""
    lon = {
        "a_lat": 1_000_000_000,
        "a_acc": 10,
        "a_brk": 10,
        "a_max": 1_000_000_000,
        "v_max": 100,
        "r_h": 1_000,
        "hold": 0,
        "ramp_in": 0,
        "ramp_out": 0,
        "r_sw_m": 0,
        "veh_width": 0,
    }
    q = {
        "base": {
            "v0": 1,
            "s": [0, 1_000],
            "k": [0, 0],
            "wl": [1_000, 1_000],
            "wr": [1_000, 1_000],
            "obstacles": [],
        },
        "lat": 0,
    }
    consts = {"lon": lon, "margin": 0, "step": 1_000, "kappa_max": 10**12}
    return optimal.Model(
        consts,
        q,
        (0,),
        (1,),
        1,
        optimal.State(0, 0, 0, 1),
        1,
    )


def test_optimal_certificate_contract_without_lean() -> None:
    solution = optimal.solve(_tiny_model())
    certificate = optimal.certificate_from_solution(solution)
    assert optimal.check_certificate(certificate)["ok"] is True

    required = {
        "consts",
        "q",
        "offsets",
        "speeds",
        "roots",
        "horizon",
        "start",
        "terminalCap",
        "states",
        "values",
        "plan",
        "claimed",
        "planCost",
    }
    assert required <= certificate.keys()

    for b, root in certificate["roots"]:
        assert root * root <= b < (root + 1) ** 2

    assert optimal.check_certificate(optimal.corrupt(certificate, "value"))["ok"] is False
    assert optimal.check_certificate(optimal.corrupt(certificate, "wrong_plan"))["ok"] is False


def test_lean_sources_contain_no_untrusted_shortcuts() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden = ("sorry", "native_decide", "axiom")
    for relative in ("lean/Corpus/Teacher.lean", "lean/TeacherCheck.lean"):
        path = root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{token!r} found in {relative}"
