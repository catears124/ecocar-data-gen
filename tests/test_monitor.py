import pytest

from autocross.backends.base import VehicleState
from autocross.monitor import RaceMonitor
from autocross.track import build


class VP:
    length, width, cg_to_front, cg_to_rear, track_width = 4.8, 2.0, 1.5, 1.5, 1.6


RULES = {"gate_spacing_m": 10.0, "off_course_rule": "all_wheels", "off_course_dnf_s": 50.0}


def straight():
    return build({"width": 5.0, "cone_spacing": 5.0, "start_offset": 10.0, "finish_runoff": 20.0,
                  "segments": [{"type": "straight", "length": 130}]})


def drive(track, lateral_fn, v=10.0, dt=0.025, t_max=30.0):
    m = RaceMonitor(track, RULES, VP, 120.0)
    t = 0.0
    while not m.done and t < t_max:
        t += dt
        x = v * t
        m.update(VehicleState(t, x, lateral_fn(x), 0.0, v, 0.0, 0.0, 0.0, 0.0), dt)
    return m


def test_clean_run_timing():
    m = drive(straight(), lambda x: 0.0)
    assert m.finish_time is not None and m.dnf is None
    # start line at 10 m, finish at 130 - 20 = 110 m -> 100 m at 10 m/s
    assert m.finish_time - m.start_time == pytest.approx(10.0, abs=0.03)
    assert not [e for e in m.events if e.kind in ("off_course", "missed_gate", "cone_hit")]


def test_excursion_counts_off_course_gates_and_cones():
    # ramp out past the left cone line (w/2 = 2.5 m), hold at 6 m, ramp back in
    m = drive(straight(), lambda x: min(6.0, max(0.0, 0.6 * min(x - 45.0, 85.0 - x))))
    kinds = [e.kind for e in m.events]
    assert kinds.count("off_course") == 1
    assert kinds.count("missed_gate") >= 1
    assert kinds.count("cone_hit") >= 2  # knocks cones going out and coming back in
    assert m.finish_time is not None


def test_wrong_way_dnf():
    m = RaceMonitor(straight(), RULES, VP, 120.0)
    for i in range(600):
        t = (i + 1) * 0.025
        x = 5.0 + 12.5 * t if t <= 2 else 30.0 - 5.0 * (t - 2)
        m.update(VehicleState(t, x, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0), 0.025)
        if m.done:
            break
    assert m.dnf == "wrong_way"
