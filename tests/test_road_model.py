import math

import numpy as np
import pytest

from autocross.perception import RoadModelSensor
from autocross.runner import _check_command
from autocross.api import Command
from autocross.track import build, load


def course():
    # 50 m straight, 90 deg left arc of R 20 m, 60 m straight
    return build({"width": 5.0, "segments": [{"type": "straight", "length": 50},
                                              {"type": "arc", "radius": 20, "angle": 90},
                                              {"type": "straight", "length": 60}]})


def test_straight_offset_heading_and_corner_ahead():
    t = course()
    rs = RoadModelSensor(t, {"range_m": 40.0}, np.random.default_rng(0))
    r = rs.measure(20.0, 1.0, math.radians(5.0), progress=20.0)  # 1 m left of centerline, pointing 5 deg left
    assert r.lateral_offset == pytest.approx(1.0, abs=0.02)
    assert math.degrees(r.heading_error) == pytest.approx(5.0, abs=0.2)
    assert r.curvature == pytest.approx(0.0, abs=1e-3)
    assert r.dist_to_left_boundary == pytest.approx(1.5, abs=0.05)
    assert r.dist_to_right_boundary == pytest.approx(3.5, abs=0.05)
    assert len(r.corners) == 1
    c = r.corners[0]
    assert c.direction == 1 and c.entry_distance == pytest.approx(30.0, abs=3.0)
    assert not c.fully_visible and math.isnan(c.exit_distance)


def test_in_corner_fully_visible():
    t = course()
    rs = RoadModelSensor(t, {"range_m": 60.0}, np.random.default_rng(0))
    s = 55.0
    p, h = t.line.point_at(s), t.line.heading_at(s)
    rs.s_hint = s
    r = rs.measure(p[0], p[1], h, progress=s)
    c = r.corners[0]
    assert c.entry_distance < 0 < c.exit_distance and c.fully_visible
    assert c.min_radius == pytest.approx(20.0, rel=0.1)
    assert r.curvature == pytest.approx(1 / 20.0, rel=0.15)


def test_obstacles_only_within_range(root):
    t = load(root / "tracks" / "practice_obstacles.yaml")
    rs = RoadModelSensor(t, {"range_m": 40.0}, np.random.default_rng(0))
    s_ob = float(t.obstacles[0]["s"])
    for s, expect in ((s_ob - 60.0, 0), (s_ob - 20.0, 1)):
        p, h = t.line.point_at(s), t.line.heading_at(s)
        rs.s_hint = s
        r = rs.measure(p[0], p[1], h, progress=s)
        assert len(r.obstacles) == expect
    ob = r.obstacles[0]
    assert ob.distance == pytest.approx(20.0, abs=0.5)
    assert ob.lateral_max - ob.lateral_min == pytest.approx(float(t.obstacles[0]["width"]), abs=0.05)


def test_noise_is_seeded_and_bounded():
    t = course()
    cfg = {"offset_noise_std_m": 0.1, "heading_noise_std_deg": 0.5, "curvature_noise_std": 0.002}
    a = RoadModelSensor(t, cfg, np.random.default_rng(3)).measure(20.0, 0.0, 0.0, 20.0)
    b = RoadModelSensor(t, cfg, np.random.default_rng(3)).measure(20.0, 0.0, 0.0, 20.0)
    assert a.lateral_offset == b.lateral_offset and abs(a.lateral_offset) < 0.5


def test_vehicle_frame_path_converted_to_world():
    cmd, _ = _check_command(Command(path=np.array([[1.0, 0.0], [2.0, 0.0]]), path_frame="vehicle"),
                            (10.0, 5.0, math.pi / 2))
    assert cmd.path == pytest.approx(np.array([[10.0, 6.0], [10.0, 7.0]]))
    cmd, _ = _check_command(Command(path=np.array([[1.0, 2.0]]), path_frame="bogus"), (0.0, 0.0, 0.0))
    assert cmd.path is None
