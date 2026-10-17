import math

import numpy as np
import pytest

from autocross import geometry as g
from autocross import trackgen
from autocross.track import build, load


def test_projection_sign_and_station():
    line = g.Polyline(np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]))
    s, lat, _ = line.project(np.array([5.0, 2.0]))
    assert s == pytest.approx(5.0) and lat == pytest.approx(2.0)
    s, lat, _ = line.project(np.array([15.0, -1.0]), s_hint=14.0, window=5.0)
    assert s == pytest.approx(15.0) and lat == pytest.approx(-1.0)


def test_curvature_of_circle():
    th = np.linspace(0, 2 * math.pi, 400, endpoint=False)
    pts = 20.0 * np.column_stack((np.cos(th), np.sin(th)))
    assert np.allclose(g.curvature(pts, closed=True), 1 / 20.0, rtol=1e-3)


def test_frames_roundtrip():
    p = np.array([[3.0, -2.0], [10.0, 5.0]])
    b = g.world_to_body(p, 1.0, 2.0, 0.7)
    assert np.allclose(g.body_to_world(b, 1.0, 2.0, 0.7), p)


@pytest.mark.parametrize("name", ["practice_basic", "practice_loop", "practice_obstacles", "hackathon_test"])
def test_shipped_tracks_valid(root, name):
    t = load(root / "tracks" / f"{name}.yaml")
    assert t.validate() == []
    info = t.info(True, 20.0)
    assert len(info.centerline) == len(info.width_left) and len(info.gates) > 5
    blind = t.info(False, 20.0)
    assert len(blind.centerline) == 0 and len(blind.cones) == 0
    assert blind.name == "course"  # a name could encode a generator seed


def test_private_tracks_valid(root):
    import yaml
    for f in sorted((root / "tracks" / "private").glob("*.yaml")):
        spec = yaml.safe_load(f.read_text())
        if "segments" in spec or "centerline" in spec:  # skip config files such as evaluation.yaml
            assert load(f).validate() == [], f.name


def test_obstacle_on_boundary_cone_is_dropped():
    spec = {"segments": [{"type": "straight", "length": 120}], "width": 5.0,
            "obstacles": [{"s": 50.0, "offset": 1.75, "width": 1.5}]}  # outer cone lands on the left cone line
    t = build(spec)
    assert t.validate() == []
    assert len(t.obstacle_cones) == 2 and len(t.obstacle_s) == 2  # cones at 1.0, 1.75; the one at 2.5 is dropped


def test_overlapping_extra_cone_rejected():
    t = build({"segments": [{"type": "straight", "length": 120}], "width": 5.0,
               "extra_cones": [{"x": 50.0, "y": 2.6}]})  # 0.1 m from a left boundary cone
    assert any("apart" in p for p in t.validate())


@pytest.mark.parametrize("seed", range(6))
def test_generated_obstacles_clear_of_cones(seed):
    assert build(trackgen.generate_open(seed, n_obstacles=4)).validate() == []
    assert build(trackgen.generate_closed(seed, n_obstacles=3)).validate() == []


def test_arc_segment_geometry():
    t = build({"segments": [{"type": "straight", "length": 50}, {"type": "arc", "radius": 20, "angle": 90},
                            {"type": "straight", "length": 50}]})
    assert t.center[-1] == pytest.approx([70.0, 70.0], abs=0.2)  # (50 + R, R + 50)


def test_generators_deterministic():
    a, b = trackgen.generate_open(5), trackgen.generate_open(5)
    assert a == b and build(a).validate() == []
    assert build(trackgen.generate_closed(2)).validate() == []


def test_overlapping_course_rejected():
    # 270 deg loop whose exit straight runs back across the entry straight
    t = build({"segments": [{"type": "straight", "length": 40}, {"type": "arc", "radius": 15, "angle": 270},
                            {"type": "straight", "length": 60}]})
    assert any("overlaps" in p for p in t.validate())
