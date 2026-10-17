"""End-to-end and invariant tests for the submitted controller ``teams/certline.py``.

The course runs use the bicycle backend (no CARLA), as the other controller tests do. The
invariant test is synthetic and fast: it checks the property the plan is built on -- the published
racing line never leaves the clearance interval that the vendored ``steer-1`` decision certifies.
"""
import math
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from autocross.api import Observation, RoadModel, ObstacleInfo, VehicleParams
from autocross.runner import run

CTRL = "teams/certline.py"


def _run(cfg, root, tmp_path, track="practice_basic", ctrl=CTRL, **kw):
    return run(cfg, str(root / ctrl), str(root / "tracks" / f"{track}.yaml"),
               out_dir=tmp_path, verbose=False, **kw)


def test_finishes_clean(cfg, root, tmp_path):
    s = _run(cfg, root, tmp_path)
    m = s["metrics"]
    assert m["completion"]["finished"]
    assert m["penalties"]["cone_hit"] == 0 and m["penalties"]["off_course"] == 0
    assert m["penalties"]["missed_gate"] == 0
    assert m["deviation"]["path_source"] == "controller"
    assert (tmp_path / "telemetry.csv").exists() and (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("track", ["practice_technical", "hackathon_test"])
def test_finishes_clean_technical(cfg, root, tmp_path, track):
    s = _run(cfg, root, tmp_path, track=track)
    m = s["metrics"]
    assert m["completion"]["finished"]
    assert m["penalties"]["off_course"] == 0 and m["penalties"]["missed_gate"] == 0


def test_obstacle_course_no_cone(cfg, root, tmp_path):
    """The sidestep plan and its obstacle memory must get past the cones without touching one."""
    s = _run(cfg, root, tmp_path, track="practice_obstacles")
    m = s["metrics"]
    assert m["completion"]["finished"]
    assert m["penalties"]["cone_hit"] == 0


def test_closed_course(cfg, root, tmp_path):
    s = _run(cfg, root, tmp_path, track="practice_loop")
    assert s["metrics"]["completion"]["finished"]


def test_never_errors_on_any_public_course(cfg, root, tmp_path):
    """A controller error is a visible, costly failure (-1 each, 20 in a row is a DNF)."""
    for track in ("practice_basic", "practice_loop", "practice_obstacles", "practice_technical",
                  "hackathon_test"):
        s = _run(cfg, root, tmp_path, track=track)
        assert s["metrics"]["errors"] == 0, (track, s["controller_error"])
        assert s["controller_error"] is None, track


# --------------------------------------------------------------------------- plan invariant
def _load_controller(root):
    spec = importlib.util.spec_from_file_location("certline_sub", root / CTRL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _vehicle():
    return VehicleParams(mass=1845.0, wheelbase=3.0, track_width=1.67, length=4.79, width=2.16,
                         cg_to_front=1.45, cg_to_rear=1.55, max_steer=math.radians(35.0),
                         max_steer_rate=2.0, mu=0.95, max_accel=5.2, max_decel=9.0, max_speed=40.0,
                         control_dt=0.05, actuation_delay_steps=0)


def _road(lateral=0.0, curvature=0.0, obstacles=(), range_m=45.0, width=3.0):
    """A vehicle-frame road model with the given constant curvature and obstacles."""
    s = np.linspace(0.0, range_m, int(range_m) + 1)
    k = np.full_like(s, curvature)
    y = 0.5 * curvature * s ** 2                      # centerline quadrature, good enough for a plan
    center = np.column_stack([s, y])
    obs = [ObstacleInfo(distance=d, lateral_min=lo, lateral_max=hi,
                        center=np.zeros(2), points=np.zeros((0, 2))) for (d, lo, hi) in obstacles]
    return RoadModel(valid=True, range_m=range_m, lateral_offset=lateral, heading_error=0.0,
                     curvature=curvature, width_left=width, width_right=width,
                     dist_to_left_boundary=width - lateral, dist_to_right_boundary=width + lateral,
                     preview_s=s, preview_curvature=k, preview_centerline=center,
                     preview_width_left=np.full_like(s, width),
                     preview_width_right=np.full_like(s, width),
                     preview_left_boundary=center + [0, width], preview_right_boundary=center - [0, width],
                     corners=[], obstacles=list(obs), finish_distance=float("nan"))


def _obs(road, v=12.0, progress=0.0):
    return Observation(t=0.0, step=0, x=0.0, y=0.0, yaw=0.0, vx=v, vy=0.0, yaw_rate=0.0, ax=0.0,
                       ay=0.0, speed=v, steer=0.0, throttle=0.0, brake=0.0, progress=progress,
                       lap=0, timing_started=True, road=road)


@pytest.mark.parametrize("line_on", [1.0, 0.0])
@pytest.mark.parametrize("road", [
    {"lateral": 0.0, "curvature": 0.0},
    {"lateral": 0.4, "curvature": 0.083},
    {"lateral": -0.5, "curvature": -0.12},
    {"lateral": 0.0, "curvature": 0.05, "obstacles": [(14.0, -2.0, -1.1)]},
    {"lateral": 0.2, "curvature": -0.05, "obstacles": [(10.0, 1.1, 2.0), (22.0, -2.2, -1.2)]},
])
def test_plan_stays_inside_certified_interval(cfg, root, road, line_on):
    """Whatever the planner does, every published offset is inside the interval that steer-1's
    clearance predicate certifies (or the plan is rejected and the centerline is published)."""
    mod = _load_controller(root)
    vp = _vehicle()
    ctrl = mod.CertifiedRacingLine({"line_on": line_on})
    ctrl.on_start(None, vp)
    rm = _road(**road)
    for _ in range(6):
        cmd = ctrl.step(_obs(rm, progress=0.0))
        assert np.isfinite([cmd.steer, cmd.throttle, cmd.brake]).all()
        assert -1.0 <= cmd.steer <= 1.0 and 0.0 <= cmd.throttle <= 1.0 and 0.0 <= cmd.brake <= 1.0
        assert mod._plausible(cmd.path), "published path must be plausible (else the metric ignores it)"
        q = mod.quantise(_obs(rm))
        hd = np.arctan2(np.gradient(rm.preview_centerline[:, 1]), np.gradient(rm.preview_centerline[:, 0]))
        nrm = np.column_stack((-np.sin(hd), np.cos(hd)))
        off_mm = np.einsum("ij,ij->i", cmd.path - rm.preview_centerline, nrm) * 1000.0
        for i in range(len(q["s"])):
            lo, hi = mod._spare_boxes(ctrl.steer_c, q, ctrl.mem, i)
            if lo > hi:
                continue
            assert lo - 100.0 <= off_mm[i] <= hi + 100.0, (i, off_mm[i], lo, hi)


def test_speed_decision_matches_published_curvature(cfg, root):
    """The published plan and its speed caps must describe the same path: under budget the
    certified decision is greatest-safe, over budget it is the braking branch (never throttle)."""
    mod = _load_controller(root)
    vp = _vehicle()
    ctrl = mod.CertifiedRacingLine({})
    ctrl.on_start(None, vp)
    rm = _road(curvature=0.09, obstacles=[(16.0, -2.0, -1.2)])
    for v, want in ((6.0, mod.KIND_GREATEST_SAFE), (20.0, mod.KIND_MIN_VIOLATION)):
        c = ctrl.step(_obs(rm, v=v))
        assert c.debug["kind"] == float(want), (v, c.debug["kind"])
        if want == mod.KIND_MIN_VIOLATION:
            assert c.brake > 0.0 and c.throttle == 0.0
