"""End-to-end runs on the bicycle backend (no CARLA needed)."""
import pytest

from autocross.runner import run


def _run(cfg, root, ctrl, tmp_path, track="practice_basic", **kw):
    return run(cfg, str(ctrl), str(root / "tracks" / f"{track}.yaml"), out_dir=tmp_path, verbose=False, **kw)


def test_baseline_finishes_clean(cfg, root, tmp_path):
    s = _run(cfg, root, root / "controllers" / "baseline_pure_pursuit.py", tmp_path)
    m = s["metrics"]
    assert m["completion"]["finished"]
    assert m["penalties"]["cone_hit"] == 0 and m["penalties"]["off_course"] == 0
    assert m["deviation"]["path_source"] == "controller"
    assert (tmp_path / "telemetry.csv").exists() and (tmp_path / "summary.json").exists()


def test_baseline_closed_course(cfg, root, tmp_path):
    s = _run(cfg, root, root / "controllers" / "baseline_pure_pursuit.py", tmp_path, track="practice_loop")
    assert s["metrics"]["completion"]["finished"]
    assert len(s["metrics"]["completion"]["lap_times_s"]) == 2


def test_baseline_finishes_test_course(cfg, root, tmp_path):
    s = _run(cfg, root, root / "controllers" / "baseline_pure_pursuit.py", tmp_path, track="hackathon_test")
    assert s["metrics"]["completion"]["finished"]
    assert s["metrics"]["penalties"]["off_course"] == 0 and s["metrics"]["penalties"]["missed_gate"] == 0


def test_baseline_finishes_technical_course(cfg, root, tmp_path):
    s = _run(cfg, root, root / "controllers" / "baseline_pure_pursuit.py", tmp_path, track="practice_technical")
    assert s["metrics"]["completion"]["finished"]
    assert s["metrics"]["penalties"]["off_course"] == 0 and s["metrics"]["penalties"]["missed_gate"] == 0


def test_controller_cannot_read_course_files(cfg, root, tmp_path):
    s = _run(cfg, root, root / "tests" / "controllers" / "snoop_controller.py", tmp_path)
    assert s["metrics"]["completion"]["dnf_reason"] == "controller_error"
    assert "may not access course or run files" in s["controller_error"]


def test_event_mode_gives_no_map(cfg, root, tmp_path):
    assert cfg["track_knowledge"] == "perception"  # EIC rules are the default
    s = _run(cfg, root, root / "tests" / "controllers" / "probe_controller.py", tmp_path)
    assert s["controller_error"] is None, s["controller_error"]
    assert s["metrics"]["completion"]["finished"]


def test_template_runs(cfg, root, tmp_path):
    cfg["sim"]["max_time_s"] = 8.0
    s = _run(cfg, root, root / "controllers" / "template_controller.py", tmp_path)
    assert s["controller_error"] is None and s["metrics"]["compute"]["errors"] == 0


def test_deadline_overruns_enforced(cfg, root, tmp_path):
    cfg["sim"]["max_time_s"] = 6.0
    s = _run(cfg, root, root / "tests" / "controllers" / "slow_controller.py", tmp_path)
    c = s["metrics"]["compute"]
    assert c["overruns"] >= 0.15 * c["steps"]
    assert s["score"]["category_scores"]["compute"] < 50


def test_crash_is_dnf(cfg, root, tmp_path):
    s = _run(cfg, root, root / "tests" / "controllers" / "crash_controller.py", tmp_path)
    assert s["metrics"]["completion"]["dnf_reason"] == "controller_error"
    assert "boom" in s["controller_error"]


def test_nan_command_rejected(cfg, root, tmp_path):
    s = _run(cfg, root, root / "tests" / "controllers" / "nan_controller.py", tmp_path)
    assert s["metrics"]["completion"]["dnf_reason"] == "controller_error"


def test_leaving_course_penalised(cfg, root, tmp_path):
    s = _run(cfg, root, root / "tests" / "controllers" / "straight_controller.py", tmp_path)
    m = s["metrics"]
    assert not m["completion"]["finished"]
    assert m["penalties"]["off_course"] >= 1
    assert s["score"]["deductions"]["off_course"] >= 5


@pytest.mark.parametrize("pert", [{"friction_scale": 0.8}, {"actuation_delay_steps": 2}, {"cone_dropout": 0.3},
                                  {"start_lateral_offset": 1.0, "start_yaw_offset_deg": 5}])
def test_perturbations_run(cfg, root, tmp_path, pert):
    s = _run(cfg, root, root / "controllers" / "baseline_pure_pursuit.py", tmp_path, perturb=pert)
    assert s["perturbation"] == pert
