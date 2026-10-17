"""Anti-cheat: a controller must not escape its sandbox, learn which course it is driving,
run code in the harness, or hide compute from the budget."""
import pytest

from autocross.runner import run


def _run(cfg, root, ctrl, tmp_path, ctrl_cfg=None, track="practice_basic", t=1.0):
    cfg["sim"]["max_time_s"] = t
    return run(cfg, str(root / "tests" / "controllers" / ctrl), str(root / "tracks" / f"{track}.yaml"),
               out_dir=tmp_path / "run", verbose=False, controller_cfg=ctrl_cfg)


@pytest.mark.parametrize("action", ["subprocess", "os_system", "socket", "write", "remove", "ctypes",
                                    "ctypes_pythonapi", "gc", "symlink"])
def test_escape_blocked(cfg, root, tmp_path, action):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    c = {"action": action, "path": str(victim if action in ("write", "remove") else tmp_path / "link"),
         "target": str(root / "tracks" / "practice_basic.yaml")}
    s = _run(cfg, root, "escape_controller.py", tmp_path, c)
    assert s["controller_error"] is None, s["controller_error"]
    assert victim.read_text() == "keep"


def test_course_file_read_blocked_through_symlink_free_path(cfg, root, tmp_path):
    c = {"action": "read_course", "path": str(root / "tracks" / ".." / "tracks" / "practice_basic.yaml")}
    s = _run(cfg, root, "escape_controller.py", tmp_path, c)
    assert s["controller_error"] is None, s["controller_error"]


def test_course_name_not_in_controller_memory(cfg, root, tmp_path):
    s = _run(cfg, root, "escape_controller.py", tmp_path, {"action": "frames", "needle_reversed": "practice_basic"[::-1]})
    assert s["controller_error"] is None, s["controller_error"]


def test_pickle_injection_rejected(cfg, root, tmp_path):
    marker = tmp_path / "pwned.txt"
    s = _run(cfg, root, "pickle_injection_controller.py", tmp_path, {"marker": str(marker)})
    assert not marker.exists()
    assert s["metrics"]["compute"]["errors"] > 0
    assert "rejected message" in (s["controller_error"] or "")


def test_background_threads_count_as_compute(cfg, root, tmp_path):
    s = _run(cfg, root, "background_cpu_controller.py", tmp_path, t=4.0)
    c = s["metrics"]["compute"]
    assert c["cpu_ms_per_step"] > 3.0  # the spinning thread is measured (an idle controller uses ~0.05 ms)
    assert s["score"]["category_scores"]["compute"] < 100.0


def test_path_deviation_cannot_be_gamed(cfg, root, tmp_path):
    def dev(ctrl, c=None):
        s = run(cfg, str(ctrl), str(root / "tracks" / "practice_basic.yaml"), out_dir=tmp_path / str(len(c or {})),
                verbose=False, controller_cfg=c)
        return s["score"]["category_scores"]["deviation"], s["metrics"]["deviation"]["path_source"]

    honest, src = dev(root / "controllers" / "baseline_pure_pursuit.py")
    assert src == "controller"
    gaming = root / "tests" / "controllers" / "path_gaming_controller.py"
    base = str(root / "controllers" / "baseline_pure_pursuit.py")
    stub, _ = dev(gaming, {"mode": "stub", "baseline": base})
    zig, zsrc = dev(gaming, {"mode": "zigzag", "baseline": base, "x": 1})
    assert stub < honest - 20          # a stub path ahead of the car scores badly
    assert zsrc == "centerline"        # an implausible path is ignored ...
    assert zig <= honest + 1e-6        # ... so it is no better than honestly tracking a plan


def test_carla_client_cannot_be_loaded(cfg, root, tmp_path, monkeypatch):
    """A stand-in 'carla' package on the path: the controller must not be able to load it."""
    pkg = tmp_path / "site" / "carla"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("raise SystemExit('carla client loaded')\n")
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    s = _run(cfg, root, "escape_controller.py", tmp_path, {"action": "carla"})
    assert s["controller_error"] is None, s["controller_error"]
