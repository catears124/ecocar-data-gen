"""Bicycle-model fitting: a bicycle run with altered parameters stands in for CARLA telemetry."""
import copy
import json

from autocross.fit import Windows, compare_summaries, find_runs, fit, write_bicycle_section
from autocross.runner import run

REF = {"mu": 0.92, "max_drive_force": 8600.0, "max_brake_force": 19500.0, "steer_time_constant": 0.11,
       "cornering_stiffness_per_load": 10.5}


def _run(cfg, root, out, track="practice_loop"):
    return run(cfg, str(root / "controllers" / "baseline_pure_pursuit.py"), str(root / "tracks" / f"{track}.yaml"),
               backend="bicycle", out_dir=out, verbose=False)


def test_fit_recovers_lap_time(cfg, root, tmp_path):
    ref_cfg = copy.deepcopy(cfg)
    ref_cfg["vehicle"]["bicycle"].update(REF)
    ref = _run(ref_cfg, root, tmp_path / "ref")
    before = compare_summaries(ref, _run(cfg, root, tmp_path / "before"))
    assert abs(before["diff_pct"]) > 2.0  # the test is only meaningful if the models start apart

    summary = json.loads((tmp_path / "ref" / "summary.json").read_text())
    summary["backend"] = "carla"
    (tmp_path / "ref" / "summary.json").write_text(json.dumps(summary))
    assert find_runs([tmp_path]) == [tmp_path / "ref"]

    w = Windows([tmp_path / "ref"], window_s=1.0, stride_s=1.0)
    new = fit(w, cfg["vehicle"], keys=tuple(REF), max_evals=250, restarts=0, verbose=False)
    for k, v in REF.items():
        assert abs(new[k] - v) / v < 0.05, (k, new[k], v)
    fit_cfg = copy.deepcopy(cfg)
    fit_cfg["vehicle"]["bicycle"].update(new)
    after = compare_summaries(ref, _run(fit_cfg, root, tmp_path / "after"))
    assert abs(after["diff_pct"]) < 0.5


def test_write_bicycle_section_keeps_layout(root, tmp_path):
    p = tmp_path / "sim.yaml"
    p.write_text((root / "config" / "sim.yaml").read_text())
    write_bicycle_section(p, {"mu": 0.91, "traction_limit": 0.9, "yaw_inertia_factor": 1.1})
    text = p.read_text()
    assert "    mu: 0.91\n" in text
    assert "traction_limit: 0.9" in text and "# TC/ABS" in text
    assert "    yaw_inertia_factor: 1.1\n" in text
    import yaml
    c = yaml.safe_load(text)
    assert c["vehicle"]["bicycle"]["yaw_inertia_factor"] == 1.1 and c["actuators"]["delay_steps"] == 0
