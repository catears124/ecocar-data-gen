"""Command line entry point:  python -m autocross <command> --help"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DEFAULT_SIM, PERTURBATION_KEYS, load_sim_config, load_yaml, parse_sets


def _kv(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k.strip()] = float(v)
    return out


def _common(p):
    p.add_argument("--controller", "-c", required=True, help="path to the team's controller .py file")
    p.add_argument("--controller-class", help="class name if the file defines several Controller subclasses")
    p.add_argument("--controller-config", help="YAML file passed to the controller constructor")
    p.add_argument("--team", help="team name used in run folders / leaderboard (default: controller file name)")
    p.add_argument("--backend", choices=["carla", "bicycle"], help="override config backend")
    p.add_argument("--config", default=str(DEFAULT_SIM), help="simulation config YAML")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override any config value, e.g. --set carla.render=false --set sim.seed=3")
    p.add_argument("--no-render", action="store_true", help="CARLA no-rendering mode (fast, headless)")
    p.add_argument("--quiet", action="store_true")


def _cfg(a):
    cfg = load_sim_config(a.config, a.set)
    if getattr(a, "no_render", False):
        cfg["carla"]["render"] = False
    return cfg


def cmd_run(a):
    from .runner import run
    from .track import load
    cfg = _cfg(a)
    if a.realtime is not None:
        if a.realtime <= 0:
            raise SystemExit("--realtime FACTOR must be > 0")
        cfg["sim"]["realtime_factor"] = a.realtime
    viewer = None
    if a.view:
        from .viewer import TopDownViewer
        viewer = TopDownViewer(load(a.track))
    try:
        s = run(cfg, a.controller, a.track, team=a.team, backend=a.backend, perturb=_kv(a.perturb),
                controller_class=a.controller_class,
                controller_cfg=load_yaml(a.controller_config) if a.controller_config else None,
                tag=a.tag or "", viewer=viewer, rubric=load_yaml(a.scoring) if a.scoring else None,
                verbose=not a.quiet)
    finally:
        if viewer:
            viewer.close()
    return 0 if s["metrics"]["completion"]["finished"] else 2


def cmd_batch(a):
    from .runner import run_batch
    batch = load_yaml(a.batch)
    if a.track:
        batch["tracks"] = a.track
    agg = run_batch(_cfg(a), a.controller, batch, team=a.team, backend=a.backend,
                    controller_class=a.controller_class,
                    controller_cfg=load_yaml(a.controller_config) if a.controller_config else None,
                    verbose=not a.quiet)
    return 0 if agg["finish_rate"] > 0 else 2


def cmd_score(a):
    from .runner import print_summary
    from .score import rescore
    s = rescore(a.run_dir, a.scoring, _kv(a.judged) or None)
    print_summary(s, a.run_dir)
    if s["score"]["judged_points"]:
        print(f"    judged {s['score']['judged_points']} -> TOTAL {s['score']['total_points']:.1f} / 100")
    return 0


def cmd_leaderboard(a):
    from .score import format_table, leaderboard
    rows = leaderboard(a.runs_dir, a.csv, a.track)
    if not rows:
        print("no runs found")
        return 1
    cols = ["team", "track", "finished", "time_s", "adjusted_time_s", "cones", "off_course", "missed_gates",
            "completion_time_score", "deviation_score", "compute_score", "dynamics_score", "technical", "total"]
    print(format_table(rows, cols))
    if a.award:
        print("\nAward tracks:")
        fin = [r for r in rows if r["finished"]]  # awards only go to clean finishes
        for title, key, rev in (("Fastest completion", "adjusted_time_s", False), ("Most physically realistic", "dynamics_score", True),
                                ("Best compute efficiency", "compute_score", True), ("Most precise", "deviation_score", True)):
            if fin:
                best = sorted(fin, key=lambda r: r[key], reverse=rev)[0]
                print(f"  {title:<28} {best['team']} ({best[key]:.2f})")
    return 0


def cmd_gen_track(a):
    from . import trackgen
    from .plot import preview_track
    from .track import build, save
    seed = a.seed
    if seed is None:  # secret course: the seed is never shown or stored, so nobody can regenerate it
        import secrets
        seed = secrets.randbits(48)
    if a.closed:
        spec = trackgen.generate_closed(seed, a.length or 600.0, a.width or 6.0, a.laps, a.obstacles)
    else:
        spec = trackgen.generate_open(seed, a.length or 700.0, a.width or 5.5, n_obstacles=a.obstacles)
    if a.name or a.seed is None:
        spec["name"] = a.name or Path(a.out).stem
    save(spec, a.out)
    t = build(spec)
    print(f"wrote {a.out}: {t.length:.0f} m, {len(t.cones)} cones, {len(t.obstacle_cones)} obstacle cones")
    if a.preview:
        png = Path(a.out).with_suffix(".png")
        preview_track(t, png)
        print(f"preview {png}")
    return 0


def cmd_show_track(a):
    from .plot import preview_track
    from .track import load
    from .api import VehicleParams
    cfg = load_sim_config(a.config)
    t = load(a.track)
    v = cfg["vehicle"]
    b = v["bicycle"]
    vp = VehicleParams(b["mass"], b["wheelbase"], b["track_width"], b["length"], b["width"], b["cg_to_front"],
                       b["wheelbase"] - b["cg_to_front"], 0.6, v["steer_rate_limit"], v["mu"], v["max_accel"],
                       v["max_decel"], v["max_speed"], 0.05, 0)
    problems = t.validate()
    print(f"{t.name}: {'closed x%d laps' % t.laps if t.closed else 'open'}, centerline {t.length:.1f} m, "
          f"timed distance {t.finish_progress - t.start_progress:.1f} m, {len(t.cones)} cones, "
          f"{len(t.obstacle_cones)} unmapped obstacle cones, {len(t.gate_progress(cfg['rules']['gate_spacing_m']))} gates")
    print(f"reference time: {t.reference_time if t.reference_time else 'auto'} "
          f"(estimate {t.reference_time_estimate(vp):.2f} s)")
    print("validation:", "OK" if not problems else "; ".join(problems))
    out = a.out or str(Path(a.track).with_suffix(".png"))
    preview_track(t, out, cfg["rules"]["gate_spacing_m"])
    print(f"preview {out}")
    return 0 if not problems else 1


def cmd_plot(a):
    from .plot import plot_run
    plot_run(a.run_dir)
    print(Path(a.run_dir) / "report.png")
    return 0


def cmd_calibrate(a):
    from .calibrate import calibrate
    cfg = _cfg(a)
    res = calibrate(cfg, a.backend or cfg["backend"])
    print(json.dumps(res, indent=2))
    print("\nSuggested config/sim.yaml vehicle values:")
    print(f"  mu: {res['mu_estimate']:.2f}\n  max_accel: {res['max_accel']:.1f}\n  max_decel: {res['max_decel']:.1f}")
    return 0


def cmd_compare(a):
    import datetime as dt_

    from .config import ROOT
    from .fit import compare_summaries
    from .runner import run
    cfg = _cfg(a)
    cfg["carla"]["render"] = False
    root = Path(a.out) if a.out else ROOT / cfg["output"]["runs_dir"] / f"{dt_.datetime.now():%Y%m%d-%H%M%S}_compare"
    ctrl_cfg = load_yaml(a.controller_config) if a.controller_config else None
    rows = []
    for tp in a.track:
        name = Path(tp).stem
        s = {}
        for be in ("carla", "bicycle"):
            if be == "carla" and a.carla_runs:
                s[be] = _carla_run_for(a.carla_runs, name)
                continue
            s[be] = run(cfg, a.controller, tp, team=a.team, backend=be, controller_class=a.controller_class,
                        controller_cfg=ctrl_cfg, out_dir=root / f"{name}_{be}", verbose=not a.quiet)
        rows.append({"track": name, **compare_summaries(s["carla"], s["bicycle"])})
    worst = 0.0
    print(f"\n{'course':<22}{'CARLA s':>10}{'bicycle s':>11}{'diff':>9}{'speed RMS':>11}")
    for r in rows:
        if r["diff_pct"] is None:
            worst = float("inf")
            print(f"{r['track']:<22}  finished: CARLA {r['ref_finished']}, bicycle {r['dev_finished']}")
            continue
        worst = max(worst, abs(r["diff_pct"]))
        rms = f"{r['speed_rms_mps']:.2f} m/s" if r["speed_rms_mps"] is not None else "-"
        print(f"{r['track']:<22}{r['ref_time_s']:>10.2f}{r['dev_time_s']:>11.2f}{r['diff_pct']:>+8.2f}%{rms:>11}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "compare.json").write_text(json.dumps({"tolerance_pct": a.tolerance, "results": rows}, indent=2))
    ok = worst <= a.tolerance
    print(f"\nworst lap-time difference {worst:.2f}% ({'within' if ok else 'OUTSIDE'} {a.tolerance:g}%) -> {root}")
    return 0 if ok else 1


def _carla_run_for(dirs, track_name):
    from .fit import find_runs
    for rd in find_runs(dirs, "carla"):
        s = json.loads((rd / "summary.json").read_text())
        if s["track"]["name"] == track_name or rd.name.startswith(track_name):
            s["out_dir"] = str(rd)
            return s
    raise SystemExit(f"no CARLA run for course '{track_name}' under {', '.join(dirs)}")


def cmd_fit_bicycle(a):
    from .fit import Windows, find_runs, fit, write_bicycle_section
    cfg = load_sim_config(a.config)
    runs = find_runs(a.runs, "carla")
    if not runs:
        raise SystemExit(f"no CARLA runs (summary.json + telemetry.csv) under {', '.join(a.runs)}")
    print(f"[fit] {len(runs)} CARLA run(s): " + ", ".join(r.name for r in runs))
    w = Windows(runs, window_s=a.window)
    new = fit(w, cfg["vehicle"], max_evals=a.max_evals)
    print("\nFitted vehicle.bicycle values:")
    for k, v in new.items():
        print(f"    {k}: {v:.6g}")
    if a.write:
        write_bicycle_section(a.config, new)
        print(f"\nwritten to {a.config}; check agreement with `python -m autocross compare`")
    return 0


def cmd_selftest(a):
    """Every public course, official settings, both backends if CARLA is available: what a team
    should run before submitting."""
    import datetime as dt_

    from .config import ROOT
    from .runner import run
    cfg = _cfg(a)
    cfg["carla"]["render"] = False
    backends = [a.backend] if a.backend else ["bicycle"] + (["carla"] if a.carla else [])
    courses = sorted((ROOT / "tracks").glob("*.yaml"))
    root = ROOT / cfg["output"]["runs_dir"] / f"{dt_.datetime.now():%Y%m%d-%H%M%S}_selftest"
    ctrl_cfg = load_yaml(a.controller_config) if a.controller_config else None
    rows, errors = [], []
    for be in backends:
        for c in courses:
            s = run(cfg, a.controller, str(c), team=a.team, backend=be, controller_class=a.controller_class,
                    controller_cfg=ctrl_cfg, out_dir=root / f"{c.stem}_{be}", verbose=False)
            m, sc = s["metrics"], s["score"]
            cm, p = m["completion"], m["penalties"]
            rows.append((c.stem, be, "finished" if cm["finished"] else f"DNF ({cm['dnf_reason']})",
                         cm["time_s"], p["cone_hit"], p["off_course"], p["missed_gate"], m["compute"]["step_ms_p99"],
                         sc["technical_points"]))
            if s.get("controller_error"):
                errors.append((c.stem, be, s["controller_error"]))
    print(f"\n{'course':<22}{'backend':<9}{'result':<26}{'time':>7}{'cones':>7}{'off':>5}{'gates':>6}{'step p99':>10}{'tech':>7}")
    for r in rows:
        t = f"{r[3]:.1f}" if r[3] else "-"
        print(f"{r[0]:<22}{r[1]:<9}{r[2]:<26}{t:>7}{r[4]:>7}{r[5]:>5}{r[6]:>6}{r[7]:>8.1f}ms{r[8]:>7.1f}")
    tech = [r[8] for r in rows]
    print(f"\nmean technical points {sum(tech) / len(tech):.1f} / 75 (the evaluation uses unreleased courses; "
          "see README 'Official evaluation')")
    for course, be, err in errors[:3]:
        last = err.strip().splitlines()[-1]
        print(f"\ncontroller error on {course} [{be}]: {last}")
        if "controllers may not" in err:
            print("  -> this breaks a sandbox rule (README 'Rules the harness enforces'); the same error happens "
                  "in the evaluation")
    if not a.carla:
        print("\nbicycle model only; add --carla to also run on CARLA (server running)")
    print(f"-> {root}")
    return 1 if errors else 0


def cmd_check(a):
    import platform
    ok = True
    print(f"python {platform.python_version()} ({sys.executable})")
    for mod in ("numpy", "yaml", "psutil", "matplotlib", "pygame", "carla"):
        try:
            m = __import__(mod)
            print(f"  [ok] {mod} {getattr(m, '__version__', '')}")
        except Exception as e:
            opt = mod in ("pygame", "carla")
            ok &= opt
            print(f"  [{'--' if opt else 'FAIL'}] {mod}: {e}")
    cfg = load_sim_config(a.config)
    try:
        import carla
        c = carla.Client(cfg["carla"]["host"], int(cfg["carla"]["port"]))
        c.set_timeout(5.0)
        print(f"  [ok] CARLA server {c.get_server_version()} at {cfg['carla']['host']}:{cfg['carla']['port']} "
              f"(client {c.get_client_version()})")
    except Exception as e:
        print(f"  [--] CARLA server not reachable ({e.__class__.__name__}); start it with scripts/start_carla.ps1 "
              "or use --backend bicycle")
    return 0 if ok else 1


def cmd_data_manifest(a):
    from .datagen.design import build_manifest, summarize
    design = load_yaml(a.design)
    if a.profile not in design.get("profiles", {}):
        raise SystemExit(f"profile '{a.profile}' not in {a.design} (have: {', '.join(design.get('profiles', {}))})")
    m = build_manifest(design, a.profile)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(m, indent=1, allow_nan=False))
    print(summarize(m))
    print(f"-> {a.out}")
    return 0


def cmd_data_check_manifest(a):
    from .datagen.design import check_manifest, summarize
    m = json.loads(Path(a.manifest).read_text())
    errs = check_manifest(m)
    if m.get("family_spec"):
        print(summarize(m))
    for e in errs[:50]:
        print(f"  ERROR: {e}")
    print("manifest OK" if not errs else f"manifest INVALID ({len(errs)} problems)")
    return 0 if not errs else 1


def cmd_data_generate(a):
    from .datagen.generate import main_generate
    return main_generate(a)


def cmd_data_validate(a):
    from .datagen.validate import main_validate
    return main_validate(a)


def cmd_data_lean(a):
    from .datagen.lean_bridge import main_lean
    return main_lean(a)


def cmd_data_report(a):
    from .datagen.report import main_report
    return main_report(a)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m autocross", description="EcoCAR autocross simulation harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run one controller on one course")
    _common(p)
    p.add_argument("--track", "-t", required=True)
    p.add_argument("--view", action="store_true", help="live top-down pygame view")
    p.add_argument("--realtime", nargs="?", type=float, const=1.0, metavar="FACTOR",
                   help="pace the run to wall-clock time (default x1.0; e.g. 0.5 = slow motion, 2 = double speed)")
    p.add_argument("--perturb", action="append", metavar="KEY=VALUE",
                   help="robustness perturbation: " + ", ".join(PERTURBATION_KEYS))
    p.add_argument("--scoring", help="scoring rubric YAML (default config/scoring.yaml)")
    p.add_argument("--tag", help="suffix for the run folder")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("batch", help="robustness sweep: courses x perturbations x seeds")
    _common(p)
    p.add_argument("--batch", default="config/robustness.yaml")
    p.add_argument("--track", action="append", help="override the batch course list")
    p.set_defaults(fn=cmd_batch)

    p = sub.add_parser("score", help="re-score a run folder (optionally adding judged points)")
    p.add_argument("run_dir")
    p.add_argument("--scoring")
    p.add_argument("--judged", action="append", metavar="CATEGORY=SCORE0-100", help="e.g. innovation=85")
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("leaderboard", help="best run per team/course under a runs folder")
    p.add_argument("runs_dir", nargs="?", default="runs")
    p.add_argument("--track")
    p.add_argument("--csv")
    p.add_argument("--award", action="store_true", help="also list the parallel award tracks")
    p.set_defaults(fn=cmd_leaderboard)

    p = sub.add_parser("gen-track", help="generate a random valid course from a seed")
    p.add_argument("--seed", type=int, help="omit for a secret random course (evaluation courses)")
    p.add_argument("--out", required=True)
    p.add_argument("--closed", action="store_true")
    p.add_argument("--laps", type=int, default=2)
    p.add_argument("--length", type=float)
    p.add_argument("--width", type=float)
    p.add_argument("--obstacles", type=int, default=0, help="number of unmapped obstacle cone groups")
    p.add_argument("--name")
    p.add_argument("--preview", action="store_true")
    p.set_defaults(fn=cmd_gen_track)

    p = sub.add_parser("show-track", help="validate a course, print stats and save a preview PNG")
    p.add_argument("track")
    p.add_argument("--out")
    p.add_argument("--config", default=str(DEFAULT_SIM))
    p.set_defaults(fn=cmd_show_track)

    p = sub.add_parser("plot", help="regenerate report.png for a run folder")
    p.add_argument("run_dir")
    p.set_defaults(fn=cmd_plot)

    p = sub.add_parser("calibrate", help="measure vehicle handling limits on a backend")
    p.add_argument("--backend", choices=["carla", "bicycle"])
    p.add_argument("--config", default=str(DEFAULT_SIM))
    p.add_argument("--set", action="append")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("compare", help="run a controller on CARLA and on the bicycle model, compare lap times")
    _common(p)
    p.add_argument("--track", "-t", action="append", required=True, help="repeat for several courses")
    p.add_argument("--carla-runs", action="append", metavar="DIR",
                   help="reuse existing CARLA runs from these folders instead of driving CARLA again")
    p.add_argument("--tolerance", type=float, default=2.5, help="allowed lap-time difference in %% (exit 1 if over)")
    p.add_argument("--out", help="output folder (default runs/<stamp>_compare)")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("fit-bicycle", help="fit the bicycle model parameters to CARLA run telemetry")
    p.add_argument("runs", nargs="+", help="CARLA run folders, or folders containing them (e.g. a compare folder)")
    p.add_argument("--config", default=str(DEFAULT_SIM))
    p.add_argument("--write", action="store_true", help="write the fitted values into the config's vehicle.bicycle")
    p.add_argument("--window", type=float, default=1.5, help="replay window length in seconds")
    p.add_argument("--max-evals", type=int, default=1500)
    p.set_defaults(fn=cmd_fit_bicycle)

    p = sub.add_parser("selftest", help="run a controller on every public course with the official settings "
                                         "(do this before submitting)")
    _common(p)
    p.add_argument("--carla", action="store_true", help="also run every course on CARLA (server running)")
    p.set_defaults(fn=cmd_selftest)

    p = sub.add_parser("check", help="verify the installation and CARLA connectivity")
    p.add_argument("--config", default=str(DEFAULT_SIM))
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("data-manifest", help="build an imitation-data scenario manifest (coverage design, contrasts, splits)")
    p.add_argument("--profile", default="smoke", help="design profile in the design file (smoke, full, large)")
    p.add_argument("--design", default="config/datagen.yaml")
    p.add_argument("--out", required=True, help="manifest JSON to write")
    p.set_defaults(fn=cmd_data_manifest)

    p = sub.add_parser("data-check-manifest", help="recompute and check every derived field of a manifest")
    p.add_argument("manifest")
    p.set_defaults(fn=cmd_data_check_manifest)

    p = sub.add_parser("data-generate", help="run a manifest's scenarios with the expert and write the corpus")
    p.add_argument("--manifest", required=True)
    p.add_argument("--backend", required=True, choices=["carla", "bicycle"],
                   help="carla = the real corpus (server required); bicycle = tooling check, labelled as such")
    p.add_argument("--out", required=True, help="corpus folder")
    p.add_argument("--design", default="config/datagen.yaml", help="design file (its 'generation' config overrides)")
    p.add_argument("--config", default=str(DEFAULT_SIM))
    p.add_argument("--set", action="append", metavar="KEY=VALUE")
    p.add_argument("--limit", type=int, help="only the first N accepted scenarios (partial corpus)")
    p.add_argument("--workers", type=int, default=1, help="parallel scenario runs (bicycle only)")
    p.add_argument("--resume", action="store_true", help="keep completed scenarios whose files are intact")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_data_generate)

    p = sub.add_parser("data-validate", help="validate a corpus: schema, hashes, provenance, labels, leakage, coverage")
    p.add_argument("corpus")
    p.add_argument("--no-replay", action="store_true", help="skip re-running the expert on every recorded observation")
    p.add_argument("--allow-missing-coverage", action="store_true", help="report uncovered cells as warnings")
    p.set_defaults(fn=cmd_data_validate)

    p = sub.add_parser("data-lean", help="check a manifest or corpus with the Lean checker (+ per-sample kernel witnesses)")
    p.add_argument("target", help="manifest JSON or corpus folder")
    p.add_argument("--out", help="interchange folder for a manifest (default: <manifest>.lean/)")
    p.add_argument("--no-witness", action="store_true",
                   help="skip the per-sample kernel witnesses (NOT a complete verification)")
    p.add_argument("--jobs", type=int, help="parallel Lean processes for the kernel witnesses")
    p.set_defaults(fn=cmd_data_lean)

    p = sub.add_parser("data-report", help="dataset report: failures by factor/bin/split with intervals, ESS, coverage")
    p.add_argument("corpus")
    p.set_defaults(fn=cmd_data_report)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
