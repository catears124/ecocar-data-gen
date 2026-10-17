from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SIM = ROOT / "config" / "sim.yaml"
DEFAULT_SCORING = ROOT / "config" / "scoring.yaml"

# Perturbation keys understood by the runner (see config/robustness.yaml)
PERTURBATION_KEYS = {
    "friction_scale": "tyre friction multiplier",
    "mass_scale": "vehicle mass multiplier",
    "cone_noise_std": "perception position noise [m] (replaces perception.noise_std_m)",
    "cone_dropout": "probability a visible cone is not reported",
    "misclassification": "probability a detected cone gets a random colour",
    "actuation_delay_steps": "extra control periods of actuation latency",
    "steer_bias": "constant steering offset added by the actuator (normalised units)",
    "road_offset_noise": "road-model centerline offset noise std [m]",
    "road_heading_noise_deg": "road-model heading error noise std [deg]",
    "road_curvature_noise": "road-model curvature noise std [1/m]",
    "road_range_m": "road-model preview range [m] (shorter sight distance)",
    "start_lateral_offset": "spawn offset to the left [m]",
    "start_yaw_offset_deg": "spawn heading offset [deg]",
}


def load_yaml(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def parse_sets(items: list[str] | None) -> dict:
    """['a.b=1', 'c=x'] -> {'a': {'b': 1}, 'c': 'x'} (values parsed as YAML scalars)."""
    out: dict = {}
    for item in items or []:
        key, _, val = item.partition("=")
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return out


def load_sim_config(path=None, sets: list[str] | None = None) -> dict:
    cfg = load_yaml(path or DEFAULT_SIM)
    return deep_update(cfg, parse_sets(sets))
