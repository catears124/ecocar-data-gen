"""Dynamic single-track (bicycle) model with saturating tyres and a friction circle.

Development backend only: fast, deterministic and dependency-free, so controllers can be
iterated and unit-tested without a CARLA server. Parameters are fitted to CARLA telemetry
(see ``autocross fit-bicycle``) but it is NOT a substitute for the scored CARLA runs.

The dynamics live in :func:`substep`, which works on floats and on numpy arrays alike, so
the fitter can simulate many telemetry windows at once with exactly the same physics."""
from __future__ import annotations

import math

import numpy as np

from .base import Backend, PhysicalParams, VehicleState

G = 9.81

# vehicle.bicycle keys and their defaults; FIT_KEYS are the ones fitted to CARLA telemetry
DEFAULTS = {
    "mass": 1845.0, "wheelbase": 3.0, "cg_to_front": 1.45, "cg_height": 0.5, "track_width": 1.67,
    "length": 4.79, "width": 2.16, "yaw_inertia_factor": 1.0, "mu": 1.0, "cornering_stiffness_per_load": 14.0,
    "max_drive_force": 11000.0, "max_power_w": 200000.0, "max_brake_force": 15000.0, "drag_area": 0.55,
    "rolling_resistance": 0.012, "steer_time_constant": 0.06, "front_drive_fraction": 0.4,
    "traction_limit": 0.85, "front_brake_fraction": 0.6,
}
FIT_KEYS = ("mu", "cornering_stiffness_per_load", "max_drive_force", "max_power_w", "max_brake_force", "drag_area",
            "rolling_resistance", "steer_time_constant", "traction_limit", "front_drive_fraction",
            "front_brake_fraction", "yaw_inertia_factor", "cg_height")
# Physical bounds, enforced by derive() during fitting and at load. Without ceilings on the
# dynamic parameters the fitter can reach a degenerate optimum (e.g. mu~4, drag~20, brake~0)
# that matches short telemetry windows but is not a real vehicle; these keep any fit physical.
BOUNDS = {"traction_limit": (0.3, 1.0), "front_drive_fraction": (0.0, 1.0), "front_brake_fraction": (0.0, 1.0),
          "steer_time_constant": (0.005, 0.5), "mu": (0.6, 1.3), "cornering_stiffness_per_load": (5.0, 25.0),
          "max_drive_force": (6000.0, 16000.0), "max_power_w": (1.0e5, 3.5e5), "max_brake_force": (1.0e4, 2.5e4),
          "drag_area": (0.3, 1.2), "rolling_resistance": (0.005, 0.03), "cg_height": (0.3, 0.7),
          "yaw_inertia_factor": (0.5, 1.5)}


def model_params(vehicle_cfg: dict, perturb: dict | None = None) -> dict:
    """Bicycle parameters from the ``vehicle`` config section, with derived quantities."""
    perturb = perturb or {}
    p = {k: float(vehicle_cfg.get("bicycle", {}).get(k, v)) for k, v in DEFAULTS.items()}
    p["mass"] *= float(perturb.get("mass_scale", 1.0))
    p["mu"] *= float(perturb.get("friction_scale", 1.0))
    p["max_steer"] = math.radians(float(vehicle_cfg.get("max_steer_deg", 35.0)))
    return derive(p)


def derive(p: dict) -> dict:
    for k, (lo, hi) in BOUNDS.items():
        p[k] = min(max(p[k], lo), hi)
    p["cg_to_rear"] = p["wheelbase"] - p["cg_to_front"]
    p["iz"] = p["yaw_inertia_factor"] * p["mass"] * p["cg_to_front"] * p["cg_to_rear"]
    return p


def substep(s, cmd, ax_prev, p, h):
    """Advance (x, y, yaw, vx, vy, r, delta) by ``h`` seconds under (steer, throttle, brake).
    ``ax_prev`` is the longitudinal acceleration of the previous tick (load transfer)."""
    x, y, yaw, vx, vy, r, delta = s
    steer, throttle, brake = cmd
    m, L, lf, lr = p["mass"], p["wheelbase"], p["cg_to_front"], p["cg_to_rear"]
    target = np.clip(steer, -1, 1) * p["max_steer"]
    delta = delta + (target - delta) * min(h / p["steer_time_constant"], 1.0)
    d = delta
    # longitudinal forces
    f_trac = throttle * np.minimum(p["max_drive_force"], p["max_power_w"] / np.maximum(vx, 1.0))
    f_brk = brake * p["max_brake_force"]
    f_res = 0.5 * 1.2 * p["drag_area"] * vx * np.abs(vx) + p["rolling_resistance"] * m * G * (vx > 0.05)
    # load transfer from previous-step accel
    fzf = m * (G * lr - ax_prev * p["cg_height"]) / L
    fzr = m * G - fzf
    # axle longitudinal demand (AWD split, front/rear brake split); traction control / ABS keep
    # the longitudinal force below traction_limit * mu * Fz so some lateral grip always remains
    fb = p["front_brake_fraction"]
    lim = p["traction_limit"] * p["mu"]
    fxf = np.clip(p["front_drive_fraction"] * f_trac - fb * f_brk, -lim * fzf, lim * fzf)
    fxr = np.clip((1.0 - p["front_drive_fraction"]) * f_trac - (1.0 - fb) * f_brk, -lim * fzr, lim * fzr)
    # dynamic model (only applied above 1 m/s)
    vxs = np.maximum(vx, 1.0)
    af = np.arctan2(vy + lf * r, vxs) - d
    ar = np.arctan2(vy - lr * r, vxs)
    fyf = _lat(af, fzf, fxf, p)
    fyr = _lat(ar, fzr, fxr, p)
    cd, sd = np.cos(d), np.sin(d)
    vx_dot = (fxr + fxf * cd - fyf * sd - f_res) / m + r * vy
    vy_dot = (fyr + fyf * cd + fxf * sd) / m - r * vx
    rdot = (lf * (fyf * cd + fxf * sd) - lr * fyr) / p["iz"]
    dyn = vx > 1.0
    vx_d = np.where(dyn, vx + vx_dot * h, vx)
    vy_d = np.where(dyn, vy + vy_dot * h, vy)
    r_d = np.where(dyn, r + rdot * h, r)
    # low speed: kinematic model (blended in over 1..3 m/s for continuity)
    w_kin = np.clip((3.0 - vx_d) / 2.0, 0.0, 1.0)
    vk = np.where(vx_d <= 1.0, vx_d + (fxr + fxf - f_res) / m * h, vx_d)
    vk = np.where((vk < 0) & (throttle <= 0.0), 0.0, vk)
    rk = vk * np.tan(d) / L
    vx = w_kin * vk + (1 - w_kin) * vx_d
    vy = w_kin * rk * lr + (1 - w_kin) * vy_d
    r = w_kin * rk + (1 - w_kin) * r_d
    vx = np.maximum(vx, 0.0)
    c, s_ = np.cos(yaw), np.sin(yaw)
    x = x + (vx * c - vy * s_) * h
    y = y + (vx * s_ + vy * c) * h
    yaw = yaw + r * h
    return x, y, yaw, vx, vy, r, delta


def _lat(alpha, fz, fx, p):
    fmax = p["mu"] * fz
    avail = np.sqrt(np.maximum(fmax ** 2 - fx ** 2, 0.0))
    c = p["cornering_stiffness_per_load"] * fz
    return -avail * np.tanh(c * alpha / np.maximum(avail, 1.0))


class BicycleBackend(Backend):
    name = "bicycle"

    def __init__(self, sim_cfg: dict):
        self.dt = float(sim_cfg["physics_dt"])
        self.sub = max(int(round(self.dt / 0.0025)), 1)

    def setup(self, track, vehicle_cfg: dict, perturb: dict, rng) -> PhysicalParams:
        self.p = p = model_params(vehicle_cfg, perturb)
        self.cmd = (0.0, 0.0, 0.0)
        return PhysicalParams(p["mass"], p["wheelbase"], p["track_width"], p["length"], p["width"],
                              p["cg_to_front"], p["cg_to_rear"], p["max_steer"])

    def reset(self, x, y, yaw) -> VehicleState:
        self.s = (float(x), float(y), float(yaw), 0.0, 0.0, 0.0, 0.0)  # x y yaw vx vy r delta
        self.t = 0.0
        self.acc = (0.0, 0.0)
        return self._state()

    def apply(self, steer, throttle, brake):
        self.cmd = (float(steer), float(throttle), float(brake))

    def tick(self) -> VehicleState:
        h = self.dt / self.sub
        v0 = self.s[3:5]
        for _ in range(self.sub):
            self.s = substep(self.s, self.cmd, self.acc[0], self.p, h)
        self.t += self.dt
        _, _, _, vx, vy, r, _ = self.s
        dvx, dvy = (vx - v0[0]) / self.dt, (vy - v0[1]) / self.dt
        self.acc = (float(dvx - r * vy), float(dvy + r * vx))
        return self._state()

    def _state(self) -> VehicleState:
        x, y, yaw, vx, vy, r, delta = (float(v) for v in self.s)
        return VehicleState(self.t, x, y, math.atan2(math.sin(yaw), math.cos(yaw)), vx, vy, r,
                            self.acc[0], self.acc[1], steer_angle=delta)
