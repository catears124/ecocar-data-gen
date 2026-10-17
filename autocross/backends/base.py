from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class VehicleState:
    """Ground-truth state in the ENU world frame; velocities/accels in the body frame."""
    t: float
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    yaw_rate: float
    ax: float
    ay: float
    roll: float = 0.0
    pitch: float = 0.0
    steer_angle: float = float("nan")  # measured road-wheel angle [rad], + left, if available

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))


@dataclass
class PhysicalParams:
    mass: float
    wheelbase: float
    track_width: float
    length: float
    width: float
    cg_to_front: float
    cg_to_rear: float
    max_steer: float


class Backend(ABC):
    name = "base"

    @abstractmethod
    def setup(self, track, vehicle_cfg: dict, perturb: dict, rng: np.random.Generator) -> PhysicalParams:
        """Build the world for ``track`` and return the vehicle's physical parameters."""

    @abstractmethod
    def reset(self, x: float, y: float, yaw: float) -> VehicleState:
        """Place the vehicle at rest and return the initial state."""

    @abstractmethod
    def apply(self, steer: float, throttle: float, brake: float) -> None:
        """Set actuator commands (steer normalised, + left) for the next tick."""

    @abstractmethod
    def tick(self) -> VehicleState:
        """Advance one physics step of ``dt`` seconds."""

    def draw_path(self, path: np.ndarray, life: float) -> None:
        pass

    def close(self) -> None:
        pass
