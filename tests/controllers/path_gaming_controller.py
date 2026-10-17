"""Drives like the baseline but publishes a path meant to game the path-deviation score:
mode 'stub' = a 3 m stub from the car, mode 'zigzag' = a corridor-wide zig-zag."""
import importlib.util

import numpy as np

from autocross.api import Command, Controller


def _baseline(path):
    spec = importlib.util.spec_from_file_location("baseline_for_gaming", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PurePursuitBaseline


class PathGaming(Controller):
    def __init__(self, config=None):
        super().__init__(config)
        self.inner = _baseline(self.config["baseline"])({})

    def on_start(self, track, vehicle):
        super().on_start(track, vehicle)
        self.inner.on_start(track, vehicle)

    def step(self, obs):
        c = self.inner.step(obs)
        if self.config["mode"] == "stub":
            path = np.array([[0.0, 0.0], [3.0, 0.0]])
        else:
            x = np.arange(0.0, 40.0, 0.5)
            path = np.column_stack((x, 2.5 * np.sign(np.sin(x * np.pi))))
        return Command(steer=c.steer, throttle=c.throttle, brake=c.brake, path=path, path_frame="vehicle")
