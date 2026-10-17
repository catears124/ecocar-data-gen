"""Asserts the event-mode information contract, then drives like the baseline."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "controllers"))
from baseline_pure_pursuit import PurePursuitBaseline  # noqa: E402


class ProbeController(PurePursuitBaseline):
    def on_start(self, track, vehicle):
        assert not track.full_knowledge and track.name == "course"
        assert len(track.centerline) == 0 and len(track.cones) == 0 and len(track.gates) == 0
        super().on_start(track, vehicle)

    def step(self, obs):
        r = obs.road
        assert r is not None and r.valid
        assert r.preview_s[-1] <= r.range_m + 1e-6 and len(r.preview_centerline) == len(r.preview_s)
        # centerline preview starts beside the vehicle, offset by -lateral_offset
        assert abs(r.preview_centerline[0, 1] + r.lateral_offset * np.cos(r.heading_error)) < 0.3
        assert r.dist_to_left_boundary + r.dist_to_right_boundary == \
            __import__("pytest").approx(r.width_left + r.width_right)
        for c in r.corners:
            assert c.direction in (-1, 1) and c.min_radius > 0 and c.entry_distance <= r.range_m
        return super().step(obs)
