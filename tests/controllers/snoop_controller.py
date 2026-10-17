"""Tries to read the course file from disk in on_start; the sandbox must refuse."""
from pathlib import Path

from autocross.api import Command, Controller


class SnoopController(Controller):
    def on_start(self, track, vehicle):
        super().on_start(track, vehicle)
        root = Path(__file__).resolve().parents[2]
        (root / "tracks" / "practice_basic.yaml").read_text()

    def step(self, obs):
        return Command(brake=1.0)
