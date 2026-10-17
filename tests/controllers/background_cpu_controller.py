"""Answers step() instantly but burns CPU in a background thread between steps."""
import threading

from autocross.api import Command, Controller


def _spin():
    x = 0
    while True:
        x += 1


class BackgroundCpu(Controller):
    def on_start(self, track, vehicle):
        super().on_start(track, vehicle)
        threading.Thread(target=_spin, daemon=True).start()

    def step(self, obs):
        return Command(throttle=0.2)
