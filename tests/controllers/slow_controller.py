import time

from autocross.api import Command, Controller


class SlowController(Controller):
    """Blows the 50 ms deadline on every 5th step."""

    def step(self, obs):
        if obs.step % 5 == 4:
            time.sleep(0.12)
        return Command(steer=0.0, throttle=0.3, brake=0.0)
