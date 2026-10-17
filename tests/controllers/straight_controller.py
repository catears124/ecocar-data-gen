from autocross.api import Command, Controller


class StraightController(Controller):
    """Full throttle, wheel straight: leaves the course at the first corner."""

    def step(self, obs):
        return Command(steer=0.0, throttle=1.0, brake=0.0)
