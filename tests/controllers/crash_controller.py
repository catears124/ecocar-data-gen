from autocross.api import Command, Controller


class CrashController(Controller):
    def step(self, obs):
        if obs.step >= 10:
            raise ValueError("boom")
        return Command(throttle=0.3)
