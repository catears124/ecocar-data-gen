from autocross.api import Command, Controller


class NanController(Controller):
    def step(self, obs):
        return Command(steer=float("nan"), throttle=0.5)
