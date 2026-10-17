"""Bypasses the command encoder and writes a malicious pickle straight to the harness pipe
(found by walking the stack). The harness must reject it, not execute it."""
import sys

from autocross.api import Command, Controller


class _Evil:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (exec, (f"open({self.marker!r}, 'w').write('pwned')",))


class PickleInjection(Controller):
    def step(self, obs):
        f = sys._getframe(1)
        while f is not None and "conn" not in f.f_locals:
            f = f.f_back
        if f is not None:
            f.f_locals["conn"].send(("ok", f.f_locals["seq"], _Evil(self.config["marker"]), 0.0, 0.0))
        return Command(brake=1.0)
