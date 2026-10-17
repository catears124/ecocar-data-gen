"""Tries one sandbox escape (config 'action') in on_start. The sandbox must refuse it with
PermissionError; if the escape works, or fails for another reason, the controller raises."""
import os
import sys

from autocross.api import Command, Controller


def _attempt(action, cfg):
    if action == "subprocess":
        import subprocess
        subprocess.run([sys.executable, "-c", "pass"], check=True)
    elif action == "os_system":
        os.system("echo hi")
    elif action == "socket":
        import socket
        socket.create_connection(("127.0.0.1", 2000), timeout=0.5)
    elif action == "write":
        with open(cfg["path"], "w") as f:
            f.write("x")
    elif action == "remove":
        os.remove(cfg["path"])
    elif action == "ctypes":
        import ctypes
        # CDLL(None) loads the main program on POSIX, but on Windows it raises TypeError
        # before reaching the audit hook, so use a real library name there.
        ctypes.CDLL("kernel32" if sys.platform == "win32" else None)
    elif action == "ctypes_pythonapi":
        import ctypes
        ctypes.pythonapi.Py_IsInitialized()
    elif action == "gc":
        import gc
        gc.get_objects()
    elif action == "symlink":
        os.symlink(cfg["target"], cfg["path"])
    elif action == "carla":
        import importlib
        sys.modules.pop("carla", None)       # undo the sys.modules block, then import another way
        importlib.import_module("carla")
    elif action == "read_course":
        open(cfg["path"]).read()
    else:
        raise ValueError(action)


def _course_name_visible(needle):
    """Search the locals of every caller's stack frame for the course file name."""
    f = sys._getframe(2)
    seen = []
    while f is not None:
        for v in list(f.f_locals.values()):
            if v is seen:
                continue
            items = tuple(v) if isinstance(v, (list, tuple)) else (v,)
            for it in items:
                if isinstance(it, (str, os.PathLike)) and needle in str(it):
                    seen.append(str(it))
        f = f.f_back
    return seen


class EscapeController(Controller):
    def on_start(self, track, vehicle):
        super().on_start(track, vehicle)
        action = self.config["action"]
        if action == "frames":
            found = _course_name_visible(self.config["needle_reversed"][::-1])  # never stored whole
            if found:
                raise RuntimeError(f"ESCAPED: course name visible in memory: {found[:2]}")
            return
        try:
            _attempt(action, self.config)
        except PermissionError:
            return
        except Exception as e:
            raise RuntimeError(f"NOT BLOCKED: {action} failed with {e!r} instead of PermissionError")
        raise RuntimeError(f"ESCAPED: {action} succeeded")

    def step(self, obs):
        return Command(brake=1.0)
