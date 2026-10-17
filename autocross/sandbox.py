"""Runs a team controller in a separate process under an embedded-style compute budget.

Budget enforcement (``compute_budget.mode``):
  * enforce - the step must answer within ``step_deadline_ms``; otherwise the previous
              command is held (like a missed ECU task deadline) and an overrun is logged.
              Late answers are discarded.
  * measure - always wait for the answer (up to ``hard_timeout_s``) and log overruns.
Independently of mode, the process is pinned to ``cpu_cores`` logical cores, BLAS/OpenMP
thread pools are limited to that count, CUDA devices are hidden unless ``allow_gpu``, and
CPU time and peak memory are sampled.

The controller may not open course files or earlier run outputs (``sandbox.protected_paths``
plus the course being driven), write files, start processes, use the network or native code: a
Python audit hook in the controller process refuses them. Course files are passed to that
process only as path digests, so it cannot learn which course it drives. Replies are decoded
without creating objects, and step time and CPU are measured by the harness, so a controller
cannot run code in the harness or under-report its compute.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import io
import multiprocessing as mp
import os
import pickle
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import psutil


@dataclass
class StepStats:
    wall_ms: float = float("nan")   # time inside step(), measured in the controller process
    cpu_ms: float = float("nan")    # process CPU time consumed by step()
    latency_ms: float = float("nan")  # round trip seen by the harness (incl. IPC)
    overrun: bool = False
    error: str | None = None


def load_controller_class(path: str, class_name: str | None = None):
    path = str(Path(path).resolve())
    spec = importlib.util.spec_from_file_location(f"team_controller_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).parent))
    spec.loader.exec_module(mod)
    from .api import Controller
    if class_name:
        return getattr(mod, class_name)
    found = [c for _, c in inspect.getmembers(mod, inspect.isclass)
             if issubclass(c, Controller) and c is not Controller and c.__module__ == mod.__name__]
    if len(found) != 1:
        raise RuntimeError(f"{path}: expected exactly one Controller subclass, found {[c.__name__ for c in found]}; "
                           "pass --controller-class")
    return found[0]


# Audit events a controller has no business raising: starting processes, networking (the CARLA
# server is on localhost), native code via ctypes, finding the guard through the garbage
# collector, and changing files.
DENIED_EVENTS = frozenset({
    "subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.fork", "os.forkpty",
    "os.startfile", "os.kill", "os.killpg", "signal.pthread_kill", "_winapi.CreateProcess",
    "_winapi.OpenProcess", "_winapi.TerminateProcess", "_posixsubprocess.fork_exec", "pty.spawn",
    "socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg", "socket.getaddrinfo",
    "http.client.connect", "urllib.Request", "ftplib.connect", "smtplib.connect", "webbrowser.open",
    "ctypes.dlopen", "ctypes.dlsym", "ctypes.dlsym/handle", "ctypes.call_function", "ctypes.cdata",
    "ctypes.cdata/buffer", "ctypes.string_at", "ctypes.wstring_at", "ctypes.get_errno", "ctypes.set_errno",
    "gc.get_objects", "gc.get_referrers", "gc.get_referents", "sys.remote_exec",
    "os.symlink", "os.link", "os.remove", "os.rename", "os.rmdir", "os.truncate", "os.chmod", "os.chown",
    "os.chflags", "os.utime", "os.mkdir", "shutil.rmtree", "shutil.move",
    "shutil.copyfile", "shutil.copymode", "shutil.copystat", "shutil.copytree", "shutil.chown",
    "winreg.CreateKey", "winreg.DeleteKey", "winreg.SetValue", "winreg.ConnectRegistry",
})
# Packages a controller may not load: the CARLA client talks to the simulator through its own
# C++ sockets, which no audit hook sees, so it could read the cone layout or move the car.
FORBIDDEN_PACKAGES = ("carla",)
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


def path_digest(p) -> str:
    """Digest of a resolved file path. Protected course files are passed to the controller
    process as digests, so it cannot learn which course it is driving from its own memory."""
    return hashlib.sha256(os.path.normcase(os.path.realpath(p)).encode("utf-8", "surrogatepass")).hexdigest()


def _install_guard(dirs, file_digests) -> None:
    """Install the controller-process audit hook. Audit hooks cannot be removed. It refuses
    reading protected directories and course files, writing any file, other processes' /proc
    entries, and the events in DENIED_EVENTS. Paths are resolved, so symlinks do not help.

    This is defence in depth, not a perfect sandbox: judges also review submissions
    (tools/evaluate.py flags suspicious code)."""
    roots = tuple(os.path.normcase(os.path.realpath(p)) for p in dirs)
    digests = frozenset(file_digests)
    devnull = os.path.normcase(os.path.realpath(os.devnull))
    pid = str(os.getpid())

    def deny(what):
        raise PermissionError(f"controllers may not {what}")

    def hook(event, args):
        if event in DENIED_EVENTS:
            deny(f"use '{event}' (no processes, networking, native code or file changes)")
        if event not in ("open", "os.listdir", "os.scandir") or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        try:
            p = os.path.normcase(os.path.realpath(os.fsdecode(args[0])))
        except (TypeError, ValueError):
            return
        if event == "open" and p != devnull:
            mode, flags = (args[1] if len(args) > 1 else None), (args[2] if len(args) > 2 else 0)
            if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & _WRITE_FLAGS):
                deny(f"write files: {os.fsdecode(args[0])}")
        for r in roots:
            if p == r or p.startswith(r.rstrip(os.sep) + os.sep):
                deny(f"access course or run files: {os.fsdecode(args[0])}")
        if p.startswith(os.sep + "proc" + os.sep):
            part = p.split(os.sep)[2] if len(p.split(os.sep)) > 2 else ""
            if part.isdigit() and part != pid:
                deny("inspect other processes")
        if os.path.isfile(p) and hashlib.sha256(p.encode("utf-8", "surrogatepass")).hexdigest() in digests:
            deny(f"access course or run files: {os.fsdecode(args[0])}")

    sys.addaudithook(hook)


def _pack(out):
    """Controller output -> plain data (numbers, strings, bytes). The harness never unpickles
    objects from the controller process, so a submission cannot run code in the harness."""
    from .api import Command
    if out is None:
        return None
    if not isinstance(out, Command):
        try:
            return {"invalid": repr(out)[:300]}
        except Exception:
            return {"invalid": f"<unprintable {type(out).__name__}>"}
    try:
        vals = {"steer": float(out.steer), "throttle": float(out.throttle), "brake": float(out.brake)}
    except (TypeError, ValueError):
        return {"invalid": f"Command with non-numeric steer/throttle/brake: {out!r}"[:300]}
    path = None
    if out.path is not None:
        try:
            import numpy as np
            a = np.asarray(out.path, dtype=float).reshape(-1, 2)[:2000]
            path = (int(a.shape[0]), a.astype("<f8").tobytes())
        except (TypeError, ValueError):
            path = None
    ts = float(out.target_speed) if isinstance(out.target_speed, (int, float)) else None
    dbg = {}
    if isinstance(out.debug, dict):
        for k, v in list(out.debug.items())[:64]:
            if isinstance(v, (int, float)):
                dbg[str(k)[:64]] = float(v)
    return {**vals, "path": path, "path_frame": str(out.path_frame)[:16], "target_speed": ts, "debug": dbg}


class _PlainUnpickler(pickle.Unpickler):
    """Safe for untrusted input: every pickle opcode that builds or calls an object (GLOBAL,
    STACK_GLOBAL, INST, OBJ, REDUCE via a global, EXT*) resolves it through find_class, which
    always refuses here. What remains can only produce tuples, lists, dicts, str, bytes,
    numbers, bools and None."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"controller sent a {module}.{name} object; only plain data is accepted")


def _unpack_message(raw: bytes):
    """Decode and validate a controller-process message: (kind, seq, payload, wall, cpu)."""
    msg = _PlainUnpickler(io.BytesIO(raw)).load()
    if not (isinstance(msg, tuple) and len(msg) == 5 and msg[0] in ("ready", "ok", "error") and isinstance(msg[1], int)):
        raise pickle.UnpicklingError("malformed message from the controller process")
    kind, seq, payload = msg[:3]
    if kind == "error":
        payload = payload if isinstance(payload, str) else "controller error"
    elif kind == "ok":
        payload = _command_from(payload)
    else:
        payload = None
    return kind, seq, payload


def _command_from(d):
    from .api import Command
    if d is None:
        return None
    if not isinstance(d, dict):
        return "invalid message"
    if "invalid" in d:
        return str(d["invalid"])
    import numpy as np
    try:
        path = None
        if d.get("path") is not None:
            n, buf = d["path"]
            path = np.frombuffer(buf, dtype="<f8", count=2 * int(n)).reshape(-1, 2).copy()
        ts = d.get("target_speed")
        dbg = {str(k): float(v) for k, v in (d.get("debug") or {}).items() if isinstance(v, (int, float))}
        return Command(steer=float(d["steer"]), throttle=float(d["throttle"]), brake=float(d["brake"]), path=path,
                       path_frame=str(d.get("path_frame", "vehicle")),
                       target_speed=float(ts) if isinstance(ts, (int, float)) else None, debug=dbg)
    except (KeyError, TypeError, ValueError):
        return "invalid message"


def _worker(conn, path, class_name, ctrl_cfg, dirs=(), file_digests=()):
    try:
        import numpy  # noqa: F401  (loaded before the guard: numpy needs ctypes at import time)
        from . import api  # noqa: F401
        for pkg in FORBIDDEN_PACKAGES:
            sys.modules[pkg] = None  # plain "import carla" now fails at once
        _install_guard(dirs, file_digests)
        del dirs, file_digests
        cls = load_controller_class(path, class_name)
        ctrl = cls(ctrl_cfg)
        conn.send(("ready", 0, None, 0.0, 0.0))
    except Exception:
        conn.send(("error", 0, traceback.format_exc(), 0.0, 0.0))
        return
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        kind, seq, payload = msg
        try:
            if kind == "start":
                ctrl.on_start(*payload)
                out = None
            elif kind == "step":
                out = _pack(ctrl.step(payload))
            elif kind == "finish":
                ctrl.on_finish()
                conn.send(("ok", seq, None, 0.0, 0.0))
                return
            else:
                continue
            conn.send(("ok", seq, out, 0.0, 0.0))
        except Exception:
            conn.send(("error", seq, traceback.format_exc(), 0.0, 0.0))


class ControllerProcess:
    def __init__(self, path: str, budget: dict, class_name: str | None = None, ctrl_cfg: dict | None = None,
                 protected_paths: list[str] | None = None):
        self.budget = budget
        self.mode = str(budget.get("mode", "enforce"))
        self.deadline = float(budget.get("step_deadline_ms", 50.0)) / 1e3
        self.hard_timeout = float(budget.get("hard_timeout_s", 10.0))
        self.seq = 0
        self.peak_rss = 0
        self.dead = False
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        cores = int(budget.get("cpu_cores", 1))
        env_overrides = {k: str(cores) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
        if not budget.get("allow_gpu", False):
            env_overrides["CUDA_VISIBLE_DEVICES"] = "-1"
        saved = {k: os.environ.get(k) for k in env_overrides}
        os.environ.update(env_overrides)
        try:
            prot = [Path(p) for p in protected_paths or []]
            # directories inside another protected directory are covered already; leaving them out
            # also keeps run-folder names (which can name the course) out of the controller process
            for pkg in FORBIDDEN_PACKAGES:  # their files cannot even be read, so no import route works
                try:
                    spec = importlib.util.find_spec(pkg)
                except (ImportError, ValueError):
                    spec = None
                prot += [Path(d) for d in (spec.submodule_search_locations or [])] if spec else []
            res = [Path(os.path.realpath(p)) for p in prot if not p.is_file()]
            dirs = [str(d) for d in res if not any(o != d and o in d.parents for o in res)]
            digests = [path_digest(p) for p in prot if p.is_file()]
            self.proc = ctx.Process(target=_worker, args=(child, path, class_name, ctrl_cfg or {}, dirs, digests),
                                    daemon=True)
            self.proc.start()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.ps = psutil.Process(self.proc.pid)
        self.drive_cpu_s = float("nan")
        self._cpu0 = None
        try:
            n = psutil.cpu_count(logical=True) or 1
            self.cores = list(range(max(n - cores, 0), n))  # leave core 0 for the simulator/harness
            self.ps.cpu_affinity(self.cores)
        except (AttributeError, psutil.Error):
            self.cores = []
        kind, _, payload = self._recv(float(budget.get("init_timeout_s", 60.0)))
        if kind != "ready":
            self.close()
            raise RuntimeError(f"controller failed to load:\n{payload}")

    MAX_MESSAGE_BYTES = 4 * 2 ** 20

    def _recv(self, timeout):
        """(kind, seq, payload). Timing is measured here, never taken from the controller."""
        if not self.conn.poll(timeout):
            return ("timeout", -1, None)
        try:
            return _unpack_message(self.conn.recv_bytes(self.MAX_MESSAGE_BYTES))
        except (EOFError, BrokenPipeError):
            self.dead = True
            return ("error", None, "controller process exited")
        except Exception as e:  # oversized, malformed or object-carrying message
            return ("error", None, f"rejected message from the controller process: {e}")

    def _call(self, kind, payload, timeout):
        self.seq += 1
        seq = self.seq
        self.conn.send((kind, seq, payload))
        end = time.perf_counter() + timeout
        while True:
            msg = self._recv(max(end - time.perf_counter(), 0.0))
            if msg[0] == "timeout" or msg[1] == seq or msg[1] is None:
                return msg
            # stale reply from an earlier overrun: drop it

    def _cpu_s(self):
        try:
            c = self.ps.cpu_times()
            return c.user + c.system
        except psutil.Error:
            return float("nan")

    def start(self, track_info, vehicle_params) -> None:
        t0 = time.perf_counter()
        kind, _, payload = self._call("start", (track_info, vehicle_params), float(self.budget.get("init_timeout_s", 60.0)))
        self.init_ms = (time.perf_counter() - t0) * 1e3
        self._sample_memory()
        self._cpu0 = self._cpu_s()
        if kind == "timeout":
            raise TimeoutError(f"on_start exceeded init_timeout_s={self.budget.get('init_timeout_s', 60.0)}")
        if kind == "error":
            raise RuntimeError(f"controller on_start raised:\n{payload}")

    def step(self, obs):
        """Return (command or None, StepStats). None means: hold the previous command.
        Step time is the round trip measured by the harness; CPU time is read from the OS, so it
        includes any threads the controller runs."""
        if self.dead:
            return None, StepStats(error="controller process not running", overrun=True)
        c0 = self._cpu_s()
        t0 = time.perf_counter()
        timeout = self.deadline if self.mode == "enforce" else self.hard_timeout
        kind, _, payload = self._call("step", obs, timeout)
        lat = (time.perf_counter() - t0) * 1e3
        cpu = (self._cpu_s() - c0) * 1e3
        if obs.step % 10 == 0:
            self._sample_memory()
        if kind == "timeout":
            if self.mode != "enforce" or not self.proc.is_alive():
                self.dead = True
            return None, StepStats(lat, cpu, lat, overrun=True)
        if kind == "error":
            return None, StepStats(lat, cpu, lat, True, error=payload)
        return payload, StepStats(lat, cpu, lat, lat > self.deadline * 1e3)

    def _sample_memory(self):
        try:
            mi = self.ps.memory_info()
            self.peak_rss = max(self.peak_rss, getattr(mi, "peak_wset", 0) or 0, mi.rss)
        except psutil.Error:
            pass

    def close(self):
        if self._cpu0 is not None:
            self.drive_cpu_s = self._cpu_s() - self._cpu0
        try:
            if self.proc.is_alive():
                self._call("finish", None, 2.0)
        except (OSError, BrokenPipeError):
            pass
        self._sample_memory()
        if self.proc.is_alive():
            self.proc.terminate()
        self.proc.join(timeout=2.0)
