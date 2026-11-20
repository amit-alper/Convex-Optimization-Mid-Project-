"""Runs a solver on one instance, exactly as in grading.

The solver runs in a separate process with one CPU thread. The data of the instance is loaded first
(this time does not count); then the process calls solver(param, max_time) and measures the time until
it returns. A solver that fails, returns after max_time, or returns something that cannot be projected
onto the constraints gets no value (0 points on that instance). The returned x is projected onto the
constraints and its objective is computed. Peak memory (resident set size) is sampled while it runs.
"""
import importlib
import importlib.util
import multiprocessing as mp
import os
import sys
import time

SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _maxrss_mb():
    """Peak memory of this process in MB; 0 if it cannot be measured. Works on Linux, macOS and Windows."""
    try:
        import resource                              # Linux and macOS (the module does not exist on Windows)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 2**20 if sys.platform == "darwin" else rss / 2**10
    except ImportError:
        pass
    try:                                             # Windows: peak working set of the process
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize / 2**20
    except Exception:
        pass
    return 0.0


def _child(root, solver_file, problem_module, path_conn, max_time, seed, conn):
    import threading
    import traceback

    import numpy as np

    sys.path.insert(0, root)
    from common.instances import to_numpy

    lock = threading.Lock()

    def send(msg):
        with lock:
            conn.send(msg)

    try:
        # The solver file is imported before this process knows which instance it will get: the path arrives
        # only afterwards, so work done at import time (which is not timed) cannot use the instance.
        problem = importlib.import_module(problem_module)
        spec = importlib.util.spec_from_file_location("student_solver", solver_file)
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, os.path.dirname(solver_file))
        spec.loader.exec_module(module)
        send(("imported",))
        param = problem.load(path_conn.recv())
        path_conn.close()
        try:
            import torch  # noqa: F401  (only if the solver uses it)
            torch.set_num_threads(1)
        except Exception:
            pass
    except Exception:
        send(("error", 0.0, traceback.format_exc()))
        return

    np.random.seed(seed)
    stop = threading.Event()

    def sample_memory():
        while not stop.is_set():
            send(("mem", _maxrss_mb()))
            stop.wait(0.05)

    threading.Thread(target=sample_memory, daemon=True).start()
    send(("ready",))
    try:
        t0 = time.perf_counter()
        x = module.solver(param, max_time)
        elapsed = time.perf_counter() - t0
        stop.set()
        send(("mem", _maxrss_mb()))
        send(("done", elapsed, to_numpy(x)))
    except Exception:
        stop.set()
        send(("error", time.perf_counter() - t0, traceback.format_exc()))


def run(solver_file, problem_module, instance_file, max_time, root, seed=0, grace=2.0, startup_timeout=180.0):
    """Run solver(param, max_time) once. Returns status ("ok", "late", "timeout", "error", "crashed"),
    the returned x (numpy), the time it took, the peak memory, and the error text if any."""
    for k, v in SINGLE_THREAD_ENV.items():
        os.environ[k] = v
    ctx = mp.get_context("spawn")
    recv, send = ctx.Pipe(duplex=False)
    path_recv, path_send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(os.path.abspath(root), os.path.abspath(solver_file), problem_module,
                                            path_recv, max_time, seed, send))
    proc.start()
    send.close()
    path_recv.close()
    out = {"status": "running", "x": None, "time": None, "peak_mem_mb": 0.0, "error": ""}
    deadline = time.monotonic() + startup_timeout
    started = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            out["status"] = "timeout" if started else "startup-timeout"
            break
        if recv.poll(min(remaining, 0.2)):
            try:
                msg = recv.recv()
            except EOFError:
                out["status"] = "crashed"
                break
            if msg[0] == "imported":
                path_send.send(os.path.abspath(instance_file))
            elif msg[0] == "ready":
                started = True
                deadline = time.monotonic() + max_time + grace
            elif msg[0] == "mem":
                out["peak_mem_mb"] = max(out["peak_mem_mb"], msg[1])
            elif msg[0] == "done":
                out["time"], out["x"] = msg[1], msg[2]
                out["status"] = "ok" if msg[1] <= max_time else "late"
                break
            elif msg[0] == "error":
                out["status"], out["time"], out["error"] = "error", msg[1], msg[2]
                break
        elif not proc.is_alive():
            out["status"] = "crashed"
            break
    if proc.is_alive():
        proc.terminate()
        proc.join(2)
        if proc.is_alive():
            proc.kill()
    proc.join()
    recv.close()
    path_send.close()
    return out


def value(problem, param, result):
    """The objective of the projected solution, or None (0 points) if the run did not give a valid x."""
    import numpy as np
    if result["status"] != "ok":
        return None
    try:
        f = float(problem.objective(problem.project(result["x"], param), param))
    except Exception as e:                      # wrong shape, not finite, ...
        result["error"] = f"invalid solution: {e}"
        return None
    return f if np.isfinite(f) else None
