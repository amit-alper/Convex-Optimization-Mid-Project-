"""Baseline 1 (handout p. 7): projected gradient descent started from classical MDS.

The initializer is MDS-MAP (initialization/mds.py) and the optimizer is the shared optimizers.pgd.run_pgd.
MDS checks the remaining time before its expensive stages and falls back to the course start.
The elapsed initialization time counts toward the internal deadline for PGD.
"""

from time import perf_counter

from initialization.baseline_initialization import mds_start
from optimizers.pgd import run_pgd


TIME_FRACTION = 0.9      # course README: return by 0.9 max_time (run_pgd alone would use 0.95)
MIN_PGD_TIME = 0.025     # leave room for at least one objective and gradient on the public graphs


def initialize(param, max_time):
    """Run MDS-MAP if time permits, otherwise return the projected course starting point."""
    deadline = perf_counter() + TIME_FRACTION * max_time
    return mds_start(param, deadline=deadline)


def solve(param, max_time, *, record=None):
    started = perf_counter()
    x0 = initialize(param, max_time)
    elapsed = perf_counter() - started
    remaining = TIME_FRACTION * max_time - elapsed
    if remaining < MIN_PGD_TIME:
        if record is not None:
            record(x0, iteration=0, elapsed_s=elapsed)
        return x0

    record_pgd = None
    if record is not None:
        def record_pgd(x, *, iteration, elapsed_s):
            record(x, iteration=iteration, elapsed_s=elapsed + elapsed_s)

    return run_pgd(param, remaining, x0=x0, record=record_pgd)


def solver(param, max_time):
    return solve(param, max_time)
