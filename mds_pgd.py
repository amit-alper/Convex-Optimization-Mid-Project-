"""Baseline 1 (handout p. 7): projected gradient descent started from classical MDS.

The initializer is MDS-MAP (core/mds.py) and the optimizer is the shared pgd.run_pgd.
MDS cannot be interrupted, so its actual time is subtracted from the budget given to PGD,
and the whole solve returns by TIME_FRACTION * max_time.
"""

from time import perf_counter

from pgd import run_pgd
from core.initialization.baseline_initialization import mds_start


TIME_FRACTION = 0.9      # course README: return by 0.9 max_time (pgd.py alone would use 0.95)


def initialize(param, max_time):
    """MDS-MAP fit to the anchors and projected onto the box (about 0.6-2.5 s on B-D)."""
    return mds_start(param)


def solve(param, max_time, *, record=None):
    started = perf_counter()
    x0 = initialize(param, max_time)
    elapsed = perf_counter() - started
    record_pgd = None
    if record is not None:
        def record_pgd(x, *, iteration, elapsed_s):
            record(x, iteration=iteration, elapsed_s=elapsed + elapsed_s)

    return run_pgd(param, max(0.0, TIME_FRACTION * max_time - elapsed), x0=x0, record=record_pgd)


def solver(param, max_time):
    return solve(param, max_time)
