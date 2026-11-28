"""Demonstrate composing a center initializer and PGD under one budget."""

from time import perf_counter

from initialization.baseline_initialization import collapsed_start
from optimizers.pgd import run_pgd


def initialize(param, max_time):
    """Use the collapsed reference as a simple demonstration start."""
    return collapsed_start(param)


def solve(param, max_time, *, record=None):
    started = perf_counter()
    x0 = initialize(param, 0.2 * max_time)
    elapsed = perf_counter() - started
    record_pgd = None
    if record is not None:
        def record_pgd(x, *, iteration, elapsed_s):
            record(x, iteration=iteration, elapsed_s=elapsed + elapsed_s)

    return run_pgd(param, max(0.0, max_time - elapsed), x0=x0, record=record_pgd)


def solver(param, max_time):
    return solve(param, max_time)
