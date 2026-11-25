"""Baseline 1 (handout p. 7): projected gradient descent started from classical MDS.

PGD with Armijo backtracking along the projection arc, following Liam's pgd.py on develop;
derivatives from core.derivatives.Model, start from baseline_initialization.mds_start (MDS-MAP in core/mds.py).
"""

import sys
from math import ceil
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root, for core/

from problem import project                                     # noqa: E402  (course package)
from core.derivatives import Model                              # noqa: E402
from core.initialization.baseline_initialization import mds_start      # noqa: E402
from baseline_solvers import constants                          # noqa: E402  (tunable values)


def run_pgd(param, x_start, deadline, *, record=None, started=None):
    """Run PGD from x_start until the perf_counter() deadline; return the last accepted point.

    record(x, iteration=, elapsed_s=) is called on a geometric schedule (convergence_trace.Trace).
    """
    started = perf_counter() if started is None else started
    model = Model(param)
    x = project(x_start, param)
    value = model.objective(x)
    if record is not None:
        record(x, iteration=0, elapsed_s=perf_counter() - started)

    step = np.inf
    iteration = last_recorded = 0
    next_snapshot = 1
    while perf_counter() < deadline:
        grad = model.gradient(x)
        scale = float(np.max(np.abs(grad)))
        if scale == 0:
            break
        step = min(step / constants.PGD_SHRINK, constants.PGD_MAX_COORDINATE_MOVE / scale)   # try a longer step first
        accepted = _armijo_step(model, param, x, value, grad, step, deadline)
        if accepted is None:
            break
        x, value, step = accepted
        iteration += 1
        if record is not None and iteration == next_snapshot:
            record(x, iteration=iteration, elapsed_s=perf_counter() - started)
            last_recorded, next_snapshot = iteration, ceil(1.25 * iteration)

    if record is not None and iteration > last_recorded:
        record(x, iteration=iteration, elapsed_s=perf_counter() - started)
    return x


def _armijo_step(model, param, x, value, grad, step, deadline):
    """First projected trial x(step) = P(x - step * grad) with sufficient decrease, or None."""
    while perf_counter() < deadline:
        candidate = project(x - step * grad, param)
        change = float(np.vdot(grad, candidate - x))
        if change >= 0:                  # projection left no descent direction
            return None
        candidate_value = model.objective(candidate)
        if candidate_value < value and candidate_value <= value + constants.PGD_ARMIJO * change:
            return candidate, candidate_value, step
        step *= constants.PGD_SHRINK
    return None


def solver(param, max_time):
    """MDS start (its time counts toward max_time), then PGD for the rest of the budget."""
    started = perf_counter()
    deadline = started + constants.TIME_FRACTION * max_time
    return run_pgd(param, mds_start(param), deadline, started=started)
