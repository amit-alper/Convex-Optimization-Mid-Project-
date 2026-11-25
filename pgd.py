"""Projected gradient descent on the course localization objective."""

from math import ceil
from time import perf_counter

import numpy as np

from problem import objective, project


ARMIJO = 1e-4
SHRINK = 0.5
MAX_COORDINATE_MOVE = 0.1
TIME_FRACTION = 0.95


def gradient(x, param):
    """Return the position gradient with anchor entries fixed at zero."""
    first, second = param.edges.T
    difference = x[first] - x[second]
    length = np.linalg.norm(difference, axis=1)
    slope = length - param.dist
    if param.loss == "huber":
        slope = np.clip(slope, -param.delta, param.delta)

    # At a coincidence the distance has no preferred direction.
    direction = np.divide(
        difference, length[:, None], out=np.zeros_like(difference), where=length[:, None] > 0,
    )
    contribution = direction * (slope / len(first))[:, None]
    result = np.zeros_like(x)
    np.add.at(result, first, contribution)
    np.add.at(result, second, -contribution)
    result[param.anchor] = 0
    return result


def armijo_step(x, value, grad, param, step, deadline):
    """Return the first accepted projected trial within the allotted time."""
    while perf_counter() < deadline:
        candidate = project(x - step * grad, param)
        directional_change = float(np.vdot(grad, candidate - x))
        if directional_change >= 0 or perf_counter() >= deadline:
            return None

        candidate_value = objective(candidate, param)
        if perf_counter() >= deadline:
            return None
        if candidate_value < value and candidate_value <= value + ARMIJO * directional_change:
            return candidate, candidate_value, step
        step *= SHRINK
    return None


def run_pgd(param, max_time, *, record=None):
    """Run PGD, optionally recording selected accepted iterates for diagnostics."""
    started = perf_counter()
    deadline = started + TIME_FRACTION * max_time
    x = project(param.x0, param)
    if record is not None:
        record(x, iteration=0, elapsed_s=0.0)
    if perf_counter() >= deadline:
        return x

    value = objective(x, param)
    step = np.inf
    iteration = last_recorded = 0
    next_snapshot = 1
    while perf_counter() < deadline:
        grad = gradient(x, param)
        scale = float(np.max(np.abs(grad)))
        if scale == 0:
            break
        step = min(step / SHRINK, MAX_COORDINATE_MOVE / scale)
        accepted = armijo_step(x, value, grad, param, step, deadline)
        if accepted is None:
            break
        x, value, step = accepted
        iteration += 1

        if record is not None and iteration == next_snapshot:
            record(x, iteration=iteration, elapsed_s=perf_counter() - started)
            last_recorded = iteration
            next_snapshot = ceil(1.25 * iteration)

    if record is not None and iteration > last_recorded:
        record(x, iteration=iteration, elapsed_s=perf_counter() - started)
    return x


def solver(param, max_time):
    return run_pgd(param, max_time)
