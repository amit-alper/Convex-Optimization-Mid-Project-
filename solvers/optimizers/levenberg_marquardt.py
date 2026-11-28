"""Levenberg-Marquardt (damped Gauss-Newton), the optimizer of baseline 2 (handout p. 7).

Each iteration solves (J^T W J + lambda D) step = -J^T W r over the free sensor coordinates:
  - J: residual Jacobian (core.derivatives.Model.jacobian)
  - W: Huber weights, i.e. iteratively reweighted least squares (all ones for squared loss)
  - D: diag(J^T W J), Marquardt's scaling of the damping
The trial point is projected onto the box and accepted only if the objective decreases;
lambda shrinks after a success and grows after a failure.
"""

from math import ceil
from time import perf_counter

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from problem import project
from core.derivatives import Model


TIME_FRACTION = 0.9      # leave time for return overhead and runtime variation
MIN_TRIAL_TIME = 0.025
TRIAL_SECONDS_PER_PAIR = 1.6e-8   # initial reserve reaches 0.4 s at 5,000 free coordinates
TRIAL_TIME_FACTOR = 2.0          # reserve at least twice the slowest completed trial
LAMBDA_START = 1e-3      # initial damping
LAMBDA_DOWN = 0.3        # multiply after an accepted step
LAMBDA_UP = 10.0         # multiply after a rejected step
LAMBDA_MAX = 1e12        # stop: no decrease even with a tiny gradient step


def run_lm(param, max_time, *, x0=None, record=None):
    """Run LM from the supplied array or the instance's default start; return the last accepted point.

    record(x, iteration=, elapsed_s=) is called on a geometric schedule (pipelines/convergence_trace.py).
    """
    started = perf_counter()
    deadline = started + TIME_FRACTION * max_time
    x = project(param.x0 if x0 is None else x0, param)
    if record is not None:
        record(x, iteration=0, elapsed_s=0.0)

    free_coordinates = len(param.free) * param.q
    trial_time = max(MIN_TRIAL_TIME, TRIAL_SECONDS_PER_PAIR * free_coordinates ** 2)
    if deadline - perf_counter() < trial_time:
        return x

    model = Model(param)
    value = model.objective(x)

    damping = LAMBDA_START
    iteration = last_recorded = 0
    next_snapshot = 1
    while deadline - perf_counter() >= trial_time and damping < LAMBDA_MAX:
        r = model.residuals(x)
        jac = model.jacobian(x)
        weights = model.huber_weights(r)
        normal = (jac.T @ sp.diags(weights) @ jac).tocsc()      # J^T W J
        rhs = -(jac.T @ (weights * r))                          # -J^T W r = -|E| * gradient
        diagonal = normal.diagonal()
        scaling = np.maximum(diagonal, 1e-12 * max(diagonal.max(), 1e-300))

        vector = model.to_vector(x)
        accepted = False
        while damping < LAMBDA_MAX:
            trial_started = perf_counter()
            if deadline - trial_started < trial_time:
                break
            system = normal + sp.diags(damping * scaling, format="csc")
            if deadline - perf_counter() < trial_time:
                break
            step = spsolve(system, rhs)
            if perf_counter() >= deadline:
                break
            candidate = project(model.to_positions(vector + step, x), param)
            candidate_value = model.objective(candidate)
            trial_finished = perf_counter()
            if trial_finished >= deadline:
                break
            trial_time = max(trial_time, TRIAL_TIME_FACTOR * (trial_finished - trial_started))
            if candidate_value < value:
                x, value = candidate, candidate_value
                damping *= LAMBDA_DOWN
                accepted = True
                break
            damping *= LAMBDA_UP
        if not accepted:
            break                                               # out of time or stalled

        iteration += 1
        if record is not None and iteration == next_snapshot:
            record(x, iteration=iteration, elapsed_s=perf_counter() - started)
            last_recorded, next_snapshot = iteration, ceil(1.25 * iteration)

    if record is not None and iteration > last_recorded:
        record(x, iteration=iteration, elapsed_s=perf_counter() - started)
    return x

