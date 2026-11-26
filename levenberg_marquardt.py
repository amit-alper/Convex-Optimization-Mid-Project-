"""Baseline 2 (handout p. 7): Levenberg-Marquardt (damped Gauss-Newton) started from x0.

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


TIME_FRACTION = 0.9      # one sparse solve can take a while on C, so keep a wider margin than PGD
LAMBDA_START = 1e-3      # initial damping
LAMBDA_DOWN = 0.3        # multiply after an accepted step
LAMBDA_UP = 10.0         # multiply after a rejected step
LAMBDA_MAX = 1e12        # stop: no decrease even with a tiny gradient step


def run_lm(param, max_time, *, x0=None, record=None):
    """Run LM from the supplied array or the instance's default start; return the last accepted point.

    record(x, iteration=, elapsed_s=) is called on a geometric schedule (convergence_trace.Trace).
    """
    started = perf_counter()
    deadline = started + TIME_FRACTION * max_time
    model = Model(param)
    x = project(param.x0 if x0 is None else x0, param)
    if record is not None:
        record(x, iteration=0, elapsed_s=0.0)
    value = model.objective(x)

    damping = LAMBDA_START
    iteration = last_recorded = 0
    next_snapshot = 1
    while perf_counter() < deadline and damping < LAMBDA_MAX:
        r = model.residuals(x)
        jac = model.jacobian(x)
        weights = model.huber_weights(r)
        normal = (jac.T @ sp.diags(weights) @ jac).tocsc()      # J^T W J
        rhs = -(jac.T @ (weights * r))                          # -J^T W r = -|E| * gradient
        diagonal = normal.diagonal()
        scaling = np.maximum(diagonal, 1e-12 * max(diagonal.max(), 1e-300))

        vector = model.to_vector(x)
        accepted = False
        while perf_counter() < deadline and damping < LAMBDA_MAX:
            step = spsolve(normal + sp.diags(damping * scaling, format="csc"), rhs)
            candidate = project(model.to_positions(vector + step, x), param)
            candidate_value = model.objective(candidate)
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


def initialize(param, max_time):
    """The course starting point x0 (random sensors in the box, anchors in place)."""
    return project(param.x0, param)


def solve(param, max_time, *, record=None):
    return run_lm(param, max_time, x0=initialize(param, max_time), record=record)


def solver(param, max_time):
    return solve(param, max_time)
