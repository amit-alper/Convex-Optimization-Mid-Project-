"""Baseline 2 (handout p. 7): Levenberg-Marquardt (damped Gauss-Newton) started from x0.

Each iteration solves (J^T W J + lambda D) step = -J^T W r over the free sensor coordinates:
  - J: residual Jacobian (core.derivatives.Model.jacobian)
  - W: Huber weights, i.e. iteratively reweighted least squares (all ones for squared loss)
  - D: diag(J^T W J), Marquardt's scaling of the damping
The trial point is projected onto the box (as the handout's LM baseline for Topic 3 does)
and accepted only if the objective decreases; lambda shrinks after a success, grows after a failure.
"""

import sys
from math import ceil
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root, for core/

from problem import project                                     # noqa: E402  (course package)
from core.derivatives import Model                              # noqa: E402
from core.initialization.baseline_initialization import from_x0  # noqa: E402
from baseline_solvers import constants                          # noqa: E402  (tunable values)


def run_lm(param, x_start, deadline, *, record=None, started=None):
    """Run LM from x_start until the perf_counter() deadline; return the last accepted point.

    record(x, iteration=, elapsed_s=) is called on a geometric schedule (convergence_trace.Trace).
    """
    started = perf_counter() if started is None else started
    model = Model(param)
    x = project(x_start, param)
    value = model.objective(x)
    if record is not None:
        record(x, iteration=0, elapsed_s=perf_counter() - started)

    damping = constants.LM_LAMBDA_START
    iteration = last_recorded = 0
    next_snapshot = 1
    while perf_counter() < deadline and damping < constants.LM_LAMBDA_MAX:
        r = model.residuals(x)
        jac = model.jacobian(x)
        weights = model.huber_weights(r)
        normal = (jac.T @ sp.diags(weights) @ jac).tocsc()      # J^T W J
        rhs = -(jac.T @ (weights * r))                          # -J^T W r = -|E| * gradient
        diagonal = normal.diagonal()
        scaling = np.maximum(diagonal, 1e-12 * max(diagonal.max(), 1e-300))

        vector = model.to_vector(x)
        while perf_counter() < deadline and damping < constants.LM_LAMBDA_MAX:
            step = spsolve(normal + sp.diags(damping * scaling, format="csc"), rhs)
            candidate = project(model.to_positions(vector + step, x), param)
            candidate_value = model.objective(candidate)
            if candidate_value < value:
                x, value = candidate, candidate_value
                damping *= constants.LM_LAMBDA_DOWN
                break
            damping *= constants.LM_LAMBDA_UP
        else:
            break                                               # out of time or stalled

        iteration += 1
        if record is not None and iteration == next_snapshot:
            record(x, iteration=iteration, elapsed_s=perf_counter() - started)
            last_recorded, next_snapshot = iteration, ceil(1.25 * iteration)

    if record is not None and iteration > last_recorded:
        record(x, iteration=iteration, elapsed_s=perf_counter() - started)
    return x


def solver(param, max_time):
    """LM from the course starting point x0 for the whole budget."""
    started = perf_counter()
    deadline = started + constants.TIME_FRACTION * max_time
    return run_lm(param, from_x0(param), deadline, started=started)
