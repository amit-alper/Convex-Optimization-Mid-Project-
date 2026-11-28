"""Baseline 2 (handout p. 7): Levenberg-Marquardt started from x0 (algorithm in optimizers/levenberg_marquardt.py)."""

from initialization.baseline_initialization import from_x0
from optimizers.levenberg_marquardt import run_lm


def initialize(param, max_time):
    """The course starting point x0 (random sensors in the box, anchors in place)."""
    return from_x0(param)


def solve(param, max_time, *, record=None):
    return run_lm(param, max_time, record=record)


def solver(param, max_time):
    return solve(param, max_time)
