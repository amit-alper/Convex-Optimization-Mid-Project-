"""Projected gradient descent from the course start x0 (algorithm in optimizers/pgd.py)."""

from optimizers.pgd import run_pgd


def solver(param, max_time):
    return run_pgd(param, max_time)


# The shared trace command calls solve with an optional recording callback.
solve = run_pgd
