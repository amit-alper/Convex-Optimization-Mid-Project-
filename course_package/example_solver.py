"""A minimal solver: it returns the starting point. Replace the body with your method."""
import time

from problem import objective, project


def solver(param, max_time):
    start = time.perf_counter()
    x = param.x0               # the starting point
    # ... your method here; return before time.perf_counter() - start reaches max_time ...
    return x
