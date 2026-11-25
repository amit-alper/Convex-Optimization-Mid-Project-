"""Baseline starting points. Each returns a projected n x q array.

The time spent here counts toward the solver's max_time, so callers should measure it.
"""

import numpy as np

from problem import project
from core.mds import mds_map


def from_x0(param):
    """The course starting point: random positions in the box, anchors in place."""
    return project(param.x0, param)


def random_start(param, seed=None):
    """Uniform random positions in the box, anchors in place."""
    rng = np.random.default_rng(seed)
    return project(rng.random((param.n, param.q)), param)


def mds_start(param, seed=0):
    """The handout's PGD baseline start: MDS-MAP (core/mds.py), projected onto the box."""
    return project(mds_map(param, seed), param)
