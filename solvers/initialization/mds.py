"""MDS-MAP [Shang03, Sec. 2.2]: positions from measured distances, one function per step.

  1. shortest_path_distances: fill in missing pairs with graph shortest paths
  2. classical_mds:           relative map from the top q eigenpairs [Shang03, Eq. 1]
  3. align_to_anchors:        scale + rotation/reflection + shift fit to the anchors
mds_map runs all three. baseline_initialization.mds_start builds on these pieces.

No local minima (eigendecomposition), but sensitive to range errors [Patwari05, p. 65];
expected error is about half the radio range at ~12 neighbors [Biswas04, Sec. 1].

References (PDFs in papers/):
  [Shang03]   Shang, Ruml, Zhang, Fromherz, Localization from mere connectivity, MobiHoc 2003.
  [Patwari05] Patwari et al., Locating the nodes, IEEE Signal Processing Magazine 2005.
  [Biswas04]  Biswas, Ye, Semidefinite programming for ad hoc wireless sensor network localization, IPSN 2004.
"""

from time import perf_counter

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import eigsh


MIN_STAGE_TIME = 0.01
PATH_SECONDS_PER_WORK_UNIT = 6e-8
EMBED_SECONDS_PER_PAIR = 8e-8


def mds_map(param, seed=0, *, deadline=None):
    """Return the aligned layout, or None if the optional deadline rules out a stage."""
    if deadline is not None:
        # Estimates with headroom for the public graphs. Neither SciPy call is interruptible here.
        path_time = max(MIN_STAGE_TIME, PATH_SECONDS_PER_WORK_UNIT * param.n * (param.n + len(param.edges)))
        embed_time = max(MIN_STAGE_TIME, EMBED_SECONDS_PER_PAIR * param.n ** 2)
        if deadline - perf_counter() < path_time + embed_time:
            return None

    distances = shortest_path_distances(param)
    if deadline is not None and deadline - perf_counter() < embed_time:
        return None
    relative = classical_mds(distances, param.q, seed)
    if deadline is not None and perf_counter() >= deadline:
        return None
    positions = align_to_anchors(relative, param)
    if deadline is not None and perf_counter() >= deadline:
        return None
    return positions


def shortest_path_distances(param):
    """Step 1: n x n shortest-path distances through the measured edges (used as measured)."""
    first, second = param.edges.T
    graph = sp.csr_matrix((param.dist, (first, second)), shape=(param.n, param.n))
    distances = shortest_path(graph, directed=False)
    finite = np.isfinite(distances)
    if not finite.all():                       # disconnected graph: treat as far apart
        distances[~finite] = distances[finite].max()
    return distances


def classical_mds(distances, q, seed=0):
    """Step 2: classical MDS of a full distance matrix (overwritten to save memory) -> n x q.

    B = -1/2 J D^2 J (double centering), coordinates = top-q eigenvectors * sqrt(eigenvalues).
    Only q eigenpairs are needed, so eigsh instead of a full SVD.
    """
    b = distances
    b **= 2
    means = b.mean(axis=1)
    b -= means[:, None]
    b -= means[None, :]
    b += means.mean()
    b *= -0.5

    start = np.random.default_rng(seed).standard_normal(len(b))   # fixed seed: reproducible
    values, vectors = eigsh(b, k=q, which="LA", v0=start)
    order = np.argsort(values)[::-1]
    return vectors[:, order] * np.sqrt(np.maximum(values[order], 0.0))


def align_to_anchors(y, param):
    """Step 3: map y by the similarity transform that best fits y[anchor] onto anchor_pos.

    Scaling, rotation and reflection by least squares [Shang03, Sec. 2.2], solved in closed form
    by orthogonal Procrustes (SVD). With collinear anchors (type B) the reflection across the
    anchor line is not determined by the anchors.
    """
    target = param.anchor_pos
    source = y[param.anchor]
    target_mean, source_mean = target.mean(axis=0), source.mean(axis=0)
    centered_target, centered_source = target - target_mean, source - source_mean

    u, s, vt = np.linalg.svd(centered_source.T @ centered_target)
    rotation = u @ vt
    spread = float(np.sum(centered_source ** 2))
    scale = s.sum() / spread if spread > 0 else 1.0
    return scale * (y - source_mean) @ rotation + target_mean
