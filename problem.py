"""Topic 4: sensor-network localization.

    minimize  (1/|E|) sum_{(i,j) in E} rho(||x_i - x_j|| - d_ij)
    s.t.      x_i in [0,1]^q for sensors,  x_i = a_i for anchors

x is an n x q array of all node positions (anchors included). rho is the squared loss t^2/2 or the
Huber loss with parameter delta. Starting point x0: random positions in the box, anchors in place.
"""
import numpy as np

from common.instances import load_npz

TOPIC = "localization"
def huber(t, delta):
    a = np.abs(t)
    return np.where(a <= delta, 0.5 * t * t, delta * a - 0.5 * delta * delta)


class Instance:
    feas_tol = 1e-8

    def __init__(self, arrays, meta):
        self.edges, self.dist = arrays["edges"], arrays["dist"]
        self.anchor, self.anchor_pos = arrays["anchor"], arrays["anchor_pos"]
        self.x0 = arrays["x0"]
        self.x_true = arrays.get("x_true")          # staff only; removed from student copies
        self.meta = meta
        self.kind = meta["kind"]
        self.n, self.q = meta["n"], meta["q"]
        self.loss, self.delta = meta["loss"], meta["delta"]
        mask = np.ones(self.n, dtype=bool)
        mask[self.anchor] = False
        self.free = np.flatnonzero(mask)

    def residuals(self, x):
        i, j = self.edges[:, 0], self.edges[:, 1]
        return np.linalg.norm(x[i] - x[j], axis=1) - self.dist

    def objective(self, x):
        r = self.residuals(np.asarray(x, dtype=float))
        if self.loss == "huber":
            return float(np.mean(huber(r, self.delta)))
        return float(0.5 * np.mean(r * r))

    def violation(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n, self.q) or not np.all(np.isfinite(x)):
            return np.inf
        box = np.max(np.maximum(-x[self.free], x[self.free] - 1.0), initial=0.0)
        anc = np.max(np.abs(x[self.anchor] - self.anchor_pos), initial=0.0)
        return float(max(box, anc, 0.0))


def load(path):
    """The data of one instance (`param`)."""
    return Instance(*load_npz(path))


load_instance = load


def objective(x, param):
    """The objective (1/|E|) sum rho(||x_i - x_j|| - d_ij)."""
    return param.objective(x)


def project(x, param):
    """The projection: sensors into the box [0, 1]^q, anchors to their positions."""
    x = np.array(x, dtype=float)
    if x.shape != (param.n, param.q) or not np.all(np.isfinite(x)):
        raise ValueError(f"x must be a finite {param.n} x {param.q} array")
    x = np.clip(x, 0.0, 1.0)
    x[param.anchor] = param.anchor_pos
    return x
