"""Residuals and derivatives of the localization objective, shared by every solver.

    f(x) = (1/|E|) sum_k rho(r_k),   r_k = ||x_i - x_j|| - d_ij   for edge k = (i, j)

gradient = J^T rho'(r) / |E|, with J the residual Jacobian: one formula serves PGD (gradient)
and Levenberg-Marquardt (jacobian; handout baseline). Huber is handled in LM by reweighting
(huber_weights, iteratively reweighted least squares).
"""

import numpy as np
import scipy.sparse as sp


class Model:
    """Per-instance structure, built once, with the residuals and derivatives at any x (n x q)."""

    def __init__(self, param):
        self.param = param
        self.n, self.q = param.n, param.q
        self.first, self.second = param.edges.T
        self.n_edges = len(param.edges)
        self.huber = param.loss == "huber"
        self.delta = param.delta
        self.anchor = param.anchor
        self.free = param.free

        # Incidence matrix B (|E| x n): row k is +1 at i and -1 at j, so B @ x gives x_i - x_j.
        edge_rows = np.repeat(np.arange(self.n_edges), 2)
        self.incidence = sp.csr_matrix(
            (np.tile([1.0, -1.0], self.n_edges), (edge_rows, param.edges.ravel())),
            shape=(self.n_edges, self.n),
        )

        # Jacobian sparsity over the free coordinates only (anchors are fixed).
        # Free node `free[p]`, coordinate c, is column p * q + c.
        column_of = np.full(self.n, -1)
        column_of[self.free] = np.arange(len(self.free))
        rows, cols, signs, edge_of = [], [], [], []
        for end, sign in ((self.first, 1.0), (self.second, -1.0)):
            keep = column_of[end] >= 0
            edges = np.flatnonzero(keep)
            for c in range(self.q):
                rows.append(edges)
                cols.append(column_of[end[keep]] * self.q + c)
                signs.append(np.full(len(edges), sign))
                edge_of.append((edges, c))
        self._jac_rows = np.concatenate(rows)
        self._jac_cols = np.concatenate(cols)
        self._jac_signs = np.concatenate(signs)
        self._jac_edge_of = edge_of
        self.n_vars = len(self.free) * self.q

    # --- residuals ----------------------------------------------------------------------------

    def edge_terms(self, x):
        """Return (residuals r, unit directions u) per edge; u is zero where x_i == x_j."""
        difference = self.incidence @ x
        length = np.linalg.norm(difference, axis=1)
        direction = np.divide(
            difference, length[:, None], out=np.zeros_like(difference), where=length[:, None] > 0,
        )
        return length - self.param.dist, direction

    def residuals(self, x):
        """r_k = ||x_i - x_j|| - d_ij for every edge."""
        return np.linalg.norm(self.incidence @ x, axis=1) - self.param.dist

    def loss_slope(self, r):
        """rho'(r): r for squared loss, clip(r, -delta, delta) for Huber."""
        return np.clip(r, -self.delta, self.delta) if self.huber else r

    def huber_weights(self, r):
        """IRLS weights w with rho'(r) = w * r: 1 inside delta, delta / |r| outside (all 1 for squared)."""
        if not self.huber:
            return np.ones_like(r)
        a = np.abs(r)
        return np.divide(self.delta, a, out=np.ones_like(r), where=a > self.delta)

    def objective(self, x):
        """Same value as problem.objective (no projection applied)."""
        r = self.residuals(x)
        if self.huber:
            a = np.abs(r)
            losses = np.where(a <= self.delta, 0.5 * r * r, self.delta * a - 0.5 * self.delta ** 2)
            return float(np.mean(losses))
        return float(0.5 * np.mean(r * r))

    # --- derivatives --------------------------------------------------------------------------

    def gradient(self, x):
        """Gradient with respect to all positions (n x q); anchor rows are zero."""
        r, u = self.edge_terms(x)
        contribution = u * (self.loss_slope(r) / self.n_edges)[:, None]
        result = self.incidence.T @ contribution
        result[self.anchor] = 0
        return result

    def jacobian(self, x):
        """Sparse Jacobian of the residuals (|E| x n_vars) with respect to the free coordinates.

        Row k is +u_k at sensor i and -u_k at sensor j. Use to_vector / to_positions to move
        between n x q positions and the free-coordinate vector.
        """
        _, u = self.edge_terms(x)
        values = np.concatenate([u[edges, c] for edges, c in self._jac_edge_of]) * self._jac_signs
        return sp.csr_matrix(
            (values, (self._jac_rows, self._jac_cols)), shape=(self.n_edges, self.n_vars),
        )

    # --- conversions between positions and the free-coordinate vector --------------------------

    def to_vector(self, x):
        """Free sensor coordinates as a flat vector (the Jacobian's column order)."""
        return np.asarray(x)[self.free].ravel()

    def to_positions(self, v, template):
        """Write a free-coordinate vector back into a copy of template (n x q)."""
        x = np.array(template, dtype=float)
        x[self.free] = np.reshape(v, (len(self.free), self.q))
        return x
