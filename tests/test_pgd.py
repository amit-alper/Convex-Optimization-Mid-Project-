"""Check PGD derivatives, feasibility, accepted steps, and timing."""

import sys
import unittest
from unittest.mock import patch

import numpy as np

from pipelines.convergence_trace import Trace
from pipelines.paths import COURSE, SOLVERS


sys.path.insert(0, str(COURSE))
sys.path.insert(0, str(SOLVERS))
import pgd as pgd_solver
import problem
from optimizers import pgd


class PGDTests(unittest.TestCase):
    def test_gradient_matches_central_differences(self):
        rng = np.random.default_rng(10)
        for filename in ("A_1.npz", "B_1.npz"):
            with self.subTest(filename=filename):
                param = problem.load(COURSE / "public" / filename)
                x = param.x0.copy()
                if param.loss == "huber":
                    # This random start has no residual inside the quadratic region.
                    free_edge = ~np.isin(param.edges, param.anchor).any(axis=1)
                    edge = np.flatnonzero(free_edge & (param.dist > 0) & (param.dist < 0.5))[0]
                    first, second = param.edges[edge]
                    x[first] = (0.25, 0.5)
                    x[second] = (0.25 + param.dist[edge] + param.delta / 2, 0.5)
                    residuals = np.abs(param.residuals(x))
                    self.assertTrue(np.any(residuals < param.delta))
                    self.assertTrue(np.any(residuals > param.delta))
                grad = pgd.gradient(x, param)
                np.testing.assert_array_equal(grad[param.anchor], 0)

                for _ in range(4):
                    direction = rng.normal(size=x.shape)
                    direction[param.anchor] = 0
                    direction /= np.linalg.norm(direction)
                    h = 1e-6
                    numerical = (problem.objective(x + h * direction, param)
                                 - problem.objective(x - h * direction, param)) / (2 * h)
                    np.testing.assert_allclose(
                        np.vdot(grad, direction), numerical, rtol=2e-5, atol=1e-10,
                    )

    def test_coincident_positions_have_finite_zero_edge_contributions(self):
        for filename in ("A_1.npz", "B_1.npz"):
            with self.subTest(filename=filename):
                param = problem.load(COURSE / "public" / filename)
                x = np.full_like(param.x0, 0.5)
                np.testing.assert_array_equal(pgd.gradient(x, param), 0)

    def test_solver_improves_feasible_output_and_preserves_input(self):
        for kind in "ABCD":
            with self.subTest(kind=kind):
                param = problem.load(COURSE / "public" / f"{kind}_1.npz")
                original = param.x0.copy()
                x = pgd_solver.solver(param, 0.12)
                self.assertEqual(param.violation(x), 0)
                self.assertLess(problem.objective(x, param), problem.objective(original, param))
                np.testing.assert_array_equal(param.x0, original)
                self.assertFalse(np.shares_memory(x, param.x0))

    def test_backtracking_does_not_accept_a_trial_past_the_deadline(self):
        param = problem.load(COURSE / "public" / "A_1.npz")
        initial = problem.objective(param.x0, param)
        elapsed = 0.0
        evaluations = 0

        def score(x, param):
            nonlocal elapsed, evaluations
            evaluations += 1
            if evaluations == 1:
                return initial
            elapsed += 0.33
            return initial + 1 if elapsed < 0.95 else initial - 1

        with (
            patch.object(pgd, "perf_counter", side_effect=lambda: elapsed),
            patch.object(pgd, "objective", score),
        ):
            x = pgd_solver.solver(param, 1)
        self.assertEqual(evaluations, 4)
        np.testing.assert_array_equal(x, param.x0)

    def test_alternative_start_is_projected_without_replacing_param_x0(self):
        param = problem.load(COURSE / "public" / "A_1.npz")
        original = param.x0.copy()
        supplied = param.x0 + 0.5
        before = supplied.copy()
        x = pgd.run_pgd(param, 0, x0=supplied)
        np.testing.assert_array_equal(x, problem.project(supplied, param))
        np.testing.assert_array_equal(param.x0, original)
        np.testing.assert_array_equal(supplied, before)
        np.testing.assert_array_equal(pgd_solver.solver(param, 0), original)

    def test_optional_trace_includes_start_and_final_accepted_positions(self):
        filename = "A_1.npz"
        param = problem.load(COURSE / "public" / filename)
        trace = Trace(filename)
        x = pgd.run_pgd(param, 0.12, record=trace.record)
        np.testing.assert_array_equal(trace.positions[0], param.x0)
        np.testing.assert_array_equal(trace.positions[-1], x)
        self.assertGreater(len(trace.positions), 1)
        self.assertLessEqual(len(trace.positions), 32)
        objectives = [problem.objective(positions, param) for positions in trace.positions]
        self.assertTrue(np.all(np.diff(objectives) < 0))


if __name__ == "__main__":
    unittest.main()
