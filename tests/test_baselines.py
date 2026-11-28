"""Check the shared core derivatives and the two course baselines (MDS + PGD, Levenberg-Marquardt)."""

import sys
import unittest
from time import perf_counter
from unittest.mock import Mock, patch

import numpy as np

from pipelines.convergence_trace import Trace
from pipelines.paths import COURSE, SOLVERS


sys.path.insert(0, str(COURSE))
sys.path.insert(0, str(SOLVERS))
import problem
import levenberg_marquardt as lm_solver
import mds_pgd
from core.derivatives import Model
from initialization import mds
from initialization.baseline_initialization import mds_start
from optimizers import levenberg_marquardt, pgd


def load(name):
    return problem.load(COURSE / "public" / name)


class Clock:
    """Advance time only at the operations a deadline test controls."""

    def __init__(self):
        self.elapsed = 0.0

    def __call__(self):
        return self.elapsed


class CoreTests(unittest.TestCase):
    def test_model_matches_course_objective_and_shared_gradient(self):
        for filename in ("A_1.npz", "B_1.npz"):
            with self.subTest(filename=filename):
                param = load(filename)
                model = Model(param)
                x = problem.project(np.random.default_rng(1).random((param.n, param.q)), param)
                self.assertAlmostEqual(model.objective(x), problem.objective(x, param), places=14)
                np.testing.assert_allclose(model.gradient(x), pgd.gradient(x, param), atol=1e-15)

    def test_jacobian_agrees_with_gradient(self):
        for filename in ("A_1.npz", "B_1.npz"):
            with self.subTest(filename=filename):
                param = load(filename)
                model = Model(param)
                x = problem.project(np.random.default_rng(2).random((param.n, param.q)), param)
                r = model.residuals(x)
                from_jacobian = model.jacobian(x).T @ model.loss_slope(r) / model.n_edges
                np.testing.assert_allclose(from_jacobian, model.to_vector(model.gradient(x)), atol=1e-15)

    def test_mds_start_is_feasible_and_better_than_x0(self):
        param = load("A_1.npz")
        x = mds_start(param)
        np.testing.assert_array_equal(x, problem.project(x, param))
        self.assertLess(problem.objective(x, param), problem.objective(param.x0, param))


class BaselineTests(unittest.TestCase):
    def test_mds_short_budget_skips_shortest_paths(self):
        param = load("C_1.npz")
        clock = Clock()
        callback = Mock()
        with (patch.object(mds_pgd, "perf_counter", clock),
              patch.object(mds, "perf_counter", clock),
              patch.object(mds, "shortest_path_distances", return_value=None) as paths,
              patch.object(mds, "classical_mds", return_value=param.x0),
              patch.object(mds_pgd, "run_pgd") as optimize):
            x = mds_pgd.solve(param, 0.02, record=callback)
        paths.assert_not_called()
        optimize.assert_not_called()
        np.testing.assert_array_equal(x, problem.project(param.x0, param))
        callback.assert_called_once_with(x, iteration=0, elapsed_s=0.0)

    def test_mds_skips_eigensolve_when_shortest_paths_use_the_reserve(self):
        param = load("A_1.npz")
        clock = Clock()

        def slow_paths(param):
            clock.elapsed = 0.085
            return np.zeros((param.n, param.n))

        with (patch.object(mds_pgd, "perf_counter", clock),
              patch.object(mds, "perf_counter", clock),
              patch.object(mds, "shortest_path_distances", side_effect=slow_paths),
              patch.object(mds, "classical_mds", return_value=param.x0) as embed):
            x = mds_pgd.initialize(param, 0.1)
        embed.assert_not_called()
        np.testing.assert_array_equal(x, problem.project(param.x0, param))

    def test_mds_discards_initialization_that_finishes_after_the_deadline(self):
        param = load("A_1.npz")
        clock = Clock()

        def slow_alignment(y, param):
            clock.elapsed = 0.095
            return np.zeros_like(y)

        with (patch.object(mds_pgd, "perf_counter", clock),
              patch.object(mds, "perf_counter", clock),
              patch.object(mds, "align_to_anchors", side_effect=slow_alignment) as align):
            x = mds_pgd.initialize(param, 0.1)
        align.assert_called_once()
        np.testing.assert_array_equal(x, problem.project(param.x0, param))

    def test_lm_short_budget_returns_projected_supplied_start(self):
        param = load("A_1.npz")
        start = np.full_like(param.x0, 1.5)
        clock = Clock()
        callback = Mock()

        def slow_step(matrix, rhs):
            clock.elapsed = 0.03
            return np.zeros_like(rhs)

        with (patch.object(levenberg_marquardt, "perf_counter", clock),
              patch.object(levenberg_marquardt, "spsolve", side_effect=slow_step) as solve):
            x = levenberg_marquardt.run_lm(param, 0.02, x0=start, record=callback)
        solve.assert_not_called()
        np.testing.assert_array_equal(x, problem.project(start, param))
        callback.assert_called_once_with(x, iteration=0, elapsed_s=0.0)

    def test_lm_rechecks_budget_after_building_the_damped_system(self):
        param = load("A_1.npz")
        clock = Clock()
        diags = levenberg_marquardt.sp.diags

        def slow_diagonal(*args, **kwargs):
            if kwargs.get("format") == "csc":
                clock.elapsed = 0.085
            return diags(*args, **kwargs)

        def slow_step(matrix, rhs):
            clock.elapsed += 0.04
            return np.zeros_like(rhs)

        with (patch.object(levenberg_marquardt, "perf_counter", clock),
              patch.object(levenberg_marquardt.sp, "diags", side_effect=slow_diagonal) as diagonal,
              patch.object(levenberg_marquardt, "spsolve", side_effect=slow_step) as solve):
            x = lm_solver.solver(param, 0.1)
        self.assertEqual(diagonal.call_count, 2)
        solve.assert_not_called()
        np.testing.assert_array_equal(x, problem.project(param.x0, param))

    def test_lm_reserves_time_for_a_retry_based_on_the_previous_trial(self):
        param = load("A_1.npz")
        clock = Clock()

        def slow_rejected_step(matrix, rhs):
            clock.elapsed += 0.04
            return np.zeros_like(rhs)

        with (patch.object(levenberg_marquardt, "perf_counter", clock),
              patch.object(levenberg_marquardt, "spsolve", side_effect=slow_rejected_step) as solve):
            x = lm_solver.solver(param, 0.1)
        solve.assert_called_once()
        np.testing.assert_array_equal(x, problem.project(param.x0, param))

    def test_lm_discards_a_trial_that_finishes_after_the_deadline(self):
        param = load("A_1.npz")
        clock = Clock()
        spsolve = levenberg_marquardt.spsolve

        def slow_step(matrix, rhs):
            clock.elapsed = 0.095
            return spsolve(matrix, rhs)

        with (patch.object(levenberg_marquardt, "perf_counter", clock),
              patch.object(levenberg_marquardt, "spsolve", side_effect=slow_step) as solve):
            x = lm_solver.solver(param, 0.1)
        solve.assert_called_once()
        np.testing.assert_array_equal(x, problem.project(param.x0, param))

    def test_baselines_improve_feasible_output_within_budget(self):
        param = load("A_1.npz")
        original = param.x0.copy()
        start_value = problem.objective(param.x0, param)
        for module in (mds_pgd, lm_solver):
            with self.subTest(solver=module.__name__):
                started = perf_counter()
                x = module.solver(param, 1.0)
                self.assertLess(perf_counter() - started, 1.0)
                self.assertEqual(x.shape, (param.n, param.q))
                np.testing.assert_array_equal(x, problem.project(x, param))
                self.assertLess(problem.objective(x, param), start_value)
                np.testing.assert_array_equal(param.x0, original)

    def test_lm_accepts_a_supplied_start_and_records_a_valid_trace(self):
        param = load("A_1.npz")
        start = mds_start(param)
        trace = Trace("A_1.npz")
        x = levenberg_marquardt.run_lm(param, 0.5, x0=start, record=trace.record)
        self.assertEqual(trace.iterations[0], 0)
        np.testing.assert_array_equal(trace.positions[0], start)
        np.testing.assert_array_equal(trace.positions[-1], x)
        self.assertLessEqual(problem.objective(x, param), problem.objective(start, param))

    def test_mds_pgd_deducts_initialization_time_and_shifts_trace_times(self):
        param = load("A_1.npz")
        start = param.x0.copy()
        callback = Mock()

        def optimize(param, max_time, *, x0, record):
            record(x0, iteration=0, elapsed_s=0.1)
            return x0

        with (patch.object(mds_pgd, "perf_counter", side_effect=[10, 10.25]),
              patch.object(mds_pgd, "initialize", return_value=start),
              patch.object(mds_pgd, "run_pgd", side_effect=optimize) as run):
            self.assertIs(mds_pgd.solve(param, 5, record=callback), start)
        self.assertAlmostEqual(run.call_args.args[1], mds_pgd.TIME_FRACTION * 5 - 0.25)
        callback.assert_called_once_with(start, iteration=0, elapsed_s=0.35)


if __name__ == "__main__":
    unittest.main()
