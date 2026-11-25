"""Check collapsed references and the grouped official edge losses."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from convergence_trace import Trace
from trace_diagnostics import collapsed_positions, edge_loss_stats, objective_ratio
from visualize import COURSE, ROOT


sys.path.insert(0, str(COURSE))
import problem


def make_instance(loss):
    arrays = {
        "x0": np.array([[0, 0], [1, 0], [0.5, 0], [0, 0]]),
        "anchor": np.array([1, 3]),
        "anchor_pos": np.array([[1, 0], [0, 0]]),
        "edges": np.array([[0, 2], [0, 1], [3, 2], [1, 3]]),
        "dist": np.array([0.25, 0.75, 1.0, 0.25]),
    }
    return problem.Instance(arrays, {"n": 4, "q": 2, "kind": "A", "loss": loss, "delta": 0.3})


class DiagnosticsTests(unittest.TestCase):
    def test_collapse_moves_only_free_sensors_and_zero_reference_has_no_ratio(self):
        param = make_instance("squared")
        original = param.x0.copy()
        x = collapsed_positions(problem, param)
        np.testing.assert_array_equal(x[param.free], 0.5)
        np.testing.assert_array_equal(x[param.anchor], param.anchor_pos)
        np.testing.assert_array_equal(param.x0, original)
        self.assertEqual(param.violation(x), 0)
        self.assertEqual(objective_ratio(3, 2), 1.5)
        self.assertIsNone(objective_ratio(3, 0))
        self.assertIsNone(objective_ratio(0, 0))

    def test_edge_groups_match_both_official_losses_and_handle_empty_groups(self):
        for loss, means in (("squared", (0.03125, 0.078125, 0.28125)),
                            ("huber", (0.03125, 0.068125, 0.18))):
            with self.subTest(loss=loss):
                param = make_instance(loss)
                x = problem.project(param.x0, param)
                groups = edge_loss_stats(problem, param, x)
                self.assertEqual([count for count, _ in groups.values()], [1, 2, 1])
                np.testing.assert_allclose([mean for _, mean in groups.values()], means)
                weighted = sum(count * mean for count, mean in groups.values()) / len(param.edges)
                self.assertAlmostEqual(weighted, problem.objective(x, param))

        param = make_instance("squared")
        param.edges, param.dist = param.edges[:1], param.dist[:1]
        groups = edge_loss_stats(problem, param, problem.project(param.x0, param))
        self.assertEqual(groups["Free-to-anchor"], (0, None))
        self.assertEqual(groups["Anchor-to-anchor"], (0, None))

    def test_cli_reports_projected_trace_endpoints_and_reference(self):
        instance = "A_1.npz"
        param = problem.load(COURSE / "public" / instance)
        trace = Trace(instance)
        trace.record(param.x0, iteration=0, elapsed_s=0)
        trace.record(param.x0 + 0.2, iteration=1, elapsed_s=1)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trace.npz"
            trace.save(source)
            completed = subprocess.run(
                [sys.executable, "-B", "trace_diagnostics.py", source],
                cwd=ROOT, capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        initial = problem.objective(param.x0, param)
        reference = problem.objective(collapsed_positions(problem, param), param)
        final = problem.objective(problem.project(trace.positions[-1], param), param)
        self.assertIn(f"| {instance} | {initial:.10g} | {reference:.10g} | {final:.10g} | "
                      f"{final / reference:.10g} |", completed.stdout)
        for group in ("Free-to-free", "Free-to-anchor", "Anchor-to-anchor"):
            self.assertIn(f"| {instance} | {group} |", completed.stdout)


if __name__ == "__main__":
    unittest.main()
