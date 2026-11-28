"""Check the composed solver and the generic diagnostic entry point."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from pipelines.convergence_trace import Trace
from pipelines.paths import COURSE, ROOT, SOLVERS

sys.path.insert(0, str(COURSE))
sys.path.insert(0, str(SOLVERS))
import example_initialized_pgd as example
import problem


class TeamWorkflowTests(unittest.TestCase):
    def test_composed_solver_deducts_initialization_and_disables_benchmark_recording(self):
        param = problem.load(COURSE / "public" / "A_1.npz")
        start = example.initialize(param, 1)
        with (
            patch.object(example, "perf_counter", side_effect=[10, 10.25]),
            patch.object(example, "initialize", return_value=start) as initialize,
            patch.object(example, "run_pgd", return_value=start) as optimize,
        ):
            self.assertIs(example.solver(param, 5), start)
        initialize.assert_called_once_with(param, 1)
        optimize.assert_called_once_with(param, 4.75, x0=start, record=None)

    def test_composed_trace_elapsed_time_includes_initialization(self):
        param = problem.load(COURSE / "public" / "A_1.npz")
        callback = Mock()

        def optimize(param, max_time, *, x0, record):
            record(x0, iteration=0, elapsed_s=0.1)
            return x0

        with (patch.object(example, "perf_counter", side_effect=[10, 10.25]),
              patch.object(example, "run_pgd", side_effect=optimize)):
            result = example.solve(param, 5, record=callback)
        callback.assert_called_once_with(result, iteration=0, elapsed_s=0.35)

    def test_generic_command_records_and_renders_a_separate_solver_file(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "my_start.py").write_text("fraction = 0.5\n", encoding="utf-8")
            solver = directory / "my_algorithm.py"
            solver.write_text(
                "from my_start import fraction\n"
                "from problem import project\n"
                "\n"
                "def solve(param, max_time, *, record=None):\n"
                "    assert max_time == 5\n"
                "    x = project(param.x0 * fraction, param)\n"
                "    record(param.x0, iteration=0, elapsed_s=0)\n"
                "    record(x, iteration=1, elapsed_s=0.1)\n"
                "    return x\n",
                encoding="utf-8",
            )
            output = directory / "trace.npz"
            completed = subprocess.run(
                [sys.executable, "-B", "-m", "pipelines.record_trace", "A_1.npz", solver, "--output", output],
                cwd=ROOT, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.with_suffix(".html").is_file())
            trace = Trace.load(output)
        self.assertEqual(trace.instance, "A_1.npz")
        self.assertEqual(trace.iterations, [0, 1])
        param = problem.load(COURSE / "public" / "A_1.npz")
        np.testing.assert_array_equal(trace.positions[0], param.x0)
        np.testing.assert_array_equal(trace.positions[-1], problem.project(param.x0 * 0.5, param))


if __name__ == "__main__":
    unittest.main()
