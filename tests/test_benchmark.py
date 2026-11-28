"""Smoke tests through the benchmark CLI and the supplied course runner."""

import csv
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pipelines.paths import ROOT


CONTROL = str(Path("solvers/solver.py"))  # the CSV records the path as given


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.output = self.directory / "results" / "runs.csv"

    def invoke(self, *arguments):
        return subprocess.run(
            [sys.executable, "-B", "-m", "pipelines.benchmark", *arguments, "--output", self.output],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )

    def read_rows(self):
        with self.output.open(encoding="utf-8", newline="") as output:
            return list(csv.DictReader(output))

    def write_solver(self, filename, body):
        path = self.directory / filename
        path.write_text(f"def solver(param, max_time):\n    {body}\n", encoding="utf-8")
        return path

    def test_compare_standalone_initializer_at_both_budgets(self):
        # Return a fixed test point. The course runner will restore the anchors.
        initializer = self.write_solver("initializer.py", "return param.x0 * 0")
        completed = self.invoke(CONTROL, initializer, "--only", "A_1.npz", "B_1.npz", "--run", "both")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(str(initializer), completed.stdout)
        rows = self.read_rows()
        expected = {
            (solver, instance, run, budget)
            for solver in (CONTROL, str(initializer))
            for instance in ("A_1.npz", "B_1.npz")
            for run, budget in (("short", 5), ("long", 20))
        }
        self.assertEqual(len(rows), len(expected))
        trials = {(r["solver"], r["instance"], r["run"], float(r["budget_s"])) for r in rows}
        self.assertEqual(trials, expected)
        objectives = {}
        for row in rows:
            self.assertEqual(row["status"], "ok", row["error"])
            self.assertTrue(math.isfinite(float(row["objective"])))
            self.assertGreaterEqual(float(row["runtime_s"]), 0)
            self.assertLessEqual(float(row["runtime_s"]), float(row["budget_s"]))
            self.assertGreaterEqual(float(row["peak_memory_mb"]), 0)
            self.assertEqual(row["seed"], "0")
            if row["run"] == "short":
                objectives[row["solver"], row["instance"]] = float(row["objective"])
        starting_objective = objectives[CONTROL, "A_1.npz"]
        self.assertAlmostEqual(starting_objective, 0.09200443128, places=10)
        self.assertNotEqual(starting_objective, objectives[str(initializer), "A_1.npz"])

    def test_failed_trials_are_recorded_and_the_experiment_continues(self):
        broken = self.write_solver("broken.py", 'raise RuntimeError("test solver failed")')
        invalid = self.write_solver("invalid.py", "return [0.0]")
        completed = self.invoke(broken, invalid, CONTROL, "--only", "A_1.npz", "--run", "short")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        rows = self.read_rows()
        self.assertEqual([row["status"] for row in rows], ["error", "invalid", "ok"])
        self.assertEqual([row["run"] for row in rows], ["short"] * 3)
        self.assertEqual([row["objective"] for row in rows[:2]], ["", ""])
        self.assertIn("test solver failed", rows[0]["error"])
        self.assertIn("invalid solution", rows[1]["error"])
        self.assertTrue(math.isfinite(float(rows[2]["objective"])))

    def test_unknown_instance_is_rejected_before_writing_csv(self):
        completed = self.invoke(CONTROL, "--only", "missing.npz")
        self.assertEqual(completed.returncode, 2)
        self.assertIn("missing.npz", completed.stderr)
        self.assertFalse(self.output.exists())

    def test_existing_csv_is_preserved(self):
        self.output.parent.mkdir()
        self.output.write_text("original\n", encoding="utf-8")
        completed = self.invoke(CONTROL, "--only", "A_1.npz")
        self.assertEqual(completed.returncode, 2)
        self.assertIn("cannot create", completed.stderr)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "original\n")


if __name__ == "__main__":
    unittest.main()
