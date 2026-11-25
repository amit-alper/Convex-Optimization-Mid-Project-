"""Exercise trace storage and offline rendering without an optimization solver."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from convergence_trace import Trace
from visualize import COURSE, ROOT


class TraceTests(unittest.TestCase):
    def test_round_trip_copies_selected_snapshots(self):
        trace = Trace(Path("course_package/public/A_1.npz"), "Artificial test trace")
        positions = np.zeros((3, 2))
        trace.record(positions, iteration=0, elapsed_s=0)
        positions[:] = 0.5
        trace.record(positions, iteration=11, elapsed_s=0.2)
        positions[:] = 1

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.npz"
            trace.save(path)
            loaded = Trace.load(path)
            with self.assertRaises(FileExistsError):
                trace.save(path)

        self.assertEqual((loaded.instance, loaded.label), ("A_1.npz", "Artificial test trace"))
        self.assertEqual(loaded.iterations, [0, 11])
        self.assertEqual(loaded.elapsed_s, [0, 0.2])
        np.testing.assert_array_equal(loaded.positions[0], np.zeros((3, 2)))
        np.testing.assert_array_equal(loaded.positions[1], np.full((3, 2), 0.5))

    def test_order_and_shape_are_checked(self):
        trace = Trace("A_1.npz")
        x = np.zeros((3, 2))
        trace.record(x, iteration=1, elapsed_s=0.1)
        with self.assertRaisesRegex(ValueError, "iterations must increase"):
            trace.record(x, iteration=1, elapsed_s=0.2)
        with self.assertRaisesRegex(ValueError, "time must be nondecreasing"):
            trace.record(x, iteration=2, elapsed_s=0)
        with self.assertRaisesRegex(ValueError, "same shape"):
            trace.record(x[:2], iteration=2, elapsed_s=0.2)
        self.assertEqual(len(trace.positions), 1)

    def test_renderer_preserves_endpoints_and_holds_the_final_frame(self):
        sys.path.insert(0, str(COURSE))
        try:
            import problem
        finally:
            sys.path.pop(0)

        param = problem.load(COURSE / "public" / "A_1.npz")
        trace = Trace("A_1.npz", "Artificial test trajectory")
        for step in range(45):
            trace.record(param.x0 + step / 44, iteration=step * 2, elapsed_s=step / 20)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trace.npz"
            output = Path(directory) / "trace.gif"
            trace.save(source)
            completed = subprocess.run(
                [sys.executable, "-B", "render_trace.py", source, "--output", output],
                cwd=ROOT, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            first, last = (problem.objective(problem.project(x, param), param)
                           for x in (trace.positions[0], trace.positions[-1]))
            self.assertIn(f"Projected objective from {first:.10g} to {last:.10g}", completed.stdout)
            with Image.open(output) as animation:
                self.assertEqual(animation.format, "GIF")
                self.assertEqual(animation.n_frames, 32)
                self.assertGreater(animation.width, animation.height)
                self.assertEqual(animation.info["duration"], 120)
                animation.seek(animation.n_frames - 1)
                self.assertEqual(animation.info["duration"], 1500)


if __name__ == "__main__":
    unittest.main()
