"""Check the visualization's shared scale, course scores, and PNG output."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from visualize import COURSE, RESULTS, ROOT, TOOLS, make_figure


sys.path.insert(0, str(COURSE))
import problem


class VisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.param = problem.load(COURSE / "public" / "A_1.npz")

    def test_panels_use_common_limits_aspect_and_residual_scale(self):
        param = self.param
        start = problem.project(param.x0, param)
        moved = problem.project(param.x0 + 0.15, param)
        figure = make_figure(param, [("Start", start), ("Moved", moved)], "A_1", show_edges=True)
        first, second, colorbar = figure.axes
        self.assertEqual(first.get_xlim(), second.get_xlim())
        self.assertEqual(first.get_ylim(), second.get_ylim())
        self.assertEqual((first.get_aspect(), second.get_aspect()), (1, 1))
        self.assertEqual(colorbar.get_ylabel(), "Absolute distance residual")

        first_edges, second_edges = first.collections[0], second.collections[0]
        self.assertIs(first_edges.norm, second_edges.norm)
        self.assertEqual(first_edges.norm.vmin, 0)
        expected_max = max(np.abs(param.residuals(x)).max() for x in (start, moved))
        self.assertAlmostEqual(first_edges.norm.vmax, expected_max)
        for axis, positions in ((first, start), (second, moved)):
            edges, sensors, anchors = axis.collections
            np.testing.assert_allclose(edges.get_array(), np.abs(param.residuals(positions)))
            np.testing.assert_allclose(sensors.get_offsets(), positions[param.free])
            np.testing.assert_allclose(anchors.get_offsets(), param.anchor_pos)
            self.assertEqual((sensors.get_label(), anchors.get_label()), ("Sensors", "Anchors"))

    def test_edges_are_optional(self):
        start = problem.project(self.param.x0, self.param)
        figure = make_figure(self.param, [("Start", start)], "A_1")
        self.assertEqual(len(figure.axes), 1)
        self.assertEqual(len(figure.axes[0].collections), 2)

    def test_cli_saves_png_and_reports_projected_score_and_invalid_output(self):
        output_name = f"test-visualize-{uuid4().hex}.png"
        output = RESULTS / output_name
        self.addCleanup(output.unlink, missing_ok=True)
        with tempfile.TemporaryDirectory() as directory:
            shifted = Path(directory) / "shifted.py"
            shifted.write_text(
                "def solver(param, max_time):\n"
                "    assert max_time == 20\n"
                "    return param.x0 + 1\n",
                encoding="utf-8",
            )
            invalid = Path(directory) / "invalid.py"
            invalid.write_text("def solver(param, max_time):\n    return [0]\n", encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "-B", "visualize.py", "A_1.npz", shifted, invalid,
                 "--run", "long", "--edges", "--output", output_name],
                cwd=TOOLS, capture_output=True, text=True, timeout=120,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        projected = problem.project(self.param.x0 + 1, self.param)
        expected = problem.objective(projected, self.param)
        self.assertIn(f"{shifted} objective {expected:.10g}", completed.stdout)
        self.assertIn("x0 objective 0.09200443128", completed.stdout)
        self.assertIn(f"{invalid} status invalid", completed.stderr)
        with Image.open(output) as figure:
            self.assertEqual(figure.format, "PNG")
            self.assertGreater(figure.width, figure.height)
            figure.verify()


if __name__ == "__main__":
    unittest.main()
