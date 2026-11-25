"""Exercise trace storage and offline rendering without an optimization solver."""

import json
import re
import subprocess
import sys
import tempfile
import unittest
from base64 import b64decode
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

from convergence_trace import Trace
from render_trace import render_trace
from visualize import COURSE, RESULTS, ROOT


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

    def test_objective_axis_preserves_values_and_uses_linear_scale_if_any_score_is_zero(self):
        problem = SimpleNamespace(
            project=lambda x, param: x,
            objective=lambda x, param: param.reference if x[1, 1] == 0.5 else x[1, 0],
        )
        original_draw = FigureCanvasAgg.draw
        cases = (
            ((0.8, 0.4, 0.2, 0.1, 0.05), 1, "log"),
            ((0.8, 0.4, 0.2, 0.1, 0.0), 1, "linear"),
            ((0.8, 0.0, 0.2, 0.1, 0.05), 1, "linear"),
            ((0.8, 0.4, 0.2, 0.1, 0.05), 0, "linear"),
        )
        for scores, reference, scale in cases:
            with self.subTest(scores=scores, reference=reference):
                param = SimpleNamespace(n=2, q=2, free=[1], anchor=[0], reference=reference)
                trace = Trace("A_1.npz")
                for i, score in enumerate(scores):
                    trace.record([[0, 0], [score, 0.25]], iteration=i, elapsed_s=i)
                axes = []

                def capture(canvas):
                    axes.append(canvas.figure.axes[1])
                    original_draw(canvas)

                with (
                    tempfile.TemporaryDirectory() as directory,
                    patch("render_trace.MAX_FRAMES", 3),
                    patch.object(FigureCanvasAgg, "draw", capture),
                    patch("builtins.print"),
                ):
                    render_trace(trace, problem, param, Path(directory) / "trace.html")

                history = axes[-1]
                self.assertEqual(history.get_yscale(), scale)
                np.testing.assert_array_equal(history.lines[0].get_ydata(), scores[::2])
                np.testing.assert_array_equal(history.lines[-1].get_ydata(), [reference, reference])
                self.assertEqual(history.lines[-1].get_label(), f"Collapsed reference {reference:g}")
                self.assertLess(history.get_ylim()[0], reference)
                self.assertGreater(history.get_ylim()[1], reference)
                ratio = f"{scores[-1] / reference:.4g}" if reference else "n/a"
                self.assertEqual(history.get_title(), f"Objective {scores[-1]:.8g}\nRatio to collapsed {ratio}")

    def test_default_html_viewer_embeds_sparse_frames_and_controls(self):
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
            source = Path(directory) / f"test-trace-{uuid4().hex}.npz"
            output = RESULTS / f"{source.stem}.html"
            self.addCleanup(output.unlink, missing_ok=True)
            trace.save(source)
            completed = subprocess.run(
                [sys.executable, "-B", "render_trace.py", source],
                cwd=ROOT, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            first, last = (problem.objective(problem.project(x, param), param)
                           for x in (trace.positions[0], trace.positions[-1]))
            self.assertIn(f"Projected objective from {first:.10g} to {last:.10g}", completed.stdout)
            html = output.read_text(encoding="utf-8")
            embedded = re.search(r'<script id="viewer-data" type="application/json">(.*?)</script>', html)
            self.assertIsNotNone(embedded)
            data = json.loads(embedded.group(1))
            self.assertEqual(len(data["frames"]), 32)
            self.assertEqual((data["frameMs"], data["finalFrameMs"]), (120, 1500))
            self.assertNotEqual(data["frames"][0], data["frames"][-1])
            for frame in data["frames"]:
                with Image.open(BytesIO(b64decode(frame, validate=True))) as plot:
                    self.assertEqual(plot.format, "PNG")
                    self.assertGreater(plot.width, plot.height)
                    plot.verify()
            for control in ("play", "previous", "next", "frame", "speed"):
                self.assertIn(f'id="{control}"', html)
            self.assertNotRegex(html, r'<(?:script|link|img)\b[^>]*\b(?:src|href)=')


if __name__ == "__main__":
    unittest.main()
