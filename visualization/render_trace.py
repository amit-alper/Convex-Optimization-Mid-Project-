"""Render a saved localization trace with official projected scores."""

import argparse
import json
import sys
from base64 import b64encode
from io import BytesIO
from pathlib import Path
from textwrap import fill

import numpy as np

from pipelines.convergence_trace import Trace
from pipelines.paths import COURSE, RESULTS
from pipelines.trace_diagnostics import collapsed_positions, objective_ratio


MAX_FRAMES = 32
FRAME_MS = 120
FINAL_FRAME_MS = 1500


def write_viewer(frames, destination):
    images = []
    for frame in frames:
        with BytesIO() as buffer:
            frame.save(buffer, format="PNG", optimize=True)
            images.append(b64encode(buffer.getvalue()).decode("ascii"))
    data = json.dumps({"frames": images, "frameMs": FRAME_MS, "finalFrameMs": FINAL_FRAME_MS})
    template = Path(__file__).with_name("trace_viewer.html").read_text(encoding="utf-8")
    destination.write(template.replace("__VIEWER_DATA__", data).encode("utf-8"))


def render_trace(trace, problem, param, output, *, baselines=()):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter, NullFormatter
    from PIL import Image

    count = len(trace.positions)
    indices = np.linspace(0, count - 1, min(count, MAX_FRAMES), dtype=int)
    frame_indices = set(indices)
    positions, scores = [], []
    for i, snapshot in enumerate(trace.positions):
        projected = problem.project(snapshot, param)
        scores.append(problem.objective(projected, param))
        if i in frame_indices:
            positions.append(projected)
    reference = problem.objective(collapsed_positions(problem, param), param)
    if not np.isfinite(scores).all() or not np.isfinite(reference):
        raise ValueError("the trace or collapsed reference has a nonfinite objective")
    references = [("Collapsed reference", reference), *baselines]
    if any(not np.isfinite(value) or value < 0 for _, value in references):
        raise ValueError("reference objectives must be finite and nonnegative")
    objectives = [scores[i] for i in indices]
    elapsed = [trace.elapsed_s[i] for i in indices]
    if reference > 0:
        plotted_scores = [100 * (score / reference) for score in scores]
        references = [(label, 100 * (value / reference)) for label, value in references]
    else:
        plotted_scores = scores
    plotted_objectives = [plotted_scores[i] for i in indices]

    figure = Figure(figsize=(9.2, 4.2), dpi=105)
    canvas = FigureCanvasAgg(figure)
    spatial, history = figure.subplots(1, 2)
    figure.subplots_adjust(left=0.07, right=0.97, bottom=0.19, top=0.78, wspace=0.32)
    figure.suptitle(fill(f"{trace.label} | {trace.instance}", width=80), y=0.96, weight="semibold")

    spatial.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), xlabel="x", ylabel="y",
                xticks=(0, 0.25, 0.5, 0.75, 1), yticks=(0, 0.25, 0.5, 0.75, 1))
    spatial.set_aspect("equal", adjustable="box")
    sensors = spatial.scatter(*positions[0][param.free, :2].T, s=9 if param.n > 200 else 24,
                              c="#284869", linewidths=0, label="Sensors")
    spatial.scatter(*positions[0][param.anchor, :2].T, s=60, marker="D", c="#ca4636",
                    edgecolors="white", linewidths=0.8, label="Anchors", zorder=3)
    spatial.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2, frameon=False)

    visible_values = plotted_scores + [value for _, value in references]
    low, high = min(visible_values), max(visible_values)
    if low > 0:
        history.set_yscale("log")
        limits = (low / 1.1, high * 1.1)
    else:
        padding = (high - low) * 0.08 or max(abs(high) * 0.03, 1e-12)
        limits = (low - padding, high + padding)
    duration = elapsed[-1] or 1
    history.set(xlim=(-0.02 * duration, 1.02 * duration), ylim=limits,
                xlabel="Elapsed time (s)",
                ylabel="Objective (% of collapsed reference)" if reference > 0 else "Official objective")
    if reference > 0:
        history.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
        history.yaxis.set_minor_formatter(NullFormatter())
    history.grid(alpha=0.25)
    line, = history.plot([], [], color="#284869", linewidth=1.6)
    current, = history.plot([], [], "o", color="#ca4636", markersize=5)
    colors = ("#596d80", "#776126", "#347762")
    styles = ("--", "-.", ":")
    unit = "%" if reference > 0 else ""
    for i, (label, value) in enumerate(references):
        history.axhline(value, color=colors[i % len(colors)], linestyle=styles[i % len(styles)],
                        linewidth=1.2, label=f"{label} {value:.5g}{unit}")
    history.legend(loc="upper right", fontsize="small", frameon=False)

    frames = []
    for frame, (index, x, time, objective) in enumerate(zip(indices, positions, elapsed, objectives)):
        sensors.set_offsets(x[param.free, :2])
        spatial.set_title(f"Iteration {trace.iterations[index]:,}  |  {time:.3f} s")
        ratio = objective_ratio(objective, reference)
        percent = f"{100 * ratio:.4g}%" if ratio is not None else "n/a"
        history.set_title(f"Objective {objective:.8g}\nOf collapsed reference {percent}")
        line.set_data(elapsed[:frame + 1], plotted_objectives[:frame + 1])
        current.set_data([time], [plotted_objectives[frame]])
        canvas.draw()
        image = Image.fromarray(np.asarray(canvas.buffer_rgba()))
        frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as destination:
        if output.suffix.lower() == ".gif":
            durations = [FRAME_MS] * (len(frames) - 1) + [FINAL_FRAME_MS]
            frames[0].save(destination, format="GIF", save_all=True, append_images=frames[1:],
                           duration=durations, loop=0)
        else:
            write_viewer(frames, destination)
    print(f"Saved {len(frames)} frames to {output}")
    print(f"Projected objective from {objectives[0]:.10g} to {objectives[-1]:.10g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="saved NPZ trace")
    parser.add_argument("--output", type=Path, help="new HTML or GIF path (defaults to HTML in results/)")
    parser.add_argument("--baseline", nargs=2, metavar=("LABEL", "OBJECTIVE"), action="append",
                        default=[], help="add a measured benchmark objective to the plot")
    args = parser.parse_args()
    output = args.output or RESULTS / f"{args.trace.stem}.html"
    if output.suffix.lower() not in (".html", ".gif") or output.exists():
        parser.error("choose a new .html or .gif output path")

    try:
        trace = Trace.load(args.trace)
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"cannot load the trace ({error})")
    public = {path.name: path for path in (COURSE / "public").glob("*.npz")}
    if trace.instance not in public or not (COURSE / "problem.py").is_file():
        parser.error(f"cannot find public instance {trace.instance} in course_package/ (see README.md)")

    sys.path.insert(0, str(COURSE))
    import problem

    try:
        baselines = [(label, float(value)) for label, value in args.baseline]
        render_trace(trace, problem, problem.load(public[trace.instance]), output, baselines=baselines)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
