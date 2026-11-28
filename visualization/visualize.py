"""Save a side-by-side PNG of a public instance and its solver results."""

import argparse
import json
import sys
from pathlib import Path
from textwrap import fill

from pipelines.paths import COURSE, RESULTS


def collect_panels(problem, param, instance, solvers, budget):
    from common.runner import run, value

    start = problem.project(param.x0, param)
    start_value = problem.objective(start, param)
    panels = [(f"x0 (random start)\nObjective {start_value:.10g}", start)]
    print(f"x0 objective {start_value:.10g}")
    for solver in solvers:
        result = run(solver, "problem", instance, budget, root=COURSE)
        objective = value(problem, param, result)
        label = fill(str(solver), width=36)
        if objective is None:
            status = "invalid" if result["status"] == "ok" else result["status"]
            panels.append((f"{label}\nStatus {status}", None))
            print(f"{solver} status {status}", file=sys.stderr)
            if result["error"]:
                print(result["error"], file=sys.stderr)
            continue
        # The runner scores projected positions. Plot those same positions.
        positions = problem.project(result["x"], param)
        panels.append((f"{label}\nObjective {objective:.10g}", positions))
        print(f"{solver} objective {objective:.10g}")
    return panels


def make_figure(param, panels, title, show_edges=False):
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure

    residuals = [None] * len(panels)
    if show_edges:
        residuals = [np.abs(param.residuals(x)) if x is not None else None for _, x in panels]
        maximum = max((r.max(initial=0) for r in residuals if r is not None), default=0)
        scale = Normalize(vmin=0, vmax=maximum or 1)

    width = 4.1 * len(panels) + (0.8 if show_edges else 0)
    figure = Figure(figsize=(width, 4.6), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, len(panels), sharex=True, sharey=True, squeeze=False)[0]
    figure.suptitle(title)
    dense = len(param.edges) > 2000
    for axis, (label, positions), errors in zip(axes, panels, residuals):
        axis.set(
            title=label, xlabel="x", xlim=(-0.03, 1.03), ylim=(-0.03, 1.03),
            xticks=(0, 0.25, 0.5, 0.75, 1), yticks=(0, 0.25, 0.5, 0.75, 1),
        )
        axis.set_aspect("equal", adjustable="box")
        if positions is None:
            axis.text(0.5, 0.5, "No valid solution", ha="center", va="center", transform=axis.transAxes)
            continue

        xy = positions[:, :2]
        if errors is not None:
            edges = LineCollection(
                xy[param.edges], array=errors, cmap="viridis", norm=scale,
                linewidths=0.24 if dense else 0.7, alpha=0.16 if dense else 0.62, zorder=1,
            )
            axis.add_collection(edges)
        axis.scatter(*xy[param.free].T, s=9 if dense else 24, c="#284869",
                     linewidths=0, label="Sensors", zorder=2)
        axis.scatter(*xy[param.anchor].T, s=60, marker="D", c="#ca4636",
                     edgecolors="white", linewidths=0.8, label="Anchors", zorder=3)

    axes[0].set_ylabel("y")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
    if show_edges:
        figure.colorbar(ScalarMappable(norm=scale, cmap="viridis"), ax=list(axes),
                        label="Absolute distance residual")
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", help="public instance filename, e.g. A_1.npz")
    parser.add_argument("solvers", nargs="+", type=Path, help="solver .py files")
    parser.add_argument("--run", choices=("short", "long"), default="short",
                        help="course time limit (default: short)")
    parser.add_argument("--edges", action="store_true", help="color edges by absolute distance residual")
    parser.add_argument("--output", type=Path, metavar="NAME.png", help="PNG filename inside results/")
    args = parser.parse_args()

    config = COURSE / "package.json"
    required = (config, COURSE / "problem.py", COURSE / "common" / "runner.py")
    if not all(path.is_file() for path in required):
        parser.error("extract the full course package into course_package/ (see README.md)")
    public = {path.name: path for path in (COURSE / "public").glob("*.npz")}
    if args.instance not in public:
        parser.error(f"unknown public instance {args.instance}")
    instance = public[args.instance]
    for solver in args.solvers:
        if solver.suffix != ".py" or not solver.is_file():
            parser.error(f"expected an existing solver .py file, got {solver}")

    edge_suffix = "-edges" if args.edges else ""
    name = args.output or Path(f"{instance.stem}-{args.run}{edge_suffix}.png")
    if name.name != str(name) or name.suffix.lower() != ".png":
        parser.error("--output expects a PNG filename inside results/")
    output = RESULTS / name
    if output.exists():
        parser.error(f"{output} already exists, choose a new --output filename")

    sys.path.insert(0, str(COURSE))
    import problem

    budget = json.loads(config.read_text(encoding="utf-8"))["runs"][args.run]
    param = problem.load(instance)
    panels = collect_panels(problem, param, instance, args.solvers, budget)
    title = f"{instance.name}  |  {param.n:,} nodes  |  {budget:g} s budget"
    figure = make_figure(param, panels, title, show_edges=args.edges)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as destination:
            figure.savefig(destination, format="png", dpi=150)
    except OSError as error:
        parser.error(f"cannot save {output} ({error.strerror})")
    print(f"PNG written to {output}")


if __name__ == "__main__":
    main()
