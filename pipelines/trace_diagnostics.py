"""Compare saved trace endpoints with a collapsed localization reference."""

import argparse
import sys
from pathlib import Path

import numpy as np

from pipelines.convergence_trace import Trace
from pipelines.paths import COURSE


EDGE_GROUPS = ("Free-to-free", "Free-to-anchor", "Anchor-to-anchor")


def collapsed_positions(problem, param):
    """Center the free sensors and restore the fixed anchors by projection."""
    return problem.project(np.full((param.n, param.q), 0.5), param)


def objective_ratio(objective, reference):
    return objective / reference if reference != 0 else None


def edge_loss_stats(problem, param, positions):
    """Return (count, mean loss) for each endpoint group at a feasible point."""
    residuals = param.residuals(positions)
    if param.loss == "huber":
        losses = problem.huber(residuals, param.delta)
    else:
        losses = 0.5 * residuals**2

    is_anchor = np.zeros(param.n, dtype=bool)
    is_anchor[param.anchor] = True
    anchors_per_edge = is_anchor[param.edges].sum(axis=1)
    stats = {}
    for anchors, group in enumerate(EDGE_GROUPS):
        members = anchors_per_edge == anchors
        count = int(members.sum())
        stats[group] = (count, float(losses[members].mean()) if count else None)
    return stats


def format_value(value):
    return f"{value:.10g}" if value is not None else "n/a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path, help="saved NPZ traces")
    args = parser.parse_args()

    if not (COURSE / "problem.py").is_file():
        parser.error("extract the course package into course_package/ (see README.md)")
    public = {path.name: path for path in (COURSE / "public").glob("*.npz")}
    sys.path.insert(0, str(COURSE))
    import problem

    objectives, edge_rows = [], []
    for path in args.traces:
        try:
            trace = Trace.load(path)
            if trace.instance not in public:
                parser.error(f"cannot find public instance {trace.instance} in course_package/")
            param = problem.load(public[trace.instance])
            points = (
                problem.project(trace.positions[0], param),
                collapsed_positions(problem, param),
                problem.project(trace.positions[-1], param),
            )
        except (OSError, ValueError, KeyError) as error:
            parser.error(f"cannot evaluate {path} ({error})")

        initial, reference, final = (problem.objective(x, param) for x in points)
        ratio = objective_ratio(final, reference)
        objectives.append((trace.instance, initial, reference, final, ratio))

        first_stats, reference_stats, last_stats = (edge_loss_stats(problem, param, x) for x in points)
        for group in EDGE_GROUPS:
            count, first_mean = first_stats[group]
            edge_rows.append((trace.instance, group, count, first_mean,
                              reference_stats[group][1], last_stats[group][1]))

    print("| Instance | Initial objective | Collapsed objective | Final objective | Final / collapsed |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for instance, *values in objectives:
        print(f"| {instance} | " + " | ".join(map(format_value, values)) + " |")

    print("\n| Instance | Edge group | Edges | Initial mean loss | Collapsed mean loss | Final mean loss |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for instance, group, count, *values in edge_rows:
        print(f"| {instance} | {group} | {count} | " + " | ".join(map(format_value, values)) + " |")


if __name__ == "__main__":
    main()
