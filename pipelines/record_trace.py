"""Record a solver's diagnostic trace and render its interactive HTML viewer."""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from pipelines.paths import COURSE, RESULTS


def main(default_solver=None, *, render=True):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", help="public instance filename")
    if default_solver is None:
        parser.add_argument("solver", type=Path, help="Python file providing solve(param, max_time, record=...)")
    parser.add_argument("--run", choices=("short", "long"), default="short")
    parser.add_argument("--output", type=Path, help="new NPZ path (defaults to results/)")
    args = parser.parse_args()

    public = {path.name: path for path in (COURSE / "public").glob("*.npz")}
    if args.instance not in public:
        parser.error(f"cannot find public instance {args.instance} in course_package/")
    instance = public[args.instance]
    solver_file = default_solver or args.solver
    if solver_file.suffix != ".py" or not solver_file.is_file():
        parser.error(f"expected an existing solver .py file, got {solver_file}")

    output = args.output or RESULTS / f"{solver_file.stem}-{instance.stem}-{args.run}.npz"
    viewer = output.with_suffix(".html")
    if output.suffix.lower() != ".npz" or output.exists() or (render and viewer.exists()):
        parser.error("choose a new .npz output path with no existing outputs")

    sys.path.insert(0, str(COURSE))
    from common.runner import SINGLE_THREAD_ENV

    os.environ.update(SINGLE_THREAD_ENV)
    import numpy as np
    import problem
    from pipelines.convergence_trace import Trace
    from visualization.render_trace import render_trace

    sys.path.insert(0, str(solver_file.resolve().parent))
    spec = importlib.util.spec_from_file_location("diagnostic_solver", solver_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    solve = getattr(module, "solve", None)
    if not callable(solve):
        parser.error(f"{solver_file} needs solve(param, max_time, *, record=None) to record a trace")

    budget = json.loads((COURSE / "package.json").read_text(encoding="utf-8"))["runs"][args.run]
    trace = Trace(instance.name, f"{solver_file.stem} ({budget:g} s diagnostic)")
    param = problem.load(instance)
    np.random.seed(0)
    solve(param, budget, record=trace.record)
    try:
        trace.save(output)
        print(f"Saved {len(trace.positions)} snapshots to {output}")
        if render:
            render_trace(trace, problem, param, viewer)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
