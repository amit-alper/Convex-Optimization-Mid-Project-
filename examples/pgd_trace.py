"""Record a separate PGD diagnostic run for a public instance."""

import argparse
import json
import os
import sys
from pathlib import Path

from visualize import COURSE, RESULTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", help="public instance filename")
    parser.add_argument("--run", choices=("short", "long"), default="short")
    parser.add_argument("--output", type=Path, help="new NPZ path (defaults to results/)")
    args = parser.parse_args()

    public = {path.name: path for path in (COURSE / "public").glob("*.npz")}
    if args.instance not in public:
        parser.error(f"cannot find public instance {args.instance} in course_package/")
    instance = public[args.instance]
    output = args.output or RESULTS / f"pgd-{instance.stem}-{args.run}.npz"
    if output.suffix.lower() != ".npz" or output.exists():
        parser.error("choose a new .npz output path")

    sys.path.insert(0, str(COURSE))
    from common.runner import SINGLE_THREAD_ENV

    os.environ.update(SINGLE_THREAD_ENV)
    import problem
    from convergence_trace import Trace
    from pgd import run_pgd

    budget = json.loads((COURSE / "package.json").read_text(encoding="utf-8"))["runs"][args.run]
    trace = Trace(instance.name, f"Projected gradient descent ({budget:g} s diagnostic)")
    run_pgd(problem.load(instance), budget, record=trace.record)
    trace.save(output)
    print(f"Saved {len(trace.positions)} snapshots to {output}")


if __name__ == "__main__":
    main()
