"""Run independent solver files on the course's public instances."""

import argparse
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COURSE = ROOT / "course_package"
SEED = 0
FIELDS = (
    "solver", "instance", "run", "budget_s", "seed", "status", "objective",
    "runtime_s", "peak_memory_mb", "error", "git_commit", "git_dirty",
    "python_version", "numpy_version", "platform",
)


def git_metadata():
    """Record the checkout, including uncommitted and untracked changes."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        changes = subprocess.check_output(
            ["git", "--no-optional-locks", "status", "--porcelain", "--untracked-files=normal"],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "", "git_dirty": ""}
    return {"git_commit": commit, "git_dirty": bool(changes)}


def benchmark(solvers, instances, budgets, output, checkout):
    # The supplied modules use top-level imports, just as in test_solver.py.
    sys.path.insert(0, str(COURSE))
    import numpy
    import problem
    from common.runner import run, value

    metadata = {
        **checkout,
        "python_version": platform.python_version(),
        "numpy_version": numpy.__version__,
        "platform": platform.platform(),
    }
    writer = csv.DictWriter(output, fieldnames=FIELDS)
    writer.writeheader()
    widths = [max(16, len(str(solver))) for solver in solvers]
    print(f"{'instance':<12} {'run':<5} {'budget_s':>8}",
          *(f"{str(solver):>{width}}" for solver, width in zip(solvers, widths)), flush=True)

    for instance in instances:
        param = problem.load(instance)
        for run_name, budget in budgets:
            comparisons = []
            for solver in solvers:
                result = run(solver, "problem", instance, budget, root=COURSE, seed=SEED)
                objective = value(problem, param, result)
                status = result["status"]
                error = result["error"]
                if status == "ok" and objective is None:
                    status = "invalid"
                    error = error or "no finite projected objective"
                writer.writerow({
                    "solver": str(solver),
                    "instance": instance.name,
                    "run": run_name,
                    "budget_s": budget,
                    "seed": SEED,
                    "status": status,
                    "objective": objective,
                    "runtime_s": result["time"],
                    "peak_memory_mb": result["peak_mem_mb"],
                    "error": error,
                    **metadata,
                })
                output.flush()
                comparisons.append(f"{objective:.10g}" if objective is not None else status)
            print(f"{instance.name:<12} {run_name:<5} {budget:>8g}",
                  *(f"{cell:>{width}}" for cell, width in zip(comparisons, widths)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solvers", nargs="+", type=Path, help="solver .py files to compare")
    parser.add_argument("--only", nargs="+", metavar="INSTANCE", help="public instance filenames")
    parser.add_argument("--run", choices=("short", "long", "both"), default="both",
                        help="course time limits (default: both)")
    parser.add_argument("--output", required=True, type=Path, metavar="CSV",
                        help="path for a new CSV file")
    args = parser.parse_args()

    for solver in args.solvers:
        if solver.suffix != ".py" or not solver.is_file():
            parser.error(f"expected an existing solver .py file, got {solver}")
    if len({solver.resolve() for solver in args.solvers}) != len(args.solvers):
        parser.error("pass each solver file once")

    available = {path.name: path for path in sorted((COURSE / "public").glob("*.npz"))}
    config = COURSE / "package.json"
    required = (config, COURSE / "problem.py", COURSE / "common" / "runner.py")
    if not available or not all(path.is_file() for path in required):
        parser.error("extract the full course package into course_package/ (see README.md)")
    names = list(dict.fromkeys(args.only)) if args.only else list(available)
    unknown = [name for name in names if name not in available]
    if unknown:
        parser.error(f"unknown public instance(s) {', '.join(unknown)}")

    limits = json.loads(config.read_text(encoding="utf-8"))["runs"]
    budgets = [(name, limits[name]) for name in ("short", "long") if args.run in (name, "both")]
    checkout = git_metadata()
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output = args.output.open("x", encoding="utf-8", newline="")
    except OSError as error:
        parser.error(f"cannot create {args.output} ({error.strerror})")

    with output:
        benchmark(args.solvers, [available[name] for name in names], budgets, output, checkout)
    print(f"CSV written to {args.output}")


if __name__ == "__main__":
    main()
