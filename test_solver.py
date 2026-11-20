"""Test your solver on the public instances, exactly as it will be run on the hidden ones.

    python test_solver.py my_solver.py                          # all public instances, short and long run
    python test_solver.py my_solver.py --only A_1.npz B_2.npz --run short

For every instance and run: the status, the time until your solver returned, the objective of the projected
solution, and the peak memory. We do not publish the values that our own solvers reach: compare your solver
with the baselines that you implement, and with your own earlier versions.

    python test_solver.py my_solver.py --compare values.json

also compares your solver with the values in a file that you wrote: the percentage of them that you match or
beat (values within 0.01% count as equal). The file has the form
{"A_1.npz": {"short": {"baseline": 1.23, "my old solver": 1.20}, "long": {...}}, ...}.
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common.points import points  # noqa: E402
from common.runner import run, value  # noqa: E402


def main():
    with open(os.path.join(HERE, "package.json")) as fh:
        cfg = json.load(fh)
    ap = argparse.ArgumentParser()
    ap.add_argument("solver")
    ap.add_argument("--only", nargs="*", help="instance files, e.g. A_1.npz")
    ap.add_argument("--run", choices=["short", "long", "both"], default="both")
    ap.add_argument("--compare", help="a JSON file with values of other solvers")
    a = ap.parse_args()
    import problem
    ours = json.load(open(a.compare)) if a.compare else {}
    files = sorted(glob.glob(os.path.join(HERE, "public", "*.npz")))
    if a.only:
        files = [f for f in files if os.path.basename(f) in a.only]
    runs = [r for r in ("short", "long") if a.run in (r, "both")]
    total = {r: [] for r in runs}
    head = f"{'instance':10s} {'run':5s} {'status':8s} {'time':>6s} {'objective':>16s}"
    print(head + (f" {'points':>6s}" if ours else "") + f" {'memory':>7s}")
    for f in files:
        name = os.path.basename(f)
        param = problem.load(f)
        for r in runs:
            res = run(a.solver, "problem", f, cfg["runs"][r], root=HERE)
            v = value(problem, param, res)
            t = f"{res['time']:.2f}" if res["time"] is not None else "-"
            obj = f"{v:16.10g}" if v is not None else f"{'-':>16s}"
            line = f"{name:10s} {r:5s} {res['status']:8s} {t:>6s} {obj}"
            if ours:
                others = list(ours.get(name, {}).get(r, {}).values())
                pts = points(v, others) if others else float("nan")
                total[r].append(pts)
                line += f" {pts:6.0f}"
            print(line + f" {res['peak_mem_mb']:5.0f}MB")
            if res["error"]:
                print(res["error"])
    for r in runs:
        if total[r]:
            print(f"{r} run (max_time {cfg['runs'][r]:g} s): average points against {a.compare} "
                  f"{sum(total[r]) / len(total[r]):.1f}")

if __name__ == "__main__":
    main()
