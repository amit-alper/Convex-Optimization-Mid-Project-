"""Write an artificial trace with prescribed positions and elapsed times."""

import argparse
import sys
from pathlib import Path

from convergence_trace import Trace
from visualize import COURSE, RESULTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "artificial-trace.npz")
    args = parser.parse_args()

    sys.path.insert(0, str(COURSE))
    import problem

    instance = "A_1.npz"
    param = problem.load(COURSE / "public" / instance)
    trace = Trace(instance, "Artificial trajectory (synthetic iterations and time)")
    for step in range(31):
        x = param.x0.copy()
        scale = 1 - 0.65 * (step / 30)
        x[param.free] = 0.5 + scale * (x[param.free] - 0.5)
        trace.record(x, iteration=10 * step, elapsed_s=step / 10)
    trace.save(args.output)
    print(f"Saved {len(trace.positions)} artificial snapshots to {args.output}")


if __name__ == "__main__":
    main()
