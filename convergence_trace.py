"""Record selected iterates and save them independently of a solver."""

import math
from pathlib import Path

import numpy as np


class Trace:
    def __init__(self, instance, label="Optimization trace"):
        self.instance = Path(instance).name
        self.label = str(label)
        self.positions = []
        self.iterations = []
        self.elapsed_s = []

    def record(self, positions, *, iteration, elapsed_s):
        """Copy a selected iterate so subsequent in-place updates are safe."""
        if not isinstance(iteration, (int, np.integer)) or iteration < 0:
            raise ValueError("iteration must be a nonnegative integer")
        if self.iterations and iteration <= self.iterations[-1]:
            raise ValueError("iterations must increase")
        elapsed_s = float(elapsed_s)
        if not math.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError("elapsed time must be finite and nonnegative")
        if self.elapsed_s and elapsed_s < self.elapsed_s[-1]:
            raise ValueError("elapsed time must be nondecreasing")

        snapshot = np.array(positions, dtype=float, copy=True)
        if snapshot.ndim != 2 or not snapshot.size or not np.isfinite(snapshot).all():
            raise ValueError("positions must be a nonempty finite two-dimensional array")
        if self.positions and snapshot.shape != self.positions[0].shape:
            raise ValueError("all snapshots must have the same shape")
        self.positions.append(snapshot)
        self.iterations.append(int(iteration))
        self.elapsed_s.append(elapsed_s)

    def save(self, path):
        """Write the label, instance filename, and snapshots to a new NPZ file."""
        if not self.positions:
            raise ValueError("record at least one snapshot before saving")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            np.savez_compressed(
                output, instance=self.instance, label=self.label,
                positions=np.stack(self.positions), iterations=self.iterations, elapsed_s=self.elapsed_s,
            )

    @classmethod
    def load(cls, path):
        """Load a trace without the solver or course package."""
        with np.load(path, allow_pickle=False) as data:
            trace = cls(data["instance"].item(), data["label"].item())
            for x, iteration, elapsed in zip(
                data["positions"], data["iterations"], data["elapsed_s"], strict=True,
            ):
                trace.record(x, iteration=iteration, elapsed_s=elapsed)
        if not trace.positions:
            raise ValueError("the trace has no snapshots")
        return trace
