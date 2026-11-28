"""Repository locations shared by the command-line tools and tests."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COURSE = ROOT / "course_package"
RESULTS = ROOT / "results"
SOLVERS = ROOT / "solvers"
