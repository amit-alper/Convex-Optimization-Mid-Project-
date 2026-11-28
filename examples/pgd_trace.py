"""Record the PGD trace for existing scripts that use this command."""

from pipelines.paths import SOLVERS
from pipelines.record_trace import main


if __name__ == "__main__":
    main(SOLVERS / "pgd.py", render=False)
