"""Record the PGD trace for existing scripts that use this command."""

from record_trace import main
from visualize import ROOT


if __name__ == "__main__":
    main(ROOT / "pgd.py", render=False)
