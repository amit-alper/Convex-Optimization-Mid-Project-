# Sensor-network localization

Mid-semester project for Optimization 67731, Topic 4.

## Course materials

Get your own copies from the course Moodle page. Place them in these directories at the repository root. Git ignores both directories.

- `course_package/` contains the extracted contents of `localization_package.zip`. Extract the ZIP so that `course_package/test_solver.py`, `course_package/example_solver.py`, and `course_package/public/A_1.npz` exist directly at those paths.
- `references/` holds `Project_handout.pdf`, `67731_Optimization_26_27.pdf`, and the original `localization_package.zip` for reference.

Use the course files as supplied.

## Set up and verify

Use Python 3.12. Run the commands from the repository root. NumPy is the only dependency for the supplied example.

On Linux, macOS, or WSL:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install numpy==2.5.5
.venv/bin/python -B course_package/test_solver.py course_package/example_solver.py --only A_1.npz B_1.npz --run short
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy==2.5.5
.\.venv\Scripts\python.exe -B course_package/test_solver.py course_package/example_solver.py --only A_1.npz B_1.npz --run short
```

Expect two rows with status `ok` and a finite objective. `A_1.npz` uses squared loss, and `B_1.npz` uses Huber loss. The `-B` flag keeps Python bytecode caches out of the course package.

See [AGENTS.md](AGENTS.md) for the team workflow.
