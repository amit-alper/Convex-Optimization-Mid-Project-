# Sensor-network localization

Mid-semester project for Optimization 67731, Topic 4.

## Clone the repository

```sh
git clone https://github.com/amit-alper/Convex-Optimization-Mid-Project-.git
cd Convex-Optimization-Mid-Project-
```

## Course materials

## Quick start
```
python test_solver.py example_solver.py
python test_solver.py my_solver.py --only B_1.npz --run short
```

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

## Verify the starting-point solver

`solver.py` implements the course interface `solver(param, max_time)` and returns a copy of `param.x0`. Run it on all 12 public instances with both time limits from the repository root.

On Linux, macOS, or WSL:

```sh
.venv/bin/python -B course_package/test_solver.py solver.py --run both
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -B course_package/test_solver.py solver.py --run both
```

Expect 24 rows with status `ok` and finite objectives, one run per instance at each time limit (5 and 20 seconds). To check the objective values, repeat the command with `course_package/example_solver.py` in place of `solver.py`. The `objective` column should match for every instance and run.

## Create a task branch and solver file

```sh
git switch main
git pull --ff-only origin main
git switch -c feature/my-algorithm
```

Copy the starter to your own file with `cp solver.py my_algorithm.py` on Linux, macOS, or WSL, or `Copy-Item solver.py my_algorithm.py` in PowerShell. Edit `my_algorithm.py` and keep the function signature `solver(param, max_time)`. Return an array of shape `(param.n, param.q)` that includes the anchors. Use a distinct branch and filename for each task. Your solver file can run directly through the benchmark.

## Run and compare solvers

On Linux, macOS, or WSL, run the following from the repository root.

```sh
.venv/bin/python -B benchmark.py solver.py my_algorithm.py --only A_1.npz B_1.npz --run both --output results/my-algorithm.csv
```

In Windows PowerShell, run the following.

```powershell
.\.venv\Scripts\python.exe -B benchmark.py solver.py my_algorithm.py --only A_1.npz B_1.npz --run both --output results/my-algorithm.csv
```

Pass one or more solver files before the options. Omit `--only` to run all public instances. Use `--run short` for 5 seconds, `--run long` for 20 seconds, or `--run both` for both budgets. Both is the default.

The terminal shows the projected objectives side by side for each instance and budget. Lower is better. Open the CSV to compare `objective`, `runtime_s`, `peak_memory_mb`, and `status` for the same instance and budget. The official runner measures the solver call and projects the returned positions before scoring. A failed trial has a blank objective. A status of `invalid` means the solver returned on time but produced no finite projected objective.

The CSV also records the runner seed (0), Git commit and working-tree state, and Python, NumPy, and platform versions. Use a fresh CSV name for each run. Git ignores `results/`.

## Benchmark a standalone initializer

Copy `solver.py` to `my_initializer.py` with `cp solver.py my_initializer.py` or `Copy-Item solver.py my_initializer.py` in PowerShell. In `solver(param, max_time)`, return the initialized coordinates as soon as initialization finishes. Compare them with the supplied random `param.x0` using the same instance and budget.

```sh
.venv/bin/python -B benchmark.py solver.py my_initializer.py --only A_1.npz B_1.npz --run short --output results/initialization.csv
```

In PowerShell, start the same command with `.\.venv\Scripts\python.exe`. The initializer's `objective` measures the quality of its starting positions, and `runtime_s` measures the time it spent computing them.

## Test and submit a pull request

With the course files in place, run the automated checks on Linux, macOS, or WSL.

```sh
.venv/bin/python -B -m unittest discover -s tests -v
```

In Windows PowerShell, run the following.

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

For the example algorithm branch above, commit and push your solver file.

```sh
git add -- my_algorithm.py
git commit -m "Add my localization solver"
git push -u origin feature/my-algorithm
```

Open the pull request link printed by `git push`. If there is no link, open the repository on GitHub and choose **Pull requests** and then **New pull request**. Set the base branch to `main` and the compare branch to `feature/my-algorithm`. Include the exact benchmark command, the results you observed, and any failed trials, then choose **Create pull request**. Ask another teammate to reproduce the checks and review the change before merging. [Team workflow](AGENTS.md)
