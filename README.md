# Sensor-network localization

Mid-semester project for Optimization 67731, Topic 4.

## Quick Start

Use Git and Python 3.12. In PowerShell, `python --version` should show Python 3.12. A clone contains the supplied runner and public instances under `course_package/`. [Course package README](course_package/README.md) · [Project handout](Project_handout.pdf)

### WSL or Linux

```sh
git clone --branch main https://github.com/amit-alper/Convex-Optimization-Mid-Project-.git
cd Convex-Optimization-Mid-Project-
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
git switch -c feature/my-algorithm
cp example_initialized_pgd.py my_algorithm.py
.venv/bin/python -B course_package/test_solver.py course_package/example_solver.py --only A_1.npz B_1.npz --run short
.venv/bin/python -B benchmark.py solver.py my_algorithm.py --only A_1.npz --run short --output results/my-algorithm.csv
.venv/bin/python -B visualize.py A_1.npz my_algorithm.py --run short --output my-algorithm.png
.venv/bin/python -B record_trace.py A_1.npz my_algorithm.py --run short --output results/my-algorithm.npz
.venv/bin/python -B trace_diagnostics.py results/my-algorithm.npz > results/my-algorithm-diagnostics.md
.venv/bin/python -B -m unittest discover -s tests -v
```

### Windows PowerShell

```powershell
git clone --branch main https://github.com/amit-alper/Convex-Optimization-Mid-Project-.git
cd Convex-Optimization-Mid-Project-
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
git switch -c feature/my-algorithm
Copy-Item example_initialized_pgd.py my_algorithm.py
.\.venv\Scripts\python.exe -B course_package/test_solver.py course_package/example_solver.py --only A_1.npz B_1.npz --run short
.\.venv\Scripts\python.exe -B benchmark.py solver.py my_algorithm.py --only A_1.npz --run short --output results/my-algorithm.csv
.\.venv\Scripts\python.exe -B visualize.py A_1.npz my_algorithm.py --run short --output my-algorithm.png
.\.venv\Scripts\python.exe -B record_trace.py A_1.npz my_algorithm.py --run short --output results/my-algorithm.npz
.\.venv\Scripts\python.exe -B trace_diagnostics.py results/my-algorithm.npz | Set-Content -Encoding UTF8 results/my-algorithm-diagnostics.md
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

The course smoke test should show `ok` for `A_1.npz` and `B_1.npz`. Open `results/my-algorithm.html` in a browser, and look in `results/` for the CSV, PNG, NPZ, and diagnostic report. Edit `my_algorithm.py` on your task branch to develop your solver. Use fresh output names on repeat runs. The `-B` flag keeps Python bytecode caches out of the course package.

For another task in an existing checkout, run `git switch main` and `git pull --ff-only origin main`, then create a fresh task branch and solver file.

## Shared tools and solver interfaces

| Role | Files |
| --- | --- |
| Course tools and data | `course_package/` contains the supplied runner, official objective, projection, and public instances. Use these files as supplied. |
| Shared tools | `benchmark.py` compares solver files, `visualize.py` creates static plots, `record_trace.py` saves traces and renders HTML, `render_trace.py` redraws saved traces, and `trace_diagnostics.py` reports scores and edge losses. `convergence_trace.py` holds the shared trace format. |
| Algorithms and examples | `solver.py` is the supplied-start control, `pgd.py` is the PGD algorithm, and `example_initialized_pgd.py` demonstrates an initializer and optimizer working together. Put your method in a separate Python file at the repository root. |
| Course baselines | `mds_pgd.py` runs PGD from classical MDS, and `levenberg_marquardt.py` runs Levenberg–Marquardt from `x0`. Both follow the interface below. |
| Shared solver library | `core/derivatives.py` (`Model`: residuals, gradient, sparse Jacobian, Huber weights), `core/mds.py` (MDS-MAP steps), and `core/initialization/` (starting points, one file per family). Root solver files import `core` directly, because the runner puts the solver file's folder on the import path. |

- **Initializer** `initialize(param, max_time)` returns a finite positions array of shape `(param.n, param.q)`, including anchors. Read `param` without mutating it and return within `max_time` seconds.
- **Optimizer** `solve(param, max_time, *, record=None)` returns the same shape and honors the same deadline. Optimizers that accept a supplied start can take an optional `x0` keyword. For example, `run_pgd(param, remaining, x0=positions)` projects and copies the supplied start. Omitting `x0` uses `param.x0` as before.
- **Official entry point** `solver(param, max_time)` is what the grader, benchmark, and static visualization call. For a trace-compatible solver, it can simply return `solve(param, max_time)`. Diagnostic recording is enabled only when `record_trace.py` supplies a callback.

The example initializer uses the existing collapsed reference. The example solver gives initialization up to one fifth of the total budget, subtracts the time actually used, and passes the remaining time and the returned positions to PGD. Its diagnostic timestamps include initialization time. Copy the example and replace its initializer or optimizer to try your own method.

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

Save this starter as `my_initializer.py` and replace `initialize` with your method. The benchmark calls `solver`, which returns the initialized coordinates as soon as initialization finishes.

```python
def initialize(param, max_time):
    return param.x0.copy()


def solver(param, max_time):
    return initialize(param, max_time)
```

Compare the result with the supplied random `param.x0` using the same instance and budget.

```sh
.venv/bin/python -B benchmark.py solver.py my_initializer.py --only A_1.npz B_1.npz --run short --output results/initialization.csv
```

In PowerShell, start the same command with `.\.venv\Scripts\python.exe`. The initializer's `objective` measures the quality of its starting positions, and `runtime_s` measures the time it spent computing them.

## Visualize a public instance

With the Quick Start environment, run these examples from the repository root and open the PNGs in `results/`.

```sh
.venv/bin/python -B visualize.py A_1.npz solver.py --run short --edges --output visualization-A_1.png
.venv/bin/python -B visualize.py C_1.npz solver.py --run long --edges --output visualization-C_1.png
```

In PowerShell, start the same commands with `.\.venv\Scripts\python.exe`.

Pass one public instance filename, followed by one or more solver `.py` files. For example, add `my_algorithm.py` after `solver.py` to see both results beside `x0`. Use `--run short` for 5 seconds (the default) or `--run long` for 20 seconds. Use `--edges` to add measured edges colored by absolute distance residual. All panels share axis limits, equal aspect ratios, and a single residual color scale when edges are shown.

The circles are sensors and the diamonds are anchors. The tool scores and plots each solver's projected result, and a failed trial shows its status. The course runner times the solver call. Plotting and saving happen afterward. Use a fresh PNG filename with `--output` or omit it to use the default instance-and-budget name, with `-edges` appended when requested.

## Record and animate an optimization trace

Give your solver file a `solve(param, max_time, *, record=None)` function and have the official `solver` call it without a callback. Use `record_trace.py INSTANCE SOLVER --run short` to run it separately for diagnostics. This command saves an NPZ and renders a standalone HTML viewer with the same filename stem. Use `--output` with a new `.npz` path to choose where both files go.

To record from Python, pass a recorder and save the trace after the algorithm returns. Here `param` is the course instance you loaded and `solve` is your own function.

```python
from convergence_trace import Trace

trace = Trace("A_1.npz", "My diagnostic run")
solve(param, 20, record=trace.record)
trace.save("results/my-trace.npz")
```

Inside the algorithm, start a `time.perf_counter()` timer. At a selected iteration, call `record(x, iteration=k, elapsed_s=time.perf_counter() - started)` if `record` is present. Pass the full `(param.n, param.q)` position array. Aim for roughly 30 snapshots, including the initial and final positions once each. The recorder copies each selected array, so the algorithm can continue updating it in place. `Trace.load("results/my-trace.npz")` reads the arrays, iterations, time stamps, and label independently of the solver.

The artificial example below uses prescribed positions and synthetic timing. The second command reads the saved NPZ, projects and scores the snapshots with the course package, and writes a self-contained HTML viewer. It keeps the first and final snapshots and uses at most 32 frames. The right panel plots the objective as a percentage of the collapsed reference on a logarithmic Y-axis, with the raw objective and percentage in the title. The dashed reference line marks 100%. Use `--baseline LABEL OBJECTIVE` once per measured baseline to add labeled lines from raw objectives. The viewer uses a linear axis when any snapshot or plotted reference scores zero. If the collapsed reference itself is zero, the Y-axis shows raw objectives and the percentage is `n/a`.

```sh
.venv/bin/python -B -m examples.artificial_trace
.venv/bin/python -B render_trace.py results/artificial-trace.npz
```

Open `results/artificial-trace.html` directly in a modern browser. The file embeds the frames and controls and works offline. Use **Play/Pause**, **Previous/Next**, the left and right arrow keys, or the **Frame** slider. The **Speed** menu ranges from 0.25× to 4×. Playback holds the final frame for 1.5 seconds at normal speed before looping.

In PowerShell, start the same commands with `.\.venv\Scripts\python.exe`. Use the Matplotlib environment above. For repeat runs, choose a fresh trace or viewer path with `--output`. An explicit `.gif` output path still exports a GIF.

## Projected gradient descent experiment

`pgd.py` starts at the supplied `x0`. For each measured edge $(i,j)$, let $r_{ij}=\lVert x_i-x_j\rVert_2-d_{ij}$. The objective is the mean of the official losses $\rho(r_{ij})$ over all $m$ edges. The derivative $\rho'(r)$ is $r$ for the squared loss and $\operatorname{clip}(r,-\delta,\delta)$ for Huber. For an edge of nonzero length, its contribution to the first endpoint is

$$
\frac{\rho'(r_{ij})}{m}\frac{x_i-x_j}{\lVert x_i-x_j\rVert_2}.
$$

Subtract the same contribution at the second endpoint. At coincident endpoints, the norm has no unique gradient direction, so the code uses zero for that edge's contribution. Anchor gradient entries are zero because anchors are fixed.

For each step size $\alpha$, form $y=P(x-\alpha g)$ with the official projection. Accept $y$ if $f(y)\le f(x)+10^{-4}\langle g,y-x\rangle$ and the computed objective strictly decreases. Halve $\alpha$ after a rejected trial. The first trial uses $0.1/\max_{i,k}|g_{ik}|$, which limits the largest coordinate move before projection to 0.1. Later trials start at twice the previous accepted step, capped by the same gradient-scaled limit. The solver checks a deadline at 95% of the supplied budget, including around trial evaluations, and returns the last accepted feasible iterate.

Run the short-budget comparison from the repository root.

```sh
.venv/bin/python -B benchmark.py solver.py pgd.py --only A_1.npz B_1.npz C_1.npz D_1.npz --run short --output results/pgd-short.csv
```

Generate the static `x0` versus PGD plots and the interactive convergence viewers. After recording the start and first accepted step, the diagnostic recorder multiplies each sampled iteration number by 1.25 and rounds up to choose the next snapshot. It also records the final accepted iterate. Official benchmarks use `solver(param, max_time)` with recording disabled. Each plot or trace runs PGD separately, so its final objective can vary with the number of steps completed in time.

```sh
for instance in A_1 B_1 C_1 D_1; do
  .venv/bin/python -B visualize.py "${instance}.npz" pgd.py --run short --output "pgd-${instance}-comparison.png"
  .venv/bin/python -B record_trace.py "${instance}.npz" pgd.py --run short
done
```

In PowerShell, run the benchmark command with `.\.venv\Scripts\python.exe` and use this loop for the plots and traces.

```powershell
foreach ($instance in 'A_1', 'B_1', 'C_1', 'D_1') {
    .\.venv\Scripts\python.exe -B visualize.py "${instance}.npz" pgd.py --run short --output "pgd-${instance}-comparison.png"
    .\.venv\Scripts\python.exe -B record_trace.py "${instance}.npz" pgd.py --run short
}
```

Open the PNGs and HTML viewers in `results/`. Use `--output` with fresh filenames for repeat runs.

## Course baselines

The handout names two baselines. `mds_pgd.py` initializes with MDS-MAP (`core/mds.py`) using shortest-path distances, classical MDS, and a similarity fit to the anchors. It passes the projected start and the remaining budget to `pgd.run_pgd`. `levenberg_marquardt.py` provides `run_lm(param, max_time, *, x0=None, record=None)` with the same calling convention as `run_pgd`. Each iteration solves the damped normal equations $(J^\top W J + \lambda\,\mathrm{diag}(J^\top W J))\,s = -J^\top W r$ over the free coordinates, where $W$ holds Huber IRLS weights (all ones for squared loss). It projects the trial point and accepts it only if the objective decreases.

Both baselines use an internal deadline at 90% of the requested time.

- MDS estimates shortest-path cost from the node and edge counts and embedding cost from the squared node count. It checks the combined estimate before starting and checks the embedding estimate again after shortest paths. If a stage cannot start or MDS finishes after the internal deadline, initialization returns the projected course `x0`. Time already spent still counts against PGD's budget. The solver returns the initialized point directly when less than 25 ms remains for PGD's first objective and gradient.
- LM starts with a trial reserve of at least 25 ms, scaled with the square of the free coordinate count. It raises the reserve to at least twice the slowest completed trial, including the damped system build, solve, projection, and scoring. It checks time before matrix work and again immediately before each sparse solve. A trial that overruns the internal deadline is discarded and LM returns the last accepted point.

The reserves are timing estimates based on public instance sizes. They may skip work that would have fit on a faster machine. SciPy calls already in progress run to completion, so machine load or an unexpectedly slow call can still cause a late return.

```sh
.venv/bin/python -B benchmark.py solver.py mds_pgd.py levenberg_marquardt.py --run both --output results/baselines.csv
.venv/bin/python -B visualize.py C_1.npz mds_pgd.py levenberg_marquardt.py --run short --output baselines-C_1.png
.venv/bin/python -B record_trace.py C_1.npz levenberg_marquardt.py --run short --output results/lm-C_1-short.npz
.venv/bin/python -B -m unittest discover -s tests -p test_baselines.py -v
```

In PowerShell, start the same commands with `.\.venv\Scripts\python.exe`. To build on a baseline, import `run_lm` or an initializer from `core/initialization/` in your own root-level solver file.

## Compare saved traces with a collapsed reference

The collapsed reference places every free sensor at `(0.5, 0.5)` and preserves the supplied anchor positions. The diagnostic ratio is the current projected objective divided by the official objective of that reference. A ratio below 1 means the current objective is lower than the reference. A zero reference gives `n/a`.

Run the report below against the saved PGD traces. It compares the first and last snapshots with the collapsed reference and reports counts and mean official losses for free-to-free, free-to-anchor, and anchor-to-anchor edges at all three configurations. An empty group has a count of zero and a mean of `n/a`. These calculations run offline on saved traces.

```sh
.venv/bin/python -B trace_diagnostics.py results/pgd-A_1-short.npz results/pgd-B_1-short.npz results/pgd-C_1-short.npz results/pgd-D_1-short.npz > results/pgd-short-diagnostics.md
```

Render viewers with the reference line from those same traces.

```sh
for instance in A_1 B_1 C_1 D_1; do
  .venv/bin/python -B render_trace.py "results/pgd-${instance}-short.npz" --output "results/pgd-${instance}-short-diagnostics.html"
done
```

In PowerShell, start the report command with `.\.venv\Scripts\python.exe` and use this loop for the viewers.

```powershell
foreach ($instance in 'A_1', 'B_1', 'C_1', 'D_1') {
    .\.venv\Scripts\python.exe -B render_trace.py "results/pgd-${instance}-short.npz" --output "results/pgd-${instance}-short-diagnostics.html"
}
```

## Test and submit a pull request

With the course files, NumPy, and Matplotlib in place, run the automated checks on Linux, macOS, or WSL.

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

Open the pull request link printed by `git push`. If there is no link, open the repository on GitHub and choose **Pull requests** and then **New pull request**. Set the base branch to `main` and the compare branch to `feature/my-algorithm`. Include the exact benchmark command, the results you observed, and any failed trials, then choose **Create pull request**. Ask another teammate to reproduce the checks and review the change. Merging requires explicit approval. [Team workflow](AGENTS.md)
