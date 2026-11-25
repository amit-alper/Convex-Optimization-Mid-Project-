# Sensor-network localization

Mid-semester project for Optimization 67731, Topic 4.

## Clone the repository

```sh
git clone https://github.com/amit-alper/Convex-Optimization-Mid-Project-.git
cd Convex-Optimization-Mid-Project-
```

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
git switch develop
git pull --ff-only origin develop
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

## Visualize a public instance

Install Matplotlib in the same virtual environment on Linux, macOS, or WSL.

```sh
.venv/bin/python -m pip install matplotlib==3.10.8
```

In Windows PowerShell, run the following.

```powershell
.\.venv\Scripts\python.exe -m pip install matplotlib==3.10.8
```

Run the examples from the repository root and open the PNGs in `results/`.

```sh
.venv/bin/python -B visualize.py A_1.npz solver.py --run short --edges --output visualization-A_1.png
.venv/bin/python -B visualize.py C_1.npz solver.py --run long --edges --output visualization-C_1.png
```

In PowerShell, start the same commands with `.\.venv\Scripts\python.exe`.

Pass one public instance filename, followed by one or more solver `.py` files. For example, add `my_algorithm.py` after `solver.py` to see both results beside `x0`. Use `--run short` for 5 seconds (the default) or `--run long` for 20 seconds. Use `--edges` to add measured edges colored by absolute distance residual. All panels share axis limits, equal aspect ratios, and a single residual color scale when edges are shown.

The circles are sensors and the diamonds are anchors. The tool scores and plots each solver's projected result, and a failed trial shows its status. The course runner times the solver call. Plotting and saving happen afterward. Use a fresh PNG filename with `--output` or omit it to use the default instance-and-budget name, with `-edges` appended when requested.

## Record and animate an optimization trace

Keep the official `solver(param, max_time)` entry point. Give your internal algorithm function an optional recording callback, for example `run_algorithm(param, max_time, *, record=None)`. Have `solver` call it with the default so grading and benchmarks run with recording disabled.

In a separate diagnostic run, pass a recorder and save the trace after the algorithm returns. Here `param` is the course instance you loaded and `run_algorithm` is your own function.

```python
from convergence_trace import Trace

trace = Trace("A_1.npz", "My diagnostic run")
run_algorithm(param, 20, record=trace.record)
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
  .venv/bin/python -B -m examples.pgd_trace "${instance}.npz" --run short
  .venv/bin/python -B render_trace.py "results/pgd-${instance}-short.npz"
done
```

In PowerShell, run the benchmark command with `.\.venv\Scripts\python.exe` and use this loop for the plots and traces.

```powershell
foreach ($instance in 'A_1', 'B_1', 'C_1', 'D_1') {
    .\.venv\Scripts\python.exe -B visualize.py "${instance}.npz" pgd.py --run short --output "pgd-${instance}-comparison.png"
    .\.venv\Scripts\python.exe -B -m examples.pgd_trace "${instance}.npz" --run short
    .\.venv\Scripts\python.exe -B render_trace.py "results/pgd-${instance}-short.npz"
}
```

Open the PNGs and HTML viewers in `results/`. Use `--output` with fresh filenames for repeat runs.

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

Open the pull request link printed by `git push`. If there is no link, open the repository on GitHub and choose **Pull requests** and then **New pull request**. Set the base branch to `develop` and the compare branch to `feature/my-algorithm`. Include the exact benchmark command, the results you observed, and any failed trials, then choose **Create pull request**. Ask another teammate to reproduce the checks and review the change. Merging requires explicit approval. [Team workflow](AGENTS.md)
