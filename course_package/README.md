# Topic 4: Sensor-network localization: student package

## Files
- `problem.py`: `load(path)` returns the data of one instance (`param`); `objective(x, param)` is the objective,
  and `project(x, param)` the projection onto the constraints.
- `public/`: the 12 public instances, three of each type (`A_1.npz` ... `D_3.npz`).
- `test_solver.py`: runs your solver exactly as in grading, and shows its status, time and objective value.
- `example_solver.py`: a minimal solver that shows the interface.
- `common/`: the code that runs and scores a solver. Do not modify it.

## Quick start
```
python test_solver.py example_solver.py
python test_solver.py my_solver.py --only B_1.npz --run short
```

## The problem
```
minimize   (1/|E|) sum_{(i,j) in E} rho(||x_i - x_j|| - d_ij)
subject to x_i in [0, 1]^q for sensors,  x_i = a_i for anchors
```
Data of an instance (`param = load("public/A_1.npz")`):
- `param.edges` (|E| x 2): the pairs (i, j) with a measured distance; `param.dist` (|E|): the distances d_ij
- `param.anchor`: indices of the anchors; `param.anchor_pos`: their positions
- `param.n` (all nodes, anchors included), `param.q` (2 or 3), `param.loss` ("squared": rho(t) = t^2/2, or
  "huber"), `param.delta` (Huber parameter: rho(t) = t^2/2 if |t| <= delta, delta|t| - delta^2/2 otherwise)
- `param.x0`: a starting point: random positions in the box, anchors in place

Solution: an n x q array of positions (anchors included). A NumPy array or a PyTorch tensor.
Projection: onto the constraints: sensors are clipped to the box [0, 1]^q, anchors are put at their positions.

Baselines (to implement yourselves): projected gradient descent started from classical multidimensional scaling (MDS), and Levenberg-Marquardt from the starting point.

Types of instances:
- A. Small, well anchored
- B. Large, narrow strip, many wrong distances
- C. Very large, U-shaped region, many wrong distances
- D. Large, very many wrong distances

## Your solver
```python
def solver(param, max_time):
    ...            # your method
    return x
```
The hidden instances are run twice: a short run with max_time = 5 seconds and a long run with
max_time = 20 seconds, on one CPU core. Loading the data does not count; the time runs from the call
until your function returns. Your function receives max_time, so it may use a different method in each run. Leave a safety margin
(for example, return by 0.9 max_time): a solver that returns late gets nothing on that instance.
We project the x that you return and compute its objective.

## Scoring
In each run, on every hidden instance, we compare the value of the objective at the point your solver returned
with the values of the other groups with your topic, and with the baselines. Lower is better; values that are
practically equal count as equal. Only the value counts, not the time: a solver that returns after 10 seconds
has no advantage over one that uses all 20.
A solver gets nothing on an instance if it fails, returns after max_time, or returns something that cannot be
projected (a wrong shape, or entries that are not finite).
We do not publish the values that our solvers reach, or the best values that we know: implement the baselines
yourselves, compare with them, and explore
how far you can get.
