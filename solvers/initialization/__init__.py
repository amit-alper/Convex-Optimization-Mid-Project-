"""Starting points, one file per family. Each function takes param and returns a projected n x q array.

baseline_initialization.py: x0, random, collapsed center, classical MDS (the course baseline start).
mds.py: the MDS-MAP steps behind mds_start.
Add new strategies as new files here. An entry file in solvers/ wraps one of them in
initialize(param, max_time) and subtracts the time it actually took from the optimizer's budget.
"""
