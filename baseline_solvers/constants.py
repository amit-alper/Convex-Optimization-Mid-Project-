"""Tunable constants of the baseline solvers, in one place for exploring their effect.

The solvers read these at call time (constants.NAME), so an experiment script can set, e.g.,
constants.PGD_ARMIJO = 1e-2 before calling run_pgd. The course runner always uses the values below.
"""

# Shared
TIME_FRACTION = 0.9           # return by 0.9 max_time (course README)

# Projected gradient descent (pgd.py), as in Liam's pgd.py on develop
PGD_ARMIJO = 1e-4             # sufficient-decrease constant
PGD_SHRINK = 0.5              # backtracking factor
PGD_MAX_COORDINATE_MOVE = 0.1 # cap on the first trial step, in box units

# Levenberg-Marquardt (levenberg_marquardt.py)
LM_LAMBDA_START = 1e-3        # initial damping
LM_LAMBDA_DOWN = 0.3          # multiply after an accepted step
LM_LAMBDA_UP = 10.0           # multiply after a rejected step
LM_LAMBDA_MAX = 1e12          # stop: no decrease even with a tiny gradient step
