"""Optimizers, one file per method. Each run_<method>(param, max_time, *, x0=None, record=None)
starts from x0 (default param.x0), honors max_time and returns a feasible n x q array.

pgd.py: projected gradient descent with an Armijo line search.
levenberg_marquardt.py: damped Gauss-Newton with Huber IRLS weights.
Add a new method as a new file here and call it from an entry file in solvers/.
"""
