"""Points of one run on one instance: 0 if the solver gave no valid solution in time; otherwise the
percentage of the other solvers whose value it matches or beats (values within 0.01% count as equal).
A solver that gave no valid solution is beaten by every valid one."""

TIE = 1e-4


def matches_or_beats(f, g):
    return g is None or f <= g + TIE * max(abs(f), abs(g))


def points(f, others):
    """f: your value (None if no valid solution); others: the values of the other solvers (None allowed)."""
    if f is None or not others:
        return 0.0 if f is None else 100.0
    return 100.0 * sum(matches_or_beats(f, g) for g in others) / len(others)
