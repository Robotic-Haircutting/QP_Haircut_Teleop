"""Primal active-set method for small box-constrained convex quadratic programs (no external dependencies).

    min  0.5 xᵀ H x + gᵀ x
    s.t. lb <= x <= ub

H must be symmetric positive definite (guaranteed in teleoperation by the damping term λI). With only 6 variables, each frame takes tens of microseconds.
"""

import numpy as np


def solve_box_qp(H, g, lb, ub, x0=None, max_iter: int = 50, tol: float = 1e-10) -> tuple[np.ndarray, int]:
    """Return (x*, number of iterations). x0 is an initial feasible point (clipped into the box first)."""
    H = np.asarray(H, dtype=float)
    g = np.asarray(g, dtype=float)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    if np.any(lb > ub):
        raise ValueError("qp: lb > ub")
    n = g.size
    x = np.clip(np.zeros(n) if x0 is None else np.asarray(x0, dtype=float), lb, ub)

    # Working set: -1 = at lower bound, +1 = at upper bound, 0 = free
    w = np.zeros(n, dtype=int)
    w[(x <= lb) & (lb < ub)] = -1
    w[(x >= ub) & (lb < ub)] = 1
    w[lb == ub] = -1  # variables with equal bounds are always fixed

    for it in range(1, max_iter + 1):
        free = w == 0
        # Solve the equality-constrained subproblem on the current working set: fixed variables stay put, free variables take the unconstrained optimum
        step = np.zeros(n)
        if free.any():
            grad = H @ x + g
            step[free] = np.linalg.solve(H[np.ix_(free, free)], -grad[free])

        if np.max(np.abs(step)) <= tol:
            # Already optimal on the working set; check the signs of the Lagrange multipliers (KKT)
            grad = H @ x + g
            # The gradient should be >= 0 at a lower bound and <= 0 at an upper bound; otherwise that constraint should be released
            viol = np.where(w == -1, -grad, np.where(w == 1, grad, 0.0))
            viol[lb == ub] = 0.0
            j = int(np.argmax(viol))
            if viol[j] <= tol:
                return x, it
            w[j] = 0
            continue

        # Step along step; the first bound hit is added to the working set
        alpha, block, side = 1.0, -1, 0
        for i in np.flatnonzero(free):
            if step[i] < 0:
                a = (lb[i] - x[i]) / step[i]
                if a < alpha:
                    alpha, block, side = a, i, -1
            elif step[i] > 0:
                a = (ub[i] - x[i]) / step[i]
                if a < alpha:
                    alpha, block, side = a, i, 1
        x = x + max(alpha, 0.0) * step
        if block >= 0:
            x[block] = lb[block] if side == -1 else ub[block]
            w[block] = side

    return np.clip(x, lb, ub), max_iter
