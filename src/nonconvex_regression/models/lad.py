import numpy as np
from scipy.optimize import linprog


def lad(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit a least-absolute-deviations linear regression model.

    The fitted model is

        y = beta[0] + X @ beta[1:],

    where ``beta[0]`` is the intercept. The coefficients minimize the sum of
    absolute residuals.

    Args:
        X: 2D array of shape ``(n, d)`` containing ``n`` observations and
            ``d`` features.
        y: 1D array of shape ``(n,)`` containing the response values.

    Returns:
        A 1D array of shape ``(d + 1,)`` containing the fitted intercept
        followed by the feature coefficients.

    Raises:
        ValueError: If ``X`` and ``y`` do not have the expected dimensions,
            have mismatched observation counts, or contain no observations.
        RuntimeError: If the linear-programming solver fails to find a solution.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if X_arr.ndim != 2:
        raise ValueError("'X' must be a two-dimensional array.")
    if y_arr.ndim != 1:
        raise ValueError("'y' must be a one-dimensional array.")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("'X' and 'y' must contain the same number of observations.")
    if X_arr.shape[0] == 0:
        raise ValueError("At least one observation must be provided.")

    design_matrix = np.column_stack((np.ones(X_arr.shape[0]), X_arr))
    n, p = design_matrix.shape

    # Variables: [beta (p coefficients), u (n absolute residuals)]
    c = np.r_[np.zeros(p), np.ones(n)]

    A_ub = np.vstack(
        [
            np.hstack([design_matrix, -np.eye(n)]),
            np.hstack([-design_matrix, -np.eye(n)]),
        ]
    )
    b_ub = np.r_[y_arr, -y_arr]

    # Regression coefficients unrestricted; absolute residuals nonnegative
    bounds = [(None, None)] * p + [(0, None)] * n

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not result.success:
        raise RuntimeError(result.message)

    solution = result.x
    if solution is None:
        raise RuntimeError("The linear-programming solver returned no solution.")

    return solution[:p]
