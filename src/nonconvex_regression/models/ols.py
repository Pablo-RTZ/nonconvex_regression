import numpy as np


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit an ordinary least-squares linear regression model.

    The fitted model is

        y = beta[0] + X @ beta[1:],

    where ``beta[0]`` is the intercept. The coefficients are computed using
    the Moore-Penrose pseudoinverse of the design matrix, which also returns a
    least-squares solution when the design matrix is rank deficient.

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

    return np.linalg.pinv(design_matrix) @ y_arr
