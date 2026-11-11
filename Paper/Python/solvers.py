import numpy as np


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit an ordinary least-squares linear regression model.

    The fitted model is

        y = beta[0] + X @ beta[1:],

    where ``beta[0]`` is the intercept. The coefficients are computed using
    the Moore-Penrose pseudoinverse of the design matrix, to also return the
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


def lp_irls(
    X: np.ndarray,
    y: np.ndarray,
    p: float,
    delta: float,
    maxiter: int = 1000,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Fit a linear regression model by minimizing a smoothed Lp objective.

    The fitted model is

        y = beta[0] + X @ beta[1:],

    where ``beta[0]`` is the intercept. Iteratively reweighted least squares
    uses weights ``(residual**2 + delta**2)**((p - 2) / 2)`` and stops when
    the relative change in the smoothed objective is below ``tolerance``.
    The initial coefficients are the ordinary least-squares solution.

    Args:
        X: 2D array of shape ``(n, d)`` containing ``n`` observations and
            ``d`` features.
        y: 1D array of shape ``(n,)`` containing the response values.
        p: Exponent of the Lp objective. Must be strictly between 0 and 1.
        delta: Positive smoothing parameter used to keep the weights finite.
        maxiter: Maximum number of reweighted least-squares updates.
        tolerance: Positive relative objective-change threshold for stopping.

    Returns:
        A 1D array of shape ``(d + 1,)`` containing the fitted intercept
        followed by the feature coefficients.

    Raises:
        ValueError: If the inputs have invalid shapes or values, or if ``p``,
            ``maxiter``, ``delta``, or ``tolerance`` are outside their valid
            ranges.
        RuntimeError: If an iteration produces non-finite coefficients.
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
    if not np.all(np.isfinite(X_arr)) or not np.all(np.isfinite(y_arr)):
        raise ValueError("'X' and 'y' must contain only finite values.")
    if not np.isfinite(p) or not 0.0 < p < 1.0:
        raise ValueError("'p' must be strictly between 0 and 1.")
    if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)):
        raise ValueError("'maxiter' must be a positive integer.")
    if maxiter <= 0:
        raise ValueError("'maxiter' must be a positive integer.")
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("'delta' must be a positive finite number.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("'tolerance' must be a positive finite number.")

    design_matrix = np.column_stack((np.ones(X_arr.shape[0]), X_arr))
    coefficients = np.linalg.lstsq(design_matrix, y_arr, rcond=None)[0]

    objective = np.sum(np.hypot(design_matrix @ coefficients - y_arr, delta) ** p)

    for _ in range(maxiter):
        residuals = design_matrix @ coefficients - y_arr
        log_weights = (p - 2.0) * np.log(np.hypot(residuals, delta))
        log_weights -= np.max(log_weights)
        sqrt_weights = np.exp(0.5 * log_weights)

        weighted_design = design_matrix * sqrt_weights[:, np.newaxis]
        weighted_y = y_arr * sqrt_weights
        next_coefficients = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]

        if not np.all(np.isfinite(next_coefficients)):
            raise RuntimeError("IRLS produced non-finite regression coefficients.")

        next_residuals = design_matrix @ next_coefficients - y_arr
        next_objective = np.sum(np.hypot(next_residuals, delta) ** p)
        relative_change = abs(next_objective - objective) / max(1.0, abs(objective))

        coefficients = next_coefficients
        objective = next_objective
        if relative_change < tolerance:
            break

    return coefficients


def huber_irls(
    X: np.ndarray,
    y: np.ndarray,
    delta: float,
    maxiter: int = 1000,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Fit a linear regression model by minimizing the Huber loss.

    The fitted model is

        y = beta[0] + X @ beta[1:],

    where ``beta[0]`` is the intercept. Iteratively reweighted least squares
    uses unit weights for residuals within ``delta`` and weights
    ``delta / abs(residual)`` for larger residuals. The initial coefficients
    are the ordinary least-squares solution.

    Args:
        X: 2D array of shape ``(n, d)`` containing ``n`` observations and
            ``d`` features.
        y: 1D array of shape ``(n,)`` containing the response values.
        delta: Positive threshold separating quadratic and linear loss.
        maxiter: Maximum number of reweighted least-squares updates.
        tolerance: Positive relative objective-change threshold for stopping.

    Returns:
        A 1D array of shape ``(d + 1,)`` containing the fitted intercept
        followed by the feature coefficients.

    Raises:
        ValueError: If the inputs have invalid shapes or values, or if
            ``delta``, ``maxiter``, or ``tolerance`` are outside their valid
            ranges.
        RuntimeError: If an iteration produces non-finite coefficients.
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
    if not np.all(np.isfinite(X_arr)) or not np.all(np.isfinite(y_arr)):
        raise ValueError("'X' and 'y' must contain only finite values.")
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("'delta' must be a positive finite number.")
    if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)):
        raise ValueError("'maxiter' must be a positive integer.")
    if maxiter <= 0:
        raise ValueError("'maxiter' must be a positive integer.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("'tolerance' must be a positive finite number.")

    design_matrix = np.column_stack((np.ones(X_arr.shape[0]), X_arr))
    coefficients = np.linalg.lstsq(design_matrix, y_arr, rcond=None)[0]

    residuals = design_matrix @ coefficients - y_arr
    absolute_residuals = np.abs(residuals)
    objective = np.sum(
        np.where(
            absolute_residuals <= delta,
            0.5 * residuals**2,
            delta * (absolute_residuals - 0.5 * delta),
        )
    )

    for _ in range(maxiter):
        residuals = design_matrix @ coefficients - y_arr
        absolute_residuals = np.abs(residuals)
        weights = np.ones_like(absolute_residuals)
        np.divide(
            delta,
            absolute_residuals,
            out=weights,
            where=absolute_residuals > delta,
        )
        sqrt_weights = np.sqrt(weights)

        weighted_design = design_matrix * sqrt_weights[:, np.newaxis]
        weighted_y = y_arr * sqrt_weights
        next_coefficients = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]

        if not np.all(np.isfinite(next_coefficients)):
            raise RuntimeError(
                "Huber IRLS produced non-finite regression coefficients."
            )

        next_residuals = design_matrix @ next_coefficients - y_arr
        absolute_next_residuals = np.abs(next_residuals)
        next_objective = np.sum(
            np.where(
                absolute_next_residuals <= delta,
                0.5 * next_residuals**2,
                delta * (absolute_next_residuals - 0.5 * delta),
            )
        )
        relative_change = abs(next_objective - objective) / max(1.0, abs(objective))

        coefficients = next_coefficients
        objective = next_objective
        if relative_change < tolerance:
            break

    return coefficients
