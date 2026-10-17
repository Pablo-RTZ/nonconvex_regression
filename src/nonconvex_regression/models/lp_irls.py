import numpy as np


def lp_irls(
    X: np.ndarray,
    y: np.ndarray,
    lp: float,
    eps: float,
    initial_coefficients: np.ndarray | None = None,
    maxiter: int = 1000,
    tolerance: float = 1e-6,
    return_iterations: bool = False,
) -> np.ndarray | tuple[np.ndarray, int]:
    """Fit a linear regression model by minimizing a smoothed Lp objective.

    The fitted model is

        y = beta[0] + X @ beta[1:],

    where ``beta[0]`` is the intercept. Iteratively reweighted least squares
    uses weights ``(residual**2 + eps**2)**((lp - 2) / 2)`` and stops when
    the relative change in the smoothed objective is below ``tolerance``.
    By default, the initial coefficients are the ordinary least-squares
    solution.

    Args:
        X: 2D array of shape ``(n, d)`` containing ``n`` observations and
            ``d`` features.
        y: 1D array of shape ``(n,)`` containing the response values.
        lp: Exponent of the Lp objective. Must be strictly between 0 and 1.
        eps: Positive smoothing parameter used to keep the weights finite.
        initial_coefficients: Optional initial intercept and feature
            coefficients. If omitted, the ordinary least-squares solution is
            used.
        maxiter: Maximum number of reweighted least-squares updates.
        tolerance: Positive relative objective-change threshold for stopping.
        return_iterations: If ``True``, also return the number of IRLS
            iterations performed. Defaults to ``False`` so that existing
            calls keep returning only the coefficients.

    Returns:
        If ``return_iterations`` is ``False``, a 1D array of shape
        ``(d + 1,)`` containing the fitted intercept followed by the feature
        coefficients. If ``return_iterations`` is ``True``, a tuple
        ``(coefficients, n_iterations)`` where ``n_iterations`` is the number
        of reweighted least-squares updates actually performed.

    Raises:
        ValueError: If the inputs have invalid shapes or values, or if ``lp``,
            ``maxiter``, ``eps``, or ``tolerance`` are outside their valid
            ranges, or if ``initial_coefficients`` has an invalid shape or
            contains non-finite values.
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
    if not np.isfinite(lp) or not 0.0 < lp < 1.0:
        raise ValueError("'lp' must be strictly between 0 and 1.")
    if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)):
        raise ValueError("'maxiter' must be a positive integer.")
    if maxiter <= 0:
        raise ValueError("'maxiter' must be a positive integer.")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("'eps' must be a positive finite number.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("'tolerance' must be a positive finite number.")

    design_matrix = np.column_stack((np.ones(X_arr.shape[0]), X_arr))
    if initial_coefficients is None:
        coefficients = np.linalg.lstsq(design_matrix, y_arr, rcond=None)[0]
    else:
        coefficients = np.asarray(initial_coefficients, dtype=float)
        if coefficients.shape != (design_matrix.shape[1],):
            raise ValueError(
                "'initial_coefficients' must have one value for the "
                "intercept and each feature."
            )
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("'initial_coefficients' must contain only finite values.")

    objective = np.sum(np.hypot(design_matrix @ coefficients - y_arr, eps) ** lp)

    iterations = 0
    for _ in range(maxiter):
        residuals = design_matrix @ coefficients - y_arr
        log_weights = (lp - 2.0) * np.log(np.hypot(residuals, eps))
        log_weights -= np.max(log_weights)
        sqrt_weights = np.exp(0.5 * log_weights)

        weighted_design = design_matrix * sqrt_weights[:, np.newaxis]
        weighted_y = y_arr * sqrt_weights
        next_coefficients = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]

        if not np.all(np.isfinite(next_coefficients)):
            raise RuntimeError("IRLS produced non-finite regression coefficients.")

        next_residuals = design_matrix @ next_coefficients - y_arr
        next_objective = np.sum(np.hypot(next_residuals, eps) ** lp)
        relative_change = abs(next_objective - objective) / max(1.0, abs(objective))

        coefficients = next_coefficients
        objective = next_objective
        iterations += 1
        if relative_change < tolerance:
            break

    if return_iterations:
        return coefficients, iterations
    return coefficients
