import numpy as np
from scipy.stats import gamma


def calculate_rmse(points: np.ndarray, coefficients: np.ndarray) -> float:
    """Calculate the root mean squared error of a linear fit.

    Computes the residuals between the observed y-values in ``points`` and the
    y-values predicted by the linear model ``y = c0 + c1 * x``, then returns the
    root mean squared error of those residuals.

    Args:
        points (np.ndarray): A 2D array of shape ``(n, 2)`` where each row is a
            data point ``[x, y]``. The first column holds the independent
            variable and the second column holds the observed dependent variable.
        coefficients (np.ndarray): A 1D array of length 2 containing the linear
            model coefficients ``[c0, c1]``, where ``c0`` is the intercept and
            ``c1`` is the slope.

    Returns:
        float: The root mean squared error between the observed and predicted
            y-values.

    Raises:
        IndexError: If ``coefficients`` has fewer than 2 elements.
        ValueError: If ``points`` does not have at least 2 columns.
    """
    residuals = points[:, 1] - (coefficients[0] + points[:, 0] * coefficients[1])
    return np.sqrt(np.mean(residuals**2))


def calculate_coefficient_recovery(
    coefficients: np.ndarray, true_coefficients: np.ndarray
) -> float:
    """Calculate the Euclidean distance between estimated and true coefficients.

    Measures how closely the estimated model coefficients match the ground-truth
    coefficients by computing the L2 (Euclidean) norm of their difference. A
    value of ``0.0`` indicates perfect recovery.

    Args:
        coefficients (np.ndarray): A 1D array of estimated model coefficients.
        true_coefficients (np.ndarray): A 1D array of the ground-truth model
            coefficients. Must have the same shape as ``coefficients``.

    Returns:
        float: The L2 norm of the difference between the estimated and true
            coefficients.

    Raises:
        ValueError: If ``coefficients`` and ``true_coefficients`` have
            incompatible shapes for element-wise subtraction.
    """
    return float(np.linalg.norm(coefficients - true_coefficients))


def gamma_kde(values: np.ndarray, n_points: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a probability density with a gamma kernel.

    Applies a gamma-kernel density estimator (KDE) to ``values``, evaluating the
    estimated density over a grid spanning from ``0`` to the maximum observed
    value. The kernel bandwidth is chosen adaptively using the smaller of the
    sample standard deviation and the robust interquartile-range-based spread,
    following a Silverman-style rule.

    Args:
        values (np.ndarray): Sample observations used to estimate the density.
            May be of any shape; the array is flattened internally before use.
        n_points (int, optional): Number of grid points at which to evaluate
            the density. Defaults to ``300``.

    Returns:
        tuple[np.ndarray, np.ndarray]: A pair ``(x_grid, density)`` where
        ``x_grid`` is a 1D array of ``n_points`` evenly spaced evaluation
        points from ``0`` to ``values.max()``, and ``density`` is a 1D array of
        the estimated density values at each corresponding grid point.

    Raises:
        ValueError: If ``values`` is empty.

    Examples:
        >>> rng = np.random.default_rng(0)
        >>> samples = rng.gamma(shape=2.0, scale=1.0, size=1000)
        >>> x_grid, density = gamma_kde(samples, n_points=100)
        >>> x_grid.shape
        (100,)
        >>> density.shape
        (100,)
    """
    values = values.ravel()
    x_grid = np.linspace(0, values.max(), n_points)
    std = values.std(ddof=1)
    iqr = np.subtract(*np.percentile(values, [75, 25]))
    spread = min(std, iqr / 1.34)
    if spread == 0:
        spread = max(values.mean(), np.finfo(float).eps)
    bandwidth = 0.9 * spread * len(values) ** (-0.2)
    density = gamma.pdf(
        x_grid[:, None],
        a=values[None, :] / bandwidth + 1,
        scale=bandwidth,
    ).mean(axis=1)
    return x_grid, density
