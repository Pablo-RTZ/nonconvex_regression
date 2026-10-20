import numpy as np


def clean_data(
    n: int,
    beta: np.ndarray,
    epsilon: float,
    min: float | np.ndarray,
    max: float | np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Generate clean linear-regression data with Gaussian noise.

    The generated model is

        y = beta[0] + X @ beta[1:] + noise,

    where each feature column is sampled uniformly from the corresponding
    interval in ``[min, max]`` and the noise is Gaussian with standard
    deviation ``epsilon``.

    Args:
        n: Number of observations to generate.
        beta: Coefficients of the linear model, with the first entry treated as
            the intercept and the remaining entries as feature coefficients.
        epsilon: Standard deviation of the Gaussian noise term.
        min: Lower bounds for each feature.
        max: Upper bounds for each feature.
        seed: Random seed used to make the data generation reproducible.

    Returns:
        A 2D NumPy array of shape ``(n, d + 1)`` whose columns are the feature
        matrix followed by the noisy response ``y``.

    Raises:
        ValueError: If ``beta`` is not one-dimensional, the feature bounds
            differ in shape, no feature bounds are provided, or ``beta`` does
            not contain an intercept and one coefficient per feature.
    """
    rng = np.random.default_rng(seed)

    beta_arr = np.asarray(beta, dtype=float)
    min_arr = np.asarray(min, dtype=float)
    max_arr = np.asarray(max, dtype=float)

    if beta_arr.ndim != 1:
        raise ValueError("'beta' must be a one-dimensional array.")
    if min_arr.shape != max_arr.shape:
        raise ValueError("'min' and 'max' must have the same shape.")
    if min_arr.ndim == 0:
        min_arr = np.array([float(min_arr)])
        max_arr = np.array([float(max_arr)])
    if min_arr.size == 0:
        raise ValueError("At least one feature bound must be provided.")
    if beta_arr.size != min_arr.size + 1:
        raise ValueError(
            "'beta' must have length d + 1 for d features when an intercept is assumed."
        )

    X = rng.uniform(min_arr, max_arr, size=(n, min_arr.size))
    y = beta_arr[0] + X @ beta_arr[1:] + rng.normal(0.0, epsilon, size=n)
    return np.column_stack((X, y))


def outlier_data(
    n: int,
    beta: np.ndarray,
    epsilon: float,
    min: float | np.ndarray,
    max: float | np.ndarray,
    p: float,
    A: float,
    seed: int = 42,
) -> np.ndarray:
    """Generate linear-regression data with Gaussian noise and random outliers.

    Each observation is generated from the clean linear model, then independently
    replaced by an outlier with probability ``p``. Conditional on an outlier,
    the response perturbation is drawn from ``N(0, A^2)``.

    Args:
        n: Number of observations to generate.
        beta: Coefficients of the linear model, with the first entry treated as
            the intercept and the remaining entries as feature coefficients.
        epsilon: Standard deviation of the Gaussian noise term in the clean
            data-generating process.
        min: Lower bounds for each feature.
        max: Upper bounds for each feature.
        p: Probability that a given observation is an outlier.
        A: Standard deviation of the Gaussian outlier contamination.
        seed: Random seed used to make the data generation reproducible.

    Returns:
        A 2D NumPy array of shape ``(n, d + 1)`` whose columns are the feature
        matrix followed by the noisy response ``y``.

    Raises:
        ValueError: If ``p`` is outside ``[0, 1]``, ``beta`` is not
            one-dimensional, the feature bounds differ in shape, no feature
            bounds are provided, or ``beta`` does not contain an intercept and
            one coefficient per feature.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("'p' must be in the interval [0, 1].")

    rng = np.random.default_rng(seed)

    beta_arr = np.asarray(beta, dtype=float)
    min_arr = np.asarray(min, dtype=float)
    max_arr = np.asarray(max, dtype=float)

    if beta_arr.ndim != 1:
        raise ValueError("'beta' must be a one-dimensional array.")
    if min_arr.shape != max_arr.shape:
        raise ValueError("'min' and 'max' must have the same shape.")
    if min_arr.ndim == 0:
        min_arr = np.array([float(min_arr)])
        max_arr = np.array([float(max_arr)])
    if min_arr.size == 0:
        raise ValueError("At least one feature bound must be provided.")
    if beta_arr.size != min_arr.size + 1:
        raise ValueError(
            "'beta' must have length d + 1 for d features when an intercept is assumed."
        )

    X = rng.uniform(min_arr, max_arr, size=(n, min_arr.size))
    y = beta_arr[0] + X @ beta_arr[1:] + rng.normal(0.0, epsilon, size=n)

    outlier_mask = rng.random(n) < p
    if np.any(outlier_mask):
        y[outlier_mask] += rng.normal(0.0, A, size=outlier_mask.sum())

    return np.column_stack((X, y))
