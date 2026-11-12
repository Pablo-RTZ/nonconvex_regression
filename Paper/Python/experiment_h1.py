from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

if __package__:
    from .data_generation import outlier_data
    from .solvers import lp_irls
else:
    from data_generation import outlier_data
    from solvers import lp_irls


# Default values for experiment in hypothesis 1
DEFAULT_SEED = 1234
DEFAULT_OBSERVATIONS = 1000
DEFAULT_BETA = (1.0, 1.0)
DEFAULT_SIGMA = 1e-2
DEFAULT_X_MIN = -2.0
DEFAULT_X_MAX = 2.0
DEFAULT_OUTLIER_PROPORTION = 0.25
DEFAULT_OUTLIER_SCALE = 10.0
DEFAULT_P_BOUNDS = (0.4, 0.9)
DEFAULT_DELTA_BOUNDS = (1e-6, 1e-1)
DEFAULT_TOLERANCE_BOUNDS = (1e-10, 1e-1)
DEFAULT_INITIAL_POINTS = 15
DEFAULT_BO_ITERATIONS = 50
DEFAULT_REPLICATES = 20
DEFAULT_CANDIDATE_POOL_SIZE = 512
DEFAULT_GP_LENGTH_SCALE = 0.25
DEFAULT_GP_JITTER = 1e-8
DEFAULT_MIN_GP_VARIANCE = 1e-12
DEFAULT_IRLS_MAXITER = 200
DEFAULT_RESULTS_PATH = (
    Path(__file__).resolve().parents[1] / "Results" / "h1_comparison.csv"
)


CSV_COLUMNS = (
    "evaluation",
    "phase",
    "p",
    "delta",
    "tolerance",
    "mean_recovery_error",
    "std_recovery_error",
    "observation_noise_variance",
    "posterior_mean_error",
    "is_best",
    "replicate_errors",
)


def _gp_posterior(
    observed_points: np.ndarray,
    observed_values: np.ndarray,
    observation_noise: np.ndarray,
    query_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate latent recovery error and variance with a heteroscedastic GP."""
    value_center = float(np.mean(observed_values))
    value_scale = max(float(np.std(observed_values)), DEFAULT_MIN_GP_VARIANCE)
    normalized_values = (observed_values - value_center) / value_scale
    model = GaussianProcessRegressor(
        kernel=Matern(
            length_scale=DEFAULT_GP_LENGTH_SCALE,
            nu=2.5,
        ),
        alpha=observation_noise / value_scale**2 + DEFAULT_GP_JITTER,
        optimizer=None,
        normalize_y=False,
    )
    model.fit(observed_points, normalized_values)
    normalized_mean, normalized_std = model.predict(
        query_points,
        return_std=True,
    )
    return (
        value_center + value_scale * normalized_mean,
        value_scale**2 * normalized_std**2,
    )


def _decode_parameters(normalized: np.ndarray) -> tuple[float, float, float]:
    """Map unit-cube coordinates to p, delta, and tolerance."""
    p = DEFAULT_P_BOUNDS[0] + normalized[0] * (
        DEFAULT_P_BOUNDS[1] - DEFAULT_P_BOUNDS[0]
    )
    delta = np.exp(
        np.log(DEFAULT_DELTA_BOUNDS[0])
        + normalized[1]
        * (np.log(DEFAULT_DELTA_BOUNDS[1]) - np.log(DEFAULT_DELTA_BOUNDS[0]))
    )
    tolerance = np.exp(
        np.log(DEFAULT_TOLERANCE_BOUNDS[0])
        + normalized[2]
        * (np.log(DEFAULT_TOLERANCE_BOUNDS[1]) - np.log(DEFAULT_TOLERANCE_BOUNDS[0]))
    )
    return float(p), float(delta), float(tolerance)


# H1 experiment
def run_bo_tuning_experiment(
    output_path: Path = DEFAULT_RESULTS_PATH,
    *,
    seed: int = DEFAULT_SEED,
    initial_points: int = DEFAULT_INITIAL_POINTS,
    bo_iterations: int = DEFAULT_BO_ITERATIONS,
    replicates: int = DEFAULT_REPLICATES,
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
    maxiter: int = DEFAULT_IRLS_MAXITER,
) -> Path:
    """Tune p, delta, and tolerance using noise-aware Bayesian optimization.

    Each candidate is scored by its mean coefficient-recovery error over
    repeated, fixed-seed data sets. The Gaussian process receives the estimated
    variance of that mean as observation noise. Delta and tolerance are searched
    in log space.

    Args:
        output_path: CSV destination; defaults to ``Paper/Results/h1_comparison.csv``.
        seed: Seed for the initial design, candidate pool, and data replicates.
        initial_points: Number of uniformly sampled initial parameter settings.
        bo_iterations: Number of Bayesian-optimization proposals after the
            initial design.
        replicates: Number of noisy data sets used to score each setting.
        candidate_pool_size: Number of unit-cube candidates considered per BO
            iteration.
        maxiter: Maximum number of local IRLS iterations per fit.

    Returns:
        The path to the CSV containing all evaluated parameter settings.

    Raises:
        ValueError: If any experiment-budget value is outside its valid range.
    """
    if initial_points <= 0 or bo_iterations < 0:
        raise ValueError(
            "'initial_points' must be positive and 'bo_iterations' non-negative."
        )
    if replicates < 2:
        raise ValueError("'replicates' must be at least 2 to estimate noise.")
    if candidate_pool_size <= 0 or maxiter <= 0:
        raise ValueError("'candidate_pool_size' and 'maxiter' must be positive.")

    rng = np.random.default_rng(seed)
    true_coefficients = np.asarray(DEFAULT_BETA, dtype=float)
    datasets = tuple(
        outlier_data(
            DEFAULT_OBSERVATIONS,
            true_coefficients,
            DEFAULT_SIGMA,
            DEFAULT_X_MIN,
            DEFAULT_X_MAX,
            DEFAULT_OUTLIER_PROPORTION,
            DEFAULT_OUTLIER_SCALE,
            seed=seed + replicate + 1,
        )
        for replicate in range(replicates)
    )
    design_sets = tuple((data[:, :-1], data[:, -1]) for data in datasets)

    observed_points: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    observation_noise: list[float] = []

    def evaluate(point: np.ndarray, phase: str) -> None:
        p, delta, tolerance = _decode_parameters(point)
        replicate_errors = tuple(
            float(
                np.linalg.norm(
                    lp_irls(
                        X,
                        y,
                        p,
                        delta,
                        maxiter=maxiter,
                        tolerance=tolerance,
                    )
                    - true_coefficients
                )
            )
            for X, y in design_sets
        )
        mean_error = float(np.mean(replicate_errors))
        std_error = float(np.std(replicate_errors, ddof=1))
        noise_variance = std_error**2 / replicates
        if not np.isfinite(mean_error) or not np.isfinite(noise_variance):
            raise RuntimeError("A BO candidate produced non-finite recovery error.")

        observed_points.append(point.copy())
        observation_noise.append(noise_variance)
        records.append(
            {
                "evaluation": len(records) + 1,
                "phase": phase,
                "p": p,
                "delta": delta,
                "tolerance": tolerance,
                "mean_recovery_error": mean_error,
                "std_recovery_error": std_error,
                "observation_noise_variance": noise_variance,
                "replicate_errors": ";".join(
                    f"{error:.12g}" for error in replicate_errors
                ),
            }
        )

    initial_design = rng.random((initial_points, 3))
    for point in initial_design:
        evaluate(point, "initial")

    for _ in range(bo_iterations):
        points = np.asarray(observed_points)
        values = np.asarray(
            [float(record["mean_recovery_error"]) for record in records]
        )
        noise = np.asarray(observation_noise)
        candidates = rng.random((candidate_pool_size, 3))
        posterior_mean, posterior_variance = _gp_posterior(
            points, values, noise, candidates
        )
        best_latent_mean, _ = _gp_posterior(points, values, noise, points)
        standard_deviation = np.sqrt(posterior_variance)
        improvement = float(np.min(best_latent_mean)) - posterior_mean
        standardized_improvement = improvement / standard_deviation
        expected_improvement = improvement * norm.cdf(
            standardized_improvement
        ) + standard_deviation * norm.pdf(standardized_improvement)
        next_point = candidates[int(np.argmax(expected_improvement))]
        evaluate(next_point, "bayesian_optimization")

    points = np.asarray(observed_points)
    values = np.asarray([float(record["mean_recovery_error"]) for record in records])
    posterior_mean, _ = _gp_posterior(
        points, values, np.asarray(observation_noise), points
    )
    best_index = int(np.argmin(posterior_mean))
    for index, record in enumerate(records):
        record["posterior_mean_error"] = float(posterior_mean[index])
        record["is_best"] = index == best_index

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    return output_path


if __name__ == "__main__":
    print(run_bo_tuning_experiment())
