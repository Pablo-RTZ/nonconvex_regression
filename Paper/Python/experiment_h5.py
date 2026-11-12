"""Estimate a Gaussian contamination regime and select Lp settings for H5."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

if __package__:
    from .data_generation import outlier_data
    from .experiment_h1 import (
        DEFAULT_BETA,
        DEFAULT_IRLS_MAXITER,
        DEFAULT_OBSERVATIONS,
        DEFAULT_SIGMA,
        DEFAULT_X_MAX,
        DEFAULT_X_MIN,
    )
    from .experiment_h1 import (
        DEFAULT_RESULTS_PATH as DEFAULT_PARAMETER_PATH,
    )
    from .experiment_h2 import load_best_lp_parameters
    from .solvers import huber_irls, lp_irls, ols
else:
    from data_generation import outlier_data
    from experiment_h1 import (
        DEFAULT_BETA,
        DEFAULT_IRLS_MAXITER,
        DEFAULT_OBSERVATIONS,
        DEFAULT_SIGMA,
        DEFAULT_X_MAX,
        DEFAULT_X_MIN,
    )
    from experiment_h1 import (
        DEFAULT_RESULTS_PATH as DEFAULT_PARAMETER_PATH,
    )
    from experiment_h2 import load_best_lp_parameters
    from solvers import huber_irls, lp_irls, ols


DEFAULT_SEED = 5085
DEFAULT_REPLICATES = 100
DEFAULT_ORACLE_REPLICATES = 20
DEFAULT_CANDIDATE_LIMIT = 15
DEFAULT_PROPORTIONS = (0.20, 0.25, 0.30)
DEFAULT_OUTLIER_SCALES = (5.0, 10.0, 20.0)
DEFAULT_RESULTS_PATH = (
    Path(__file__).resolve().parents[1] / "Results" / "h5_comparison.csv"
)
DEFAULT_ORACLE_PATH = (
    Path(__file__).resolve().parents[1] / "Results" / "h5_oracle_map.csv"
)

ORACLE_COLUMNS = (
    "outlier_proportion",
    "outlier_scale",
    "lp_p",
    "lp_delta",
    "lp_tolerance",
    "tuning_mean_recovery_error",
    "tuning_std_recovery_error",
    "candidate_count",
    "tuning_replicates",
)
RESULT_COLUMNS = (
    "method",
    "regime",
    "replicate",
    "seed",
    "outlier_proportion",
    "outlier_scale",
    "estimated_proportion",
    "estimated_scale",
    "estimated_sigma",
    "lookup_proportion",
    "lookup_scale",
    "recovery_error",
    "estimated_intercept",
    "estimated_slope",
    "lp_p",
    "lp_delta",
    "lp_tolerance",
    "huber_delta",
    "observations",
    "sigma",
)


def load_h1_candidates(
    parameters_path: Path, candidate_limit: int
) -> tuple[tuple[float, float, float], ...]:
    """Load a spread of valid H1 evaluations, including its selected setting."""
    required = {"p", "delta", "tolerance", "is_best"}
    rows: list[tuple[float, float, float, bool]] = []
    with Path(parameters_path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{parameters_path} is missing required columns: "
                f"{', '.join(sorted(missing))}"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                candidate = (
                    float(row["p"] or ""),
                    float(row["delta"] or ""),
                    float(row["tolerance"] or ""),
                )
            except ValueError as parse_error:
                raise ValueError(
                    f"{parameters_path}, line {line_number}: invalid parameter."
                ) from parse_error
            if not np.all(np.isfinite(candidate)):
                raise ValueError(
                    f"{parameters_path}, line {line_number}: parameters must be finite."
                )
            p, delta, tolerance = candidate
            if not 0.0 < p < 1.0 or delta <= 0.0 or tolerance <= 0.0:
                raise ValueError(
                    f"{parameters_path}, line {line_number}: invalid Lp parameters."
                )
            rows.append(
                (
                    p,
                    delta,
                    tolerance,
                    (row["is_best"] or "").strip().lower() == "true",
                )
            )

    unique = list(dict.fromkeys(row[:3] for row in rows))
    if not unique:
        raise ValueError(f"No valid H1 candidates found in {parameters_path}.")
    if candidate_limit <= 0:
        raise ValueError("'candidate_limit' must be positive.")
    if len(unique) <= candidate_limit:
        return tuple(unique)

    best = next((row[:3] for row in rows if row[3]), unique[0])
    remaining = [candidate for candidate in unique if candidate != best]
    selection = np.linspace(0, len(remaining) - 1, candidate_limit - 1, dtype=int)
    candidates = [best]
    candidates.extend(remaining[index] for index in selection)
    return tuple(dict.fromkeys(candidates))


def _robust_scale(residuals: np.ndarray) -> float:
    """Estimate a Gaussian scale from the residual median absolute deviation."""
    centered = residuals - np.median(residuals)
    scale = float(np.median(np.abs(centered)) / 0.6744897501960817)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.sqrt(np.mean(np.square(centered))))
    return max(scale, np.finfo(float).eps)


def fit_adaptive_huber(
    X: np.ndarray, y: np.ndarray, maxiter: int
) -> tuple[np.ndarray, float]:
    """Fit Huber regression while updating its threshold from robust scale."""
    coefficients = ols(X, y)
    scale = _robust_scale(y - (coefficients[0] + X @ coefficients[1:]))
    huber_delta = 1.345 * scale
    for _ in range(4):
        coefficients = huber_irls(
            X,
            y,
            huber_delta,
            maxiter=maxiter,
            tolerance=1e-10,
        )
        residuals = y - (coefficients[0] + X @ coefficients[1:])
        next_scale = _robust_scale(residuals)
        next_delta = 1.345 * next_scale
        if abs(next_delta - huber_delta) <= 1e-3 * huber_delta:
            scale = next_scale
            huber_delta = next_delta
            break
        scale = next_scale
        huber_delta = next_delta
    return coefficients, huber_delta


def fit_zero_mean_gaussian_mixture(
    residuals: np.ndarray, *, maxiter: int = 500
) -> tuple[float, float, float]:
    """Fit a two-component, zero-mean Gaussian scale mixture by EM.

    Returns the broad-component weight, the narrow-component standard
    deviation, and the added-noise standard deviation. The data generator
    adds independent clean and contamination noise, so the broad variance is
    the sum of their variances.
    """
    values = np.asarray(residuals, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Residuals must be a finite one-dimensional sample.")

    robust_variance = _robust_scale(values) ** 2
    total_variance = float(np.mean(np.square(values)))
    variance_floor = max(robust_variance * 1e-8, np.finfo(float).tiny)
    variances = np.array(
        [
            max(robust_variance, variance_floor),
            max(total_variance / 0.25, 100.0 * robust_variance),
        ]
    )
    weights = np.array([0.75, 0.25])

    for _ in range(maxiter):
        log_probabilities = (
            np.log(weights)[None, :]
            - 0.5 * np.log(2.0 * np.pi * variances)[None, :]
            - 0.5 * np.square(values[:, None]) / variances[None, :]
        )
        row_maximum = np.max(log_probabilities, axis=1, keepdims=True)
        responsibilities = np.exp(log_probabilities - row_maximum)
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

        component_counts = np.sum(responsibilities, axis=0)
        next_weights = component_counts / values.size
        next_variances = (
            np.sum(responsibilities * np.square(values[:, None]), axis=0)
            / component_counts
        )
        next_variances = np.maximum(next_variances, variance_floor)
        if next_variances[0] > next_variances[1]:
            next_variances = next_variances[::-1]
            next_weights = next_weights[::-1]
        converged = np.allclose(
            variances,
            next_variances,
            rtol=1e-7,
            atol=variance_floor,
        )
        variances = next_variances
        weights = next_weights
        if converged:
            break

    clean_scale = float(np.sqrt(variances[0]))
    contamination_scale = float(np.sqrt(max(0.0, variances[1] - variances[0])))
    contamination_proportion = float(weights[1])
    if not np.all(
        np.isfinite([clean_scale, contamination_scale, contamination_proportion])
    ):
        raise RuntimeError("The Gaussian-mixture estimate was non-finite.")
    return contamination_proportion, clean_scale, contamination_scale


def _nearest_regime(
    proportion: float,
    scale: float,
    regimes: tuple[tuple[float, float], ...],
) -> tuple[float, float]:
    """Return the closest oracle regime in normalized proportion/log-scale space."""
    proportions = np.asarray([regime[0] for regime in regimes])
    scales = np.asarray([regime[1] for regime in regimes])
    proportion_range = max(float(np.ptp(proportions)), 1e-12)
    log_scales = np.log(scales)
    scale_range = max(float(np.ptp(log_scales)), 1e-12)
    distances = np.square((proportions - proportion) / proportion_range) + np.square(
        (log_scales - np.log(max(scale, np.finfo(float).tiny))) / scale_range
    )
    return regimes[int(np.argmin(distances))]


def _interpolate_oracle_parameters(
    proportion: float,
    scale: float,
    proportions: tuple[float, ...],
    scales: tuple[float, ...],
    oracle_settings: dict[tuple[float, float], tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Interpolate oracle settings at a continuous contamination estimate.

    Interpolation is linear in proportion and log scale. The positive-valued
    smoothing and stopping parameters are interpolated in log space. Estimates
    outside the calibrated grid are clipped to its boundary.
    """
    proportion_values = np.asarray(sorted(proportions), dtype=float)
    scale_values = np.asarray(sorted(scales), dtype=float)
    log_scale_values = np.log(scale_values)
    if proportion_values.size < 2 or log_scale_values.size < 2:
        raise ValueError("Oracle interpolation requires at least a 2-by-2 grid.")

    clipped_proportion = float(
        np.clip(proportion, proportion_values[0], proportion_values[-1])
    )
    clipped_log_scale = float(
        np.clip(
            np.log(max(scale, np.finfo(float).tiny)),
            log_scale_values[0],
            log_scale_values[-1],
        )
    )

    def bracket(values: np.ndarray, target: float) -> tuple[int, int, float]:
        lower = int(
            np.clip(
                np.searchsorted(values, target, side="right") - 1,
                0,
                len(values) - 2,
            )
        )
        upper = lower + 1
        fraction = (target - values[lower]) / (values[upper] - values[lower])
        return lower, upper, float(fraction)

    rho_low, rho_high, rho_fraction = bracket(proportion_values, clipped_proportion)
    scale_low, scale_high, scale_fraction = bracket(log_scale_values, clipped_log_scale)
    corners = (
        (proportion_values[rho_low], scale_values[scale_low]),
        (proportion_values[rho_high], scale_values[scale_low]),
        (proportion_values[rho_low], scale_values[scale_high]),
        (proportion_values[rho_high], scale_values[scale_high]),
    )
    weights = (
        (1.0 - rho_fraction) * (1.0 - scale_fraction),
        rho_fraction * (1.0 - scale_fraction),
        (1.0 - rho_fraction) * scale_fraction,
        rho_fraction * scale_fraction,
    )
    corner_parameters = [oracle_settings[corner] for corner in corners]
    interpolated_p = sum(
        weight * parameters[0] for weight, parameters in zip(weights, corner_parameters, strict=True)
    )
    interpolated_log_delta = sum(
        weight * np.log(parameters[1])
        for weight, parameters in zip(weights, corner_parameters, strict=True)
    )
    interpolated_log_tolerance = sum(
        weight * np.log(parameters[2])
        for weight, parameters in zip(weights, corner_parameters, strict=True)
    )
    return (
        float(interpolated_p),
        float(np.exp(interpolated_log_delta)),
        float(np.exp(interpolated_log_tolerance)),
    )


def run_h5_experiment(
    output_path: Path = DEFAULT_RESULTS_PATH,
    *,
    oracle_path: Path = DEFAULT_ORACLE_PATH,
    parameters_path: Path = DEFAULT_PARAMETER_PATH,
    replicates: int = DEFAULT_REPLICATES,
    oracle_replicates: int = DEFAULT_ORACLE_REPLICATES,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    seed: int = DEFAULT_SEED,
    maxiter: int = DEFAULT_IRLS_MAXITER,
) -> tuple[Path, Path]:
    """Build a regime-specific H1-candidate oracle map and evaluate H5."""
    if replicates <= 0 or oracle_replicates <= 0:
        raise ValueError("'replicates' and 'oracle_replicates' must be positive.")
    if maxiter <= 0:
        raise ValueError("'maxiter' must be positive.")

    candidates = load_h1_candidates(parameters_path, candidate_limit)
    proportions = DEFAULT_PROPORTIONS
    scales = DEFAULT_OUTLIER_SCALES
    regimes = tuple(
        (proportion, scale) for proportion in proportions for scale in scales
    )
    true_coefficients = np.asarray(DEFAULT_BETA, dtype=float)
    oracle_records: list[dict[str, int | float]] = []
    oracle_settings: dict[tuple[float, float], tuple[float, float, float]] = {}

    for regime_index, (proportion, scale) in enumerate(regimes):
        candidate_scores: list[tuple[float, float, tuple[float, float, float]]] = []
        for candidate_index, (p, delta, tolerance) in enumerate(candidates):
            errors = []
            for replicate in range(oracle_replicates):
                data_seed = (
                    seed
                    + 100_000
                    + regime_index * len(candidates) * oracle_replicates
                    + candidate_index * oracle_replicates
                    + replicate
                    + 1
                )
                data = outlier_data(
                    DEFAULT_OBSERVATIONS,
                    true_coefficients,
                    DEFAULT_SIGMA,
                    DEFAULT_X_MIN,
                    DEFAULT_X_MAX,
                    proportion,
                    scale,
                    seed=data_seed,
                )
                coefficients = lp_irls(
                    data[:, :-1],
                    data[:, -1],
                    p,
                    delta,
                    maxiter=maxiter,
                    tolerance=tolerance,
                )
                if not np.all(np.isfinite(coefficients)):
                    raise RuntimeError(
                        f"Lp-IRLS returned non-finite coefficients for oracle "
                        f"regime (rho={proportion:g}, A={scale:g})."
                    )
                errors.append(float(np.linalg.norm(coefficients - true_coefficients)))

            candidate_scores.append(
                (
                    float(np.mean(errors)),
                    float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0,
                    (p, delta, tolerance),
                )
            )

        best_mean, best_std, best_parameters = min(candidate_scores, key=lambda x: x[0])
        oracle_settings[(proportion, scale)] = best_parameters
        oracle_records.append(
            {
                "outlier_proportion": proportion,
                "outlier_scale": scale,
                "lp_p": best_parameters[0],
                "lp_delta": best_parameters[1],
                "lp_tolerance": best_parameters[2],
                "tuning_mean_recovery_error": best_mean,
                "tuning_std_recovery_error": best_std,
                "candidate_count": len(candidates),
                "tuning_replicates": oracle_replicates,
            }
        )

    oracle_path = Path(oracle_path)
    oracle_path.parent.mkdir(parents=True, exist_ok=True)
    with oracle_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=ORACLE_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(oracle_records)

    anchor_parameters = load_best_lp_parameters(parameters_path)
    records: list[dict[str, int | float | str]] = []
    for regime_index, (proportion, scale) in enumerate(regimes):
        oracle_parameters = oracle_settings[(proportion, scale)]
        for replicate in range(replicates):
            data_seed = seed + regime_index * replicates + replicate + 1
            data = outlier_data(
                DEFAULT_OBSERVATIONS,
                true_coefficients,
                DEFAULT_SIGMA,
                DEFAULT_X_MIN,
                DEFAULT_X_MAX,
                proportion,
                scale,
                seed=data_seed,
            )
            X, y = data[:, :-1], data[:, -1]
            huber_coefficients, huber_delta = fit_adaptive_huber(X, y, maxiter)
            residuals = y - (huber_coefficients[0] + X @ huber_coefficients[1:])
            estimated_proportion, estimated_sigma, estimated_scale = (
                fit_zero_mean_gaussian_mixture(residuals)
            )
            lookup_regime = _nearest_regime(
                estimated_proportion, estimated_scale, regimes
            )
            estimated_parameters = _interpolate_oracle_parameters(
                estimated_proportion,
                estimated_scale,
                proportions,
                scales,
                oracle_settings,
            )
            true_parameters = _interpolate_oracle_parameters(
                proportion,
                scale,
                proportions,
                scales,
                oracle_settings,
            )
            nearest_parameters = oracle_settings[lookup_regime]
            estimates = (
                ("Huber", huber_coefficients, None, None),
                (
                    "Lp_estimated_regime_interpolated",
                    lp_irls(
                        X,
                        y,
                        *estimated_parameters[:2],
                        maxiter=maxiter,
                        tolerance=estimated_parameters[2],
                    ),
                    estimated_parameters,
                    (estimated_proportion, estimated_scale),
                ),
                (
                    "Lp_estimated_regime_nearest",
                    lp_irls(
                        X,
                        y,
                        *nearest_parameters[:2],
                        maxiter=maxiter,
                        tolerance=nearest_parameters[2],
                    ),
                    nearest_parameters,
                    lookup_regime,
                ),
                (
                    "Lp_true_regime_interpolated",
                    lp_irls(
                        X,
                        y,
                        *true_parameters[:2],
                        maxiter=maxiter,
                        tolerance=true_parameters[2],
                    ),
                    true_parameters,
                    (proportion, scale),
                ),
                (
                    "Lp_H1_anchor",
                    lp_irls(
                        X,
                        y,
                        *anchor_parameters[:2],
                        maxiter=maxiter,
                        tolerance=anchor_parameters[2],
                    ),
                    anchor_parameters,
                    (DEFAULT_PROPORTIONS[1], DEFAULT_OUTLIER_SCALES[1]),
                ),
            )
            for method, coefficients, parameters, selected_regime in estimates:
                if not np.all(np.isfinite(coefficients)):
                    raise RuntimeError(
                        f"{method} returned non-finite coefficients for "
                        f"regime (rho={proportion:g}, A={scale:g}), "
                        f"replicate {replicate + 1}."
                    )
                lp_p, lp_delta, lp_tolerance = (
                    parameters if parameters is not None else ("NA", "NA", "NA")
                )
                lookup_proportion, lookup_scale = (
                    selected_regime if selected_regime is not None else ("NA", "NA")
                )
                records.append(
                    {
                        "method": method,
                        "regime": regime_index + 1,
                        "replicate": replicate + 1,
                        "seed": data_seed,
                        "outlier_proportion": proportion,
                        "outlier_scale": scale,
                        "estimated_proportion": estimated_proportion,
                        "estimated_scale": estimated_scale,
                        "estimated_sigma": estimated_sigma,
                        "lookup_proportion": lookup_proportion,
                        "lookup_scale": lookup_scale,
                        "recovery_error": float(
                            np.linalg.norm(coefficients - true_coefficients)
                        ),
                        "estimated_intercept": float(coefficients[0]),
                        "estimated_slope": float(coefficients[1]),
                        "lp_p": lp_p,
                        "lp_delta": lp_delta,
                        "lp_tolerance": lp_tolerance,
                        "huber_delta": huber_delta if method == "Huber" else "NA",
                        "observations": DEFAULT_OBSERVATIONS,
                        "sigma": DEFAULT_SIGMA,
                    }
                )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=RESULT_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)
    return output_path, oracle_path


def main() -> None:
    """Parse command-line options and run the H5 experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, default=DEFAULT_PARAMETER_PATH)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument(
        "--oracle-replicates", type=int, default=DEFAULT_ORACLE_REPLICATES
    )
    parser.add_argument("--candidate-limit", type=int, default=DEFAULT_CANDIDATE_LIMIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_IRLS_MAXITER)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--oracle-output", type=Path, default=DEFAULT_ORACLE_PATH)
    args = parser.parse_args()
    result_path, oracle_path = run_h5_experiment(
        args.output,
        oracle_path=args.oracle_output,
        parameters_path=args.parameters,
        replicates=args.replicates,
        oracle_replicates=args.oracle_replicates,
        candidate_limit=args.candidate_limit,
        seed=args.seed,
        maxiter=args.maxiter,
    )
    print(result_path)
    print(oracle_path)


if __name__ == "__main__":
    main()
