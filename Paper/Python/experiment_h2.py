"""Compare tuned Lp-IRLS with OLS and Huber regression for hypothesis 2."""

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
        DEFAULT_OUTLIER_PROPORTION,
        DEFAULT_OUTLIER_SCALE,
        DEFAULT_RESULTS_PATH as DEFAULT_PARAMETER_PATH,
        DEFAULT_SEED,
        DEFAULT_SIGMA,
        DEFAULT_X_MAX,
        DEFAULT_X_MIN,
    )
    from .solvers import huber_irls, lp_irls, ols
else:
    from data_generation import outlier_data
    from experiment_h1 import (
        DEFAULT_BETA,
        DEFAULT_IRLS_MAXITER,
        DEFAULT_OBSERVATIONS,
        DEFAULT_OUTLIER_PROPORTION,
        DEFAULT_OUTLIER_SCALE,
        DEFAULT_RESULTS_PATH as DEFAULT_PARAMETER_PATH,
        DEFAULT_SEED,
        DEFAULT_SIGMA,
        DEFAULT_X_MAX,
        DEFAULT_X_MIN,
    )
    from solvers import huber_irls, lp_irls, ols


DEFAULT_REPLICATES = 1000
DEFAULT_HUBER_DELTA_MULTIPLIER = 1.345
DEFAULT_RESULTS_PATH = (
    Path(__file__).resolve().parents[1] / "Results" / "h2_comparison.csv"
)

CSV_COLUMNS = (
    "method",
    "replicate",
    "seed",
    "recovery_error",
    "estimated_intercept",
    "estimated_slope",
    "lp_p",
    "lp_delta",
    "lp_tolerance",
    "huber_delta",
    "observations",
    "sigma",
    "outlier_proportion",
    "outlier_scale",
)


def load_best_lp_parameters(parameters_path: Path) -> tuple[float, float, float]:
    """Load the unique parameter setting selected by the H1 search."""
    required_columns = {"p", "delta", "tolerance", "is_best"}
    with Path(parameters_path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = required_columns - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{parameters_path} is missing required columns: "
                f"{', '.join(sorted(missing))}"
            )
        best_rows = [
            row
            for row in reader
            if (row["is_best"] or "").strip().lower() == "true"
        ]

    if len(best_rows) != 1:
        raise ValueError(
            f"{parameters_path} must mark exactly one H1 setting with is_best=true; "
            f"found {len(best_rows)}."
        )

    try:
        p = float(best_rows[0]["p"] or "")
        delta = float(best_rows[0]["delta"] or "")
        tolerance = float(best_rows[0]["tolerance"] or "")
    except ValueError as error:
        raise ValueError(
            f"{parameters_path} contains invalid values for the selected H1 setting."
        ) from error

    if not np.all(np.isfinite([p, delta, tolerance])):
        raise ValueError(
            f"{parameters_path} contains non-finite values for the selected H1 setting."
        )
    if not 0.0 < p < 1.0 or delta <= 0.0 or tolerance <= 0.0:
        raise ValueError(
            f"{parameters_path} contains an invalid selected H1 setting: "
            "p must be in (0, 1), and delta and tolerance must be positive."
        )
    return p, delta, tolerance


def run_h2_experiment(
    output_path: Path = DEFAULT_RESULTS_PATH,
    *,
    parameters_path: Path = DEFAULT_PARAMETER_PATH,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
    maxiter: int = DEFAULT_IRLS_MAXITER,
) -> Path:
    """Compare OLS, Huber, and H1-tuned Lp coefficient recovery.

    All methods are fitted to the same generated sample in each replicate.
    Huber's threshold is fixed at 1.345 times the known Gaussian noise
    standard deviation, a conventional choice under Gaussian clean noise.

    Args:
        output_path: CSV destination for the per-replicate estimates.
        parameters_path: H1 Bayesian-optimization CSV containing its selected
            setting, marked by ``is_best``.
        replicates: Number of paired contaminated datasets to generate.
        seed: Base seed; replicate ``i`` uses ``seed + i + 1``.
        maxiter: Maximum IRLS updates for Huber and Lp.

    Returns:
        The path to the CSV containing the per-method recovery errors.

    Raises:
        ValueError: If the replicate count or iteration limit is invalid, or
            the H1 result file does not contain one valid selected setting.
        RuntimeError: If a solver returns a non-finite coefficient estimate.
    """
    if replicates <= 0:
        raise ValueError("'replicates' must be positive.")
    if maxiter <= 0:
        raise ValueError("'maxiter' must be positive.")

    p, lp_delta, tolerance = load_best_lp_parameters(parameters_path)
    huber_delta = DEFAULT_HUBER_DELTA_MULTIPLIER * DEFAULT_SIGMA
    true_coefficients = np.asarray(DEFAULT_BETA, dtype=float)
    records: list[dict[str, int | float | str]] = []

    for replicate in range(replicates):
        data_seed = seed + replicate + 1
        data = outlier_data(
            DEFAULT_OBSERVATIONS,
            true_coefficients,
            DEFAULT_SIGMA,
            DEFAULT_X_MIN,
            DEFAULT_X_MAX,
            DEFAULT_OUTLIER_PROPORTION,
            DEFAULT_OUTLIER_SCALE,
            seed=data_seed,
        )
        X, y = data[:, :-1], data[:, -1]
        estimates = (
            ("OLS", ols(X, y)),
            (
                "Huber",
                huber_irls(
                    X,
                    y,
                    huber_delta,
                    maxiter=maxiter,
                    tolerance=tolerance,
                ),
            ),
            (
                "Lp",
                lp_irls(
                    X,
                    y,
                    p,
                    lp_delta,
                    maxiter=maxiter,
                    tolerance=tolerance,
                ),
            ),
        )

        for method, coefficients in estimates:
            if not np.all(np.isfinite(coefficients)):
                raise RuntimeError(
                    f"{method} returned non-finite coefficients for replicate "
                    f"{replicate + 1}."
                )
            records.append(
                {
                    "method": method,
                    "replicate": replicate + 1,
                    "seed": data_seed,
                    "recovery_error": float(
                        np.linalg.norm(coefficients - true_coefficients)
                    ),
                    "estimated_intercept": float(coefficients[0]),
                    "estimated_slope": float(coefficients[1]),
                    "lp_p": p,
                    "lp_delta": lp_delta,
                    "lp_tolerance": tolerance,
                    "huber_delta": huber_delta,
                    "observations": DEFAULT_OBSERVATIONS,
                    "sigma": DEFAULT_SIGMA,
                    "outlier_proportion": DEFAULT_OUTLIER_PROPORTION,
                    "outlier_scale": DEFAULT_OUTLIER_SCALE,
                }
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    return output_path


def main() -> None:
    """Parse command-line options and run the H2 comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, default=DEFAULT_PARAMETER_PATH)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_IRLS_MAXITER)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    args = parser.parse_args()
    print(
        run_h2_experiment(
            args.output,
            parameters_path=args.parameters,
            replicates=args.replicates,
            seed=args.seed,
            maxiter=args.maxiter,
        )
    )


if __name__ == "__main__":
    main()
