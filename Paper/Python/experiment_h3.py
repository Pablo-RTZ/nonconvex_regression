"""Measure transfer of H1-tuned Lp-IRLS settings across nearby linear models."""

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
        DEFAULT_SEED,
        DEFAULT_SIGMA,
        DEFAULT_X_MAX,
        DEFAULT_X_MIN,
    )
    from .experiment_h1 import (
        DEFAULT_RESULTS_PATH as DEFAULT_PARAMETER_PATH,
    )
    from .experiment_h2 import load_best_lp_parameters
    from .solvers import lp_irls
else:
    from data_generation import outlier_data
    from experiment_h1 import (
        DEFAULT_BETA,
        DEFAULT_IRLS_MAXITER,
        DEFAULT_OBSERVATIONS,
        DEFAULT_OUTLIER_PROPORTION,
        DEFAULT_OUTLIER_SCALE,
        DEFAULT_SEED,
        DEFAULT_SIGMA,
        DEFAULT_X_MAX,
        DEFAULT_X_MIN,
    )
    from experiment_h1 import (
        DEFAULT_RESULTS_PATH as DEFAULT_PARAMETER_PATH,
    )
    from experiment_h2 import load_best_lp_parameters
    from solvers import lp_irls


# Default values for the fixed-contamination model-transfer experiment.
DEFAULT_REPLICATES = 200
DEFAULT_GRID_SIZE = 15
DEFAULT_INTERCEPT_MULTIPLIERS = tuple(np.linspace(0.8, 1.2, DEFAULT_GRID_SIZE))
DEFAULT_SLOPE_MULTIPLIERS = tuple(np.linspace(0.8, 1.2, DEFAULT_GRID_SIZE))
DEFAULT_RESULTS_PATH = (
    Path(__file__).resolve().parents[1] / "Results" / "h3_comparison.csv"
)

CSV_COLUMNS = (
    "replicate",
    "seed",
    "intercept_multiplier",
    "slope_multiplier",
    "anchor_intercept",
    "anchor_slope",
    "true_intercept",
    "true_slope",
    "recovery_error",
    "estimated_intercept",
    "estimated_slope",
    "lp_p",
    "lp_delta",
    "lp_tolerance",
    "observations",
    "sigma",
    "outlier_proportion",
    "outlier_scale",
)


def run_h3_experiment(
    output_path: Path = DEFAULT_RESULTS_PATH,
    *,
    parameters_path: Path = DEFAULT_PARAMETER_PATH,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
    maxiter: int = DEFAULT_IRLS_MAXITER,
) -> Path:
    """Evaluate the H1-selected Lp settings over nearby coefficient values.

    The intercept and slope are independently varied over their configured
    multiplier grids. Contamination and all other data-generation settings
    stay fixed. Each replicate uses the same seed across the model grid for
    paired comparisons.
    """
    if replicates <= 0:
        raise ValueError("'replicates' must be positive.")
    if maxiter <= 0:
        raise ValueError("'maxiter' must be positive.")

    base_coefficients = np.asarray(DEFAULT_BETA, dtype=float)
    if base_coefficients.shape != (2,) or not np.all(np.isfinite(base_coefficients)):
        raise ValueError("H3 requires finite intercept and slope in DEFAULT_BETA.")

    p, delta, tolerance = load_best_lp_parameters(parameters_path)
    records: list[dict[str, int | float]] = []

    for intercept_multiplier in DEFAULT_INTERCEPT_MULTIPLIERS:
        for slope_multiplier in DEFAULT_SLOPE_MULTIPLIERS:
            true_coefficients = base_coefficients * np.asarray(
                (intercept_multiplier, slope_multiplier)
            )
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
                coefficients = lp_irls(
                    X,
                    y,
                    p,
                    delta,
                    maxiter=maxiter,
                    tolerance=tolerance,
                )
                if not np.all(np.isfinite(coefficients)):
                    raise RuntimeError(
                        "Lp-IRLS returned non-finite coefficients for "
                        f"intercept multiplier {intercept_multiplier}, "
                        f"slope multiplier {slope_multiplier}, replicate "
                        f"{replicate + 1}."
                    )

                records.append(
                    {
                        "replicate": replicate + 1,
                        "seed": data_seed,
                        "intercept_multiplier": intercept_multiplier,
                        "slope_multiplier": slope_multiplier,
                        "anchor_intercept": float(base_coefficients[0]),
                        "anchor_slope": float(base_coefficients[1]),
                        "true_intercept": float(true_coefficients[0]),
                        "true_slope": float(true_coefficients[1]),
                        "recovery_error": float(
                            np.linalg.norm(coefficients - true_coefficients)
                        ),
                        "estimated_intercept": float(coefficients[0]),
                        "estimated_slope": float(coefficients[1]),
                        "lp_p": p,
                        "lp_delta": delta,
                        "lp_tolerance": tolerance,
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
    """Parse command-line options and run the H3 model-transfer experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, default=DEFAULT_PARAMETER_PATH)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_IRLS_MAXITER)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    args = parser.parse_args()
    print(
        run_h3_experiment(
            args.output,
            parameters_path=args.parameters,
            replicates=args.replicates,
            seed=args.seed,
            maxiter=args.maxiter,
        )
    )


if __name__ == "__main__":
    main()
