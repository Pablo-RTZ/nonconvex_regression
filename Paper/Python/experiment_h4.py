"""Measure H1-tuned Lp-IRLS recovery across contamination regimes."""

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


DEFAULT_REPLICATES = 200
DEFAULT_GRID_SIZE = 15
DEFAULT_PROPORTION_MULTIPLIERS = tuple(np.linspace(0.8, 1.2, DEFAULT_GRID_SIZE))
DEFAULT_AMPLITUDE_MULTIPLIERS = tuple(np.linspace(0.5, 10, DEFAULT_GRID_SIZE))
DEFAULT_RESULTS_PATH = (
    Path(__file__).resolve().parents[1] / "Results" / "h4_comparison.csv"
)

CSV_COLUMNS = (
    "replicate",
    "seed",
    "proportion_multiplier",
    "amplitude_multiplier",
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


def run_h4_experiment(
    output_path: Path = DEFAULT_RESULTS_PATH,
    *,
    parameters_path: Path = DEFAULT_PARAMETER_PATH,
    replicates: int = DEFAULT_REPLICATES,
    grid_size: int | None = None,
    seed: int = DEFAULT_SEED,
    maxiter: int = DEFAULT_IRLS_MAXITER,
) -> Path:
    """Evaluate the H1-selected Lp settings across contamination regimes.

    Outlier proportion and amplitude are independently varied on a square grid
    using the configured multiplier grids. The model and other
    data-generation settings remain fixed. Replicates use sequential,
    non-overlapping seeds across the grid. When ``grid_size`` is provided, each
    configured multiplier range is resampled to that many evenly spaced
    values.
    """
    if replicates <= 0:
        raise ValueError("'replicates' must be positive.")
    if grid_size is not None and grid_size <= 0:
        raise ValueError("'grid_size' must be positive.")
    if maxiter <= 0:
        raise ValueError("'maxiter' must be positive.")

    true_coefficients = np.asarray(DEFAULT_BETA, dtype=float)
    if true_coefficients.ndim != 1 or not np.all(np.isfinite(true_coefficients)):
        raise ValueError("H4 requires finite coefficients in DEFAULT_BETA.")

    p, delta, tolerance = load_best_lp_parameters(parameters_path)
    proportion_multipliers = DEFAULT_PROPORTION_MULTIPLIERS
    amplitude_multipliers = DEFAULT_AMPLITUDE_MULTIPLIERS
    if grid_size is not None:
        proportion_multipliers = tuple(
            np.linspace(
                min(DEFAULT_PROPORTION_MULTIPLIERS),
                max(DEFAULT_PROPORTION_MULTIPLIERS),
                grid_size,
            )
        )
        amplitude_multipliers = tuple(
            np.linspace(
                min(DEFAULT_AMPLITUDE_MULTIPLIERS),
                max(DEFAULT_AMPLITUDE_MULTIPLIERS),
                grid_size,
            )
        )
    records: list[dict[str, int | float]] = []
    next_seed = seed + 1

    for proportion_multiplier in proportion_multipliers:
        outlier_proportion = DEFAULT_OUTLIER_PROPORTION * proportion_multiplier
        for amplitude_multiplier in amplitude_multipliers:
            outlier_scale = DEFAULT_OUTLIER_SCALE * amplitude_multiplier
            for replicate in range(replicates):
                data_seed = next_seed
                next_seed += 1
                data = outlier_data(
                    DEFAULT_OBSERVATIONS,
                    true_coefficients,
                    DEFAULT_SIGMA,
                    DEFAULT_X_MIN,
                    DEFAULT_X_MAX,
                    outlier_proportion,
                    outlier_scale,
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
                        f"proportion multiplier {proportion_multiplier}, "
                        f"amplitude multiplier {amplitude_multiplier}, "
                        f"replicate {replicate + 1}."
                    )

                records.append(
                    {
                        "replicate": replicate + 1,
                        "seed": data_seed,
                        "proportion_multiplier": proportion_multiplier,
                        "amplitude_multiplier": amplitude_multiplier,
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
                        "outlier_proportion": outlier_proportion,
                        "outlier_scale": outlier_scale,
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
    """Parse command-line options and run the H4 contamination-transfer experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, default=DEFAULT_PARAMETER_PATH)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help=(
            "resample both configured multiplier ranges to this many points "
            "(default: use the configured multiplier values)"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_IRLS_MAXITER)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    args = parser.parse_args()
    print(
        run_h4_experiment(
            args.output,
            parameters_path=args.parameters,
            replicates=args.replicates,
            grid_size=args.grid_size,
            seed=args.seed,
            maxiter=args.maxiter,
        )
    )


if __name__ == "__main__":
    main()
