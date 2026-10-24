"""Test transfer of tuned Lp parameters to nearby contamination conditions."""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nonconvex_regression.data.generators import clean_data, outlier_data
from nonconvex_regression.helpers.helpers import calculate_coefficient_recovery
from nonconvex_regression.models.lp_irls import lp_irls
from nonconvex_regression.models.ols import ols


@dataclass(frozen=True)
class ContaminationScenario:
    """An anchor contamination condition and its tuned Lp parameters."""

    name: str
    proportion: float
    magnitude: float
    lp: float
    eps: float
    tolerance: float


def default_scenarios() -> tuple[tuple[str, float, float], ...]:
    """Return the three anchor conditions used by the tuning experiments."""
    return (
        ("scenario_1", 0.02, 0.1),
        ("scenario_2", 0.1, 0.5),
        ("scenario_3", 0.25, 5.0),
    )


def read_frozen_parameters(csv_path: Path) -> tuple[ContaminationScenario, ...]:
    """Read each scenario's Lp, smoothing, and elbow tolerance from tuning output."""
    required_columns = {
        "scenario",
        "proportion",
        "magnitude",
        "lp",
        "eps",
        "tolerance",
        "is_error_elbow",
    }
    selected: dict[str, set[tuple[float, float, float, float, float]]] = {}
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = required_columns - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            if row["is_error_elbow"].strip().lower() != "true":
                continue
            selected.setdefault(row["scenario"], set()).add(
                (
                    float(row["proportion"]),
                    float(row["magnitude"]),
                    float(row["lp"]),
                    float(row["eps"]),
                    float(row["tolerance"]),
                )
            )

    scenarios: list[ContaminationScenario] = []
    for name, proportion, magnitude in default_scenarios():
        values = selected.get(name, set())
        if len(values) != 1:
            raise ValueError(
                f"Expected one selected parameter set for {name} in {csv_path}; "
                f"found {len(values)}."
            )
        saved_proportion, saved_magnitude, lp, eps, tolerance = next(iter(values))
        if not np.isclose(saved_proportion, proportion) or not np.isclose(
            saved_magnitude, magnitude
        ):
            raise ValueError(
                f"The saved contamination settings for {name} do not match the "
                "expected anchor conditions."
            )
        scenarios.append(
            ContaminationScenario(name, proportion, magnitude, lp, eps, tolerance)
        )
    return tuple(scenarios)


def _target_conditions(
    scenario: ContaminationScenario,
    proportion_shift_fraction: float,
    magnitude_multipliers: tuple[float, ...],
) -> tuple[tuple[float, float], ...]:
    """Build the local proportion-by-magnitude grid around an anchor."""
    proportions = {
        scenario.proportion * (1.0 + shift)
        for shift in (
            -proportion_shift_fraction,
            0.0,
            proportion_shift_fraction,
        )
    }
    magnitudes = {
        scenario.magnitude * multiplier for multiplier in magnitude_multipliers
    }
    return tuple(itertools.product(sorted(proportions), sorted(magnitudes)))


def run_experiment(
    scenarios: tuple[ContaminationScenario, ...],
    *,
    experiments: int = 1000,
    seed: int = 42,
    proportion_shift_fraction: float = 0.1,
    magnitude_multipliers: tuple[float, ...] = (0.8, 1.0, 1.2),
    n_train: int = 1000,
    n_test: int = 1000,
    beta: tuple[float, ...] = (1.0, 1.0),
    noise_std: float = 1e-2,
    x_min: float = -2.0,
    x_max: float = 2.0,
    maxiter: int = 200,
) -> list[dict[str, object]]:
    """Evaluate frozen anchor parameters and OLS on paired clean test sets."""
    if experiments <= 0 or n_train <= 0 or n_test <= 0 or maxiter <= 0:
        raise ValueError("'experiments', sample sizes, and 'maxiter' must be positive.")
    if seed < 0:
        raise ValueError("'seed' must be non-negative.")
    if (
        not np.isfinite(proportion_shift_fraction)
        or not 0.0 <= proportion_shift_fraction <= 1.0
    ):
        raise ValueError("'proportion_shift_fraction' must be finite and in [0, 1].")
    multipliers = np.asarray(magnitude_multipliers, dtype=float)
    if (
        multipliers.ndim != 1
        or multipliers.size == 0
        or not np.all(np.isfinite(multipliers))
        or np.any(multipliers <= 0.0)
    ):
        raise ValueError("'magnitude_multipliers' must be positive finite values.")
    true_coefficients = np.asarray(beta, dtype=float)
    if true_coefficients.ndim != 1 or true_coefficients.size != 2:
        raise ValueError(
            "'beta' must contain an intercept and one slope for scalar x bounds."
        )
    if not np.all(np.isfinite(true_coefficients)):
        raise ValueError("'beta' must contain only finite values.")
    if not x_min < x_max:
        raise ValueError("'x_min' must be less than 'x_max'.")
    if not np.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("'noise_std' must be finite and non-negative.")

    rows: list[dict[str, object]] = []
    for anchor in scenarios:
        targets = _target_conditions(
            anchor,
            proportion_shift_fraction,
            tuple(float(item) for item in multipliers),
        )
        for target_proportion, target_magnitude in targets:
            for experiment_index in range(experiments):
                experiment_seed = seed + experiment_index
                test_seed = seed + experiments + experiment_index
                training = outlier_data(
                    n_train,
                    true_coefficients,
                    noise_std,
                    x_min,
                    x_max,
                    target_proportion,
                    target_magnitude,
                    seed=experiment_seed,
                )
                test = clean_data(
                    n_test,
                    true_coefficients,
                    noise_std,
                    x_min,
                    x_max,
                    seed=test_seed,
                )
                X_train, y_train = training[:, :-1], training[:, -1]
                X_test, y_test = test[:, :-1], test[:, -1]

                for method in ("Lp", "OLS"):
                    start_time = time.perf_counter()
                    if method == "Lp":
                        coefficients, iterations = lp_irls(
                            X_train,
                            y_train,
                            anchor.lp,
                            anchor.eps,
                            maxiter=maxiter,
                            tolerance=anchor.tolerance,
                            return_iterations=True,
                        )
                    else:
                        coefficients = ols(X_train, y_train)
                        iterations = None
                    fit_seconds = time.perf_counter() - start_time

                    predictions = coefficients[0] + X_test @ coefficients[1:]
                    test_rmse = float(np.sqrt(np.mean((predictions - y_test) ** 2)))
                    rows.append(
                        {
                            "anchor_scenario": anchor.name,
                            "anchor_proportion": anchor.proportion,
                            "anchor_magnitude": anchor.magnitude,
                            "target_proportion": target_proportion,
                            "target_magnitude": target_magnitude,
                            "proportion_relative_shift": (
                                target_proportion / anchor.proportion - 1.0
                                if anchor.proportion > 0.0
                                else 0.0
                            ),
                            "magnitude_ratio": target_magnitude / anchor.magnitude,
                            "lp": anchor.lp,
                            "eps": anchor.eps,
                            "tolerance": anchor.tolerance,
                            "method": method,
                            "experiment": experiment_index + 1,
                            "seed": experiment_seed,
                            "test_seed": test_seed,
                            "clean_test_rmse": test_rmse,
                            "coefficient_recovery_error": (
                                calculate_coefficient_recovery(
                                    coefficients, true_coefficients
                                )
                            ),
                            "fit_seconds": fit_seconds,
                            "iterations": iterations,
                        }
                    )
    return rows


def write_results_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write per-fit Monte Carlo measurements to CSV."""
    if not rows:
        raise ValueError("No experiment results were provided.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the local hyperparameter-stability experiment."""
    parser = argparse.ArgumentParser(
        description="Test tuned Lp parameters on nearby contamination conditions."
    )
    parser.add_argument(
        "--parameters", type=Path, default=Path("results/tolerance_tuning.csv")
    )
    parser.add_argument("--experiments", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proportion-shift-fraction", type=float, default=0.1)
    parser.add_argument(
        "--magnitude-multipliers",
        type=float,
        nargs=3,
        default=(0.8, 1.0, 1.2),
        metavar=("LOWER", "CENTER", "UPPER"),
    )
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--noise-std", type=float, default=1e-2)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hyperparameter_stability.csv"),
    )
    args = parser.parse_args()

    scenarios = read_frozen_parameters(args.parameters)
    start_time = time.perf_counter()
    rows = run_experiment(
        scenarios,
        experiments=args.experiments,
        seed=args.seed,
        proportion_shift_fraction=args.proportion_shift_fraction,
        magnitude_multipliers=tuple(args.magnitude_multipliers),
        n_train=args.n_train,
        n_test=args.n_test,
        noise_std=args.noise_std,
        maxiter=args.maxiter,
    )
    write_results_csv(rows, args.output)
    elapsed_seconds = time.perf_counter() - start_time
    fit_count = len(rows)
    print(f"Per-fit results written to {args.output}")
    print(f"Fits run: {fit_count}")
    print(f"Total experiment time: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
