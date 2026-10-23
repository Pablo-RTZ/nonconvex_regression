"""Measure coefficient recovery across solver tolerances."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nonconvex_regression.data.generators import outlier_data
from nonconvex_regression.helpers.helpers import calculate_coefficient_recovery
from nonconvex_regression.models.lp_irls import lp_irls

DEFAULT_TOLERANCE_BOUNDS = (1e-10, 1e-1)
DEFAULT_TOLERANCE_COUNT = 31


@dataclass(frozen=True)
class ContaminationScenario:
    """Outlier proportion and magnitude defining one contamination scenario."""

    name: str
    proportion: float
    magnitude: float


@dataclass(frozen=True)
class ToleranceEvaluation:
    """Per-experiment measurements for one solver tolerance."""

    tolerance: float
    experiment_errors: tuple[float, ...]
    solver_seconds: tuple[float, ...]
    iterations: tuple[int, ...]

    @property
    def mean_error(self) -> float:
        """Return mean coefficient recovery error."""
        return float(np.mean(self.experiment_errors))

    @property
    def mean_solver_seconds(self) -> float:
        """Return mean time spent inside the IRLS solver."""
        return float(np.mean(self.solver_seconds))

    @property
    def mean_iterations(self) -> float:
        """Return mean number of IRLS updates."""
        return float(np.mean(self.iterations))


@dataclass(frozen=True)
class ScenarioToleranceResult:
    """Tolerance measurements for one contamination scenario."""

    scenario: ContaminationScenario
    lp: float
    eps: float
    evaluations: tuple[ToleranceEvaluation, ...]
    experiment_seeds: tuple[int, ...]
    elbow_tolerance: float | None


def tolerance_scenarios() -> tuple[ContaminationScenario, ...]:
    """Return the contamination scenarios used by notebook 06."""
    return (
        ContaminationScenario("scenario_1", 0.02, 0.1),
        ContaminationScenario("scenario_2", 0.1, 0.5),
        ContaminationScenario("scenario_3", 0.25, 5.0),
    )


def read_best_parameters(
    csv_path: Path,
    scenarios: tuple[ContaminationScenario, ...],
) -> dict[str, tuple[float, float]]:
    """Read the unique best ``(lp, eps)`` pair for each requested scenario."""
    required_columns = {
        "scenario",
        "proportion",
        "magnitude",
        "lp",
        "eps",
        "is_best",
    }
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = required_columns - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: "
                f"{', '.join(sorted(missing))}"
            )
        best_rows: dict[str, set[tuple[float, float, float, float]]] = {}
        for row in reader:
            if row["is_best"].strip().lower() != "true":
                continue
            best_rows.setdefault(row["scenario"], set()).add(
                (
                    float(row["proportion"]),
                    float(row["magnitude"]),
                    float(row["lp"]),
                    float(row["eps"]),
                )
            )

    parameters: dict[str, tuple[float, float]] = {}
    for scenario in scenarios:
        rows = best_rows.get(scenario.name, set())
        if len(rows) != 1:
            raise ValueError(
                f"Expected one best parameter pair for {scenario.name} in "
                f"{csv_path}; found {len(rows)}."
            )
        proportion, magnitude, lp, eps = next(iter(rows))
        if not np.isclose(proportion, scenario.proportion) or not np.isclose(
            magnitude, scenario.magnitude
        ):
            raise ValueError(
                f"The saved contamination settings for {scenario.name} do not "
                "match the notebook 06 scenarios."
            )
        parameters[scenario.name] = (lp, eps)
    return parameters


def _find_elbow(evaluations: tuple[ToleranceEvaluation, ...]) -> float | None:
    """Find the largest interior distance from the error curve's endpoint chord."""
    if len(evaluations) < 3:
        return None
    log_tolerances = np.log10([item.tolerance for item in evaluations])
    errors = np.asarray([item.mean_error for item in evaluations])
    error_range = float(np.ptp(errors))
    if error_range == 0.0:
        return None

    points = np.column_stack(
        (
            (log_tolerances - log_tolerances[0])
            / (log_tolerances[-1] - log_tolerances[0]),
            (errors - errors.min()) / error_range,
        )
    )
    chord = points[-1] - points[0]
    chord_length = float(np.linalg.norm(chord))
    if chord_length == 0.0:
        return None
    offsets = points[1:-1] - points[0]
    distances = np.abs(chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0])
    distances /= chord_length
    elbow_index = int(np.argmax(distances)) + 1
    if distances[elbow_index - 1] <= np.finfo(float).eps:
        return None
    return evaluations[elbow_index].tolerance


def run_scenario(
    scenario: ContaminationScenario,
    lp: float,
    eps: float,
    tolerances: tuple[float, ...],
    *,
    experiments: int = 50,
    seed: int = 42,
    n: int = 1000,
    beta: tuple[float, ...] = (1.0, 1.0),
    noise_std: float = 1e-2,
    x_min: float = -2.0,
    x_max: float = 2.0,
    maxiter: int = 200,
) -> ScenarioToleranceResult:
    """Run each solver tolerance on the same seeded data sets."""
    tolerance_array = np.asarray(tolerances, dtype=float)
    if tolerance_array.ndim != 1 or tolerance_array.size < 3:
        raise ValueError("'tolerances' must contain at least three values.")
    if (
        not np.all(np.isfinite(tolerance_array))
        or np.any(tolerance_array <= 0.0)
        or np.any(np.diff(tolerance_array) <= 0.0)
    ):
        raise ValueError("'tolerances' must be positive, finite, and increasing.")
    if experiments <= 0 or n <= 0 or maxiter <= 0:
        raise ValueError("'experiments', 'n', and 'maxiter' must be positive.")
    if seed < 0:
        raise ValueError("'seed' must be non-negative.")
    if not 0.0 <= scenario.proportion <= 1.0:
        raise ValueError("'scenario.proportion' must be in [0, 1].")
    if not np.isfinite(scenario.magnitude) or scenario.magnitude < 0.0:
        raise ValueError("'scenario.magnitude' must be finite and non-negative.")

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

    datasets = []
    experiment_seeds = tuple(seed + index for index in range(experiments))
    for experiment_seed in experiment_seeds:
        points = outlier_data(
            n,
            true_coefficients,
            noise_std,
            x_min,
            x_max,
            scenario.proportion,
            scenario.magnitude,
            seed=experiment_seed,
        )
        datasets.append((points[:, :-1], points[:, -1]))

    evaluations: list[ToleranceEvaluation] = []
    for tolerance in tolerance_array:
        errors: list[float] = []
        solver_seconds: list[float] = []
        iteration_counts: list[int] = []
        for X, y in datasets:
            start_time = time.perf_counter()
            coefficients, iterations = lp_irls(
                X,
                y,
                lp,
                eps,
                maxiter=maxiter,
                tolerance=float(tolerance),
                return_iterations=True,
            )
            solver_seconds.append(time.perf_counter() - start_time)
            errors.append(
                calculate_coefficient_recovery(coefficients, true_coefficients)
            )
            iteration_counts.append(iterations)
        evaluations.append(
            ToleranceEvaluation(
                float(tolerance),
                tuple(errors),
                tuple(solver_seconds),
                tuple(iteration_counts),
            )
        )

    evaluation_tuple = tuple(evaluations)
    return ScenarioToleranceResult(
        scenario=scenario,
        lp=lp,
        eps=eps,
        evaluations=evaluation_tuple,
        experiment_seeds=experiment_seeds,
        elbow_tolerance=_find_elbow(evaluation_tuple),
    )


def write_results_csv(
    results: tuple[ScenarioToleranceResult, ...],
    output_path: Path,
) -> None:
    """Write per-experiment and aggregate tolerance measurements to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = (
            "scenario",
            "proportion",
            "magnitude",
            "lp",
            "eps",
            "tolerance_index",
            "tolerance",
            "experiment",
            "seed",
            "coefficient_recovery_error",
            "solver_seconds",
            "iterations",
            "mean_coefficient_recovery_error",
            "mean_solver_seconds",
            "mean_iterations",
            "comparison_tolerance",
            "log10_tolerance_step",
            "marginal_error_improvement",
            "marginal_improvement_per_log10_tolerance",
            "marginal_solver_time_increase_seconds",
            "marginal_improvement_per_added_second",
            "relative_improvement_per_log10_tolerance",
            "relative_improvement_per_added_second",
            "is_error_elbow",
        )
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            comparisons: dict[
                float,
                tuple[
                    float,
                    float,
                    float,
                    float,
                    float | None,
                    float | None,
                    float | None,
                ],
            ] = {}
            for tight, loose in zip(
                result.evaluations[:-1], result.evaluations[1:], strict=True
            ):
                log_step = float(np.log10(loose.tolerance / tight.tolerance))
                error_improvement = loose.mean_error - tight.mean_error
                time_increase = tight.mean_solver_seconds - loose.mean_solver_seconds
                relative_improvement = (
                    error_improvement / loose.mean_error
                    if loose.mean_error > 0.0
                    else None
                )
                comparisons[tight.tolerance] = (
                    loose.tolerance,
                    log_step,
                    error_improvement,
                    time_increase,
                    error_improvement / time_increase if time_increase > 0.0 else None,
                    (
                        relative_improvement / log_step
                        if relative_improvement is not None and log_step > 0.0
                        else None
                    ),
                    (
                        relative_improvement / time_increase
                        if relative_improvement is not None and time_increase > 0.0
                        else None
                    ),
                )

            for tolerance_index, evaluation in enumerate(result.evaluations):
                comparison = comparisons.get(evaluation.tolerance)
                for experiment_index, error in enumerate(evaluation.experiment_errors):
                    writer.writerow(
                        {
                            "scenario": result.scenario.name,
                            "proportion": result.scenario.proportion,
                            "magnitude": result.scenario.magnitude,
                            "lp": result.lp,
                            "eps": result.eps,
                            "tolerance_index": tolerance_index,
                            "tolerance": evaluation.tolerance,
                            "experiment": experiment_index + 1,
                            "seed": result.experiment_seeds[experiment_index],
                            "coefficient_recovery_error": error,
                            "solver_seconds": evaluation.solver_seconds[
                                experiment_index
                            ],
                            "iterations": evaluation.iterations[experiment_index],
                            "mean_coefficient_recovery_error": evaluation.mean_error,
                            "mean_solver_seconds": evaluation.mean_solver_seconds,
                            "mean_iterations": evaluation.mean_iterations,
                            "comparison_tolerance": comparison[0] if comparison else "",
                            "log10_tolerance_step": comparison[1] if comparison else "",
                            "marginal_error_improvement": (
                                comparison[2] if comparison else ""
                            ),
                            "marginal_improvement_per_log10_tolerance": (
                                comparison[2] / comparison[1] if comparison else ""
                            ),
                            "marginal_solver_time_increase_seconds": (
                                comparison[3] if comparison else ""
                            ),
                            "marginal_improvement_per_added_second": (
                                comparison[4]
                                if comparison and comparison[4] is not None
                                else ""
                            ),
                            "relative_improvement_per_log10_tolerance": (
                                comparison[5]
                                if comparison and comparison[5] is not None
                                else ""
                            ),
                            "relative_improvement_per_added_second": (
                                comparison[6]
                                if comparison and comparison[6] is not None
                                else ""
                            ),
                            "is_error_elbow": (
                                result.elbow_tolerance == evaluation.tolerance
                            ),
                        }
                    )


def main() -> None:
    """Run the default notebook 06 scenarios and save tolerance results."""
    parser = argparse.ArgumentParser(
        description="Measure solver-tolerance trade-offs for notebook 06 scenarios."
    )
    parser.add_argument(
        "--parameters", type=Path, default=Path("results/parameter_tuning.csv")
    )
    parser.add_argument("--experiments", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tolerance-min", type=float, default=DEFAULT_TOLERANCE_BOUNDS[0]
    )
    parser.add_argument(
        "--tolerance-max", type=float, default=DEFAULT_TOLERANCE_BOUNDS[1]
    )
    parser.add_argument("--tolerance-count", type=int, default=DEFAULT_TOLERANCE_COUNT)
    parser.add_argument(
        "--output", type=Path, default=Path("results/tolerance_tuning.csv")
    )
    args = parser.parse_args()
    if args.tolerance_count < 3:
        parser.error("--tolerance-count must be at least 3.")
    if (
        not np.isfinite(args.tolerance_min)
        or not np.isfinite(args.tolerance_max)
        or not 0.0 < args.tolerance_min < args.tolerance_max
    ):
        parser.error("tolerance bounds must satisfy 0 < min < max < infinity.")

    scenarios = tolerance_scenarios()
    parameters = read_best_parameters(args.parameters, scenarios)
    tolerances = tuple(
        np.logspace(
            np.log10(args.tolerance_min),
            np.log10(args.tolerance_max),
            args.tolerance_count,
        )
    )
    start_time = time.perf_counter()
    results = tuple(
        run_scenario(
            scenario,
            *parameters[scenario.name],
            tolerances,
            experiments=args.experiments,
            seed=args.seed,
        )
        for scenario in scenarios
    )
    write_results_csv(results, args.output)
    elapsed_seconds = time.perf_counter() - start_time

    for result in results:
        elbow = (
            f"{result.elbow_tolerance:.3g}"
            if result.elbow_tolerance is not None
            else "not identifiable"
        )
        print(
            f"{result.scenario.name} (p={result.scenario.proportion:g}, "
            f"A={result.scenario.magnitude:g}): lp={result.lp:.6g}, "
            f"eps={result.eps:.6g}, error elbow tolerance={elbow}"
        )
    simulation_count = sum(
        len(evaluation.experiment_errors)
        for result in results
        for evaluation in result.evaluations
    )
    print(f"Per-experiment results saved to {args.output}")
    print(f"Solver fits run: {simulation_count}")
    print(f"Total experiment time: {elapsed_seconds:.3f} seconds")


if __name__ == "__main__":
    main()
