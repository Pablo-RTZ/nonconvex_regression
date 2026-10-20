"""Tune the smoothed Lp estimator over contamination scenarios."""

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

DEFAULT_LP_BOUNDS = (0.4, 0.9)


@dataclass(frozen=True)
class ContaminationScenario:
    """Outlier proportion and magnitude defining one contamination scenario."""

    name: str
    proportion: float
    magnitude: float


@dataclass(frozen=True)
class ParameterEvaluation:
    """Recovery errors for one parameter pair across repeated experiments."""

    round_number: int
    lp: float
    eps: float
    experiment_errors: tuple[float, ...]

    @property
    def mean_error(self) -> float:
        """Return the mean coefficient recovery error across experiments."""
        return float(np.mean(self.experiment_errors))


@dataclass(frozen=True)
class ScenarioSearchResult:
    """Search results and the best evaluated parameters for a scenario."""

    scenario: ContaminationScenario
    best: ParameterEvaluation
    evaluations: tuple[ParameterEvaluation, ...]
    experiment_seeds: tuple[int, ...]


def default_scenarios() -> tuple[ContaminationScenario, ...]:
    """Return the three proportion/magnitude pairs in the tuning notebook."""
    return (
        ContaminationScenario("low", 0.02, 0.1),
        ContaminationScenario("medium", 0.1, 0.5),
        ContaminationScenario("high", 0.25, 2.5),
    )


def tune_scenario(
    scenario: ContaminationScenario,
    *,
    experiments: int = 20,
    rounds: int = 5,
    seed: int = 42,
    n: int = 1000,
    beta: tuple[float, ...] = (1.0, 1.0),
    noise_std: float = 1e-2,
    x_min: float = -2.0,
    x_max: float = 2.0,
    lp_bounds: tuple[float, float] = DEFAULT_LP_BOUNDS,
    eps_bounds: tuple[float, float] = (1e-6, 1e-1),
    maxiter: int = 200,
    tolerance: float = 1e-6,
) -> ScenarioSearchResult:
    """Find lp and eps by quadtree search over repeated seeded data sets.

    By default, the search covers ``lp`` from 0.4 to 0.9 and ``eps`` from
    1e-6 to 1e-1.

    The eps dimension is split in log space because the search range spans
    several orders of magnitude. In each round, the four child-cell centers
    are evaluated, then the best center's quadrant becomes the next search
    region. Candidate scores are the mean coefficient recovery error over the
    same generated data sets.
    """
    if experiments <= 0:
        raise ValueError("'experiments' must be positive.")
    if rounds <= 0:
        raise ValueError("'rounds' must be positive.")
    if n <= 0:
        raise ValueError("'n' must be positive.")
    if maxiter <= 0:
        raise ValueError("'maxiter' must be positive.")
    if seed < 0:
        raise ValueError("'seed' must be non-negative.")
    if not 0.0 <= scenario.proportion <= 1.0:
        raise ValueError("'scenario.proportion' must be in [0, 1].")
    if not np.isfinite(scenario.magnitude) or scenario.magnitude < 0.0:
        raise ValueError("'scenario.magnitude' must be finite and non-negative.")

    lp_low, lp_high = lp_bounds
    if not 0.0 < lp_low < lp_high < 1.0:
        raise ValueError("'lp_bounds' must satisfy 0 < low < high < 1.")
    eps_low, eps_high = eps_bounds
    if not 0.0 < eps_low < eps_high or not np.isfinite(eps_high):
        raise ValueError("'eps_bounds' must satisfy 0 < low < high < infinity.")
    if not x_min < x_max:
        raise ValueError("'x_min' must be less than 'x_max'.")
    if not np.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("'noise_std' must be finite and non-negative.")

    true_coefficients = np.asarray(beta, dtype=float)
    if true_coefficients.ndim != 1 or true_coefficients.size != 2:
        raise ValueError(
            "'beta' must contain an intercept and one slope for scalar x bounds."
        )
    if not np.all(np.isfinite(true_coefficients)):
        raise ValueError("'beta' must contain only finite values.")

    datasets = []
    for experiment_index in range(experiments):
        experiment_seed = seed + experiment_index
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

    evaluations: list[ParameterEvaluation] = []
    best: ParameterEvaluation | None = None
    log_eps_low = float(np.log10(eps_low))
    log_eps_high = float(np.log10(eps_high))

    for round_number in range(1, rounds + 1):
        lp_mid = (lp_low + lp_high) / 2.0
        log_eps_mid = (log_eps_low + log_eps_high) / 2.0
        lp_centers = (
            (lp_low + lp_mid) / 2.0,
            (lp_mid + lp_high) / 2.0,
        )
        log_eps_centers = (
            (log_eps_low + log_eps_mid) / 2.0,
            (log_eps_mid + log_eps_high) / 2.0,
        )

        round_evaluations: list[ParameterEvaluation] = []
        for lp in lp_centers:
            for log_eps in log_eps_centers:
                eps = float(10.0**log_eps)
                errors = tuple(
                    calculate_coefficient_recovery(
                        lp_irls(
                            X,
                            y,
                            lp,
                            eps,
                            maxiter=maxiter,
                            tolerance=tolerance,
                        ),
                        true_coefficients,
                    )
                    for X, y in datasets
                )
                evaluation = ParameterEvaluation(round_number, lp, eps, errors)
                round_evaluations.append(evaluation)
                evaluations.append(evaluation)

        round_best = min(round_evaluations, key=lambda item: item.mean_error)
        if best is None or round_best.mean_error < best.mean_error:
            best = round_best

        if round_best.lp < lp_mid:
            lp_high = lp_mid
        else:
            lp_low = lp_mid
        best_log_eps = float(np.log10(round_best.eps))
        if best_log_eps < log_eps_mid:
            log_eps_high = log_eps_mid
        else:
            log_eps_low = log_eps_mid

    if best is None:
        raise RuntimeError("The parameter search completed without evaluations.")
    return ScenarioSearchResult(
        scenario,
        best,
        tuple(evaluations),
        tuple(seed + index for index in range(experiments)),
    )


def write_results_csv(
    results: tuple[ScenarioSearchResult, ...], output_path: Path
) -> None:
    """Write per-experiment candidate errors and scenario-best markers to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "scenario",
                "proportion",
                "magnitude",
                "round",
                "lp",
                "eps",
                "experiment",
                "seed",
                "coefficient_recovery_error",
                "mean_coefficient_recovery_error",
                "is_best",
            ),
        )
        writer.writeheader()
        for result in results:
            for evaluation in result.evaluations:
                is_best = evaluation is result.best
                for experiment_index, error in enumerate(evaluation.experiment_errors):
                    writer.writerow(
                        {
                            "scenario": result.scenario.name,
                            "proportion": result.scenario.proportion,
                            "magnitude": result.scenario.magnitude,
                            "round": evaluation.round_number,
                            "lp": evaluation.lp,
                            "eps": evaluation.eps,
                            "experiment": experiment_index + 1,
                            "seed": result.experiment_seeds[experiment_index],
                            "coefficient_recovery_error": error,
                            "mean_coefficient_recovery_error": evaluation.mean_error,
                            "is_best": is_best,
                        }
                    )


def main() -> None:
    """Run the default three-scenario search and save detailed CSV results."""
    parser = argparse.ArgumentParser(
        description="Tune lp and eps separately for three contamination scenarios."
    )
    parser.add_argument("--experiments", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("results/parameter_tuning.csv")
    )
    args = parser.parse_args()

    start_time = time.perf_counter()
    results = tuple(
        tune_scenario(
            scenario,
            experiments=args.experiments,
            rounds=args.rounds,
            seed=args.seed,
        )
        for scenario in default_scenarios()
    )
    write_results_csv(results, args.output)
    elapsed_seconds = time.perf_counter() - start_time
    simulation_count = sum(
        len(evaluation.experiment_errors)
        for result in results
        for evaluation in result.evaluations
    )
    for result in results:
        print(
            f"{result.scenario.name} (p={result.scenario.proportion:g}, "
            f"A={result.scenario.magnitude:g}): "
            f"lp={result.best.lp:.6g}, eps={result.best.eps:.6g}, "
            f"mean recovery error={result.best.mean_error:.6g}"
        )
    print(f"Detailed candidate errors written to {args.output}")
    print(f"Simulations run: {simulation_count}")
    print(f"Total experiment time: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
