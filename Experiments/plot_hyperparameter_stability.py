"""Plot local hyperparameter-stability results from a CSV file."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np

REQUIRED_COLUMNS = {
    "anchor_scenario",
    "target_proportion",
    "target_magnitude",
    "method",
    "clean_test_rmse",
}


class _GroupedMeasurements(TypedDict):
    """Measurements for one method and target contamination condition."""

    errors: list[float]
    proportion: float
    magnitude: float


def _read_summaries(
    csv_path: Path,
) -> dict[str, dict[tuple[str, float, float], tuple[float, float, int]]]:
    """Aggregate clean-test RMSE into means, standard errors, and sample counts."""
    grouped: dict[tuple[str, str, float, float], _GroupedMeasurements] = {}
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            scenario = row["anchor_scenario"]
            method = row["method"]
            proportion = float(row["target_proportion"])
            magnitude = float(row["target_magnitude"])
            rmse = float(row["clean_test_rmse"])
            if not np.isfinite(rmse):
                raise ValueError(f"{csv_path} contains a non-finite clean-test RMSE.")
            key = (scenario, method, proportion, magnitude)
            grouped.setdefault(
                key,
                {
                    "errors": [],
                    "proportion": proportion,
                    "magnitude": magnitude,
                },
            )["errors"].append(rmse)

    summaries: dict[str, dict[tuple[str, float, float], tuple[float, float, int]]] = (
        defaultdict(dict)
    )
    for (scenario, method, proportion, magnitude), entry in grouped.items():
        errors = np.asarray(entry["errors"], dtype=float)
        standard_error = (
            float(np.std(errors, ddof=1) / np.sqrt(errors.size))
            if errors.size > 1
            else 0.0
        )
        summaries[scenario][(method, proportion, magnitude)] = (
            float(np.mean(errors)),
            standard_error,
            int(errors.size),
        )
    if not summaries:
        raise ValueError(f"No experiment results found in {csv_path}.")
    return dict(summaries)


def _scenario_figures(
    summaries: dict[str, dict[tuple[str, float, float], tuple[float, float, int]]],
) -> tuple[plt.Figure, ...]:
    """Create RMSE-grid, Lp-versus-OLS, and neighborhood trend figures."""
    scenarios = list(summaries)
    figures: list[plt.Figure] = []

    figure, axes = plt.subplots(
        len(scenarios),
        2,
        figsize=(11, 4 * len(scenarios)),
        squeeze=False,
        layout="constrained",
    )
    for row_index, scenario in enumerate(scenarios):
        entries = summaries[scenario]
        proportions = sorted({key[1] for key in entries})
        magnitudes = sorted({key[2] for key in entries})
        for column_index, method in enumerate(("Lp", "OLS")):
            values = np.full((len(proportions), len(magnitudes)), np.nan)
            for i, proportion in enumerate(proportions):
                for j, magnitude in enumerate(magnitudes):
                    measurement = entries.get((method, proportion, magnitude))
                    if measurement is not None:
                        values[i, j] = measurement[0]
            image = axes[row_index, column_index].imshow(
                values,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
            )
            axes[row_index, column_index].set_xticks(
                range(len(magnitudes)),
                [f"{value:g}" for value in magnitudes],
            )
            axes[row_index, column_index].set_yticks(
                range(len(proportions)),
                [f"{value:g}" for value in proportions],
            )
            axes[row_index, column_index].set_xlabel("Outlier standard deviation A")
            axes[row_index, column_index].set_ylabel("Contamination proportion c")
            axes[row_index, column_index].set_title(
                f"{scenario}: {method} clean-test RMSE"
            )
            figure.colorbar(image, ax=axes[row_index, column_index], label="RMSE")
    figure.suptitle("Clean-test RMSE across local contamination conditions")
    figures.append(figure)

    figure, axes = plt.subplots(
        1,
        len(scenarios),
        figsize=(5.5 * len(scenarios), 4.5),
        squeeze=False,
        layout="constrained",
    )
    for axis, scenario in zip(axes[0], scenarios, strict=True):
        entries = summaries[scenario]
        proportions = sorted({key[1] for key in entries})
        magnitudes = sorted({key[2] for key in entries})
        differences = np.full((len(proportions), len(magnitudes)), np.nan)
        for i, proportion in enumerate(proportions):
            for j, magnitude in enumerate(magnitudes):
                lp_result = entries.get(("Lp", proportion, magnitude))
                ols_result = entries.get(("OLS", proportion, magnitude))
                if lp_result is not None and ols_result is not None:
                    differences[i, j] = lp_result[0] - ols_result[0]
        limit = float(np.nanmax(np.abs(differences)))
        if limit == 0.0:
            limit = 1.0
        image = axis.imshow(
            differences,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_xticks(range(len(magnitudes)), [f"{value:g}" for value in magnitudes])
        axis.set_yticks(
            range(len(proportions)), [f"{value:g}" for value in proportions]
        )
        axis.set_xlabel("Outlier standard deviation A")
        axis.set_ylabel("Contamination proportion c")
        axis.set_title(scenario)
        figure.colorbar(
            image,
            ax=axis,
            label="Lp RMSE − OLS RMSE (negative favors Lp)",
        )
    figure.suptitle("Lp vs OLS RMSE across local contamination conditions")
    figures.append(figure)

    figure, axes = plt.subplots(
        len(scenarios),
        2,
        figsize=(11, 4 * len(scenarios)),
        squeeze=False,
        layout="constrained",
    )
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, 3))
    for row_index, scenario in enumerate(scenarios):
        entries = summaries[scenario]
        proportions = sorted({key[1] for key in entries})
        magnitudes = sorted({key[2] for key in entries})
        for column_index, method in enumerate(("Lp", "OLS")):
            axis = axes[row_index, column_index]
            for color, magnitude in zip(colors, magnitudes, strict=False):
                measurements = [
                    entries.get((method, proportion, magnitude))
                    for proportion in proportions
                ]
                means = np.asarray(
                    [
                        value[0] if value is not None else np.nan
                        for value in measurements
                    ]
                )
                errors = np.asarray(
                    [
                        value[1] if value is not None else np.nan
                        for value in measurements
                    ]
                )
                axis.errorbar(
                    proportions,
                    means,
                    yerr=errors,
                    marker="o",
                    capsize=2,
                    color=color,
                    label=f"A={magnitude:g}",
                )
            axis.set_xticks(
                proportions, [f"{proportion:g}" for proportion in proportions]
            )
            axis.set_xlabel("Contamination proportion c")
            axis.set_ylabel("Mean clean-test RMSE ± SE")
            axis.set_title(f"{scenario}: {method}")
            axis.grid(True, alpha=0.3)
            axis.legend()
    figure.suptitle("Clean-test error trends through each local neighborhood")
    figures.append(figure)
    return tuple(figures)


def plot_hyperparameter_stability(csv_path: Path) -> tuple[plt.Figure, ...]:
    """Create figures from a hyperparameter-stability CSV file."""
    return _scenario_figures(_read_summaries(csv_path))


def main() -> None:
    """Display plots for a hyperparameter-stability results CSV."""
    parser = argparse.ArgumentParser(
        description="Plot local hyperparameter-stability Monte Carlo results."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        default=Path("results/hyperparameter_stability.csv"),
    )
    args = parser.parse_args()
    plot_hyperparameter_stability(args.csv_path)
    plt.show()


if __name__ == "__main__":
    main()
