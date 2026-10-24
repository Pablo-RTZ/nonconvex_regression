"""Plot solver-tolerance results and marginal improvements from CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np

REQUIRED_COLUMNS = {
    "scenario",
    "lp",
    "eps",
    "tolerance",
    "coefficient_recovery_error",
    "solver_seconds",
    "relative_improvement_per_log10_tolerance",
    "relative_improvement_per_added_second",
    "is_error_elbow",
}


class _GroupedEntry(TypedDict):
    """Temporary intermediate data for aggregated tolerance results."""

    errors: list[float]
    solver_seconds: list[float]
    lp: float
    eps: float
    relative_per_log_tolerance: float
    relative_per_second: float
    is_error_elbow: bool


def _csv_float(row: dict[str, object], key: str) -> float:
    """Safely coerce a CSV cell to float, treating blanks as NaN."""
    value = row.get(key)
    if value is None:
        return float("nan")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return float("nan")
        return float(stripped)
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(
        f"Expected a numeric CSV value for {key!r}, got {type(value).__name__}."
    )


def _read_summaries(csv_path: Path) -> dict[str, list[dict[str, float | bool]]]:
    """Aggregate per-experiment rows into tolerance-level measurements."""
    grouped: dict[tuple[str, float], _GroupedEntry] = {}
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            scenario = str(row.get("scenario") or "")
            tolerance = _csv_float(row, "tolerance")
            key = (scenario, tolerance)
            entry = grouped.setdefault(
                key,
                {
                    "errors": [],
                    "solver_seconds": [],
                    "lp": _csv_float(row, "lp"),
                    "eps": _csv_float(row, "eps"),
                    "relative_per_log_tolerance": _csv_float(
                        row, "relative_improvement_per_log10_tolerance"
                    ),
                    "relative_per_second": _csv_float(
                        row, "relative_improvement_per_added_second"
                    ),
                    "is_error_elbow": str(row.get("is_error_elbow") or "")
                    .strip()
                    .lower()
                    == "true",
                },
            )
            entry["errors"].append(_csv_float(row, "coefficient_recovery_error"))
            entry["solver_seconds"].append(_csv_float(row, "solver_seconds"))

    summaries: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
    for (scenario, tolerance), entry in grouped.items():
        errors = np.asarray(entry["errors"], dtype=float)
        solver_seconds = np.asarray(entry["solver_seconds"], dtype=float)
        summaries[scenario].append(
            {
                "tolerance": tolerance,
                "mean_error": float(np.mean(errors)),
                "standard_error": (
                    float(np.std(errors, ddof=1) / np.sqrt(errors.size))
                    if errors.size > 1
                    else 0.0
                ),
                "mean_solver_seconds": float(np.mean(solver_seconds)),
                "lp": float(entry["lp"]),
                "eps": float(entry["eps"]),
                "relative_per_log_tolerance": float(
                    entry["relative_per_log_tolerance"]
                ),
                "relative_per_second": float(entry["relative_per_second"]),
                "is_error_elbow": bool(entry["is_error_elbow"]),
            }
        )

    if not summaries:
        raise ValueError(f"No tolerance results found in {csv_path}.")
    for scenario, evaluations in summaries.items():
        evaluations.sort(key=lambda item: float(item["tolerance"]))
        if len(evaluations) < 3:
            raise ValueError(f"{scenario} has fewer than three tolerance measurements.")
    return dict(summaries)


def _make_figures(
    summaries: dict[str, list[dict[str, float | bool]]],
) -> tuple[plt.Figure, ...]:
    """Create recovery, log-tolerance marginal, and time-adjusted figures."""
    scenarios = list(summaries)
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(scenarios)))
    figures: list[plt.Figure] = []

    figure, axes = plt.subplots(
        1,
        len(scenarios),
        figsize=(5.5 * len(scenarios), 4.5),
        squeeze=False,
        layout="constrained",
    )
    for axis, scenario, color in zip(axes[0], scenarios, colors, strict=True):
        evaluations = summaries[scenario]
        tolerances = np.asarray([item["tolerance"] for item in evaluations])
        errors = np.asarray([item["mean_error"] for item in evaluations])
        standard_errors = np.asarray([item["standard_error"] for item in evaluations])
        axis.errorbar(
            tolerances,
            errors,
            yerr=standard_errors,
            marker="o",
            markersize=4,
            linewidth=1.5,
            capsize=2,
            color=color,
            label="Mean +/- SE",
        )
        elbow = next((item for item in evaluations if item["is_error_elbow"]), None)
        if elbow is not None:
            elbow_tolerance = float(elbow["tolerance"])
            axis.axvline(
                elbow_tolerance,
                color="tab:red",
                linestyle="--",
                label=f"Error elbow: {elbow_tolerance:.2g}",
            )
        first = evaluations[0]
        axis.set_xscale("log")
        axis.set_xlabel("Solver tolerance")
        axis.set_ylabel("Coefficient recovery error")
        axis.set_title(
            f"{scenario}\n$ l_p={float(first['lp']):.4g}, "
            f"\\epsilon={float(first['eps']):.3g}$"
        )
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.suptitle("Coefficient recovery across solver tolerances")
    figures.append(figure)

    figure, axes = plt.subplots(
        1,
        len(scenarios),
        figsize=(5.5 * len(scenarios), 4.5),
        squeeze=False,
        layout="constrained",
    )
    for axis, scenario, color in zip(axes[0], scenarios, colors, strict=True):
        evaluations = summaries[scenario]
        tolerance_values = [item["tolerance"] for item in evaluations]
        improvement_values = [
            item["relative_per_log_tolerance"] for item in evaluations
        ]
        improvements_percent = 100.0 * np.asarray(improvement_values, dtype=float)
        axis.plot(
            tolerance_values,
            improvements_percent,
            marker="o",
            color=color,
        )
        finite_improvements = improvements_percent
        finite_indices = np.flatnonzero(np.isfinite(finite_improvements))
        if finite_indices.size:
            maximum_index = int(
                finite_indices[np.argmax(finite_improvements[finite_indices])]
            )
            maximum_tolerance = tolerance_values[maximum_index]
            axis.axvline(
                maximum_tolerance,
                color="red",
                linestyle="--",
                label=f"Maximum at tolerance: {maximum_tolerance:.3g}",
            )
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        axis.set_xscale("log")
        axis.set_xlabel("Solver tolerance")
        axis.set_ylabel("Error reduction per log$_{10}$ tolerance unit (%)")
        axis.set_title(scenario)
        axis.grid(True, alpha=0.3)
        if finite_indices.size:
            axis.legend()
    figure.suptitle("Marginal recovery improvement across tolerance decades")
    figures.append(figure)

    figure, axes = plt.subplots(
        1,
        len(scenarios),
        figsize=(5.5 * len(scenarios), 4.5),
        squeeze=False,
        layout="constrained",
    )
    for axis, scenario, color in zip(axes[0], scenarios, colors, strict=True):
        evaluations = summaries[scenario]
        tolerances = np.asarray([item["tolerance"] for item in evaluations])
        improvements_per_second = 100.0 * np.asarray(
            [item["relative_per_second"] for item in evaluations]
        )
        valid = np.isfinite(improvements_per_second)
        if np.any(valid):
            valid_indices = np.flatnonzero(valid)
            maximum_index = int(
                valid_indices[np.argmax(improvements_per_second[valid])]
            )
            maximum_tolerance = tolerances[maximum_index]
            axis.plot(
                tolerances[valid],
                improvements_per_second[valid],
                marker="o",
                color=color,
            )
            axis.axvline(
                maximum_tolerance,
                color="red",
                linestyle="--",
                label=f"Maximum at tolerance: {maximum_tolerance:.3g}",
            )
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        axis.set_xscale("log")
        axis.set_xlabel("Solver tolerance")
        axis.set_ylabel("Error reduction per added solver second (%)")
        axis.set_title(scenario)
        axis.grid(True, alpha=0.3)
        if np.any(valid):
            axis.legend()
    figure.suptitle("Marginal recovery improvement adjusted for solver time")
    figures.append(figure)
    return tuple(figures)


def plot_tolerance_tuning(csv_path: Path) -> tuple[plt.Figure, ...]:
    """Create figures from a tolerance-tuning CSV file."""
    return _make_figures(_read_summaries(csv_path))


def main() -> None:
    """Display plots for one solver-tolerance CSV file."""
    parser = argparse.ArgumentParser(
        description="Plot solver-tolerance results and marginal improvements."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        default=Path("results/tolerance_tuning.csv"),
    )
    args = parser.parse_args()
    plot_tolerance_tuning(args.csv_path)
    plt.show()


if __name__ == "__main__":
    main()
