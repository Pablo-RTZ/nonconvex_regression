"""Plot H1 Bayesian-optimization tuning results from a CSV file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

DEFAULT_RESULTS_PATH = Path(__file__).resolve().parents[1] / "Results" / "h1_comparison.csv"
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "Assets" / "h1_comparison.png"
)

REQUIRED_COLUMNS = {
    "evaluation",
    "phase",
    "p",
    "delta",
    "tolerance",
    "mean_recovery_error",
    "std_recovery_error",
    "is_best",
}


class _Evaluation(TypedDict):
    """Fields used to visualize one evaluated BO configuration."""

    evaluation: int
    phase: str
    p: float
    delta: float
    tolerance: float
    mean_recovery_error: float
    std_recovery_error: float
    is_best: bool


def _read_evaluations(csv_path: Path) -> list[_Evaluation]:
    """Read and validate Bayesian-optimization results."""
    evaluations: list[_Evaluation] = []
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {', '.join(sorted(missing))}"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                evaluation = int(row["evaluation"] or "")
                phase = row["phase"] or ""
                p = float(row["p"] or "")
                delta = float(row["delta"] or "")
                tolerance = float(row["tolerance"] or "")
                error = float(row["mean_recovery_error"] or "")
                std_error = float(row["std_recovery_error"] or "")
                best_value = (row["is_best"] or "").strip().lower()
            except ValueError as error:
                raise ValueError(
                    f"{csv_path}, line {line_number}: invalid numeric value."
                ) from error

            if not phase:
                raise ValueError(f"{csv_path}, line {line_number}: phase is empty.")
            if best_value not in {"true", "false"}:
                raise ValueError(
                    f"{csv_path}, line {line_number}: is_best must be true or false."
                )
            if not np.all(np.isfinite([p, delta, tolerance, error, std_error])):
                raise ValueError(
                    f"{csv_path}, line {line_number}: parameter values and error "
                    "must be finite."
                )
            if delta <= 0.0 or tolerance <= 0.0 or error < 0.0 or std_error < 0.0:
                raise ValueError(
                    f"{csv_path}, line {line_number}: delta and tolerance must be "
                    "positive; recovery error and its standard deviation must be "
                    "non-negative."
                )

            evaluations.append(
                {
                    "evaluation": evaluation,
                    "phase": phase,
                    "p": p,
                    "delta": delta,
                    "tolerance": tolerance,
                    "mean_recovery_error": error,
                    "std_recovery_error": std_error,
                    "is_best": best_value == "true",
                }
            )

    if not evaluations:
        raise ValueError(f"No BO results found in {csv_path}.")
    evaluations.sort(key=lambda result: result["evaluation"])
    evaluation_numbers = [result["evaluation"] for result in evaluations]
    if len(set(evaluation_numbers)) != len(evaluation_numbers):
        raise ValueError(f"{csv_path} contains duplicate evaluation numbers.")
    if sum(result["is_best"] for result in evaluations) != 1:
        raise ValueError(f"{csv_path} must mark exactly one BO-selected result.")
    return evaluations


def plot_bo_tuning(csv_path: Path) -> plt.Figure:
    """Create a dashboard of optimization progress and parameter-space results."""
    evaluations = _read_evaluations(csv_path)
    phases = list(dict.fromkeys(result["phase"] for result in evaluations))
    markers = {"initial": "o", "bayesian_optimization": "^"}
    phase_markers = {phase: markers.get(phase, "s") for phase in phases}
    best = next(result for result in evaluations if result["is_best"])

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5.5),
        gridspec_kw={"width_ratios": (1, 1.5)},
        layout="constrained",
    )
    progress_axis, parameter_axis = axes
    errors = np.asarray([result["mean_recovery_error"] for result in evaluations])
    best_so_far = np.minimum.accumulate(errors)

    for phase in phases:
        phase_results = [result for result in evaluations if result["phase"] == phase]
        progress_axis.errorbar(
            [result["evaluation"] for result in phase_results],
            [result["mean_recovery_error"] for result in phase_results],
            yerr=[result["std_recovery_error"] for result in phase_results],
            fmt=phase_markers[phase],
            markersize=5,
            markerfacecolor="white" if phase == "initial" else "black",
            markeredgecolor="black",
            color="black",
            ecolor="black",
            elinewidth=0.8,
            capsize=2,
            capthick=0.8,
            linestyle="none",
            alpha=0.85,
            label=phase.replace("_", " ").title(),
        )
    progress_axis.plot(
        [result["evaluation"] for result in evaluations],
        best_so_far,
        color="black",
        linewidth=1.8,
        label="Best observed so far",
    )
    bo_evaluations = [
        result["evaluation"]
        for result in evaluations
        if result["phase"] == "bayesian_optimization"
    ]
    if bo_evaluations:
        progress_axis.axvline(
            min(bo_evaluations) - 0.5,
            color="gray",
            linestyle=":",
            linewidth=1,
        )
    progress_axis.set_yscale("log")
    progress_axis.set_xlabel("Evaluation")
    progress_axis.set_ylabel("Mean coefficient recovery error (bars: ±1 SD)")
    progress_axis.set_title("Optimization progress")
    progress_axis.grid(True, alpha=0.3, color="0.75")
    progress_axis.legend()

    parameters = ("p", "delta", "tolerance")
    transformed: dict[str, np.ndarray] = {}
    for parameter in parameters:
        values = np.asarray([result[parameter] for result in evaluations])
        if parameter in {"delta", "tolerance"}:
            values = np.log10(values)
        low = float(np.min(values))
        high = float(np.max(values))
        transformed[parameter] = (
            np.full(values.shape, 0.5)
            if high == low
            else (values - low) / (high - low)
        )

    coordinates = np.arange(len(parameters))
    for index in range(len(evaluations)):
        parameter_axis.plot(
            coordinates,
            [transformed[parameter][index] for parameter in parameters],
            color="0.72",
            linewidth=0.8,
            alpha=0.45,
            zorder=1,
        )

    top_count = min(5, len(evaluations))
    top_indices = sorted(
        range(len(evaluations)),
        key=lambda index: evaluations[index]["mean_recovery_error"],
    )[:top_count]
    for rank, index in reversed(list(enumerate(top_indices, start=1))):
        parameter_axis.plot(
            coordinates,
            [transformed[parameter][index] for parameter in parameters],
            color="0.12",
            linewidth=1.0 + 0.45 * (top_count - rank + 1),
            alpha=0.8,
            zorder=2 + rank,
        )

    best_index = evaluations.index(best)
    parameter_axis.plot(
        coordinates,
        [transformed[parameter][best_index] for parameter in parameters],
        color="black",
        linewidth=2.2,
        linestyle="--",
        marker="*",
        markersize=9,
        zorder=top_count + 4,
    )
    for x_position, parameter in zip(coordinates, parameters, strict=True):
        values = np.asarray([result[parameter] for result in evaluations])
        use_log = parameter in {"delta", "tolerance"}
        axis_values = np.log10(values) if use_log else values
        low = float(np.min(axis_values))
        high = float(np.max(axis_values))
        tick_positions = (0.0, 0.5, 1.0)
        tick_values = (
            (low, (low + high) / 2.0, high)
            if high != low
            else (low, low, low)
        )
        parameter_axis.axvline(x_position, color="black", linewidth=1.0, zorder=0)
        for y_position, tick_value in zip(tick_positions, tick_values, strict=True):
            parameter_axis.plot(
                [x_position - 0.025, x_position + 0.025],
                [y_position, y_position],
                color="black",
                linewidth=0.7,
                zorder=0,
            )
            displayed_value = 10.0**tick_value if use_log else tick_value
            parameter_axis.annotate(
                f"{displayed_value:.2g}",
                (x_position, y_position),
                xytext=(-7, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=8,
            )

    parameter_axis.set_xlim(-0.3, len(parameters) - 0.7)
    parameter_axis.set_ylim(-0.08, 1.08)
    parameter_axis.set_xticks(coordinates, parameters)
    parameter_axis.set_yticks([])
    parameter_axis.set_title(
        "Parallel coordinates (p linear; delta and tolerance in log space)"
    )
    parameter_axis.grid(axis="y", color="0.88", linewidth=0.6)

    legend_handles = [
        Line2D([], [], color="0.72", linewidth=1, label="All evaluations"),
        Line2D(
            [],
            [],
            color="0.12",
            linewidth=2,
            label=f"Top {top_count} by observed error",
        ),
        Line2D(
            [],
            [],
            color="black",
            linewidth=2,
            linestyle="--",
            marker="*",
            label="BO-selected configuration",
        ),
    ]
    parameter_axis.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=len(legend_handles),
        frameon=False,
    )
    return figure


def main() -> None:
    """Save and display a visualization for a BO tuning results CSV."""
    parser = argparse.ArgumentParser(
        description="Plot Bayesian-optimization tuning results from CSV."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=f"results CSV (default: {DEFAULT_RESULTS_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"output image path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="save the figure without opening a display window",
    )
    args = parser.parse_args()

    figure = plot_bo_tuning(args.csv_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    print(f"Saved BO visualization to {args.output}")
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
