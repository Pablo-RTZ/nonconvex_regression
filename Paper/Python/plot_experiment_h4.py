"""Plot H4 recovery errors across contamination regimes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter

if __package__:
    from .experiment_h1 import DEFAULT_OUTLIER_PROPORTION, DEFAULT_OUTLIER_SCALE
else:
    from experiment_h1 import DEFAULT_OUTLIER_PROPORTION, DEFAULT_OUTLIER_SCALE

DEFAULT_RESULTS_PATH = (
    Path(__file__).resolve().parents[1] / "Results" / "h4_comparison.csv"
)
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "Assets" / "h4_comparison.png"
)
REQUIRED_COLUMNS = {
    "replicate",
    "proportion_multiplier",
    "amplitude_multiplier",
    "recovery_error",
    "lp_p",
    "lp_delta",
    "lp_tolerance",
    "sigma",
    "outlier_proportion",
    "outlier_scale",
}


def _read_h4_results(
    csv_path: Path,
) -> tuple[dict[tuple[float, float], dict[int, float]], dict[str, float]]:
    """Read and validate per-replicate H4 recovery errors."""
    errors: dict[tuple[float, float], dict[int, float]] = {}
    settings: dict[str, float] = {}
    setting_columns = (
        "lp_p",
        "lp_delta",
        "lp_tolerance",
        "sigma",
    )

    with Path(csv_path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {', '.join(sorted(missing))}"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                replicate = int(row["replicate"] or "")
                proportion_multiplier = float(row["proportion_multiplier"] or "")
                amplitude_multiplier = float(row["amplitude_multiplier"] or "")
                outlier_proportion = float(row["outlier_proportion"] or "")
                outlier_scale = float(row["outlier_scale"] or "")
                error = float(row["recovery_error"] or "")
                row_settings = {
                    column: float(row[column] or "") for column in setting_columns
                }
            except ValueError as parse_error:
                raise ValueError(
                    f"{csv_path}, line {line_number}: invalid numeric value."
                ) from parse_error

            values = (
                proportion_multiplier,
                amplitude_multiplier,
                outlier_proportion,
                outlier_scale,
                error,
                *row_settings.values(),
            )
            if replicate <= 0 or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{csv_path}, line {line_number}: values must be finite and "
                    "replicate must be positive."
                )
            if (
                error < 0.0
                or proportion_multiplier <= 0.0
                or amplitude_multiplier <= 0.0
                or not 0.0 <= outlier_proportion <= 1.0
                or outlier_scale < 0.0
                or row_settings["lp_delta"] <= 0.0
                or row_settings["lp_tolerance"] <= 0.0
                or row_settings["sigma"] < 0.0
                or not 0.0 < row_settings["lp_p"] < 1.0
            ):
                raise ValueError(
                    f"{csv_path}, line {line_number}: invalid experiment parameter."
                )
            if not np.isclose(
                outlier_proportion,
                DEFAULT_OUTLIER_PROPORTION * proportion_multiplier,
            ) or not np.isclose(
                outlier_scale, DEFAULT_OUTLIER_SCALE * amplitude_multiplier
            ):
                raise ValueError(
                    f"{csv_path}, line {line_number}: contamination values do not "
                    "match their multipliers."
                )

            if not settings:
                settings = row_settings
            elif settings != row_settings:
                raise ValueError(
                    f"{csv_path}, line {line_number}: fixed experiment settings "
                    "changed between rows."
                )

            condition = (outlier_proportion, outlier_scale)
            condition_errors = errors.setdefault(condition, {})
            if replicate in condition_errors:
                raise ValueError(
                    f"{csv_path}, line {line_number}: duplicate replicate "
                    f"{replicate} for condition {condition}."
                )
            condition_errors[replicate] = error

    if not errors:
        raise ValueError(f"No H4 results found in {csv_path}.")
    replicate_ids = set(next(iter(errors.values())))
    if any(set(condition) != replicate_ids for condition in errors.values()):
        raise ValueError(
            f"{csv_path} must contain the same replicates for every condition."
        )
    proportions = {condition[0] for condition in errors}
    amplitudes = {condition[1] for condition in errors}
    if (
        len(errors) != len(proportions) * len(amplitudes)
        or len(proportions) < 2
        or len(amplitudes) < 2
    ):
        raise ValueError(f"{csv_path} does not contain a complete 2D condition grid.")
    return errors, settings


def plot_h4_recovery(csv_path: Path) -> plt.Figure:
    """Plot recovery-error changes from the original contamination setting."""
    errors, settings = _read_h4_results(csv_path)
    proportions = sorted({condition[0] for condition in errors})
    amplitudes = sorted({condition[1] for condition in errors})
    mean_errors = np.empty((len(amplitudes), len(proportions)))
    standard_deviations = np.empty_like(mean_errors)
    for row, amplitude in enumerate(amplitudes):
        for column, proportion in enumerate(proportions):
            values = np.asarray(list(errors[(proportion, amplitude)].values()))
            mean_errors[row, column] = np.mean(values)
            standard_deviations[row, column] = (
                np.std(values, ddof=1) if values.size > 1 else 0.0
            )

    reference_proportion = DEFAULT_OUTLIER_PROPORTION
    reference_amplitude = DEFAULT_OUTLIER_SCALE
    if not proportions[0] <= reference_proportion <= proportions[-1] or not (
        amplitudes[0] <= reference_amplitude <= amplitudes[-1]
    ):
        raise ValueError(
            f"{csv_path} does not cover the original contamination setting "
            f"(rho={reference_proportion:g}, A={reference_amplitude:g}); "
            "include that setting within the multiplier ranges to plot "
            "deviations from it."
        )

    def interpolate_at_reference(values: np.ndarray) -> float:
        """Bilinearly interpolate a heatmap at the original setting."""
        proportion_index = int(
            np.clip(
                np.searchsorted(proportions, reference_proportion, side="right") - 1,
                0,
                len(proportions) - 2,
            )
        )
        amplitude_index = int(
            np.clip(
                np.searchsorted(amplitudes, reference_amplitude, side="right") - 1,
                0,
                len(amplitudes) - 2,
            )
        )
        proportion_fraction = (reference_proportion - proportions[proportion_index]) / (
            proportions[proportion_index + 1] - proportions[proportion_index]
        )
        amplitude_fraction = (reference_amplitude - amplitudes[amplitude_index]) / (
            amplitudes[amplitude_index + 1] - amplitudes[amplitude_index]
        )
        lower = (
            values[amplitude_index, proportion_index] * (1.0 - proportion_fraction)
            + values[amplitude_index, proportion_index + 1] * proportion_fraction
        )
        upper = (
            values[amplitude_index + 1, proportion_index] * (1.0 - proportion_fraction)
            + values[amplitude_index + 1, proportion_index + 1] * proportion_fraction
        )
        return float(lower * (1.0 - amplitude_fraction) + upper * amplitude_fraction)

    mean_change = mean_errors - interpolate_at_reference(mean_errors)
    standard_deviation_change = standard_deviations - interpolate_at_reference(
        standard_deviations
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(11, 5), layout="constrained", sharex=True, sharey=True
    )
    panels = (
        (axes[0], mean_change, "Mean recovery-error change from original"),
        (
            axes[1],
            standard_deviation_change,
            "Recovery-error SD change from original",
        ),
    )
    for axis, values, title in panels:
        max_deviation = float(np.max(np.abs(values)))
        normalization = (
            TwoSlopeNorm(
                vmin=-max_deviation,
                vcenter=0.0,
                vmax=max_deviation,
            )
            if max_deviation > 0.0
            else None
        )
        mesh = axis.pcolormesh(
            proportions,
            amplitudes,
            values,
            shading="nearest",
            cmap="Greys",
            norm=normalization,
        )
        axis.scatter(
            reference_proportion,
            reference_amplitude,
            marker="o",
            s=45,
            facecolor="white",
            edgecolor="black",
            linewidth=1.0,
            zorder=3,
        )
        axis.set_title(title)
        axis.set_xlabel(r"Outlier proportion $\rho$")
        axis.set_xticks(np.linspace(proportions[0], proportions[-1], 5))
        axis.set_yticks(np.linspace(amplitudes[0], amplitudes[-1], 5))
        axis.grid(color="0.75", linewidth=0.35)
        colorbar = figure.colorbar(mesh, ax=axis)
        colorbar.formatter = FormatStrFormatter("%.4g")
        colorbar.update_ticks()
    axes[0].set_ylabel(r"Outlier amplitude $A$")
    figure.suptitle(
        f"H1-tuned Lp-IRLS: p={settings['lp_p']:.3g}, "
        f"delta={settings['lp_delta']:.3g}, "
        f"tolerance={settings['lp_tolerance']:.3g}; "
        f"base contamination: rho={DEFAULT_OUTLIER_PROPORTION:g}, "
        f"A={DEFAULT_OUTLIER_SCALE:g}"
    )
    return figure


def main() -> None:
    """Parse command-line options and save or display the H4 plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--show",
        action="store_true",
        help="display the plot after saving it",
    )
    args = parser.parse_args()

    figure = plot_h4_recovery(args.csv_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    print(args.output)
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
