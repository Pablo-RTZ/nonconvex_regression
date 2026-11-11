"""Plot H3 recovery errors across nearby models from an experiment CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter

DEFAULT_RESULTS_PATH = (
    Path(__file__).resolve().parents[1] / "Results" / "h3_comparison.csv"
)
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "Assets" / "h3_comparison.png"
)
REQUIRED_COLUMNS = {
    "replicate",
    "anchor_intercept",
    "anchor_slope",
    "true_intercept",
    "true_slope",
    "recovery_error",
    "lp_p",
    "lp_delta",
    "lp_tolerance",
    "sigma",
    "outlier_proportion",
    "outlier_scale",
}
def _read_h3_results(
    csv_path: Path,
) -> tuple[dict[tuple[float, float], dict[int, float]], dict[str, float]]:
    """Read and validate per-replicate H3 recovery errors."""
    errors: dict[tuple[float, float], dict[int, float]] = {}
    settings: dict[str, float] = {}
    setting_columns = (
        "lp_p",
        "lp_delta",
        "lp_tolerance",
        "sigma",
        "outlier_proportion",
        "outlier_scale",
        "anchor_intercept",
        "anchor_slope",
    )

    with Path(csv_path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: "
                f"{', '.join(sorted(missing))}"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                replicate = int(row["replicate"] or "")
                anchor_intercept = float(row["anchor_intercept"] or "")
                anchor_slope = float(row["anchor_slope"] or "")
                intercept = float(row["true_intercept"] or "")
                slope = float(row["true_slope"] or "")
                error = float(row["recovery_error"] or "")
                row_settings = {
                    column: float(row[column] or "") for column in setting_columns
                }
            except ValueError as parse_error:
                raise ValueError(
                    f"{csv_path}, line {line_number}: invalid numeric value."
                ) from parse_error

            if replicate <= 0 or not np.all(
                np.isfinite(
                    [
                        anchor_intercept,
                        anchor_slope,
                        intercept,
                        slope,
                        error,
                        *row_settings.values(),
                    ]
                )
            ):
                raise ValueError(
                    f"{csv_path}, line {line_number}: values must be finite and "
                    "replicate must be positive."
                )
            if (
                error < 0.0
                or row_settings["lp_delta"] <= 0.0
                or row_settings["lp_tolerance"] <= 0.0
                or row_settings["sigma"] < 0.0
                or not 0.0 <= row_settings["outlier_proportion"] <= 1.0
                or row_settings["outlier_scale"] < 0.0
                or not 0.0 < row_settings["lp_p"] < 1.0
            ):
                raise ValueError(
                    f"{csv_path}, line {line_number}: invalid model or "
                    "experiment parameter."
                )

            if not settings:
                settings = row_settings
            elif settings != row_settings:
                raise ValueError(
                    f"{csv_path}, line {line_number}: experiment settings changed "
                    "between rows."
                )

            model_errors = errors.setdefault((intercept, slope), {})
            if replicate in model_errors:
                raise ValueError(
                    f"{csv_path}, line {line_number}: duplicate replicate "
                    f"{replicate} for model ({intercept}, {slope})."
                )
            model_errors[replicate] = error

    if not errors:
        raise ValueError(f"No H3 results found in {csv_path}.")
    replicate_ids = set(next(iter(errors.values())))
    if any(set(model_errors) != replicate_ids for model_errors in errors.values()):
        raise ValueError(
            f"{csv_path} must contain the same replicates for every model."
        )
    intercepts = {model[0] for model in errors}
    slopes = {model[1] for model in errors}
    if len(errors) != len(intercepts) * len(slopes):
        raise ValueError(f"{csv_path} does not contain a complete model grid.")
    if len(intercepts) < 2 or len(slopes) < 2:
        raise ValueError(f"{csv_path} must contain at least a 2x2 model grid.")
    return errors, settings


def plot_h3_recovery(csv_path: Path) -> plt.Figure:
    """Plot mean and standard-deviation recovery-error heatmaps in grayscale."""
    errors, settings = _read_h3_results(csv_path)
    intercepts = sorted({model[0] for model in errors})
    slopes = sorted({model[1] for model in errors})
    mean_errors = np.empty((len(slopes), len(intercepts)))
    standard_deviations = np.empty_like(mean_errors)
    for row, slope in enumerate(slopes):
        for column, intercept in enumerate(intercepts):
            values = np.asarray(list(errors[(intercept, slope)].values()))
            mean_errors[row, column] = np.mean(values)
            standard_deviations[row, column] = (
                np.std(values, ddof=1) if values.size > 1 else 0.0
            )

    anchor = (settings["anchor_intercept"], settings["anchor_slope"])
    reference = min(
        errors,
        key=lambda model: np.hypot(model[0] - anchor[0], model[1] - anchor[1]),
    )
    reference_row = slopes.index(reference[1])
    reference_column = intercepts.index(reference[0])
    mean_error_deviation = mean_errors - mean_errors[reference_row, reference_column]
    standard_deviation_change = (
        standard_deviations
        - standard_deviations[reference_row, reference_column]
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(11, 5), layout="constrained", sharex=True, sharey=True
    )
    panels = (
        (
            axes[0],
            mean_error_deviation,
            "Mean recovery-error change from reference",
        ),
        (
            axes[1],
            standard_deviation_change,
            "Recovery-error SD change from reference",
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
            intercepts,
            slopes,
            values,
            shading="nearest",
            cmap="Greys",
            norm=normalization,
        )
        axis.set_title(title)
        axis.set_xlabel(r"True intercept $\beta_0$")
        axis.set_xticks(np.linspace(intercepts[0], intercepts[-1], 5))
        axis.set_yticks(np.linspace(slopes[0], slopes[-1], 5))
        axis.grid(color="0.75", linewidth=0.35)
        colorbar = figure.colorbar(mesh, ax=axis)
        colorbar.formatter = FormatStrFormatter("%.4g")
        colorbar.update_ticks()
    axes[0].set_ylabel(r"True slope $\beta_1$")
    figure.suptitle(
        f"Contamination: rho={settings['outlier_proportion']:g}, "
        f"A={settings['outlier_scale']:g}; "
        f"Lp-IRLS: p={settings['lp_p']:.3g}, "
        f"delta={settings['lp_delta']:.3g}, "
        f"tolerance={settings['lp_tolerance']:.3g}; "
        f"reference model=({reference[0]:g}, {reference[1]:g}), "
        f"mean recovery error={mean_errors[reference_row, reference_column]:.4g}"
    )
    return figure


def main() -> None:
    """Parse command-line options and save or display the H3 plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--show",
        action="store_true",
        help="display the plot after saving it",
    )
    args = parser.parse_args()

    figure = plot_h3_recovery(args.csv_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    print(args.output)
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
