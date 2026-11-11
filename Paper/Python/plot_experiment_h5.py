"""Plot contamination-regime estimates and recovery errors for H5."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_RESULTS_PATH = (
    Path(__file__).resolve().parents[1] / "Results" / "h5_comparison.csv"
)
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "Assets" / "h5_comparison.png"
)
METHODS = (
    "Huber",
    "Lp_estimated_regime",
    "Lp_true_regime_oracle",
    "Lp_H1_anchor",
)
METHOD_LABELS = (
    "Adaptive Huber",
    "Estimated-regime $L_p$",
    "True-regime oracle $L_p$",
    "H1 anchor $L_p$",
)
REQUIRED_COLUMNS = {
    "method",
    "replicate",
    "outlier_proportion",
    "outlier_scale",
    "estimated_proportion",
    "estimated_scale",
    "recovery_error",
}


def _read_results(
    csv_path: Path,
) -> tuple[dict[str, list[dict[str, float]]], tuple[float, ...], tuple[float, ...]]:
    """Load and validate per-method H5 results."""
    records = {method: [] for method in METHODS}
    proportions: set[float] = set()
    scales: set[float] = set()
    with Path(csv_path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: "
                f"{', '.join(sorted(missing))}"
            )
        for line_number, row in enumerate(reader, start=2):
            method = row["method"]
            if method not in records:
                raise ValueError(
                    f"{csv_path}, line {line_number}: unknown method {method!r}."
                )
            try:
                numeric = {
                    name: float(row[name] or "")
                    for name in (
                        "outlier_proportion",
                        "outlier_scale",
                        "estimated_proportion",
                        "estimated_scale",
                        "recovery_error",
                    )
                }
                replicate = int(row["replicate"] or "")
            except ValueError as parse_error:
                raise ValueError(
                    f"{csv_path}, line {line_number}: invalid numeric value."
                ) from parse_error
            if (
                replicate <= 0
                or not np.all(np.isfinite(list(numeric.values())))
                or numeric["outlier_scale"] <= 0.0
                or numeric["estimated_scale"] < 0.0
                or numeric["recovery_error"] < 0.0
                or not 0.0 <= numeric["outlier_proportion"] <= 1.0
                or not 0.0 <= numeric["estimated_proportion"] <= 1.0
            ):
                raise ValueError(
                    f"{csv_path}, line {line_number}: invalid experiment result."
                )
            records[method].append(
                {
                    **numeric,
                    "replicate": float(replicate),
                }
            )
            proportions.add(numeric["outlier_proportion"])
            scales.add(numeric["outlier_scale"])

    if any(not values for values in records.values()):
        raise ValueError(f"{csv_path} does not contain every expected method.")
    lengths = {len(values) for values in records.values()}
    if len(lengths) != 1:
        raise ValueError(f"{csv_path} has mismatched replicate counts by method.")
    return records, tuple(sorted(proportions)), tuple(sorted(scales))


def plot_h5_results(csv_path: Path) -> plt.Figure:
    """Plot estimated-versus-true contamination and recovery-error distributions."""
    records, _, _ = _read_results(csv_path)
    estimates = records["Huber"]
    figure, axes = plt.subplots(
        1, 3, figsize=(13, 4.5), layout="constrained"
    )

    for axis, actual_name, estimate_name, label in (
        (
            axes[0],
            "outlier_proportion",
            "estimated_proportion",
            r"Contamination proportion $\rho$",
        ),
        (
            axes[1],
            "outlier_scale",
            "estimated_scale",
            r"Added-noise scale $A$",
        ),
    ):
        actual_values = sorted({row[actual_name] for row in estimates})
        means = []
        standard_deviations = []
        for actual in actual_values:
            sample = np.asarray(
                [
                    row[estimate_name]
                    for row in estimates
                    if row[actual_name] == actual
                ]
            )
            means.append(float(np.mean(sample)))
            standard_deviations.append(
                float(np.std(sample, ddof=1)) if sample.size > 1 else 0.0
            )
        axis.errorbar(
            actual_values,
            means,
            yerr=standard_deviations,
            marker="o",
            capsize=3,
            label="Estimated mean $\\pm$ SD",
        )
        lower = min(actual_values[0], min(means))
        upper = max(actual_values[-1], max(means))
        axis.plot([lower, upper], [lower, upper], linestyle="--", color="0.4")
        axis.set_xlabel(f"True {label}")
        axis.set_ylabel(f"Estimated {label}")
        axis.set_title(f"Mixture estimate of {label}")
        axis.grid(alpha=0.25)
        if actual_name == "outlier_scale":
            axis.set_xscale("log")
            axis.set_yscale("log")

    error_values = [
        [row["recovery_error"] for row in records[method]]
        for method in METHODS
    ]
    axes[2].boxplot(
        error_values,
        tick_labels=METHOD_LABELS,
        showfliers=False,
        patch_artist=True,
    )
    axes[2].set_yscale("log")
    axes[2].set_ylabel(r"Coefficient-recovery error $\|\hat{\beta}-\beta\|_2$")
    axes[2].set_title("Recovery across all regimes")
    axes[2].tick_params(axis="x", labelrotation=25)
    axes[2].grid(axis="y", alpha=0.25)
    figure.suptitle("H5: Gaussian-mixture regime estimation and parameter lookup")
    return figure


def main() -> None:
    """Parse command-line options and save or display the H5 figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--show",
        action="store_true",
        help="display the plot after saving it",
    )
    args = parser.parse_args()
    figure = plot_h5_results(args.csv_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    print(args.output)
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
