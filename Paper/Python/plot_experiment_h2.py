"""Plot H2 coefficient-recovery comparisons from an experiment CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_RESULTS_PATH = (
    Path(__file__).resolve().parents[1] / "Results" / "h2_comparison.csv"
)
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "Assets" / "h2_comparison.png"
)
METHODS = ("OLS", "Huber", "Lp")
REQUIRED_COLUMNS = {
    "method",
    "replicate",
    "recovery_error",
    "lp_p",
    "lp_delta",
    "lp_tolerance",
    "huber_delta",
}


def read_h2_results(csv_path: Path) -> dict[str, dict[int, float]]:
    """Read and validate per-replicate H2 coefficient-recovery errors."""
    errors = {method: {} for method in METHODS}
    with Path(csv_path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {', '.join(sorted(missing))}"
            )

        for line_number, row in enumerate(reader, start=2):
            method = row["method"] or ""
            if method not in errors:
                raise ValueError(
                    f"{csv_path}, line {line_number}: unknown method {method!r}."
                )
            try:
                replicate = int(row["replicate"] or "")
                error = float(row["recovery_error"] or "")
            except ValueError as parse_error:
                raise ValueError(
                    f"{csv_path}, line {line_number}: invalid replicate or "
                    "recovery error."
                ) from parse_error
            if replicate <= 0 or not np.isfinite(error) or error < 0.0:
                raise ValueError(
                    f"{csv_path}, line {line_number}: replicate must be positive "
                    "and recovery error must be finite and non-negative."
                )
            if replicate in errors[method]:
                raise ValueError(
                    f"{csv_path}, line {line_number}: duplicate {method} "
                    f"replicate {replicate}."
                )
            errors[method][replicate] = error

    if any(not method_errors for method_errors in errors.values()):
        raise ValueError(f"{csv_path} must contain results for OLS, Huber, and Lp.")
    replicate_ids = set(errors["OLS"])
    if any(set(method_errors) != replicate_ids for method_errors in errors.values()):
        raise ValueError(
            f"{csv_path} must contain the same replicates for all methods."
        )
    return errors


def plot_h2_comparison(csv_path: Path) -> plt.Figure:
    """Plot monochrome recovery-error distributions for all three methods."""
    errors = read_h2_results(csv_path)
    replicate_ids = sorted(errors["OLS"])
    values = [
        np.asarray([errors[method][replicate] for replicate in replicate_ids])
        for method in METHODS
    ]

    with Path(csv_path).open(newline="", encoding="utf-8") as csv_file:
        first_row = next(csv.DictReader(csv_file), None)
    if first_row is None:
        raise ValueError(f"No H2 results found in {csv_path}.")
    try:
        p = float(first_row["lp_p"])
        lp_delta = float(first_row["lp_delta"])
        tolerance = float(first_row["lp_tolerance"])
        huber_delta = float(first_row["huber_delta"])
    except (KeyError, ValueError) as error:
        raise ValueError(
            f"{csv_path} contains invalid comparison parameters."
        ) from error
    if not np.all(np.isfinite([p, lp_delta, tolerance, huber_delta])):
        raise ValueError(f"{csv_path} contains non-finite comparison parameters.")
    if min(lp_delta, tolerance, huber_delta) <= 0.0:
        raise ValueError(f"{csv_path} contains non-positive comparison parameters.")

    figure, distribution_axis = plt.subplots(figsize=(7, 5), layout="constrained")
    labels = [
        f"{method}\nmean={np.mean(method_errors):.3g}"
        for method, method_errors in zip(METHODS, values, strict=True)
    ]
    boxes = distribution_axis.boxplot(
        values,
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
        boxprops={"color": "black"},
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
    )
    for box in boxes["boxes"]:
        box.set_facecolor("none")
    positive_errors = np.concatenate(values)
    positive_errors = positive_errors[positive_errors > 0.0]
    if positive_errors.size:
        distribution_axis.set_yscale(
            "symlog",
            linthresh=max(float(np.min(positive_errors)) / 2.0, 1e-12),
        )
    distribution_axis.set_ylabel(
        r"Coefficient recovery error $\|\hat{\beta}-\beta\|_2$"
    )
    distribution_axis.set_title(
        f"Recovery error across {len(replicate_ids)} paired samples"
    )
    distribution_axis.grid(axis="y", alpha=0.3)

    figure.suptitle(
        f"p={p:.4g}, delta={lp_delta:.3g}, tolerance={tolerance:.3g}; "
        f"Huber delta={huber_delta:.3g}"
    )
    return figure


def main() -> None:
    """Parse command-line options and save or display the H2 comparison plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--show",
        action="store_true",
        help="display the plot after saving it",
    )
    args = parser.parse_args()

    figure = plot_h2_comparison(args.csv_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    print(args.output)
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
