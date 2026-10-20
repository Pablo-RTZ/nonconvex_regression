"""Plot parameter-tuning results and the inferred quadtree search."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from Experiments.parameter_tuning import DEFAULT_LP_BOUNDS

REQUIRED_COLUMNS = {
    "scenario",
    "round",
    "lp",
    "eps",
    "coefficient_recovery_error",
    "mean_coefficient_recovery_error",
}


class _CandidateAggregate(TypedDict):
    mean_error: float
    errors: list[float]


def _read_candidates(
    csv_path: Path,
) -> dict[str, dict[int, list[dict[str, float]]]]:
    """Read and aggregate experiment rows into unique round candidates."""
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {', '.join(sorted(missing))}"
            )
        grouped: dict[
            tuple[str, int, float, float],
            _CandidateAggregate,
        ] = {}
        for row in reader:
            key = (
                row["scenario"],
                int(row["round"]),
                float(row["lp"]),
                float(row["eps"]),
            )
            candidate = grouped.setdefault(
                key,
                {
                    "mean_error": float(row["mean_coefficient_recovery_error"]),
                    "errors": [],
                },
            )
            errors = candidate["errors"]
            errors.append(float(row["coefficient_recovery_error"]))

    candidates: dict[str, dict[int, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (scenario, round_number, lp, eps), values in grouped.items():
        errors = values["errors"]
        if not errors:
            raise ValueError(
                f"No experiment errors were recorded for {scenario}, "
                f"round {round_number}, lp={lp}, eps={eps}."
            )
        candidates[scenario][round_number].append(
            {
                "lp": lp,
                "eps": eps,
                "mean_error": values["mean_error"],
                "standard_error": float(np.std(errors, ddof=1) / np.sqrt(len(errors)))
                if len(errors) > 1
                else 0.0,
            }
        )

    if not candidates:
        raise ValueError(f"No tuning results found in {csv_path}.")
    for scenario, rounds in candidates.items():
        for round_number, round_candidates in rounds.items():
            if len(round_candidates) != 4:
                raise ValueError(
                    f"{scenario}, round {round_number} has "
                    f"{len(round_candidates)} candidates; expected four."
                )
    return dict(candidates)


def _make_figures(
    candidates: dict[str, dict[int, list[dict[str, float]]]],
    lp_bounds: tuple[float, float],
    eps_bounds: tuple[float, float],
) -> tuple[plt.Figure, ...]:
    """Create quadtree, score progression, and parameter progression figures."""
    scenarios = sorted(candidates)
    all_means = [
        candidate["mean_error"]
        for rounds in candidates.values()
        for round_candidates in rounds.values()
        for candidate in round_candidates
    ]
    normalization = Normalize(vmin=min(all_means), vmax=max(all_means))
    color_map = plt.get_cmap("viridis")
    figures: list[plt.Figure] = []

    figure, axes = plt.subplots(
        1,
        len(scenarios),
        figsize=(6 * len(scenarios), 5),
        squeeze=False,
        layout="constrained",
    )
    scatter = None
    for axis, scenario in zip(axes[0], scenarios, strict=True):
        lp_low, lp_high = lp_bounds
        log_eps_low, log_eps_high = np.log10(eps_bounds)
        scenario_candidates = candidates[scenario]

        for round_number in sorted(scenario_candidates):
            lp_mid = (lp_low + lp_high) / 2.0
            log_eps_mid = (log_eps_low + log_eps_high) / 2.0
            round_candidates = scenario_candidates[round_number]
            expected_centers = {
                (lp_center, 10.0**log_eps_center)
                for lp_center in ((lp_low + lp_mid) / 2.0, (lp_mid + lp_high) / 2.0)
                for log_eps_center in (
                    (log_eps_low + log_eps_mid) / 2.0,
                    (log_eps_mid + log_eps_high) / 2.0,
                )
            }
            for candidate in round_candidates:
                if not any(
                    np.isclose(candidate["lp"], lp_center)
                    and np.isclose(candidate["eps"], eps_center, rtol=1e-8)
                    for lp_center, eps_center in expected_centers
                ):
                    raise ValueError(
                        f"{scenario}, round {round_number} candidates do not "
                        "match the supplied search bounds. Pass the bounds "
                        "used to generate this CSV."
                    )
            round_best = min(
                round_candidates, key=lambda candidate: candidate["mean_error"]
            )

            for candidate in round_candidates:
                lp_index = int(candidate["lp"] >= lp_mid)
                eps_index = int(np.log10(candidate["eps"]) >= log_eps_mid)
                cell_lp_low, cell_lp_high = (
                    (lp_low, lp_mid) if lp_index == 0 else (lp_mid, lp_high)
                )
                cell_eps_low, cell_eps_high = (
                    (log_eps_low, log_eps_mid)
                    if eps_index == 0
                    else (log_eps_mid, log_eps_high)
                )
                axis.add_patch(
                    Rectangle(
                        (cell_lp_low, 10.0**cell_eps_low),
                        cell_lp_high - cell_lp_low,
                        10.0**cell_eps_high - 10.0**cell_eps_low,
                        fill=False,
                        edgecolor=color_map(
                            (round_number - 1) / max(1, len(scenario_candidates))
                        ),
                        linewidth=2.0 if candidate is round_best else 0.8,
                        linestyle="-" if candidate is round_best else ":",
                        alpha=0.8,
                    )
                )

            scatter = axis.scatter(
                [candidate["lp"] for candidate in round_candidates],
                [candidate["eps"] for candidate in round_candidates],
                c=[candidate["mean_error"] for candidate in round_candidates],
                cmap=color_map,
                norm=normalization,
                s=48,
                edgecolors="black",
                linewidths=0.35,
                zorder=3,
            )
            axis.scatter(
                [round_best["lp"]],
                [round_best["eps"]],
                marker="*",
                s=150,
                color="red",
                edgecolors="white",
                linewidths=0.7,
                zorder=4,
            )

            if round_best["lp"] < lp_mid:
                lp_high = lp_mid
            else:
                lp_low = lp_mid
            if np.log10(round_best["eps"]) < log_eps_mid:
                log_eps_high = log_eps_mid
            else:
                log_eps_low = log_eps_mid

        axis.set_yscale("log")
        axis.set_xlabel("$l_p$")
        axis.set_ylabel("$\\epsilon$")
        axis.set_title(f"{scenario}: quadtree candidates and selected cells")
        axis.grid(True, alpha=0.2)

    if scatter is not None:
        figure.colorbar(
            scatter, ax=list(axes[0]), label="Mean coefficient recovery error"
        )
    figure.suptitle(
        "Quadtree parameter search (stars mark each round's best candidate)"
    )
    figures.append(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    markers = ("o", "s", "^", "D", "v")
    line_styles = ("-", "--", "-.", ":")
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(scenarios)))
    for scenario_index, scenario in enumerate(scenarios):
        rounds = sorted(candidates[scenario])
        best_candidates = [
            min(
                candidates[scenario][round_number],
                key=lambda candidate: candidate["mean_error"],
            )
            for round_number in rounds
        ]
        errorbar = axis.errorbar(
            rounds,
            [candidate["mean_error"] for candidate in best_candidates],
            yerr=[candidate["standard_error"] for candidate in best_candidates],
            marker=markers[scenario_index % len(markers)],
            linestyle=line_styles[scenario_index % len(line_styles)],
            color=colors[scenario_index],
            alpha=0.65,
            capsize=3,
            label=scenario,
        )
        for cap in errorbar[1]:
            cap.set_alpha(0.65)
        for bar_collection in errorbar[2]:
            bar_collection.set_alpha(0.65)
    axis.set_xlabel("Quadtree round")
    axis.set_ylabel("Best mean coefficient recovery error")
    axis.set_title("Best recovery error by round (bars: standard error across seeds)")
    axis.set_xticks(
        sorted(
            {
                round_number
                for scenario_rounds in candidates.values()
                for round_number in scenario_rounds
            }
        )
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figures.append(figure)

    figure, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(scenarios)))
    for scenario_index, scenario in enumerate(scenarios):
        rounds = sorted(candidates[scenario])
        best_candidates = [
            min(
                candidates[scenario][round_number],
                key=lambda candidate: candidate["mean_error"],
            )
            for round_number in rounds
        ]
        axes[0].plot(
            rounds,
            [candidate["lp"] for candidate in best_candidates],
            marker=markers[scenario_index % len(markers)],
            linestyle=line_styles[scenario_index % len(line_styles)],
            color=colors[scenario_index],
            alpha=0.65,
            label=scenario,
        )
        axes[1].plot(
            rounds,
            [candidate["eps"] for candidate in best_candidates],
            marker=markers[scenario_index % len(markers)],
            linestyle=line_styles[scenario_index % len(line_styles)],
            color=colors[scenario_index],
            alpha=0.65,
            label=scenario,
        )
    axes[0].set_ylabel("Selected $l_p$")
    axes[1].set_ylabel("Selected $\\epsilon$")
    axes[1].set_xlabel("Quadtree round")
    axes[1].set_yscale("log")
    axes[0].set_title("Best candidate parameter path by round")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figures.append(figure)
    return tuple(figures)


def plot_parameter_tuning(
    csv_path: Path,
    *,
    lp_bounds: tuple[float, float] = DEFAULT_LP_BOUNDS,
    eps_bounds: tuple[float, float] = (1e-6, 1e-1),
) -> tuple[plt.Figure, ...]:
    """Create figures from search results and reconstruct quadtree cells.

    The current CSV format stores each round's candidates and scores, but not
    search-cell boundaries. The bounds must therefore match those used by the
    run; defaults match ``tune_scenario`` (``lp`` from 0.4 to 0.9 and ``eps``
    from 1e-6 to 1e-1). The figures are returned without being saved so
    notebooks can render them inline.
    """
    if not 0.0 < lp_bounds[0] < lp_bounds[1] < 1.0:
        raise ValueError("'lp_bounds' must satisfy 0 < low < high < 1.")
    if not 0.0 < eps_bounds[0] < eps_bounds[1]:
        raise ValueError("'eps_bounds' must satisfy 0 < low < high.")
    candidates = _read_candidates(csv_path)
    return _make_figures(candidates, lp_bounds, eps_bounds)


def main() -> None:
    """Display plots for one parameter-tuning CSV file."""
    parser = argparse.ArgumentParser(
        description="Plot quadtree parameter-tuning output from CSV."
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--lp-min", type=float, default=DEFAULT_LP_BOUNDS[0])
    parser.add_argument("--lp-max", type=float, default=DEFAULT_LP_BOUNDS[1])
    parser.add_argument("--eps-min", type=float, default=1e-6)
    parser.add_argument("--eps-max", type=float, default=1e-1)
    args = parser.parse_args()

    plot_parameter_tuning(
        args.csv_path,
        lp_bounds=(args.lp_min, args.lp_max),
        eps_bounds=(args.eps_min, args.eps_max),
    )
    plt.show()


if __name__ == "__main__":
    main()
