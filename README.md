# README

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE.md)

This repository explores non-convex $L_p$ objectives for robust linear
regression in the presence of outliers. It implements ordinary least squares
(OLS), least absolute deviations (LAD), and a smoothed $L_p$ estimator fitted
with iteratively reweighted least squares (IRLS). The examples use synthetic
data to compare coefficient recovery and prediction error on clean test data
under varying outlier proportions, outlier magnitudes, and smoothing
parameters.

## Quick start

Clone the repository and install the package and notebook dependencies:

```bash
git clone https://github.com/Pablo-RTZ/nonconvex_regression.git
cd nonconvex_regression
python -m pip install -e . -r requirements.txt
```

The editable install makes changes to `src/nonconvex_regression/` available
immediately. The example notebooks are in [`Examples/`](Examples/).

## Contents

The repository contains the regression implementation in
[`src/nonconvex_regression/`](src/nonconvex_regression/) and the following
Jupyter notebooks, ordered as a suggested walkthrough:

1. [`01_data_generation.ipynb`](Examples/01_data_generation.ipynb) — generates
   clean and contaminated linear-regression data.
2. [`02_model_testing.ipynb`](Examples/02_model_testing.ipynb) — compares OLS,
   LAD, and $L_p$ fits and visualize their objectives.
3. [`03_parameter_sweep.ipynb`](Examples/03_parameter_sweep.ipynb) — compares
   prediction error and coefficient recovery across outlier settings.
4. [`04_lp_sweep.ipynb`](Examples/04_lp_sweep.ipynb) — studies how the $L_p$
   exponent affects the results.
5. [`05_lp_optimization.ipynb`](Examples/05_lp_optimization.ipynb) — examines
   the effect of smoothing on test error and IRLS iteration counts.
6. [`06_parameter_tuning.ipynb`](Examples/06_parameter_tuning.ipynb) — sets up
   repeated-seed parameter tuning for three contamination scenarios.
7. [`07_tolerance_tuning.ipynb`](Examples/07_tolerance_tuning.ipynb) — compares
   solver tolerances using the best `lp` and `eps` values from notebook 06.

in order to simplify the code in the notebooks, the parameter tuning and its plots are defined in a `.py` file in the Experiments folder instead of the notebooks. Thus, the quadtree parameter search can also be run from the repository root after installing the package:

```bash
python Experiments/parameter_tuning.py
```

The program searches each contamination scenario separately over `lp` in `[0.4, 0.9]` and `eps` in `[1e-6, 1e-1]`, using the mean coefficient recovery error over a fixed number of seeded experiments (20 by default). It refines the best quadrant for a determined amount of rounds (5 by default) and writes per-candidate, per-experiment errors to `results/parameter_tuning.csv`. Use `--experiments`, `--rounds`, `--seed`, and `--output` to change those run settings.

Similarly, the plots from a tuning CSV can be generated and displayed with:

```bash
python Experiments/plot_parameter_tuning.py results/parameter_tuning.csv
```

The notebook displays a quadtree candidate/cell plot, best recovery error by
round, and the best `lp`/`eps` path by round inline. The quadtree cell
boundaries are reconstructed from the initial search bounds, which default to
the values used by the tuning program; pass `--lp-min`, `--lp-max`, `--eps-min`,
and `--eps-max` if a run used different bounds.

Notebook 07's experiment and plotting code are also separate programs. Run the
experiment from the repository root with:

```bash
python Experiments/tolerance_tuning.py
```

It reads each scenario's best `lp` and `eps` from
`results/parameter_tuning.csv`, evaluates 31 solver tolerances (by default) logarithmically
spaced from `1e-10` to `1e-1` on the same 50 (by default) seeded data sets, and writes
per-fit recovery error, solver time, iteration count, and marginal improvement
metrics to `results/tolerance_tuning.csv`. Marginal recovery improvement is
the relative error reduction from a looser to the adjacent tighter tolerance,
divided by the interval in `log10(tolerance)`. It is shown as a percentage;
the time-adjusted value divides that relative improvement by the added mean
solver time and is shown as a percentage per second. The elbow is estimated as the point
of maximum distance from the normalized recovery-error curve to its endpoint
chord. Tolerance bounds, grid size, experiment count, seed, parameter CSV, and
output path can be changed with the corresponding CLI options. Plot the saved
experiment independently with:

```bash
python Experiments/plot_tolerance_tuning.py results/tolerance_tuning.csv
```

## License

This project is licensed under the GNU General Public License v3.0 or later
(GPL-3.0-or-later). See [LICENSE.md](LICENSE.md) for details.

If you use this repository, please cite it using the metadata in
[`citation.cff`](citation.cff).
