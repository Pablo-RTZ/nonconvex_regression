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

The main goal of the documentation and notebooks is to serve as a chronological
walkthrough of the exploratory analysis prior to the paper. The repository
contains the regression implementation in
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
8. [`08_hyperparameter_stability.ipynb`](Examples/08_hyperparameter_stability.ipynb)
   — evaluates each scenario's frozen `lp`, `eps`, and tolerance on nearby
   contamination conditions.

## Running experiments

Detailed instructions for running the experiment scripts and generating their
plots are in [`Docs/Experiments.md`](Docs/Experiments.md).

## License

This project is licensed under the GNU General Public License v3.0 or later
(GPL-3.0-or-later). See [LICENSE.md](LICENSE.md) for details.

If you use this repository, please cite it using the metadata in
[`citation.cff`](citation.cff).
