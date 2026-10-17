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

## License

This project is licensed under the GNU General Public License v3.0 or later
(GPL-3.0-or-later). See [LICENSE.md](LICENSE.md) for details.

If you use this repository, please cite it using the metadata in
[`citation.cff`](citation.cff).
