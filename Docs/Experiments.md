# Running experiments

Install the package and notebook dependencies as described in the
[README quick start](../README.md#quick-start). Run the experiment commands
from the repository root.

## Parameter tuning

To keep the notebooks focused, parameter-tuning code and plots are defined in
Python scripts in the `Experiments/` directory. Run the quadtree parameter
search with:

```bash
python Experiments/parameter_tuning.py
```

The program searches each contamination scenario separately over `lp` in
`[0.4, 0.9]` and `eps` in `[1e-6, 1e-1]`, using the mean coefficient recovery
error over a fixed number of seeded experiments (20 by default). It refines
the best quadrant for a set number of rounds (5 by default) and writes
per-candidate, per-experiment errors to `results/parameter_tuning.csv`. Use
`--experiments`, `--rounds`, `--seed`, and `--output` to change those run
settings.

Generate and display plots from a tuning CSV with:

```bash
python Experiments/plot_parameter_tuning.py results/parameter_tuning.csv
```

The script displays a quadtree candidate/cell plot, best recovery error by
round, and the best `lp`/`eps` path by round. The quadtree cell boundaries are
reconstructed from the initial search bounds, which default to the values
used by the tuning program; pass `--lp-min`, `--lp-max`, `--eps-min`, and
`--eps-max` if a run used different bounds.

## Tolerance tuning

Notebook 07's experiment and plotting code are also separate programs. Run
the experiment with:

```bash
python Experiments/tolerance_tuning.py
```

It reads each scenario's best `lp` and `eps` from
`results/parameter_tuning.csv`, evaluates 31 solver tolerances (by default)
logarithmically spaced from `1e-10` to `1e-1` on the same 50 (by default)
seeded data sets, and writes per-fit recovery error, solver time, iteration
count, and marginal improvement metrics to `results/tolerance_tuning.csv`.
Marginal recovery improvement is the relative error reduction from a looser
to the adjacent tighter tolerance, divided by the interval in
`log10(tolerance)`. It is shown as a percentage; the time-adjusted value
divides that relative improvement by the added mean solver time and is shown
as a percentage per second. The elbow is estimated as the point of maximum
distance from the normalized recovery-error curve to its endpoint chord.
Tolerance bounds, grid size, experiment count, seed, parameter CSV, and output
path can be changed with the corresponding CLI options. Plot the saved
experiment independently with:

```bash
python Experiments/plot_tolerance_tuning.py results/tolerance_tuning.csv
```

## Hyperparameter stability

The hyperparameter-stability experiment reuses the selected `lp`, `eps`, and
error-elbow tolerance from `results/tolerance_tuning.csv`. For each of the
three anchor scenarios, it tests a local grid using contamination fractions
at the anchor and at ±10% of the anchor proportion, and outlier magnitudes at
0.8, 1.0, and 1.2 times the anchor magnitude. Each target condition is
repeated 1,000 times by default, with a separate clean test set for every
repetition. It compares the frozen $L_p$ fit against OLS and records clean-test
RMSE, coefficient recovery, fit time, and IRLS iterations:

```bash
python Experiments/hyperparameter_stability.py
python Experiments/plot_hyperparameter_stability.py results/hyperparameter_stability.csv
```

Use `--experiments`, `--proportion-shift-fraction`, `--magnitude-multipliers`,
`--n-train`, `--n-test`, `--seed`, `--parameters`, and `--output` to adjust
the run. Clean-test prediction error is reported as RMSE, consistently with
the preceding experiment notebooks. The plotting script shows RMSE heatmaps
for both methods, the Lp-minus-OLS RMSE difference, and RMSE trends with
standard errors across contamination fractions.
