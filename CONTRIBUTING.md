# CONTRIBUTING

Thanks for your interest in this project!

This is primarily a personal data science project, meant to improve my coding and project organization skills. Nonetheless feedback and collaboration are welcome. Contributions may include code improvements, exploratory analyses, visualizations, experiments, documentation, bug reports, and suggestions for future work.

## Getting started

Read the project overview and setup instructions in [README.md](README.md). Set up the required environment and dependencies in order to be able to run the notebooks.

## Project conventions

The project structure is meant to be consistent, with example notebooks ordered in suggested reading.

Notebooks are meant for exploration, with all implementation of functions organized in `src/nonconvex_regression/`.

Google docstrings are used for all functions, as well as type definitions, in order to provide clear and usable code, specially for exploration.

## Code Quality Tools

This project uses **Ruff** for linting/formatting and **mypy** for static type checking. Both are configured in `pyproject.toml`. Please run them before opening a pull request.

### Ruff (linter & formatter)

Ruff checks code style, catches common errors, and enforces docstrings and type annotations.

- **Line length:** 88 characters.
- **Enabled rules:**
  - `E` – style errors (whitespace, indentation, etc.)
  - `F` – common errors (unused imports/variables, undefined names)
  - `I` – import sorting
  - `ANN` – every function must have type annotations
  - `D` – docstrings required on public classes and functions
  - `UP` – modernize syntax
  - `B` – catches common bug patterns
  - `W` – pycodestyle warnings
- **Ignored:** module-level (`D100`) and package-level (`D104`) docstrings.
- **Docstring style:** Google convention.

```bash
ruff check . --fix    # lint
ruff format .         # auto-format
```

### mypy (type checker)

mypy verifies that your type annotations are correct and complete.

- `disallow_untyped_defs = true` – every function must have full type annotations.
- `check_untyped_defs = true` – even unannotated functions are type-checked internally.

```bash
mypy src/
```

### Summary

| Tool | Purpose                                 | Run with             |
|------|-----------------------------------------|----------------------|
| Ruff | Style, imports, docstrings, annotations | `ruff check . --fix` |
| Ruff | Linting                                 | `ruff format .`      |
| mypy | Type correctness                        | `mypy src/`          |

Both must pass before your PR can be merged.

## Changes and collaboration

For small fixes, a concise description of the change is usually sufficient. For larger changes, new analyses, or alternative modeling approaches, please discuss the idea first so that we can agree on its scope and direction.

If using Git, work on a separate branch for substantial changes and open a pull request when feedback or review would be useful.

Pull requests should briefly describe:

What changed and why.
How the change can be run or validated.
Any relevant results, assumptions, limitations, or trade-offs.

Questions, suggestions, and feedback are welcome even if you do not intend to submit code.
. Clearly identify anything that has not been tested.
