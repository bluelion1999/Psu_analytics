# Sub-project 1: Engineering foundation — design

Date: 2026-09-23. Status: approved in chat, awaiting spec review.

## Context: the enhancement roadmap

Phases 1-5 are merged. The project is being enhanced along three goals at once
(portfolio polish, an in-season tool, analytics depth), split into sub-projects that each get
their own spec -> plan -> build -> PR:

1. **Engineering foundation** (this spec)
2. ~~Scheduled refresh~~: deferred; refresh stays manual (`psu refresh`)
3. Model upgrade: preseason priors, calibration, per-week sim history
4. 4th-down decision model
5. Recruiting-to-production analysis
6. Dashboard polish and public deploy

## Goal

One command refreshes everything, and every push is linted and tested in CI.

## Non-goals

- Scheduling or automation of the refresh (deferred).
- Splitting `simulate.py`; sub-project 3 reshapes it.
- Type checking (mypy/pyright), pre-commit hooks, coverage reporting.
- Skipping pipeline steps whose inputs did not change.

## 1. CLI restructure and `psu refresh`

### Handlers

`src/psu/cli.py` keeps the argparse setup. Each command body moves into a handler
`cmd_<name>(args, settings) -> int`: `cmd_status`, `cmd_ingest`, `cmd_build`, `cmd_train`,
`cmd_simulate`, `cmd_refresh`. `main()` parses args, sets up logging, loads settings and
dispatches through a dict. The behaviour of the existing commands, including exit codes and
messages, does not change.

### Shared defaults

These module constants move to `src/psu/config.py`. The argparse defaults and `refresh` both read
them.

| Constant | Value | Used by |
|---|---|---|
| `GARBAGE` | `"38,28,22"` | `build --garbage` |
| `BUILD_ALPHA` | `50.0` | `build --alpha` |
| `TRAIN_ALPHA` | `20.0` | `train --alpha` |
| `SHRINK_PLAYS` | `75` | `train --shrink-plays` |
| `SIM_N` | `10_000` | `simulate --sims` |
| `SIM_SEED` | `0` | `simulate --seed` |
| `SIM_TAU` | `5.0` | `simulate --tau` |

### `psu refresh`

```
psu refresh [--skip-ingest] [--max-calls N] [--sims N] [--seed N]
```

- Steps, in order: ingest -> build -> train -> simulate.
  - ingest covers the current season only (`settings.current_season`). Past seasons are already
    final and cached.
  - build, train and simulate use the shared defaults above. `--sims` and `--seed` pass through to
    simulate, and `--max-calls` passes through to ingest.
- Before each step, print a `== <step> ==` header to stdout.
- Stop at the first step that returns non-zero, and return that code. Codes are unchanged: 2 for a
  usage or data error, 3 for an ingest budget stop.
- `--skip-ingest` leaves ingest out, for offline reruns. Build still fails with code 2 if no plays
  are loaded.
- On success, print one summary line with the API calls made (0 when skipped), the elapsed
  seconds, and the team's simulated mean wins, for example:
  `refresh ok in 94s: 4 API calls, Penn State mean wins 9.47`.
- To produce that line, the ingest and simulate handlers expose their results to `cmd_refresh`.
  Each one splits into an inner function (`_ingest`, `_simulate`) that does the work, prints the
  same output as today and returns `(code, result)`. `cmd_ingest` and `cmd_simulate` become thin
  wrappers around them, so the standalone commands print exactly what they print today.
- If a step stops the run, `refresh` prints `refresh stopped at <step> (exit <code>)` to stderr.

`refresh` calls `_ingest` and `_simulate` directly, and calls `cmd_build` and `cmd_train` with an
`argparse.Namespace` built from the shared defaults. It does not re-invoke `main()`.

### Database in use

If DuckDB can't open `data/psu.duckdb` because another process holds it (for example, a second
`psu` command writing), `main()` catches `duckdb.IOException`, prints
`error: data/psu.duckdb is in use by another process; close it and retry (<detail>)` and returns 2,
instead of showing a traceback. This applies to every command.

## 2. Lint and CI

### Ruff

Add to `pyproject.toml`:

```toml
[tool.ruff]
line-length = 120
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B"]
ignore = ["B008"]
```

`ruff` is added to the `dev` extras. A single, separate commit applies
`ruff format` and `ruff check --fix` across `src`, `tests` and `app`, plus hand fixes for the rest
(the baseline has 52 findings, 33 of them auto-fixable). It has no behaviour changes, and
`pytest` must pass unchanged afterwards. `zip()` calls get `strict=True` only where the lengths
must match; the rest get an explicit `strict=False`.

### GitHub Actions

`.github/workflows/ci.yml`:

- Triggers: push to `main`, and pull requests into `main`.
- Matrix: `ubuntu-latest` x Python `3.11`, `3.13`.
- Steps: checkout, setup-python with the pip cache, `pip install -e ".[dev]"`, `ruff check .`,
  `ruff format --check .`, `pytest`.
- No secrets. Tests that need `data/psu.duckdb` already skip when it is absent.
- The README gets a CI status badge.

## 3. Testing

New tests in `tests/test_cli.py`, with step inner functions monkeypatched to record calls and
return canned results:

- `refresh` runs ingest, build, train, simulate in that order and returns 0.
- `refresh` stops at the first failing step and returns its code. Build returning 2 means train
  and simulate never run.
- An ingest budget stop returns 3 through `refresh`, and later steps do not run.
- `--skip-ingest` never calls ingest.
- The success summary line contains the API-call count and the mean wins.
- Argparse defaults equal the `config` constants.

The existing CLI, train and simulate tests must pass unchanged. That proves the handler split
changed no behaviour.

## 4. Docs

- README: a new **Daily use** section near the top for `psu refresh` (what it runs, flags, exit
  codes, and typical cost of about 5 API calls), and a **Development** section for
  `ruff check`, `ruff format` and `pytest`.
- The CI badge goes under the title.

## Delivery

A plan in `docs/superpowers/plans/`, built by `psu-implementer` and `psu-reviewer` subagents, with
one Conventional Commit per task on `feat/foundation` and a PR into `main`.
