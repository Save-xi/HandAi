# Contributing

This subproject is maintained as a single-right-hand baseline.

## Scope

- Keep the baseline path stable: input -> detection -> features -> gesture -> visualization / JSON.
- Treat `control_representation` and `svh_preview` as optional extensions.
- Do not make Unity, real SVH hardware, or network transport a runtime requirement for the baseline.

## Recommended environment

- Python `3.10`
- `conda activate single-right-hand-baseline`
  or install with:
  `python -m pip install -r requirements.txt`

## Before opening a PR

Run these checks from `single-hand-teleop-baseline/`:

```bash
python -m compileall -q src
pytest -q
python src/main.py --help
```

If you have CI-only tooling installed locally, also run:

```bash
python -m ruff check src tests
```

## Code style

- Prefer small, targeted patches over large rewrites.
- Keep payload field names aligned with the frozen frame payload contract.
- Add tests for behavior changes, especially around gesture logic, payload export, and extension fallbacks.
- When adding optional features, make sure baseline-only mode still works without hardware, Unity, or networking.
