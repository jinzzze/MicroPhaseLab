# Contributing to MicroPhaseLab

Thank you for contributing. This project prioritizes reproducible teaching workflows,
clear scientific limits, and approachable code.

## Before opening a pull request

1. Discuss substantial changes in an issue first.
2. Keep raw data, generated masks, model checkpoints, and outputs out of Git.
3. Add or update tests for behavioral changes.
4. Run python -m pytest -q and ruff check . locally.
5. Explain how the change affects reproducibility, group leakage, or scientific scope.

## Development setup

Install development dependencies with pip install -e ".[dev]". Install
pip install -e ".[torch]" only when working on optional U-Net features.

## Pull request expectations

Use a focused title, describe the motivation, link related issues, and include test
results. Do not claim real-data performance without a frozen test split, recorded
configuration, and visual quality control.
