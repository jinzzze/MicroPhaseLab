"""Deterministic group-aware dataset splitting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _validate_ratios(train: float, val: float, test: float) -> None:
    ratios = (train, val, test)
    if any(ratio < 0 or ratio > 1 for ratio in ratios):
        raise ValueError("Split ratios must lie between 0 and 1")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"Split ratios must sum to 1; got {sum(ratios):.6f}")


def _counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    raw = np.asarray(ratios, dtype=float) * total
    base = np.floor(raw).astype(int)
    remainder = total - int(base.sum())
    order = np.argsort(-(raw - base), kind="stable")
    for index in order[:remainder]:
        base[index] += 1
    return int(base[0]), int(base[1]), int(base[2])


def create_group_splits(
    manifest_path: Path,
    output_dir: Path,
    *,
    group_column: str = "group_id",
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> dict[str, int]:
    _validate_ratios(train, val, test)
    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise ValueError("Manifest is empty")
    if group_column not in manifest.columns:
        raise KeyError(f"Group column {group_column!r} is missing")
    if manifest[group_column].isna().any():
        raise ValueError(f"Group column {group_column!r} contains missing values")

    groups = sorted(manifest[group_column].astype(str).unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    train_count, val_count, _ = _counts(len(groups), (train, val, test))
    group_sets = {
        "train": set(groups[:train_count]),
        "val": set(groups[train_count : train_count + val_count]),
        "test": set(groups[train_count + val_count :]),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {"groups_total": len(groups), "rows_total": len(manifest)}
    assigned_groups: set[str] = set()
    for split_name, selected_groups in group_sets.items():
        if assigned_groups & selected_groups:
            raise AssertionError("Internal error: a group was assigned to multiple splits")
        assigned_groups |= selected_groups
        subset = manifest[manifest[group_column].astype(str).isin(selected_groups)].copy()
        subset.insert(0, "split", split_name)
        subset.to_csv(output_dir / f"{split_name}.csv", index=False)
        summary[f"{split_name}_groups"] = len(selected_groups)
        summary[f"{split_name}_rows"] = len(subset)

    (output_dir / "split_summary.json").write_text(
        json.dumps({**summary, "seed": seed}, indent=2), encoding="utf-8"
    )
    return summary
