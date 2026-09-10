"""Train/test splits. The whole thesis lives here: test IDs never leak into train."""
from __future__ import annotations

from .io import Clip


def _guard(train: list[Clip], test: list[Clip]) -> None:
    if any(c.y == 1 for c in train):
        raise ValueError("leak: train set contains anomalies (y==1)")
    train_ids = {c.model_id for c in train}
    test_ids = {c.model_id for c in test}
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f"leak: test model_id(s) {overlap} also in train")


def loio(clips: list[Clip], test_id: str) -> tuple[list[Clip], list[Clip]]:
    """Leave-one-model-ID-out: train on normals of the other IDs, test all y of test_id."""
    train = [c for c in clips if c.model_id != test_id and c.y == 0]
    test = [c for c in clips if c.model_id == test_id]
    _guard(train, test)
    return train, test


def same_id(clips: list[Clip], the_id: str, train_frac: float = 0.5) -> tuple[list[Clip], list[Clip]]:
    """Debug split: train on half of one ID's normals, test the rest + its anomalies.

    Deliberately WRONG generalization test. If its AUC ~= loio AUC, the loio split
    is suspect. Does not go through _guard's ID check (same ID by design).
    """
    normals = [c for c in clips if c.model_id == the_id and c.y == 0]
    anoms = [c for c in clips if c.model_id == the_id and c.y == 1]
    cut = int(len(normals) * train_frac)
    train, held_normals = normals[:cut], normals[cut:]
    if any(c.y == 1 for c in train):
        raise ValueError("leak: same_id train contains anomalies")
    return train, held_normals + anoms
