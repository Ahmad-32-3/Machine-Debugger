"""The leak evidence. Run: python tests/test_eval.py  (or pytest)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mimii import split as splits  # noqa: E402
from mimii.eval import evaluate  # noqa: E402
from mimii.io import Clip  # noqa: E402


def _clips():
    out = []
    for mid in ("00", "02", "06"):
        for i in range(5):
            out.append(Clip(f"{mid}_n{i}.wav", "fan", mid, 0, 0))
        for i in range(3):
            out.append(Clip(f"{mid}_a{i}.wav", "fan", mid, 0, 1))
    return out


def test_loio_ids_disjoint():
    train, test = splits.loio(_clips(), test_id="06")
    assert {c.model_id for c in train}.isdisjoint({c.model_id for c in test})
    assert all(c.y == 0 for c in train)          # normal-only train
    assert {c.model_id for c in test} == {"06"}  # test is the held-out ID


def test_injected_leak_fails():
    """Put a test-ID clip into train -> guard must raise."""
    clips = _clips()
    train = [c for c in clips if c.model_id in ("00", "02") and c.y == 0]
    train.append(Clip("06_leak.wav", "fan", "06", 0, 0))  # inject the held-out ID
    test = [c for c in clips if c.model_id == "06"]
    try:
        splits._guard(train, test)
    except ValueError:
        return
    raise AssertionError("leak not detected: test ID present in train but guard passed")


def test_anomaly_in_train_fails():
    clips = _clips()
    try:
        splits._guard([c for c in clips if c.model_id == "00"], [])  # includes y==1
    except ValueError:
        return
    raise AssertionError("anomaly-in-train not detected")


def test_metrics_separable():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.7])  # anomalies score higher
    m = evaluate(y, scores)
    assert m["auc_roc"] == 1.0 and m["pr_auc"] == 1.0


if __name__ == "__main__":
    test_loio_ids_disjoint()
    test_injected_leak_fails()
    test_anomaly_in_train_fails()
    test_metrics_separable()
    print("PASS: 4/4")
