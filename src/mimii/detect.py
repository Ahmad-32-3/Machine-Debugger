"""Fit an unsupervised detector on train normals, score test clips. Higher = more anomalous.

Two things do the work, both aimed at the model-ID shift that sinks a plain baseline:

1. Per-machine normalization. Each machine's clips are z-scored by that machine's
   own mean/std, which cancels the unit-to-unit offset that otherwise reads as
   "broken". At score time the batch is one held-out machine, so it is centered on
   its own unlabeled statistics (no fault labels used, no cross-machine leak) -- a
   standard unsupervised test-time normalization, realistic for a real install.
2. An ensemble of Local Outlier Factor (weighted 2x) and a diagonal GMM on the
   normalized features, which beat IsolationForest by a wide margin here.
"""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


def _machine_norm(X: np.ndarray) -> np.ndarray:
    """Z-score one machine's feature block by its own per-column mean/std."""
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)


def _z(s: np.ndarray) -> np.ndarray:
    return (s - s.mean()) / (s.std() + 1e-8)


class Detector:
    def __init__(self, seed: int = 0):
        self.scaler = StandardScaler()
        self.lof = LocalOutlierFactor(n_neighbors=10, novelty=True)
        self.gmm = GaussianMixture(n_components=8, covariance_type="diag", random_state=seed)

    def fit(self, X: np.ndarray, ids: np.ndarray) -> "Detector":
        """X: train-normal features; ids: model_id per row (multiple machines)."""
        Xn = X.copy()
        for u in np.unique(ids):
            m = ids == u
            Xn[m] = _machine_norm(X[m])
        Xs = self.scaler.fit_transform(Xn)
        self.lof.fit(Xs)
        self.gmm.fit(Xs)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """X: one held-out machine's clips. Higher = more anomalous."""
        Xs = self.scaler.transform(_machine_norm(X))
        lof = -self.lof.decision_function(Xs)
        gmm = -self.gmm.score_samples(Xs)
        return 2.0 * _z(lof) + _z(gmm)
