"""Small deterministic soft-voting ensemble for BTC research/production artifacts."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class SoftVotingEnsemble:
    """Average base-model probabilities without changing the canonical feature schema.

    Self-contained and pickle-safe so the resulting joblib artifact can be loaded
    by the production runtime and provenance audit.
    """

    def __init__(self, weights=(0.34, 0.33, 0.33)):
        self.weights = tuple(float(x) for x in weights)

    @staticmethod
    def _factories():
        return (
            lambda: Pipeline([
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.3, max_iter=3000)),
            ]),
            lambda: ExtraTreesClassifier(
                n_estimators=220,
                max_depth=12,
                min_samples_leaf=12,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
            lambda: HistGradientBoostingClassifier(
                max_iter=160,
                max_leaf_nodes=15,
                learning_rate=0.04,
                l2_regularization=1.5,
                random_state=42,
            ),
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("invalid_training_matrix")
        if len(set(y.tolist())) < 3:
            raise ValueError("ensemble_requires_three_classes")
        if len(self.weights) != 3 or any(x < 0 for x in self.weights):
            raise ValueError("invalid_ensemble_weights")
        total = float(sum(self.weights))
        if total <= 0:
            raise ValueError("ensemble_weights_sum_nonpositive")

        self.models_ = []
        for factory in self._factories():
            model = factory()
            model.fit(X, y)
            self.models_.append(model)

        self.classes_ = np.asarray(sorted(set(y.tolist())))
        if list(self.classes_) != ["DOWN", "FLAT", "UP"]:
            raise ValueError("unexpected_class_order")
        self.weights_ = np.asarray(self.weights, dtype=float) / total
        self.n_features_in_ = int(X.shape[1])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if not hasattr(self, "models_"):
            raise RuntimeError("ensemble_not_fitted")
        parts = []
        for model in self.models_:
            raw = np.asarray(model.predict_proba(X), dtype=float)
            aligned = np.zeros((len(X), len(self.classes_)), dtype=float)
            for j, cls in enumerate(model.classes_):
                idx = int(np.where(self.classes_ == cls)[0][0])
                aligned[:, idx] = raw[:, j]
            parts.append(aligned)

        out = sum(w * p for w, p in zip(self.weights_, parts))
        out = np.clip(out, 1e-7, 1.0)
        return out / out.sum(axis=1, keepdims=True)

    def predict(self, X):
        p = self.predict_proba(X)
        return self.classes_[np.argmax(p, axis=1)]
