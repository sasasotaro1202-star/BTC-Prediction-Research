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

    def __init__(self, weights=(0.34, 0.33, 0.33), learn_weights=True):
        self.weights = tuple(float(x) for x in weights)
        self.learn_weights = bool(learn_weights)

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

    @staticmethod
    def _weight_grid():
        grid = []
        values = np.arange(0.1, 0.81, 0.1)
        for a in values:
            for b in values:
                cc = 1.0 - float(a) - float(b)
                if 0.1 - 1e-12 <= cc <= 0.8 + 1e-12:
                    grid.append((float(a), float(b), float(cc)))
        return grid

    @staticmethod
    def _learn_validation_weights(parts, y):
        n = len(y)
        if n < 120:
            return None
        split = int(n * 0.75)
        if n - split < 30:
            return None
        yi = np.asarray([("DOWN", "FLAT", "UP").index(str(v)) for v in y[split:]], dtype=int)
        best = None
        prior = np.asarray((0.34, 0.33, 0.33), dtype=float)
        for weights in SoftVotingEnsemble._weight_grid():
            mix = sum(w * p for w, p in zip(weights, parts))
            mix = np.clip(mix, 1e-7, 1.0)
            mix /= mix.sum(axis=1, keepdims=True)
            ll = float(-np.mean(np.log(mix[np.arange(len(yi)), yi])))
            one = np.eye(3)[yi]
            br = float(np.mean(np.sum((mix - one) ** 2, axis=1)))
            distance = float(np.sum((np.asarray(weights) - prior) ** 2))
            key = (ll + 0.15 * br + 0.01 * distance, ll, br, tuple(round(x, 6) for x in weights))
            if best is None or key < best[0]:
                best = (key, weights)
        return best[1] if best else None

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
        validation_parts = []
        split = int(len(y) * 0.75) if len(y) >= 120 else None
        for factory in self._factories():
            model = factory()
            if split is not None and len(y) - split >= 30:
                probe = factory()
                probe.fit(X[:split], y[:split])
                raw = np.asarray(probe.predict_proba(X[split:]), dtype=float)
                aligned = np.zeros((len(raw), 3), dtype=float)
                idx = {"DOWN": 0, "FLAT": 1, "UP": 2}
                for j, cls in enumerate(probe.classes_):
                    if str(cls) in idx:
                        aligned[:, idx[str(cls)]] = raw[:, j]
                aligned = np.clip(aligned, 1e-7, 1.0)
                aligned /= aligned.sum(axis=1, keepdims=True)
                validation_parts.append(aligned)
            model.fit(X, y)
            self.models_.append(model)

        self.classes_ = np.asarray(sorted(set(y.tolist())))
        if list(self.classes_) != ["DOWN", "FLAT", "UP"]:
            raise ValueError("unexpected_class_order")
        chosen = self._learn_validation_weights(validation_parts, y) if self.learn_weights and len(validation_parts) == 3 else None
        if chosen is None:
            chosen = self.weights
            self.weight_learning_ = "constructor_default"
        else:
            self.weight_learning_ = "chronological_internal_validation"
        if len(chosen) != 3 or any(x < 0 for x in chosen):
            raise ValueError("invalid_ensemble_weights")
        total = float(sum(chosen))
        if total <= 0:
            raise ValueError("ensemble_weights_sum_nonpositive")
        self.weights_ = np.asarray(chosen, dtype=float) / total
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
