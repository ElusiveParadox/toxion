
from __future__ import annotations

from typing import Callable

import numpy as np


class MaskedMultitaskBaseline:
    def __init__(self, model_factory: Callable[[], object], task_names: list[str]):
        self.model_factory = model_factory
        self.task_names = task_names
        self.models: dict[str, object | None] = {}

    def fit(self, X: np.ndarray, y_matrix: np.ndarray) -> None:
        for t, task in enumerate(self.task_names):
            col = y_matrix[:, t]
            rows = ~np.isnan(col)
            y_col = col[rows].astype(int)
            if len(np.unique(y_col)) < 2:
                self.models[task] = None
                continue
            model = self.model_factory()
            model.fit(X[rows], y_col)
            self.models[task] = model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        out = np.full((X.shape[0], len(self.task_names)), 0.5, dtype=np.float32)
        for t, task in enumerate(self.task_names):
            model = self.models.get(task)
            if model is None:
                continue
            proba = model.predict_proba(X)
            classes = list(model.classes_)
            if 1 in classes:
                out[:, t] = proba[:, classes.index(1)]
        return out


def make_logreg_factory(**kwargs):
    from sklearn.linear_model import LogisticRegression
    params = dict(max_iter=1000, class_weight="balanced") | kwargs
    return lambda: LogisticRegression(**params)


def make_random_forest_factory(**kwargs):
    from sklearn.ensemble import RandomForestClassifier
    params = dict(n_estimators=300, class_weight="balanced", n_jobs=-1) | kwargs
    return lambda: RandomForestClassifier(**params)


def make_xgboost_factory(**kwargs):
    from xgboost import XGBClassifier
    params = dict(n_estimators=300, max_depth=6, learning_rate=0.05, eval_metric="logloss") | kwargs
    return lambda: XGBClassifier(**params)
