"""Builds the fold-wise design matrices once and caches them, so every model sees the same folds."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from .common import RESULTS, folds, load
from .features import supervised, unsupervised

CACHE = RESULTS / "cache"


def design(seed: int = 0, n_splits: int = 5, text: bool = True):
    """Returns y, a list of (train_idx, val_idx, X_train, X_val) and (X_full_train, X_test)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"design_s{seed}_k{n_splits}_t{int(text)}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())

    train, test = load()
    y = train.target.values
    U = unsupervised(train, test)
    Utr, Ute = U.iloc[: len(train)].reset_index(drop=True), U.iloc[len(train):].reset_index(drop=True)

    out = []
    for i, (a, b) in enumerate(folds(y, n_splits, seed)):
        F, A = supervised(train.iloc[a].reset_index(drop=True), y[a],
                          train.iloc[b].reset_index(drop=True), seed=seed + i, text=text)
        Xa = pd.concat([Utr.iloc[a].reset_index(drop=True), F], axis=1)
        Xb = pd.concat([Utr.iloc[b].reset_index(drop=True), A], axis=1)
        out.append((a, b, Xa, Xb))
    F, A = supervised(train, y, test, seed=seed, text=text)
    full = (pd.concat([Utr, F], axis=1), pd.concat([Ute, A], axis=1))
    res = (y, out, full)
    path.write_bytes(pickle.dumps(res))
    return res


def run_cv(make_model, cols, seed=0, n_splits=5, fit_kwargs=None, predict_test=True):
    """Out-of-fold predictions and a test prediction from a model trained on all of train."""
    y, fs, (Xfull, Xtest) = design(seed, n_splits)
    oof = np.zeros(len(y))
    for a, b, Xa, Xb in fs:
        m = make_model()
        m.fit(Xa[cols], y[a], **(fit_kwargs or {}))
        oof[b] = m.predict_proba(Xb[cols])[:, 1]
    test_pred = None
    if predict_test:
        m = make_model()
        m.fit(Xfull[cols], y, **(fit_kwargs or {}))
        test_pred = m.predict_proba(Xtest[cols])[:, 1]
    return y, oof, test_pred
