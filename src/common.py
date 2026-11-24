"""Shared pieces: data loading, the profit metric, cross-validation and run logging."""
from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# Economics of the decision (in $M per song)
GAIN_HIT = 2.5 - 1.0    # invest in a hit
LOSS_FLOP = 1.0 - 0.1   # invest in a flop
BREAK_EVEN = LOSS_FLOP / (GAIN_HIT + LOSS_FLOP)  # invest when P(hit) > 0.375


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_excel(DATA / "train.xlsx")
    test = pd.read_excel(DATA / "test.xlsx")
    for df in (train, test):
        df["track"] = df["track"].astype(str)
        df["artist"] = df["artist"].astype(str)
    return train, test


def profit(y: np.ndarray, invest: np.ndarray) -> float:
    invest = np.asarray(invest).astype(bool)
    y = np.asarray(y)
    return GAIN_HIT * np.sum(invest & (y == 1)) - LOSS_FLOP * np.sum(invest & (y == 0))


def folds(y: np.ndarray, n_splits: int = 5, seed: int = 0):
    return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ths = np.round(np.arange(0.20, 0.70, 0.005), 3)
    prof = np.array([profit(y, p > t) for t in ths])
    return {
        "auc": roc_auc_score(y, p),
        "acc": accuracy_score(y, p > 0.5),
        "logloss": log_loss(y, p),
        "profit_375": profit(y, p > BREAK_EVEN),
        "profit_best": prof.max(),
        "th_best": float(ths[prof.argmax()]),
        "profit_oracle": GAIN_HIT * y.sum(),
    }


def report(name: str, y: np.ndarray, p: np.ndarray, notes: str = "") -> dict:
    m = evaluate(y, p)
    line = (f"{name:<40} AUC {m['auc']:.4f}  acc {m['acc']:.4f}  ll {m['logloss']:.4f}  "
            f"profit@.375 {m['profit_375']:.1f}M  best {m['profit_best']:.1f}M@{m['th_best']}")
    print(line, flush=True)
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "runs.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "name", "auc", "acc", "logloss", "profit_375", "profit_best", "th_best", "notes"])
        w.writerow([dt.datetime.now().isoformat(timespec="seconds"), name, f"{m['auc']:.4f}", f"{m['acc']:.4f}",
                    f"{m['logloss']:.4f}", f"{m['profit_375']:.1f}", f"{m['profit_best']:.1f}", m["th_best"], notes])
    return m


def save_oof(name: str, oof: np.ndarray, test_pred: np.ndarray | None = None) -> None:
    d = RESULTS / "oof"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{name}_oof.npy", oof)
    if test_pred is not None:
        np.save(d / f"{name}_test.npy", test_pred)
