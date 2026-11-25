"""Run 09: v5 text model + the hyper-parameters found in run 08, seed-bagged, and a stacked blend.

Usage: python run09_text5_tuned.py [n_splits]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import catboost as cb
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from src.common import RESULTS, folds, report, save_oof
from src.pipeline import design, run_cv

K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SEEDS = 3
y, fs, (Xfull, Xtest) = design(seed=0, n_splits=K, version=5)
cols = list(Xfull.columns)
best = json.loads((RESULTS / "best_params.json").read_text())
print(len(cols), "features;", K, "folds", flush=True)

makers = {
    "lgbm": lambda s: (lambda: lgb.LGBMClassifier(**best["lgbm"], subsample_freq=1, verbose=-1, random_state=s, n_jobs=20)),
    "cat": lambda s: (lambda: cb.CatBoostClassifier(**best["cat"], verbose=0, random_seed=s, thread_count=20)),
    "xgb": lambda s: (lambda: xgb.XGBClassifier(n_estimators=700, learning_rate=0.025, max_depth=6, min_child_weight=3,
                                                subsample=0.8, colsample_bytree=0.6, reg_lambda=2.0, tree_method="hist",
                                                n_jobs=20, random_state=s)),
}
Z, ZT = {}, {}
for name, mk in makers.items():
    runs = [run_cv(mk(s), cols, n_splits=K, version=5) for s in range(SEEDS)]
    o, t = np.mean([r[1] for r in runs], 0), np.mean([r[2] for r in runs], 0)
    report(f"r09 {name} v5 k{K} x{SEEDS}", y, o)
    save_oof(f"r09_{name}_k{K}", o, t)
    Z[name], ZT[name] = logit(np.clip(o, 1e-6, 1 - 1e-6)), logit(np.clip(t, 1e-6, 1 - 1e-6))

names = list(Z)
report(f"r09 mean-logit k{K}", y, expit(np.mean([Z[n] for n in names], 0)))

# Stacked blend: logistic regression on the three logits, itself fitted out of fold. It learns
# the weights and recalibrates at once (Platt scaling of the blend).
S, ST = np.column_stack([Z[n] for n in names]), np.column_stack([ZT[n] for n in names])
oof2 = np.zeros(len(y))
for a, b in folds(y, 5, seed=99):
    oof2[b] = LogisticRegression(C=1.0).fit(S[a], y[a]).predict_proba(S[b])[:, 1]
report(f"r09 stack LR k{K}", y, oof2)
m = LogisticRegression(C=1.0).fit(S, y)
print("stack weights", dict(zip(names, np.round(m.coef_[0], 3))), "intercept", round(m.intercept_[0], 3))
save_oof(f"r09_stack_k{K}", oof2, m.predict_proba(ST)[:, 1])
