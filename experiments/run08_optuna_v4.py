"""Run 08 (v4): Optuna on LightGBM and CatBoost over the honest (title-grouped) CV, then a seed bag.

The objective is CV log-loss, not AUC: the decision uses the probability against 0.375, so
calibration matters as much as ranking.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import catboost as cb
import lightgbm as lgb
import numpy as np
import optuna
from scipy.special import expit, logit
from sklearn.metrics import log_loss

from src.common import RESULTS, report, save_oof
from src.pipeline import design, run_cv

optuna.logging.set_verbosity(optuna.logging.WARNING)
y, fs, (Xfull, Xtest) = design(seed=0, version=4)
cols = list(Xfull.columns)
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 60


def lgb_model(p, seed=0):
    return lambda: lgb.LGBMClassifier(**p, subsample_freq=1, verbose=-1, random_state=seed, n_jobs=20)


def cat_model(p, seed=0):
    return lambda: cb.CatBoostClassifier(**p, verbose=0, random_seed=seed, thread_count=20)


def lgb_space(t):
    return dict(n_estimators=t.suggest_int("n_estimators", 300, 2000, log=True),
                learning_rate=t.suggest_float("learning_rate", 0.005, 0.08, log=True),
                num_leaves=t.suggest_int("num_leaves", 7, 96, log=True),
                min_child_samples=t.suggest_int("min_child_samples", 10, 150, log=True),
                subsample=t.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=t.suggest_float("colsample_bytree", 0.2, 0.9),
                reg_lambda=t.suggest_float("reg_lambda", 1e-3, 30, log=True),
                reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True))


def cat_space(t):
    return dict(iterations=t.suggest_int("iterations", 500, 3000, log=True),
                learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
                depth=t.suggest_int("depth", 4, 8),
                l2_leaf_reg=t.suggest_float("l2_leaf_reg", 0.5, 30, log=True),
                rsm=t.suggest_float("rsm", 0.3, 1.0),
                random_strength=t.suggest_float("random_strength", 0.1, 5, log=True))


best = {}
for name, space, mk, trials in [("lgbm", lgb_space, lgb_model, N_TRIALS), ("cat", cat_space, cat_model, N_TRIALS // 3)]:
    def objective(t):
        _, oof, _ = run_cv(mk(space(t)), cols, predict_test=False)
        return log_loss(y, oof)
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    st.optimize(objective, n_trials=trials)
    best[name] = st.best_params
    print(name, "best logloss", round(st.best_value, 4), st.best_params, flush=True)

(RESULTS / "best_params.json").write_text(json.dumps(best, indent=2))

# Seed bag of the tuned models
zs, zt = [], []
for name, mk in [("lgbm", lgb_model), ("cat", cat_model)]:
    o_s, t_s = [], []
    for seed in range(5):
        _, o, t = run_cv(mk(best[name], seed), cols)
        o_s.append(o); t_s.append(t)
    o, t = np.mean(o_s, 0), np.mean(t_s, 0)
    report(f"r08 {name} tuned x5 seeds", y, o)
    save_oof(f"r08_{name}", o, t)
    zs.append(logit(np.clip(o, 1e-6, 1 - 1e-6))); zt.append(logit(np.clip(t, 1e-6, 1 - 1e-6)))
report("r08 tuned lgbm+cat mean-logit", y, expit(np.mean(zs, 0)))
save_oof("r08_blend", expit(np.mean(zs, 0)), expit(np.mean(zt, 0)))
