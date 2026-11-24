"""Run 07: masked titles (v4) - train titles masked like the test ones, same three boostings."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import catboost as cb
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from scipy.special import logit

from src.common import report, save_oof
from src.pipeline import design, run_cv

y, fs, (Xfull, Xtest) = design(seed=0, version=4)
cols = list(Xfull.columns)
print(len(cols), "features")

models = {
    "lgbm": lambda: lgb.LGBMClassifier(n_estimators=700, learning_rate=0.025, num_leaves=31, min_child_samples=30,
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=2.0,
                                       verbose=-1, random_state=0),
    "xgb": lambda: xgb.XGBClassifier(n_estimators=700, learning_rate=0.025, max_depth=6, min_child_weight=3,
                                     subsample=0.8, colsample_bytree=0.6, reg_lambda=2.0, tree_method="hist", n_jobs=16,
                                     random_state=0),
    "cat": lambda: cb.CatBoostClassifier(iterations=1500, learning_rate=0.04, depth=6, l2_leaf_reg=3,
                                         verbose=0, random_seed=0, task_type="CPU", thread_count=16),
}
oofs, tests = {}, {}
for name, mk in models.items():
    _, oofs[name], tests[name] = run_cv(mk, cols, version=4)
    report(f"r07 {name} v4", y, oofs[name], f"{len(cols)} features")
    save_oof(f"r07_{name}", oofs[name], tests[name])

avg = np.mean([logit(np.clip(o, 1e-6, 1 - 1e-6)) for o in oofs.values()], axis=0)
report("r07 mean-logit lgbm+xgb+cat", y, 1 / (1 + np.exp(-avg)))

