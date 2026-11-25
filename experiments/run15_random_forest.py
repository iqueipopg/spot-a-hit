"""Run 15: random forest on the same 10-fold design as run 10, to compare bagging with boosting."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pickle

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from src.common import RESULTS, load, report, save_oof

tr, te = load()
y = tr.target.values
_, fs, (Xfull, Xtest) = pickle.load(open(RESULTS / "cache" / "design_v5_s0_k10_t1.pkl", "rb"))
for name, mk in {"rf": lambda: RandomForestClassifier(800, min_samples_leaf=3, max_features=0.3, n_jobs=6, random_state=0),
                 "et": lambda: ExtraTreesClassifier(800, min_samples_leaf=3, max_features=0.5, n_jobs=6, random_state=0)}.items():
    oof = np.zeros(len(y))
    for a, b, Xa, Xb in fs:
        oof[b] = mk().fit(Xa.fillna(-1), y[a]).predict_proba(Xb.fillna(-1))[:, 1]
    report(f"r15 {name} v5 k10", y, oof)
    save_oof(f"r15_{name}_k10", oof)
