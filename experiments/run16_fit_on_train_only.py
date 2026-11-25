"""Run 16: label-free features fitted on the training rows only ("fit on train only", slide 32).

Baseline (runs 09-13): group statistics of the Kind-A features (audio vs decade, the artist's
catalogue, title counts) are computed on train + test together. Here, in every fold, they are
computed on the training part only and mapped onto the validation rows, exactly as one would do
for test. The label-based features are taken unchanged from the cached 10-fold design, so the only
difference is where the Kind-A statistics come from. Same LightGBM (Optuna parameters, one seed).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.common import RESULTS, load, report
from src.features import CONTINUOUS, unsupervised

ART_STATS = ["danceability", "energy", "acousticness", "instrumentalness", "speechiness", "valence", "loudness"]


def fit_apply(fit_df, app_df):
    """Kind-A features with every group statistic estimated on fit_df only."""
    Uf = unsupervised(fit_df, fit_df.iloc[:0])
    Ua = unsupervised(app_df, app_df.iloc[:0])           # row-wise columns are right; group ones get replaced
    for c in CONTINUOUS:
        g = fit_df.groupby("decade")[c]
        mu, sd = app_df.decade.map(g.mean()), app_df.decade.map(g.std())
        Ua[f"{c}_zdec"] = ((app_df[c] - mu) / sd).values
    art = fit_df.groupby("artist")
    a = app_df.artist
    Ua["art_n"] = a.map(art.size()).fillna(0).values
    Ua["art_n_dec"] = a.map(art.decade.nunique()).fillna(0).values
    Ua["art_first_dec"] = a.map(art.decade.min()).values
    Ua["art_span"] = (a.map(art.decade.max()) - a.map(art.decade.min())).values
    for c in ART_STATS:
        m = a.map(art[c].mean())
        Ua[f"art_mean_{c}"] = m.values
        Ua[f"dev_art_{c}"] = (app_df[c] - m).values
    Ua["art_std_energy"] = a.map(art.energy.std()).fillna(0).values
    key = lambda s: s.str.lower().str.strip()
    Ua["tr_title_n"] = key(app_df.track).map(key(fit_df.track).value_counts()).fillna(0).values
    return Uf.reset_index(drop=True), Ua.reset_index(drop=True)


tr, te = load()
y = tr.target.values
_, fs, _ = pickle.load(open(RESULTS / "cache" / "design_v5_s0_k10_t1.pkl", "rb"))
mk = lambda: lgb.LGBMClassifier(n_estimators=545, learning_rate=0.0137, num_leaves=30, min_child_samples=32,
                                subsample=0.994, subsample_freq=1, colsample_bytree=0.271, reg_lambda=0.0086,
                                reg_alpha=0.0044, verbose=-1, random_state=0, n_jobs=20)
base, strict = np.zeros(len(y)), np.zeros(len(y))
for a, b, Xa, Xb in fs:
    base[b] = mk().fit(Xa, y[a]).predict_proba(Xb)[:, 1]
    Uf, Ua = fit_apply(tr.iloc[a].reset_index(drop=True), tr.iloc[b].reset_index(drop=True))
    cols = [c for c in Uf.columns if c in Xa.columns]
    Xa2, Xb2 = Xa.copy(), Xb.copy()
    Xa2[cols], Xb2[cols] = Uf[cols].values, Ua[cols].values
    strict[b] = mk().fit(Xa2, y[a]).predict_proba(Xb2)[:, 1]
report("r16 lgbm k10 Kind A on train+test (base)", y, base)
report("r16 lgbm k10 Kind A fit on train only", y, strict)
