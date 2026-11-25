"""Gain importance of LightGBM (Optuna params, full train, design v5) grouped by feature family."""
import sys, pickle, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/home/queipo/spot-a-hit')
from src.common import RESULTS, load
tr, te = load(); y = tr.target.values
_, fs, (Xfull, Xtest) = pickle.load(open(RESULTS / 'cache' / 'design_v5_s0_k10_t1.pkl', 'rb'))
m = lgb.LGBMClassifier(n_estimators=545, learning_rate=0.0137, num_leaves=30, min_child_samples=32, subsample=0.994,
    subsample_freq=1, colsample_bytree=0.271, reg_lambda=0.0086, reg_alpha=0.0044, verbose=-1, random_state=0, n_jobs=20).fit(Xfull, y)
imp = pd.Series(m.booster_.feature_importance('gain'), index=Xfull.columns); imp /= imp.sum()
B = ['art_te','art_seen','artdec_te','text_p','art_te_k1','art_hits','art_flops','knn_p','artname_p','text5_p','prim_te','prim_seen','cred_max','cred_mean','cred_seen']
AUDIO = ["danceability","energy","key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence","tempo","duration_ms","time_signature","chorus_hit","sections"]
def fam(c):
    if c in ('text_p','text5_p','artname_p'): return 'B: text models (title, artist name)'
    if c == 'knn_p': return 'B: audio neighbours'
    if c in B: return 'B: artist track record (hits/flops)'
    if c in AUDIO or c in ('decade','log_dur','chorus_frac','sec_per_min','dance_x_energy','acoustic_minus_energy','vocal'): return 'A: raw audio + simple ratios'
    if c.endswith('_zdec'): return 'A: audio vs its decade'
    if c.startswith('art_') or c.startswith('dev_art'): return 'A: artist catalogue (no labels)'
    return 'A: title and name shape'
g = imp.groupby(imp.index.map(fam)).agg(['sum','size']).sort_values('sum', ascending=False)
print(g.round(3)); print(imp.sort_values(ascending=False).head(10).round(3))
