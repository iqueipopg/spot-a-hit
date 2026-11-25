"""Run 14: the char BiGRU as a feature inside the trees, nested out of fold.

In each outer fold (the same 10 folds as the v5 design): inner 5-fold BiGRU on the fit part gives
'gru_p' for those rows, a BiGRU on the whole fit part gives 'gru_p' for the held-out rows. For the
final train->test design, the train column is the 10-fold OOF of run 11 and the test column its
full-train prediction. Then LightGBM and CatBoost with the run-08 parameters, 3 seeds.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import pickle

import catboost as cb
import lightgbm as lgb
import numpy as np
from scipy.special import expit, logit

from src import gru as G
from src.common import RESULTS, folds, report, save_oof, title_key

y, fs, (Xfull, Xtest) = pickle.load(open(RESULTS / 'cache' / 'design_v5_s0_k10_t1.pkl', 'rb'))
best = json.loads((RESULTS / 'best_params.json').read_text())
cache = RESULTS / 'cache' / 'gru_nested_k10.pkl'
groups = G.tr.track_raw.map(title_key).values
if cache.exists():
    nested = pickle.load(open(cache, 'rb'))
else:
    nested = []
    for i, (a, b, Xa, Xb) in enumerate(fs):
        fa = np.zeros(len(a))
        for ia, ib in folds(y[a], 5, seed=i, groups=groups[a]):
            (p,) = G.fit_predict(a[ia], G.Xtr[a[ia]], G.dec_tr[a[ia]], G.y[a[ia]], [(G.Xtr[a[ib]], G.dec_tr[a[ib]])], seed=i)
            fa[ib] = p
        (fb,) = G.fit_predict(a, G.Xtr[a], G.dec_tr[a], G.y[a], [(G.Xtr[b], G.dec_tr[b])], seed=i + 50)
        nested.append((fa, fb)); print('outer fold', i, flush=True)
    pickle.dump(nested, open(cache, 'wb'))

lg = lambda p: logit(np.clip(p, 1e-5, 1 - 1e-5))
gfull = np.load(RESULTS / 'oof' / 'r11_gru_k10_oof.npy'); gtest = np.load(RESULTS / 'oof' / 'r11_gru_k10_test.npy')
makers = {
    'lgbm': lambda s: lgb.LGBMClassifier(**best['lgbm'], subsample_freq=1, verbose=-1, random_state=s, n_jobs=20),
    'cat': lambda s: cb.CatBoostClassifier(**best['cat'], verbose=0, random_seed=s, thread_count=20),
}
Z = {}
for name, mk in makers.items():
    oof = np.zeros(len(y)); tst = np.zeros(len(Xtest))
    for s in range(3):
        for (a, b, Xa, Xb), (fa, fb) in zip(fs, nested):
            m = mk(s).fit(Xa.assign(gru_p=lg(fa)), y[a])
            oof[b] += m.predict_proba(Xb.assign(gru_p=lg(fb)))[:, 1] / 3
        m = mk(s).fit(Xfull.assign(gru_p=lg(gfull)), y)
        tst += m.predict_proba(Xtest.assign(gru_p=lg(gtest)))[:, 1] / 3
    report(f'r14 {name} v5+gru k10 x3', y, oof)
    save_oof(f'r14_{name}_k10', oof, tst)
    Z[name] = lg(oof)
report('r14 mean-logit lgbm+cat', y, expit(np.mean(list(Z.values()), 0)))
