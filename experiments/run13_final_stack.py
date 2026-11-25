"""Run 13: final stack on the 10-fold OOFs.

Inputs (all out of fold): LightGBM, CatBoost, XGBoost (run 10), the char BiGRU (run 11, k=10) and
the TF-IDF text model that the trees also see as a feature, plus interactions with seen-artist. Second level: logistic regression on
logits, itself evaluated out of fold, then refitted on all of train and applied to test.
Writes results/decisions.csv (id, p_hit, invest) at the break-even threshold 0.375.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pickle

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

from src.common import BREAK_EVEN, RESULTS, folds, load, report, save_oof

tr, te = load()
y = tr.target.values
lg = lambda p: logit(np.clip(p, 1e-5, 1 - 1e-5))
oof = lambda n: lg(np.load(RESULTS / 'oof' / f'{n}_oof.npy'))
tst = lambda n: lg(np.load(RESULTS / 'oof' / f'{n}_test.npy'))

_, fs, (Xfull, Xtest) = pickle.load(open(RESULTS / 'cache' / 'design_v5_s0_k10_t1.pkl', 'rb'))
text_oof = np.zeros(len(y)); seen_oof = np.zeros(len(y))
for a, b, Xa, Xb in fs:
    text_oof[b] = Xb['text5_p'].values; seen_oof[b] = Xb['cred_seen'].values
inputs = {
    'trees': (np.mean([oof(f'r09_{m}_k10') for m in ('lgbm', 'cat', 'xgb')], 0),
              np.mean([tst(f'r09_{m}_k10') for m in ('lgbm', 'cat', 'xgb')], 0)),
    'gru': (oof('r11_gru_k10'), tst('r11_gru_k10')),
    'text': (lg(text_oof), lg(Xtest['text5_p'].values)),
}
# 'seen': some credited artist of the song is in the training part. The stack may weigh each
# model differently for known and new artists (interaction terms).
u, uT = (seen_oof > 0).astype(float), (Xtest['cred_seen'].values > 0).astype(float)
for k in ('trees', 'gru', 'text'):
    inputs[k + '*seen'] = (inputs[k][0] * u, inputs[k][1] * uT)
inputs['seen'] = (u, uT)
for subset in (['trees'], ['trees', 'gru'], ['trees', 'gru', 'text'],
               ['trees', 'gru', 'text', 'seen', 'trees*seen', 'gru*seen', 'text*seen']):
    S = np.column_stack([inputs[k][0] for k in subset])
    ST = np.column_stack([inputs[k][1] for k in subset])
    o = np.zeros(len(y))
    for rep in range(5):  # repeat the second-level CV to average out the split
        for a, b in folds(y, 10, seed=rep):
            o[b] += LogisticRegression(max_iter=2000).fit(S[a], y[a]).predict_proba(S[b])[:, 1] / 5
    report(f'r13 stack k10 {"+".join(subset)}'[:60], y, o)
m = LogisticRegression(max_iter=2000).fit(S, y)
print('weights', dict(zip(subset, np.round(m.coef_[0], 3))), 'intercept', round(m.intercept_[0], 3))
p = m.predict_proba(ST)[:, 1]
save_oof('r13_final', o, p)
out = pd.DataFrame({'id': te['id'], 'p_hit': p.round(4), 'invest': (p > BREAK_EVEN).astype(int)})
out.to_csv(RESULTS / 'decisions.csv', index=False)
print('invest in', out.invest.sum(), 'of', len(out), '| mean p', p.mean().round(3), '| train hit rate', y.mean().round(3))
