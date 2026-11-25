"""Second-level experiments on saved 10-fold OOFs (no retraining). Prints one line per stack."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pickle

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

from src.common import RESULTS, evaluate, folds, load

tr, te = load(); y = tr.target.values
lg = lambda p: logit(np.clip(p, 1e-5, 1 - 1e-5))
L = lambda n: lg(np.load(RESULTS / 'oof' / f'{n}_oof.npy'))
_, fs, _ = pickle.load(open(RESULTS / 'cache' / 'design_v5_s0_k10_t1.pkl', 'rb'))
t = np.zeros(len(y)); cs = np.zeros(len(y))
for a, b, Xa, Xb in fs:
    t[b] = Xb['text5_p'].values; cs[b] = Xb['cred_seen'].values
u = (cs > 0).astype(float)
I = {'t09': np.mean([L(f'r09_{m}_k10') for m in ('lgbm', 'cat', 'xgb')], 0),
     't14': np.mean([L('r14_lgbm_k10'), L('r14_cat_k10')], 0),
     't14x': np.mean([L('r14_lgbm_k10'), L('r14_cat_k10'), L('r09_xgb_k10')], 0),
     'gru': L('r11_gru_k10'), 'text': lg(t)}


def cv(cols, reps=5):
    S = np.column_stack(cols); o = np.zeros(len(y))
    for r in range(reps):
        for a, b in folds(y, 10, seed=r):
            o[b] += LogisticRegression(max_iter=2000).fit(S[a], y[a]).predict_proba(S[b])[:, 1] / reps
    e = evaluate(y, o)
    return f"AUC {e['auc']:.4f} acc {e['acc']:.4f} ll {e['logloss']:.4f} profit {e['profit_375']:.1f}"


inter = lambda k: [I[k], I['gru'], I['text'], u, I[k] * u, I['gru'] * u, I['text'] * u]
for k in ('t09', 't14', 't14x'):
    print(k, 'plain', cv([I[k], I['gru'], I['text']]))
    print(k, 'inter', cv(inter(k)))
print('t14+t09 inter', cv(inter('t14') + [I['t09']]))
