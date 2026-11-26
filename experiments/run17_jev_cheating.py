"""Run 17: how far a model with outside knowledge goes. Jev (pretrained, zero-shot) on all 10,000 train songs,
stacked with our out-of-fold predictions. Usage: python experiments/run17_jev_cheating.py results/jev_train.tsv"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/home/queipo/spot-a-hit')
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from src.common import load, profit
J = sys.argv[1]
tr, te = load(); y_all = tr.target.values
ours = pd.Series(np.load('/home/queipo/spot-a-hit/results/oof/r13_final_oof.npy'), index=tr.id.astype(str))
jev = pd.read_csv(J, sep='\t', dtype={'id': str}).set_index('id')['hit'].dropna()
ids = jev.index.intersection(ours.index)
y = tr.set_index(tr.id.astype(str)).loc[ids, 'target'].values
o, j = ours[ids].values, jev[ids].values
lg = lambda p: logit(np.clip(p, 1e-4, 1 - 1e-4))
def stack(cols):
    X = np.column_stack(cols); out = np.zeros(len(y))
    for r in range(5):
        for a, b in StratifiedKFold(10, shuffle=True, random_state=r).split(X, y):
            out[b] += LogisticRegression(max_iter=2000).fit(X[a], y[a]).predict_proba(X[b])[:, 1] / 5
    return out
scale = 10000 / len(y)
rows = {'Ours (honest)': o, 'Jev raw': j, 'Jev recalibrated': stack([lg(j)]), 'Ours + Jev (cheating)': stack([lg(o), lg(j)])}
print(f'n={len(y)}  perfect foresight x{scale:.0f}: {1.5 * y.sum() * scale:,.0f}M')
for k, p in rows.items():
    print(f'{k:<24} AUC {roc_auc_score(y, p):.4f}  profit@.375 {profit(y, p > 0.375) * scale:,.0f}M (per 10k songs)')
c = rows['Ours + Jev (cheating)']
r = np.random.default_rng(0); d, dp = [], []
for _ in range(1000):
    i = r.integers(0, len(y), len(y))
    d.append(roc_auc_score(y[i], c[i]) - roc_auc_score(y[i], o[i])); dp.append(profit(y[i], c[i] > .375) - profit(y[i], o[i] > .375))
print(f'paired gain AUC {np.mean(d):.4f} +- {np.std(d, ddof=1):.4f} | profit {np.mean(dp):.0f} +- {np.std(dp, ddof=1):.0f}M')
_, fs, _ = __import__('pickle').load(open('/home/queipo/spot-a-hit/results/cache/design_v5_s0_k10_t1.pkl', 'rb'))
seen = np.zeros(len(tr))
for a, b, Xa, Xb in fs: seen[b] = Xb['cred_seen'].values
s = pd.Series(seen > 0, index=tr.id.astype(str))[ids].values
for nm, m in (('known artists', s), ('new artists', ~s)):
    print(f'{nm}: ours {roc_auc_score(y[m], o[m]):.4f} -> with Jev {roc_auc_score(y[m], c[m]):.4f}; Jev alone {roc_auc_score(y[m], j[m]):.4f}')
