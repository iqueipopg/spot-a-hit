"""Run 03: second stage that lets songs of the same artist talk to each other.

Stage 1 gives every unlabeled song (validation fold or test) a probability. Songs by one artist
are almost always all hits or all flops, so the stage-1 probabilities of an artist's *other*
unlabeled songs are evidence about this one. That uses no label: only the model's own guesses on
the closed box we are given. Stage 2 is a small logistic/GBM model on top, fitted out of fold.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.common import folds, load, report, save_oof
from src.pipeline import design

train, test = load()
y, fs, (Xfull, Xtest) = design(seed=0, version=2)
names = ["lgbm", "xgb", "cat"]
oof = np.mean([logit(np.clip(np.load(f"results/oof/r02_{n}_oof.npy"), 1e-6, 1 - 1e-6)) for n in names], axis=0)
tst = np.mean([logit(np.clip(np.load(f"results/oof/r02_{n}_test.npy"), 1e-6, 1 - 1e-6)) for n in names], axis=0)


def peer_features(artists: np.ndarray, z: np.ndarray) -> pd.DataFrame:
    """Leave-one-out mean of stage-1 logits among the other unlabeled songs of the same artist."""
    df = pd.DataFrame({"a": artists, "z": z})
    g = df.groupby("a").z
    s, n = g.transform("sum"), g.transform("size")
    peer_n = n - 1
    peer_mean = ((s - df.z) / peer_n.replace(0, np.nan)).fillna(0)
    return pd.DataFrame({"peer_n": peer_n, "peer_mean": peer_mean, "peer_has": (peer_n > 0).astype(int)})


# Peer features inside each validation fold (the songs that were unlabeled together)
P = pd.DataFrame(index=range(len(y)), columns=["peer_n", "peer_mean", "peer_has"], dtype=float)
for a, b, _, _ in fs:
    P.iloc[b] = peer_features(train.artist.values[b], oof[b]).values
Pt = peer_features(test.artist.values, tst)

# Seen-in-train evidence is already inside stage 1; the peer block adds the unlabeled neighbours.
def stage2_X(z, P, seen):
    return np.column_stack([z, P.peer_mean * P.peer_has, P.peer_has, np.log1p(P.peer_n),
                            P.peer_mean * P.peer_has * (seen == 0)])

seen_tr = np.zeros(len(y))
for a, b, _, Xb in fs:
    seen_tr[b] = Xb.art_seen.values
X2, X2t = stage2_X(oof, P, seen_tr), stage2_X(tst, Pt, Xtest.art_seen.values)

report("r03 stage1 mean-logit", y, 1 / (1 + np.exp(-oof)))
oof2 = np.zeros(len(y))
for a, b in folds(y, 5, seed=42):
    m = LogisticRegression(C=1.0, max_iter=1000).fit(X2[a], y[a])
    oof2[b] = m.predict_proba(X2[b])[:, 1]
report("r03 stage2 peer logistic", y, oof2, "peer = other unlabeled songs of same artist")
m = LogisticRegression(C=1.0, max_iter=1000).fit(X2, y)
save_oof("r03_peer_lr", oof2, m.predict_proba(X2t)[:, 1])
print("coefs", np.round(m.coef_, 3))
print("test songs with peers:", (Pt.peer_n > 0).mean().round(3), " CV songs with peers:", (P.peer_n > 0).mean().round(3))
