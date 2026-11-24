"""Run 06: neural network with entity embeddings, on the GPU.

Same fold-safe design matrix as the boosting models, plus learned embeddings for the artist, the
primary artist and the decade. Artist ids are dropped to an "unknown" token at random during
training (30%) so the network also learns to judge songs of artists it has never seen, which is
42% of the test. Its value is diversity for the final blend, not beating the trees alone.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import QuantileTransformer

from src.common import load, report, save_oof
from src.features import primary_artist
from src.pipeline import design

DEV = "cuda" if torch.cuda.is_available() else "cpu"
train, test = load()
y, fs, (Xfull, Xtest) = design(seed=0)
cols = list(Xfull.columns)


class Net(nn.Module):
    def __init__(self, n_num, n_art, n_prim, d=16, h=256, p=0.3):
        super().__init__()
        self.art = nn.Embedding(n_art, d)
        self.prim = nn.Embedding(n_prim, d)
        self.dec = nn.Embedding(6, 4)
        self.mlp = nn.Sequential(
            nn.Linear(n_num + 2 * d + 4, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(p),
            nn.Linear(h, h // 2), nn.BatchNorm1d(h // 2), nn.SiLU(), nn.Dropout(p),
            nn.Linear(h // 2, 1))

    def forward(self, x, a, pr, dc):
        return self.mlp(torch.cat([x, self.art(a), self.prim(pr), self.dec(dc)], 1)).squeeze(1)


def ids(fit_keys, keys):
    """Index 0 is 'unknown': anything not in the fit set."""
    vocab = {k: i + 1 for i, k in enumerate(pd.unique(fit_keys))}
    return np.array([vocab.get(k, 0) for k in keys]), len(vocab) + 1


def fit_predict(Xa, ya, Xb, art_a, art_b, prim_a, prim_b, dec_a, dec_b, seed, epochs=60, drop_id=0.3):
    torch.manual_seed(seed); np.random.seed(seed)
    qt = QuantileTransformer(n_quantiles=500, output_distribution="normal", random_state=seed)
    xa = torch.tensor(qt.fit_transform(Xa), dtype=torch.float32, device=DEV)
    xb = torch.tensor(qt.transform(Xb), dtype=torch.float32, device=DEV)
    aa, n_art = ids(art_a, art_a); ab, _ = ids(art_a, art_b)
    pa, n_prim = ids(prim_a, prim_a); pb, _ = ids(prim_a, prim_b)
    t = lambda v: torch.tensor(v, dtype=torch.long, device=DEV)
    aa, ab, pa, pb = t(aa), t(ab), t(pa), t(pb)
    da, db = t((dec_a - 1960) // 10), t((dec_b - 1960) // 10)
    yt = torch.tensor(ya, dtype=torch.float32, device=DEV)

    net = Net(xa.shape[1], n_art, n_prim).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * ((len(ya) + 255) // 256))
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(ya), device=DEV)
        for i in range(0, len(ya), 256):
            b = perm[i:i + 256]
            mask = (torch.rand(len(b), device=DEV) < drop_id)
            a_in = torch.where(mask, torch.zeros_like(aa[b]), aa[b])
            p_in = torch.where(mask, torch.zeros_like(pa[b]), pa[b])
            loss = lossf(net(xa[b], a_in, p_in, da[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(xb, ab, pb, db)).cpu().numpy()


SEEDS = 5
prim_tr = train.artist.map(primary_artist).values
prim_te = test.artist.map(primary_artist).values
oof = np.zeros(len(y))
for a, b, Xa, Xb in fs:
    oof[b] = np.mean([fit_predict(Xa[cols].values, y[a], Xb[cols].values, train.artist.values[a],
                                  train.artist.values[b], prim_tr[a], prim_tr[b], train.decade.values[a],
                                  train.decade.values[b], s) for s in range(SEEDS)], 0)
report("r06 nn entity embeddings x5", y, oof, DEV)
tp = np.mean([fit_predict(Xfull[cols].values, y, Xtest[cols].values, train.artist.values, test.artist.values,
                          prim_tr, prim_te, train.decade.values, test.decade.values, s) for s in range(SEEDS)], 0)
save_oof("r06_nn", oof, tp)
