"""Run 12: hybrid net = BiGRU(title) + BiGRU(artist) + audio MLP; was run 11 over 'masked title || artist' on the GPU.

A different inductive bias from TF-IDF + logistic regression: it reads the string in order, so it
can learn patterns such as 'Lil ', ' Orchestra', '(R***x)' as sequences rather than bags of
n-grams. Trained out of fold on the same title-grouped folds; the OOF probability is then a
candidate feature / stack input. Usage: python run11_char_gru.py [n_splits]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn

from src.common import folds, load, report, save_oof, title_key

K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
tr, te = load()
y = tr.target.values.astype(np.float32)
chars = sorted(set(''.join(tr.track.tolist() + te.track.tolist() + tr.artist.tolist() + te.artist.tolist())))
idx = {c: i + 1 for i, c in enumerate(chars)}
L = 64


def enc(ss):
    X = np.zeros((len(ss), L), dtype=np.int64)
    for i, s in enumerate(ss):
        v = [idx[c] for c in s][:L]
        X[i, :len(v)] = v
    return torch.tensor(X)


Ttr, Tte = enc(tr.track.tolist()), enc(te.track.tolist())
Atr, Ate = enc(tr.artist.tolist()), enc(te.artist.tolist())
from sklearn.preprocessing import QuantileTransformer
from src.features import AUDIO
qt = QuantileTransformer(output_distribution='normal', n_quantiles=500).fit(np.vstack([tr[AUDIO].values, te[AUDIO].values]))
Ntr, Nte = torch.tensor(qt.transform(tr[AUDIO].values), dtype=torch.float32), torch.tensor(qt.transform(te[AUDIO].values), dtype=torch.float32)
Xtr, Xte = (Ttr, Atr, Ntr), (Tte, Ate, Nte)
dec_tr = torch.tensor(((tr.decade.values - 1960) // 10).astype(np.int64))
dec_te = torch.tensor(((te.decade.values - 1960) // 10).astype(np.int64))


NA = len(AUDIO)


class Net(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.emb = nn.Embedding(V + 1, 48, padding_idx=0)
        self.dec = nn.Embedding(6, 8)
        self.gt = nn.GRU(48, 64, batch_first=True, bidirectional=True)
        self.ga = nn.GRU(48, 64, batch_first=True, bidirectional=True)
        self.num = nn.Sequential(nn.Linear(NA, 64), nn.ReLU())
        self.drop = nn.Dropout(0.3)
        self.out = nn.Sequential(nn.Linear(4 * 128 + 64 + 8, 96), nn.ReLU(), nn.Dropout(0.3), nn.Linear(96, 1))

    def pool(self, gru, x):
        m = (x > 0).unsqueeze(-1).float()
        h, _ = gru(self.drop(self.emb(x)))
        return torch.cat([(h * m).sum(1) / m.sum(1).clamp(min=1), (h - 1e4 * (1 - m)).max(1).values], 1)

    def forward(self, x, d):
        t, a, n = x
        return self.out(torch.cat([self.pool(self.gt, t), self.pool(self.ga, a), self.num(n), self.dec(d)], 1)).squeeze(-1)


def fit_predict(a, Xa, Da, ya, apply_sets, seed, epochs=14):
    torch.manual_seed(seed)
    net = Net(len(chars)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    n = len(a); bs = 128
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * ((n + bs - 1) // bs))
    Xa, Da, ya = tuple(z.to(dev) for z in Xa), Da.to(dev), torch.tensor(ya).to(dev)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            opt.zero_grad(); loss = lossf(net(tuple(z[j] for z in Xa), Da[j]), ya[j]); loss.backward(); opt.step(); sched.step()
    net.eval(); outs = []
    with torch.no_grad():
        for Xb, Db in apply_sets:
            outs.append(torch.sigmoid(net(tuple(z.to(dev) for z in Xb), Db.to(dev))).cpu().numpy())
    return outs


SEEDS = 5
groups = tr.track_raw.map(title_key).values
oof = np.zeros(len(y))
for s in range(SEEDS):
    for a, b in folds(y, K, seed=0, groups=groups):
        (pb,) = fit_predict(a, tuple(z[a] for z in Xtr), dec_tr[a], y[a], [(tuple(z[b] for z in Xtr), dec_tr[b])], seed=100 * s)
        oof[b] += pb / SEEDS
    print('seed', s, 'done', flush=True)
# full-train model for the test set
tst = np.mean([fit_predict(np.arange(len(y)), Xtr, dec_tr, y, [(Xte, dec_te)], seed=900 + s)[0] for s in range(SEEDS)], 0)
report(f'r12 hybrid GRU+audio k{K} x{SEEDS}', y, oof)
save_oof(f'r12_hyb_k{K}', oof, tst)
