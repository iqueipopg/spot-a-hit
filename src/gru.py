"""Char BiGRU over masked title || artist (the model of run 11), importable."""
import numpy as np
import torch
import torch.nn as nn

from src.common import load

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
tr, te = load()
y = tr.target.values.astype(np.float32)
join = lambda d: d.track.str.slice(0, 60) + ' || ' + d.artist.str.slice(0, 40)
s_tr = join(tr).tolist()
s_te = join(te).tolist()
chars = sorted(set(''.join(s_tr + s_te)))
idx = {c: i + 1 for i, c in enumerate(chars)}
L = 104


def enc(ss):
    X = np.zeros((len(ss), L), dtype=np.int64)
    for i, s in enumerate(ss):
        v = [idx[c] for c in s][:L]
        X[i, :len(v)] = v
    return torch.tensor(X)


Xtr, Xte = enc(s_tr), enc(s_te)
dec_tr = torch.tensor(((tr.decade.values - 1960) // 10).astype(np.int64))
dec_te = torch.tensor(((te.decade.values - 1960) // 10).astype(np.int64))


class Net(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.emb = nn.Embedding(V + 1, 48, padding_idx=0)
        self.dec = nn.Embedding(6, 8)
        self.gru = nn.GRU(48, 96, batch_first=True, bidirectional=True, num_layers=1)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Sequential(nn.Linear(2 * 96 * 2 + 8, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x, d):
        m = (x > 0).unsqueeze(-1).float()
        h, _ = self.gru(self.drop(self.emb(x)))
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        mx = (h - 1e4 * (1 - m)).max(1).values
        return self.out(torch.cat([mean, mx, self.dec(d)], 1)).squeeze(-1)


def fit_predict(a, Xa, Da, ya, apply_sets, seed, epochs=14):
    torch.manual_seed(seed)
    net = Net(len(chars)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    n = len(a); bs = 128
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * ((n + bs - 1) // bs))
    Xa, Da, ya = Xa.to(dev), Da.to(dev), torch.tensor(ya).to(dev)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            opt.zero_grad(); loss = lossf(net(Xa[j], Da[j]), ya[j]); loss.backward(); opt.step(); sched.step()
    net.eval(); outs = []
    with torch.no_grad():
        for Xb, Db in apply_sets:
            outs.append(torch.sigmoid(net(Xb.to(dev), Db.to(dev))).cpu().numpy())
    return outs
