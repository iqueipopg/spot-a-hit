"""Builds the submission notebook (one self-contained .ipynb) from the cells below.

The notebook only needs train.xlsx and test.xlsx next to it. It is the final pipeline of the
experiments (runs 09-13 in DIARIO.md) with nothing else: no cache, no imports from src/.
"""
from pathlib import Path

import nbformat as nbf

NAME = "QueipoDeLlano_Ignacio"
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Spot-a-hit — Ignacio Queipo de Llano

**Goal.** For each of the 2,000 test songs, invest (1) or pass (0). A hit returns +\$1.5M net, a
flop −\$0.9M, so we invest when **P(hit) > 0.9 / 2.4 = 0.375**. The score is profit, which rewards
two things: ranking songs well (AUC) and well-calibrated probabilities around 0.375.

**Pipeline.**
1. Read both files. Test titles are *masked* (every lowercase a, b, c, e, h, i, m, o, s, u, v is
   shown as `*`); train titles are masked the same way so every title feature means the same in both.
2. Features that never look at the target, built on train + test together (audio, audio relative
   to its decade, artist catalogue statistics, name and title shape).
3. Target-based features (artist hit rates, text models on title and artist, audio neighbours),
   always out of fold with an inner K-fold: no song ever sees its own label.
4. Three gradient-boosted tree models (LightGBM, CatBoost, XGBoost) and a character-level BiGRU.
5. A logistic-regression stack of the four, with interactions for *artist already seen*, and the
   decision at the 0.375 break-even threshold.

Validation is 10-fold, stratified and **grouped by normalised title**, as in the official split (no
test title appears in train). Expected CV: AUC ≈ 0.948, accuracy ≈ 0.904, profit ≈ \$6.28bn on the
10,000 training songs (86.6% of the perfect-foresight ceiling).

**Runtime.** With the defaults below, about 60 min on a 20-core CPU (+ ~15 min for the BiGRU
with a GPU, longer without). Set `QUICK = True` for a ~10 min run with 5 folds and one seed (slightly
lower scores, same method).
""")

code(r"""
import os, re, warnings
QUICK = False
N_FOLDS = 5 if QUICK else 10
SEEDS = 1 if QUICK else 3        # seed bagging of the tree models
GRU_SEEDS = 1 if QUICK else 5
N_JOBS = max(1, (os.cpu_count() or 2) - 2)   # -1 makes LightGBM/CatBoost crawl on some machines

import numpy as np, pandas as pd
import lightgbm as lgb, catboost as cb, xgboost as xgb
import torch, torch.nn as nn
from scipy.sparse import hstack
from scipy.special import logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
warnings.filterwarnings("ignore")
""")

md("## 1. Data, economics and validation")

code(r"""
GAIN_HIT, LOSS_FLOP = 2.5 - 1.0, 1.0 - 0.1
BREAK_EVEN = LOSS_FLOP / (GAIN_HIT + LOSS_FLOP)          # 0.375

# Test titles are masked; the rule was inferred from letter survival rates and, applied to train,
# reproduces the test's share of masked characters (38.6% vs 39.0%).
MASKED = set("abcehimosuv")
mask_text = lambda s: "".join("*" if ch in MASKED else ch for ch in s)

train, test = pd.read_excel("train.xlsx"), pd.read_excel("test.xlsx")
for df in (train, test):
    df["track"], df["artist"] = df["track"].astype(str), df["artist"].astype(str)
train["track_raw"] = train["track"]                      # only used to group covers in the CV
train["track"] = train["track"].map(mask_text)
y = train["target"].values
print(train.shape, test.shape, "hit rate", y.mean().round(3))

TITLE_RE = re.compile(r"\s*[\(\[].*?[\)\]]|\s+-\s+.*$")
title_key = lambda t: TITLE_RE.sub("", t.lower()).strip()   # '(Remastered)', ' - Live' removed
GROUPS = train["track_raw"].map(title_key).values

def folds(y, k, seed=0, groups=None):
    # Stratified and grouped: versions of the same title never fall on both sides
    g = groups if groups is not None else np.arange(len(y))
    return list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, g))

def profit(y, invest):
    invest = np.asarray(invest, bool)
    return GAIN_HIT * np.sum(invest & (y == 1)) - LOSS_FLOP * np.sum(invest & (y == 0))

def evaluate(name, y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    r = dict(model=name, AUC=roc_auc_score(y, p), accuracy=accuracy_score(y, p > 0.5),
             logloss=log_loss(y, p), profit_M=profit(y, p > BREAK_EVEN))
    print(f"{name:<34} AUC {r['AUC']:.4f}  acc {r['accuracy']:.4f}  logloss {r['logloss']:.4f}  profit {r['profit_M']:.0f}M")
    return r
""")

md(r"""
## 2. Features that do not use the target

Built on train + test together: they never look at a label, and the test songs are part of the
closed box we are given (no external data).
""")

code(r"""
AUDIO = ["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness",
         "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "time_signature",
         "chorus_hit", "sections"]
CONTINUOUS = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness",
              "liveness", "valence", "tempo", "duration_ms", "chorus_hit", "sections"]
masked_re = lambda words: re.compile("|".join(re.escape(mask_text(w)) for w in words), re.I)
FEAT_RE = re.compile(r"\b(?:feat\.?|featuring|ft\.?|with|x|vs\.?)\b|&|,|\band\b", re.I)
CLASSICAL_RE = re.compile(r"\b(?:op\.|bwv|k\.\s?\d|no\.\s?\d|sonata|concerto|symphon|allegro|adagio|andante|"
                          r"prelude|nocturne|etude|minor|major|act \w+|scene|aria|movement|mvt)\b", re.I)
ENSEMBLE_RE = re.compile(r"orchestra|quartet|trio|ensemble|choir|philharmonic|symphony|band|chorus|"
                         r"singers|brothers|sisters|boys|girls", re.I)
TR_FEAT_M = masked_re(["feat"])
CLASSICAL_M = masked_re(["op.", "bwv", "no. ", "sonata", "concerto", "symphon", "allegro", "adagio", "andante",
                         "prelude", "nocturne", "etude", "minor", "major", "aria", "movement"])
VERSION_M = masked_re(["remaster", "live", "version", "mix", "edit", "demo", "acoustic", "instrumental",
                       "mono", "stereo", "take"])
nonascii = lambda s: sum(ord(c) > 127 for c in s) / max(len(s), 1)

def unsupervised(train, test):
    al = pd.concat([train.drop(columns=["target", "track_raw"]), test], ignore_index=True)
    X = al[["decade"] + AUDIO].copy()
    dur = al.duration_ms / 1000
    X["log_dur"], X["chorus_frac"], X["sec_per_min"] = np.log(dur), al.chorus_hit / dur, al.sections / (dur / 60)
    X["dance_x_energy"] = al.danceability * al.energy
    X["acoustic_minus_energy"] = al.acousticness - al.energy
    X["vocal"] = (al.instrumentalness < 0.5).astype(int)
    for c in CONTINUOUS:                                   # a 1965 hit does not sound like a 2015 hit
        g = al.groupby("decade")[c]
        X[f"{c}_zdec"] = (al[c] - g.transform("mean")) / g.transform("std")
    art = al.groupby("artist")                             # how prolific, how long, how they sound
    X["art_n"], X["art_n_dec"] = art.artist.transform("size"), art.decade.transform("nunique")
    X["art_first_dec"] = art.decade.transform("min")
    X["art_span"] = art.decade.transform("max") - X["art_first_dec"]
    for c in ["danceability", "energy", "acousticness", "instrumentalness", "speechiness", "valence", "loudness"]:
        X[f"art_mean_{c}"] = art[c].transform("mean")
        X[f"dev_art_{c}"] = al[c] - X[f"art_mean_{c}"]
    X["art_std_energy"] = art.energy.transform("std").fillna(0)
    a, t = al.artist, al.track
    X["art_multi"], X["art_ensemble"] = a.str.contains(FEAT_RE).astype(int), a.str.contains(ENSEMBLE_RE).astype(int)
    X["art_the"] = a.str.lower().str.startswith("the ").astype(int)
    X["art_len"], X["art_words"], X["art_nonascii"] = a.str.len(), a.str.split().str.len(), a.map(nonascii)
    X["tr_len"], X["tr_words"], X["tr_nonascii"] = t.str.len(), t.str.split().str.len(), t.map(nonascii)
    X["tr_paren"] = t.str.contains(r"[\(\[]").astype(int)
    X["tr_dash"] = t.str.contains(r" - ").astype(int)
    X["tr_digits"] = t.str.count(r"\d")
    X["tr_feat"] = t.str.contains(TR_FEAT_M).astype(int)  # keyword flags in their masked spelling
    X["tr_classical"] = (t.str.contains(CLASSICAL_M) | a.str.contains(CLASSICAL_RE)).astype(int)
    X["tr_version"] = t.str.contains(VERSION_M).astype(int)
    X["tr_star_share"] = t.str.count(r"\*") / t.str.len().clip(lower=1)
    X["tr_upper"] = t.map(lambda s: sum(c.isupper() for c in s) / max(sum(c.isalpha() for c in s), 1))
    X["tr_colon"] = t.str.contains(":").astype(int)
    tl = t.str.lower().str.strip()
    X["tr_title_n"] = tl.map(tl.value_counts())            # same title by several artists: standards
    return X

U = unsupervised(train, test)
U_tr, U_te = U.iloc[:len(train)].reset_index(drop=True), U.iloc[len(train):].reset_index(drop=True)
print(U.shape[1], "unsupervised features")
""")

md(r"""
## 3. Target-based features, fold-safe

`supervised(fit, y_fit, apply)` returns the features for the *fit* rows out of fold (inner 5-fold,
grouped by title) and for the *apply* rows from a fit on all of *fit*. In the CV it is called once
per outer fold, so the validation rows are always an *apply* set.

* **Artist hit rate** (smoothed), raw hit and flop counts, artist × decade, and the same for the
  *primary* and every *credited* artist ("Kid Ink Featuring Chris Brown" → kid ink, chris brown).
  83% of the artists with several songs are all-hit or all-flop, so this is the strongest signal.
* **Text models**: TF-IDF + logistic regression on masked title and artist name.
* **Audio neighbours**: hit rate of the 50 songs that sound most alike.
""")

code(r"""
def inner(k, seed, n, y, groups):
    return StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(np.zeros(n), y, groups)

def smoothed_rate(kf, yf, ka, prior, k):
    g = pd.DataFrame({"k": kf, "y": yf}).groupby("k").y.agg(["sum", "count"])
    s = pd.Series(ka)
    return s.map((g["sum"] + prior * k) / (g["count"] + k)).fillna(prior).values, s.map(g["count"]).fillna(0).values

def oof_apply(fn, n_fit, y_fit, seed, groups):
    # Generic out-of-fold wrapper: fn(fit_idx, apply_idx or None) -> predictions
    out = None
    for a, b in inner(5, seed, n_fit, y_fit, groups):
        r = fn(a, b)
        out = np.zeros((n_fit,) + np.shape(r)[1:]) if out is None else out
        out[b] = r
    return out, fn(np.arange(n_fit), None)

def target_encode(kf, yf, ka, k, seed, groups):
    kf, prior = np.asarray(kf), yf.mean()
    f = lambda a, b: np.column_stack(smoothed_rate(kf[a], yf[a], kf[b] if b is not None else ka, prior, k))
    fo, ao = oof_apply(f, len(kf), yf, seed, groups)
    return fo[:, 0], fo[:, 1], ao[:, 0], ao[:, 1]

SPLIT_RE = re.compile(r"\s+(?:featuring|feat\.?|ft\.?|with|x|vs\.?|and|duet with)\s+|\s*[,&/]\s*", re.I)
credited = lambda a: [p.strip().lower() for p in SPLIT_RE.split(a) if p and p.strip()] or [a.lower()]

def credited_encode(af, yf, aa, seed, groups, k=2):
    def enc(af, yf, aa):
        d = pd.DataFrame([(i, p) for i, a in enumerate(af) for p in credited(a)], columns=["i", "p"])
        d["y"] = yf[d.i]
        g = d.groupby("p").y.agg(["sum", "count"]); prior = yf.mean()
        rate = ((g["sum"] + prior * k) / (g["count"] + k)).to_dict()
        out = np.zeros((len(aa), 3))
        for j, a in enumerate(aa):
            r = [rate[p] for p in credited(a) if p in rate]
            out[j] = (max(r), np.mean(r), len(r)) if r else (prior, prior, 0)
        return out
    af = np.asarray(af)
    fo, ao = oof_apply(lambda a, b: enc(af[a], yf[a], af[b] if b is not None else np.asarray(aa)), len(af), yf, seed, groups)
    return fo.T, ao.T

KNN_COLS = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness",
            "liveness", "valence", "tempo", "duration_ms", "decade"]

def knn_feature(fit, yf, app, seed, groups):
    Xf, Xa = fit[KNN_COLS].values, app[KNN_COLS].values
    def f(a, b):
        m = make_pipeline(QuantileTransformer(n_quantiles=200), KNeighborsClassifier(50, weights="distance"))
        return m.fit(Xf[a], yf[a]).predict_proba(Xf[b] if b is not None else Xa)[:, 1]
    return oof_apply(f, len(Xf), yf, seed, groups)

def text_joint(tf, af, yf, ta, aa, seed, groups, C=2.0):
    # Title || artist in one char 2-5 + word 1-2 vocabulary
    txt_f = (tf.str.lower() + " || " + af.str.lower()).values
    txt_a = (ta.str.lower() + " || " + aa.str.lower()).values
    def f(a, b):
        v1 = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
        v2 = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        tb = txt_f[b] if b is not None else txt_a
        Xa = hstack([v1.fit_transform(txt_f[a]), v2.fit_transform(txt_f[a])]).tocsr()
        Xb = hstack([v1.transform(tb), v2.transform(tb)]).tocsr()
        return LogisticRegression(C=C, max_iter=3000).fit(Xa, yf[a]).predict_proba(Xb)[:, 1]
    return oof_apply(f, len(txt_f), yf, seed, groups)

def text_separate(tf, af, yf, ta, aa, seed, groups, C=3.0):
    # Masked titles and clean artist names are different alphabets: separate vocabularies
    tf, af, ta, aa = (s.str.lower().values for s in (tf, af, ta, aa))
    def f(a, b):
        vs = [(TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), min_df=2, sublinear_tf=True), 0),
              (TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), min_df=2, sublinear_tf=True), 1),
              (TfidfVectorizer(token_pattern=r"[^ ]+", ngram_range=(1, 3), min_df=1, sublinear_tf=True), 1),
              (TfidfVectorizer(token_pattern=r"[^ ]+", min_df=2, sublinear_tf=True), 0)]
        fit_txt = (tf[a], af[a])
        app_txt = (tf[b], af[b]) if b is not None else (ta, aa)
        Xa = hstack([v.fit_transform(fit_txt[s]) for v, s in vs]).tocsr()
        Xb = hstack([v.transform(app_txt[s]) for v, s in vs]).tocsr()
        return LogisticRegression(C=C, max_iter=4000).fit(Xa, yf[a]).predict_proba(Xb)[:, 1]
    return oof_apply(f, len(tf), yf, seed, groups)

def supervised(fit, yf, app, seed):
    G = fit.track_raw.map(title_key).values
    F, A = pd.DataFrame(index=range(len(fit))), pd.DataFrame(index=range(len(app)))
    def put(name, res):
        F[name], A[name] = res
    fa, aa = fit.artist.values, app.artist.values
    te_f, cnt_f, te_a, cnt_a = target_encode(fa, yf, aa, 5, seed, G)
    F["art_te"], F["art_seen"], A["art_te"], A["art_seen"] = te_f, cnt_f, te_a, cnt_a
    kf, ka = (fit.artist + "|" + fit.decade.astype(str)).values, (app.artist + "|" + app.decade.astype(str)).values
    F["artdec_te"], _, A["artdec_te"], _ = target_encode(kf, yf, ka, 5, seed, G)
    put("text_p", text_joint(fit.track, fit.artist, yf, app.track, app.artist, seed, G))
    F["art_te_k1"], _, A["art_te_k1"], _ = target_encode(fa, yf, aa, 1, seed, G)
    hf, _, ha, _ = target_encode(fa, yf, aa, 1e-9, seed, G)
    F["art_hits"], A["art_hits"] = hf * F.art_seen, ha * A.art_seen
    F["art_flops"], A["art_flops"] = F.art_seen - F.art_hits, A.art_seen - A.art_hits
    put("knn_p", knn_feature(fit, yf, app, seed, G))
    blank_f, blank_a = pd.Series("", index=fit.index), pd.Series("", index=app.index)
    put("artname_p", text_joint(blank_f, fit.artist, yf, blank_a, app.artist, seed, G))
    put("text5_p", text_separate(fit.track, fit.artist, yf, app.track, app.artist, seed, G))
    pf, pa = [credited(a)[0] for a in fa], [credited(a)[0] for a in aa]
    F["prim_te"], F["prim_seen"], A["prim_te"], A["prim_seen"] = target_encode(np.array(pf), yf, np.array(pa), 2, seed, G)
    (F["cred_max"], F["cred_mean"], F["cred_seen"]), (A["cred_max"], A["cred_mean"], A["cred_seen"]) = \
        credited_encode(fa, yf, aa, seed, G)
    return F, A
""")

md(r"""
## 4. Design matrices

One design per outer fold (target features fitted on the training part only) plus the final
train → test design. Every model below sees exactly the same folds.
""")

code(r"""
FOLDS = folds(y, N_FOLDS, seed=0, groups=GROUPS)
design = []
for i, (a, b) in enumerate(FOLDS):
    F, A = supervised(train.iloc[a].reset_index(drop=True), y[a], train.iloc[b].reset_index(drop=True), seed=i)
    design.append((a, b, pd.concat([U_tr.iloc[a].reset_index(drop=True), F], axis=1),
                         pd.concat([U_tr.iloc[b].reset_index(drop=True), A], axis=1)))
    print("fold", i, "done", flush=True)
F, A = supervised(train, y, test, seed=0)
X_full, X_test = pd.concat([U_tr, F], axis=1), pd.concat([U_te, A], axis=1)
print(X_full.shape[1], "features")
""")

md(r"""
## 5. Gradient-boosted trees

Hyper-parameters for LightGBM and CatBoost were tuned with Optuna minimising log-loss (the
calibration is what the 0.375 threshold pays for); they learn slowly and LightGBM sees only ~27% of
the columns per tree, a strong regulariser with many correlated artist features. Each model is
bagged over `SEEDS` seeds.
""")

code(r"""
MODELS = {
    "LightGBM": lambda s: lgb.LGBMClassifier(n_estimators=545, learning_rate=0.0137, num_leaves=30, min_child_samples=32,
                                             subsample=0.994, subsample_freq=1, colsample_bytree=0.271, reg_lambda=0.0086,
                                             reg_alpha=0.0044, verbose=-1, random_state=s, n_jobs=N_JOBS),
    "CatBoost": lambda s: cb.CatBoostClassifier(iterations=968, learning_rate=0.0105, depth=7, l2_leaf_reg=5.85, rsm=0.788,
                                                random_strength=0.711, verbose=0, random_seed=s, thread_count=N_JOBS,
                                                allow_writing_files=False),
    "XGBoost": lambda s: xgb.XGBClassifier(n_estimators=700, learning_rate=0.025, max_depth=6, min_child_weight=3,
                                           subsample=0.8, colsample_bytree=0.6, reg_lambda=2.0, tree_method="hist",
                                           n_jobs=N_JOBS, random_state=s),
}
OOF, TEST, rows = {}, {}, []
for name, mk in MODELS.items():
    oof, tst = np.zeros(len(y)), np.zeros(len(test))
    for s in range(SEEDS):
        for a, b, Xa, Xb in design:
            oof[b] += mk(s).fit(Xa, y[a]).predict_proba(Xb)[:, 1] / SEEDS
        tst += mk(s).fit(X_full, y).predict_proba(X_test)[:, 1] / SEEDS
    OOF[name], TEST[name] = oof, tst
    rows.append(evaluate(name, y, oof))
""")

md(r"""
## 6. Character-level BiGRU

Reads "masked title || artist" in order, character by character. On its own it is much weaker
than the trees (AUC ≈ 0.88), but it makes *different* mistakes (correlation 0.76 with the trees),
which is what a stack needs. Same folds as above.
""")

code(r"""
dev = "cuda" if torch.cuda.is_available() else "cpu"
join = lambda d: (d.track.str.slice(0, 60) + " || " + d.artist.str.slice(0, 40)).tolist()
s_tr, s_te = join(train), join(test)
idx = {c: i + 1 for i, c in enumerate(sorted(set("".join(s_tr + s_te))))}
L = 104
def enc(ss):
    X = np.zeros((len(ss), L), dtype=np.int64)
    for i, s in enumerate(ss):
        v = [idx[c] for c in s][:L]; X[i, :len(v)] = v
    return torch.tensor(X)
S_tr, S_te = enc(s_tr), enc(s_te)
D_tr = torch.tensor(((train.decade.values - 1960) // 10).astype(np.int64))
D_te = torch.tensor(((test.decade.values - 1960) // 10).astype(np.int64))

class CharGRU(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.emb, self.dec = nn.Embedding(V + 1, 48, padding_idx=0), nn.Embedding(6, 8)
        self.gru = nn.GRU(48, 96, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Sequential(nn.Linear(4 * 96 + 8, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, x, d):
        m = (x > 0).unsqueeze(-1).float()
        h, _ = self.gru(self.drop(self.emb(x)))
        pooled = torch.cat([(h * m).sum(1) / m.sum(1).clamp(min=1), (h - 1e4 * (1 - m)).max(1).values], 1)
        return self.out(torch.cat([pooled, self.dec(d)], 1)).squeeze(-1)

def gru_fit_predict(Xa, Da, ya, Xb, Db, seed, epochs=14, bs=128):
    torch.manual_seed(seed)
    net = CharGRU(len(idx)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    n = len(ya)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * ((n + bs - 1) // bs))
    Xa, Da, ya = Xa.to(dev), Da.to(dev), torch.tensor(ya, dtype=torch.float32).to(dev)
    for _ in range(epochs):
        net.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            opt.zero_grad(); nn.functional.binary_cross_entropy_with_logits(net(Xa[j], Da[j]), ya[j]).backward()
            opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(Xb.to(dev), Db.to(dev))).cpu().numpy()

oof = np.zeros(len(y))
for s in range(GRU_SEEDS):
    for a, b in FOLDS:
        oof[b] += gru_fit_predict(S_tr[a], D_tr[a], y[a], S_tr[b], D_tr[b], seed=100 * s) / GRU_SEEDS
OOF["BiGRU"] = oof
TEST["BiGRU"] = np.mean([gru_fit_predict(S_tr, D_tr, y, S_te, D_te, seed=900 + s) for s in range(GRU_SEEDS)], 0)
rows.append(evaluate("BiGRU", y, oof))
""")

md(r"""
## 7. Stack and decision

Second level: logistic regression on logits of (mean of the three trees, BiGRU, the separate-space
text model), plus an indicator *some credited artist of the song is already in train* and its
interactions. For a known artist the stack leans on the trees' artist hit rates; for a new one it
gives more weight to the BiGRU. The stack itself is evaluated out of fold (10 folds × 5 repeats),
then refitted on all of train and applied to test. Decision: invest if P(hit) > 0.375.
""")

code(r"""
lg = lambda p: logit(np.clip(p, 1e-5, 1 - 1e-5))
text_oof, seen_oof = np.zeros(len(y)), np.zeros(len(y))
for a, b, Xa, Xb in design:
    text_oof[b], seen_oof[b] = Xb["text5_p"].values, Xb["cred_seen"].values
u, uT = (seen_oof > 0).astype(float), (X_test["cred_seen"].values > 0).astype(float)
trees, treesT = np.mean([lg(OOF[m]) for m in MODELS], 0), np.mean([lg(TEST[m]) for m in MODELS], 0)
g, gT, t, tT = lg(OOF["BiGRU"]), lg(TEST["BiGRU"]), lg(text_oof), lg(X_test["text5_p"].values)
S = np.column_stack([trees, g, t, u, trees * u, g * u, t * u])
ST = np.column_stack([treesT, gT, tT, uT, treesT * uT, gT * uT, tT * uT])

rows.append(evaluate("Trees (mean of logits)", y, 1 / (1 + np.exp(-trees))))
stack_oof = np.zeros(len(y))
for r in range(5):
    for a, b in folds(y, 10, seed=r):
        stack_oof[b] += LogisticRegression(max_iter=2000).fit(S[a], y[a]).predict_proba(S[b])[:, 1] / 5
rows.append(evaluate("Final stack", y, stack_oof))

stack = LogisticRegression(max_iter=2000).fit(S, y)
p_test = stack.predict_proba(ST)[:, 1]
decisions = pd.DataFrame({"id": test["id"], "p_hit": p_test.round(4), "invest": (p_test > BREAK_EVEN).astype(int)})
decisions.to_csv("decisions.csv", index=False)
print(f"\nInvest in {decisions.invest.sum()} of {len(decisions)} test songs -> decisions.csv")
print(f"CV profit {profit(y, stack_oof > BREAK_EVEN):.0f}M vs perfect foresight {GAIN_HIT * y.sum():.0f}M "
      f"and investing in everything {profit(y, np.ones(len(y))):.0f}M (10,000 train songs)")
""")

code(r"""
pd.DataFrame(rows).set_index("model").round(4)
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parent / f"{NAME}.ipynb"
nbf.write(nb, out)
print("wrote", out)
