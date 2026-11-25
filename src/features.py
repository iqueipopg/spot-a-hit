"""Feature engineering.

Two kinds of features:
* unsupervised ones, built on train + test together (they never look at the target, so using the
  test songs' own features is legitimate: it is the closed box we are given);
* target-based ones (artist hit rate, title text model), always computed out of fold so that a
  song never sees its own label.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import hstack

AUDIO = ["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness",
         "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "time_signature",
         "chorus_hit", "sections"]
CONTINUOUS = ["danceability", "energy", "loudness", "speechiness", "acousticness",
              "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "chorus_hit", "sections"]

from .common import mask_text


def masked_re(words, flags=re.I):
    """Keyword regex that matches the masked spelling ('remaster' -> 'r*****t*r')."""
    return re.compile("|".join(re.escape(mask_text(w)) for w in words), flags)


FEAT_RE = re.compile(r"\b(feat\.?|featuring|ft\.?|with|x|vs\.?)\b|&|,|\band\b", re.I)
CLASSICAL_RE = re.compile(r"\b(op\.|bwv|k\.\s?\d|no\.\s?\d|sonata|concerto|symphon|allegro|adagio|andante|"
                          r"prelude|nocturne|etude|minor|major|act \w+|scene|aria|movement|mvt)\b", re.I)
VERSION_RE = re.compile(r"remaster|live|version|mix|edit|demo|acoustic|instrumental|mono|stereo|take", re.I)
ENSEMBLE_RE = re.compile(r"orchestra|quartet|trio|ensemble|choir|philharmonic|symphony|band|chorus|"
                         r"singers|brothers|sisters|boys|girls", re.I)


TR_FEAT_M = masked_re(["feat"])
CLASSICAL_M = masked_re(["op.", "bwv", "no. ", "sonata", "concerto", "symphon", "allegro", "adagio", "andante",
                         "prelude", "nocturne", "etude", "minor", "major", "aria", "movement"])
VERSION_M = masked_re(["remaster", "live", "version", "mix", "edit", "demo", "acoustic", "instrumental",
                       "mono", "stereo", "take"])


def _nonascii(s: str) -> float:
    return sum(ord(c) > 127 for c in s) / max(len(s), 1)


def unsupervised(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Features that do not use the target. Returns one frame for train+test (train first)."""
    al = pd.concat([train.drop(columns=["target", "track_raw"], errors="ignore"), test], ignore_index=True)
    X = al[["decade"] + AUDIO].copy()

    # Song structure
    dur_s = al.duration_ms / 1000
    X["log_dur"] = np.log(dur_s)
    X["chorus_frac"] = al.chorus_hit / dur_s
    X["sec_per_min"] = al.sections / (dur_s / 60)
    X["dance_x_energy"] = al.danceability * al.energy
    X["acoustic_minus_energy"] = al.acousticness - al.energy
    X["vocal"] = (al.instrumentalness < 0.5).astype(int)

    # Each audio feature relative to its decade: a 1965 hit does not sound like a 2015 hit
    for c in CONTINUOUS:
        g = al.groupby("decade")[c]
        X[f"{c}_zdec"] = (al[c] - g.transform("mean")) / g.transform("std")

    # Artist: how prolific, how long the career, how their catalogue sounds
    art = al.groupby("artist")
    X["art_n"] = art.artist.transform("size")
    X["art_n_dec"] = art.decade.transform("nunique")
    X["art_first_dec"] = art.decade.transform("min")
    X["art_span"] = art.decade.transform("max") - X["art_first_dec"]
    for c in ["danceability", "energy", "acousticness", "instrumentalness", "speechiness", "valence", "loudness"]:
        X[f"art_mean_{c}"] = art[c].transform("mean")
        X[f"dev_art_{c}"] = al[c] - X[f"art_mean_{c}"]
    X["art_std_energy"] = art.energy.transform("std").fillna(0)

    # Artist name
    a = al.artist
    X["art_multi"] = a.str.contains(FEAT_RE).astype(int)
    X["art_ensemble"] = a.str.contains(ENSEMBLE_RE).astype(int)
    X["art_the"] = a.str.lower().str.startswith("the ").astype(int)
    X["art_len"] = a.str.len()
    X["art_words"] = a.str.split().str.len()
    X["art_nonascii"] = a.map(_nonascii)

    # Track title
    t = al.track
    X["tr_len"] = t.str.len()
    X["tr_words"] = t.str.split().str.len()
    X["tr_nonascii"] = t.map(_nonascii)
    X["tr_paren"] = t.str.contains(r"[\(\[]").astype(int)
    X["tr_dash"] = t.str.contains(r" - ").astype(int)
    X["tr_digits"] = t.str.count(r"\d")
    # Keyword flags look for the masked spelling, which is what both train and test now show
    X["tr_feat"] = t.str.contains(TR_FEAT_M).astype(int)
    X["tr_classical"] = (t.str.contains(CLASSICAL_M) | a.str.contains(CLASSICAL_RE)).astype(int)
    X["tr_version"] = t.str.contains(VERSION_M).astype(int)
    X["tr_star_share"] = t.str.count(r"\*") / t.str.len().clip(lower=1)
    X["tr_upper"] = t.map(lambda s: sum(c.isupper() for c in s) / max(sum(c.isalpha() for c in s), 1))
    X["tr_colon"] = t.str.contains(":").astype(int)
    # Same title by several artists: covers and standards
    X["tr_title_n"] = t.str.lower().str.strip().map(t.str.lower().str.strip().value_counts())
    return X


# ---------------------------------------------------------------- target-based, fold-safe

def _inner(n_splits, seed, X, y, groups):
    from sklearn.model_selection import StratifiedGroupKFold
    if groups is None:
        return StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y)
    return StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, groups)


def _smoothed_rate(keys_fit, y_fit, keys_apply, prior, k):
    g = pd.DataFrame({"k": keys_fit, "y": y_fit}).groupby("k").y.agg(["sum", "count"])
    rate = (g["sum"] + prior * k) / (g["count"] + k)
    s = pd.Series(keys_apply)
    return s.map(rate).fillna(prior).values, s.map(g["count"]).fillna(0).values


def target_encode(keys_fit, y_fit, keys_apply, k=5, inner=5, seed=0, groups=None):
    """Smoothed target encoding: out-of-fold on the fit set, full fit on the apply set."""
    keys_fit = np.asarray(keys_fit)
    prior = y_fit.mean()
    fit_enc = np.zeros(len(keys_fit))
    fit_cnt = np.zeros(len(keys_fit))
    for a, b in _inner(inner, seed, keys_fit, y_fit, groups):
        fit_enc[b], fit_cnt[b] = _smoothed_rate(keys_fit[a], y_fit[a], keys_fit[b], prior, k)
    app_enc, app_cnt = _smoothed_rate(keys_fit, y_fit, keys_apply, prior, k)
    return fit_enc, fit_cnt, app_enc, app_cnt


def _text(df_tracks, df_artists):
    return (df_tracks.str.lower() + " || " + df_artists.str.lower()).values


def text_model(tr_track, tr_artist, y_fit, ap_track, ap_artist, inner=5, seed=0, C=2.0, groups=None):
    """Char + word TF-IDF logistic regression on title and artist: OOF on fit, full fit on apply."""
    def vec():
        return (TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
                TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True))

    def fit_predict(txt_a, ya, txt_b):
        v1, v2 = vec()
        Xa = hstack([v1.fit_transform(txt_a), v2.fit_transform(txt_a)]).tocsr()
        Xb = hstack([v1.transform(txt_b), v2.transform(txt_b)]).tocsr()
        m = LogisticRegression(C=C, max_iter=3000).fit(Xa, ya)
        return m.predict_proba(Xb)[:, 1]

    txt_fit = _text(tr_track, tr_artist)
    txt_app = _text(ap_track, ap_artist)
    fit_p = np.zeros(len(txt_fit))
    for a, b in _inner(inner, seed, txt_fit, y_fit, groups):
        fit_p[b] = fit_predict(txt_fit[a], y_fit[a], txt_fit[b])
    return fit_p, fit_predict(txt_fit, y_fit, txt_app)


def text_model_v5(tr_track, tr_artist, y_fit, ap_track, ap_artist, inner=5, seed=0, C=3.0, groups=None):
    """Title and artist in separate vector spaces (char 1-5 each, plus whole-token words).
    Masked titles and clean artist names are different alphabets; mixing them in one vocabulary
    blurred both. CV AUC of the text model alone: 0.883 -> 0.886."""
    def vecs():
        return [(TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), min_df=2, sublinear_tf=True), 0),
                (TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), min_df=2, sublinear_tf=True), 1),
                (TfidfVectorizer(token_pattern=r"[^ ]+", ngram_range=(1, 3), min_df=1, sublinear_tf=True), 1),
                (TfidfVectorizer(token_pattern=r"[^ ]+", min_df=2, sublinear_tf=True), 0)]

    def fit_predict(ta, aa, ya, tb, ab):
        vs = vecs()
        Xa = hstack([v.fit_transform((ta, aa)[s]) for v, s in vs]).tocsr()
        Xb = hstack([v.transform((tb, ab)[s]) for v, s in vs]).tocsr()
        return LogisticRegression(C=C, max_iter=4000).fit(Xa, ya).predict_proba(Xb)[:, 1]

    tf, af = tr_track.str.lower().values, tr_artist.str.lower().values
    tp, ap = ap_track.str.lower().values, ap_artist.str.lower().values
    out = np.zeros(len(tf))
    for a, b in _inner(inner, seed, tf, y_fit, groups):
        out[b] = fit_predict(tf[a], af[a], y_fit[a], tf[b], af[b])
    return out, fit_predict(tf, af, y_fit, tp, ap)


def supervised(fit_df: pd.DataFrame, y_fit: np.ndarray, app_df: pd.DataFrame, seed=0, text=True,
               version=1):
    """Target-based features for a fit set (OOF) and an apply set."""
    from .common import title_key
    G = fit_df.track_raw.map(title_key).values if version >= 3 else None
    F, A = pd.DataFrame(index=range(len(fit_df))), pd.DataFrame(index=range(len(app_df)))
    F["art_te"], F["art_seen"], A["art_te"], A["art_seen"] = target_encode(
        fit_df.artist.values, y_fit, app_df.artist.values, seed=seed, groups=G)
    # Artist x decade: an artist can be a star in one decade and forgotten in the next
    kf = (fit_df.artist + "|" + fit_df.decade.astype(str)).values
    ka = (app_df.artist + "|" + app_df.decade.astype(str)).values
    F["artdec_te"], _, A["artdec_te"], _ = target_encode(kf, y_fit, ka, seed=seed, groups=G)
    if text:
        F["text_p"], A["text_p"] = text_model(fit_df.track, fit_df.artist, y_fit,
                                              app_df.track, app_df.artist, seed=seed, groups=G)
    if version >= 2:
        # Raw evidence instead of a smoothed rate: 3 hits of 3 is not the same as 1 of 1
        F["art_te_k1"], _, A["art_te_k1"], _ = target_encode(fit_df.artist.values, y_fit,
                                                             app_df.artist.values, k=1, seed=seed, groups=G)
        hf, _, ha, _ = target_encode(fit_df.artist.values, y_fit, app_df.artist.values, k=1e-9, seed=seed, groups=G)
        F["art_hits"], A["art_hits"] = hf * F["art_seen"], ha * A["art_seen"]
        F["art_flops"], A["art_flops"] = F["art_seen"] - F["art_hits"], A["art_seen"] - A["art_hits"]
        # Neighbours in audio space: hit rate of the songs that sound most alike
        F["knn_p"], A["knn_p"] = knn_feature(fit_df, y_fit, app_df, seed=seed, groups=G)
        # Artist-name text alone (orchestra, DJ, Lil...), useful for artists never seen
        blank_f = pd.Series("", index=fit_df.index)
        blank_a = pd.Series("", index=app_df.index)
        F["artname_p"], A["artname_p"] = text_model(blank_f, fit_df.artist, y_fit,
                                                    blank_a, app_df.artist, seed=seed, groups=G)
    if version >= 5:
        F["text5_p"], A["text5_p"] = text_model_v5(fit_df.track, fit_df.artist, y_fit,
                                                   app_df.track, app_df.artist, seed=seed, groups=G)
    if version >= 3:
        # "Kid Ink Featuring Chris Brown" -> primary "kid ink"; credited {"kid ink", "chris brown"}
        pf, pa = fit_df.artist.map(primary_artist).values, app_df.artist.map(primary_artist).values
        F["prim_te"], F["prim_seen"], A["prim_te"], A["prim_seen"] = target_encode(
            pf, y_fit, pa, k=2, seed=seed, groups=G)
        F["cred_max"], F["cred_mean"], F["cred_seen"], A["cred_max"], A["cred_mean"], A["cred_seen"] =             credited_encode(fit_df.artist.values, y_fit, app_df.artist.values, seed=seed, groups=G)
    return F, A


SPLIT_RE = re.compile(r"\s+(?:featuring|feat\.?|ft\.?|with|x|vs\.?|and|duet with)\s+|\s*[,&/]\s*", re.I)


def credited(artist: str) -> list[str]:
    return [p.strip().lower() for p in SPLIT_RE.split(artist) if p and p.strip()] or [artist.lower()]


def primary_artist(artist: str) -> str:
    return credited(artist)[0]


def credited_encode(art_fit, y_fit, art_apply, k=2, inner=5, seed=0, groups=None):
    """Hit rate of every credited artist; summarised by max, mean and how many were seen."""
    def enc(af, yf, aa):
        rows = [(i, p) for i, a in enumerate(af) for p in credited(a)]
        d = pd.DataFrame(rows, columns=["i", "p"]); d["y"] = yf[d.i]
        g = d.groupby("p").y.agg(["sum", "count"]); prior = yf.mean()
        rate = (g["sum"] + prior * k) / (g["count"] + k)
        out = np.zeros((len(aa), 3))
        for j, a in enumerate(aa):
            ps = [p for p in credited(a) if p in rate.index]
            out[j] = (max(rate[p] for p in ps), np.mean([rate[p] for p in ps]), len(ps)) if ps else (prior, prior, 0)
        return out
    art_fit = np.asarray(art_fit)
    fo = np.zeros((len(art_fit), 3))
    for a, b in _inner(inner, seed, art_fit, y_fit, groups):
        fo[b] = enc(art_fit[a], y_fit[a], art_fit[b])
    ao = enc(art_fit, y_fit, np.asarray(art_apply))
    return fo[:, 0], fo[:, 1], fo[:, 2], ao[:, 0], ao[:, 1], ao[:, 2]


KNN_COLS = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness",
            "liveness", "valence", "tempo", "duration_ms", "decade"]


def knn_feature(fit_df, y_fit, app_df, k=50, inner=5, seed=0, groups=None):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer

    def fp(Xa, ya, Xb):
        m = make_pipeline(QuantileTransformer(n_quantiles=200), KNeighborsClassifier(k, weights="distance"))
        return m.fit(Xa, ya).predict_proba(Xb)[:, 1]

    Xf, Xa = fit_df[KNN_COLS].values, app_df[KNN_COLS].values
    out = np.zeros(len(Xf))
    for a, b in _inner(inner, seed, Xf, y_fit, groups):
        out[b] = fp(Xf[a], y_fit[a], Xf[b])
    return out, fp(Xf, y_fit, Xa)
