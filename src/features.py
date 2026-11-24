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

FEAT_RE = re.compile(r"\b(feat\.?|featuring|ft\.?|with|x|vs\.?)\b|&|,|\band\b", re.I)
CLASSICAL_RE = re.compile(r"\b(op\.|bwv|k\.\s?\d|no\.\s?\d|sonata|concerto|symphon|allegro|adagio|andante|"
                          r"prelude|nocturne|etude|minor|major|act \w+|scene|aria|movement|mvt)\b", re.I)
VERSION_RE = re.compile(r"remaster|live|version|mix|edit|demo|acoustic|instrumental|mono|stereo|take", re.I)
ENSEMBLE_RE = re.compile(r"orchestra|quartet|trio|ensemble|choir|philharmonic|symphony|band|chorus|"
                         r"singers|brothers|sisters|boys|girls", re.I)


def _nonascii(s: str) -> float:
    return sum(ord(c) > 127 for c in s) / max(len(s), 1)


def unsupervised(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Features that do not use the target. Returns one frame for train+test (train first)."""
    al = pd.concat([train.drop(columns="target"), test], ignore_index=True)
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
    X["tr_feat"] = t.str.contains(r"\bfeat", case=False).astype(int)
    X["tr_classical"] = (t + " " + a).str.contains(CLASSICAL_RE).astype(int)
    X["tr_version"] = t.str.contains(VERSION_RE).astype(int)
    X["tr_upper"] = t.map(lambda s: sum(c.isupper() for c in s) / max(sum(c.isalpha() for c in s), 1))
    X["tr_colon"] = t.str.contains(":").astype(int)
    # Same title by several artists: covers and standards
    X["tr_title_n"] = t.str.lower().str.strip().map(t.str.lower().str.strip().value_counts())
    return X


# ---------------------------------------------------------------- target-based, fold-safe

def _smoothed_rate(keys_fit, y_fit, keys_apply, prior, k):
    g = pd.DataFrame({"k": keys_fit, "y": y_fit}).groupby("k").y.agg(["sum", "count"])
    rate = (g["sum"] + prior * k) / (g["count"] + k)
    s = pd.Series(keys_apply)
    return s.map(rate).fillna(prior).values, s.map(g["count"]).fillna(0).values


def target_encode(keys_fit, y_fit, keys_apply, k=5, inner=5, seed=0):
    """Smoothed target encoding: out-of-fold on the fit set, full fit on the apply set."""
    keys_fit = np.asarray(keys_fit)
    prior = y_fit.mean()
    fit_enc = np.zeros(len(keys_fit))
    fit_cnt = np.zeros(len(keys_fit))
    for a, b in StratifiedKFold(inner, shuffle=True, random_state=seed).split(keys_fit, y_fit):
        fit_enc[b], fit_cnt[b] = _smoothed_rate(keys_fit[a], y_fit[a], keys_fit[b], prior, k)
    app_enc, app_cnt = _smoothed_rate(keys_fit, y_fit, keys_apply, prior, k)
    return fit_enc, fit_cnt, app_enc, app_cnt


def _text(df_tracks, df_artists):
    return (df_tracks.str.lower() + " || " + df_artists.str.lower()).values


def text_model(tr_track, tr_artist, y_fit, ap_track, ap_artist, inner=5, seed=0, C=2.0):
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
    for a, b in StratifiedKFold(inner, shuffle=True, random_state=seed).split(txt_fit, y_fit):
        fit_p[b] = fit_predict(txt_fit[a], y_fit[a], txt_fit[b])
    return fit_p, fit_predict(txt_fit, y_fit, txt_app)


def supervised(fit_df: pd.DataFrame, y_fit: np.ndarray, app_df: pd.DataFrame, seed=0, text=True):
    """Target-based features for a fit set (OOF) and an apply set."""
    F, A = pd.DataFrame(index=range(len(fit_df))), pd.DataFrame(index=range(len(app_df)))
    F["art_te"], F["art_seen"], A["art_te"], A["art_seen"] = target_encode(
        fit_df.artist.values, y_fit, app_df.artist.values, seed=seed)
    # Artist x decade: an artist can be a star in one decade and forgotten in the next
    kf = (fit_df.artist + "|" + fit_df.decade.astype(str)).values
    ka = (app_df.artist + "|" + app_df.decade.astype(str)).values
    F["artdec_te"], _, A["artdec_te"], _ = target_encode(kf, y_fit, ka, seed=seed)
    if text:
        F["text_p"], A["text_p"] = text_model(fit_df.track, fit_df.artist, y_fit,
                                              app_df.track, app_df.artist, seed=seed)
    return F, A
