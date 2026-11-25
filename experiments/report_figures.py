"""Figures and numbers for the 2-page report, all from saved out-of-fold predictions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logit
from sklearn.metrics import roc_auc_score

from src.common import BREAK_EVEN, RESULTS, load, profit

OUT = Path(__file__).resolve().parents[1] / "report" / "fig"
OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID = "#1f2933", "#6b7280", "#e5e7eb"
BLUE, LIGHT, ORANGE = "#2563eb", "#93b4f5", "#d97706"
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})

tr, te = load()
y = tr.target.values
ld = lambda n: np.load(RESULTS / "oof" / f"{n}_oof.npy")
lg = lambda p: logit(np.clip(p, 1e-5, 1 - 1e-5))
final = ld("r13_final")

# ---------- Fig 1: how the CV AUC got built
steps = [("Audio only (LightGBM)", 0.8501, 4930),
         ("+ artist & title, no target", 0.9246, 5776),
         ("+ artist target enc. + text model", 0.9439, 6123),
         ("Honest CV: grouped + masked titles", 0.9449, 6170),
         ("Optuna, 3 boosting families", 0.9463, 6189),
         ("Split text spaces, 10 folds", 0.9474, 6234),
         ("+ char BiGRU in the stack", 0.9483, 6249),
         ("+ seen-artist interactions (final)", 0.9485, 6281)]
fig, ax = plt.subplots(figsize=(3.4, 2.0))
yy = np.arange(len(steps))[::-1]
for i, (lab, auc, pr) in zip(yy, steps):
    c = BLUE if "final" in lab else (MUTED if "Honest" in lab else LIGHT)
    ax.barh(i, auc - 0.84, left=0.84, height=0.62, color=c)
    ax.text(auc + 0.002, i, f"{auc:.4f} | {pr:,}", va="center", fontsize=6.2, color=INK)
ax.set_yticks(yy, [s[0] for s in steps], fontsize=6.8)
ax.set_xlim(0.84, 1.0); ax.set_xticks([0.85, 0.90, 0.95]); ax.set_xlabel("CV AUC  (label: AUC | CV profit, $M)")
ax.axvline(0.95, color=ORANGE, lw=0.8, ls="--"); 
ax.tick_params(axis="y", length=0); ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(pad=0.3); fig.savefig(OUT / "progression.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# ---------- Fig 2: profit vs threshold (a) and calibration (b)
ths = np.linspace(0.05, 0.95, 181)
prof = np.array([profit(y, final > t) for t in ths])
oracle, allin = 1.5 * y.sum(), profit(y, np.ones_like(y))
fig, (a, b) = plt.subplots(1, 2, figsize=(3.35, 1.75))
a.plot(ths, prof, color=BLUE, lw=1.4)
a.axhline(oracle, color=MUTED, lw=0.7, ls=":"); a.text(0.06, oracle + 120, f"perfect foresight {oracle:,.0f}", fontsize=5.8, color=MUTED)
a.axhline(allin, color=MUTED, lw=0.7, ls=":"); a.text(0.06, allin + 120, f"invest in all {allin:,.0f}", fontsize=5.8, color=MUTED)
a.axvline(BREAK_EVEN, color=ORANGE, lw=0.8, ls="--")
p375 = profit(y, final > BREAK_EVEN)
a.plot([BREAK_EVEN], [p375], "o", color=ORANGE, ms=3.5)
a.text(BREAK_EVEN + 0.03, p375 - 900, f"0.375 → {p375:,.0f}", fontsize=5.8, color=INK)
a.set_xlabel("Invest if P(hit) >"); a.set_ylabel("CV profit, $M"); a.set_ylim(2000, 7900)
a.set_title("(a) Profit vs threshold", fontsize=7, color=INK)
bins = np.linspace(0, 1, 11); idx = np.digitize(final, bins[1:-1])
mp = [final[idx == k].mean() for k in range(10)]; fr = [y[idx == k].mean() for k in range(10)]
b.plot([0, 1], [0, 1], color=GRID, lw=0.8)
b.plot(mp, fr, "o-", color=BLUE, ms=3, lw=1.1)
b.set_xlabel("Predicted P(hit)"); b.set_ylabel("Observed hit rate"); b.set_title("(b) Calibration", fontsize=7, color=INK)
for z in (a, b): z.grid(True, color=GRID, lw=0.5); z.set_axisbelow(True)
fig.tight_layout(pad=0.3, w_pad=0.8); fig.savefig(OUT / "profit_calibration.pdf"); plt.close(fig)

# ---------- numbers for the text
_, fs, _ = pickle.load(open(RESULTS / "cache" / "design_v5_s0_k10_t1.pkl", "rb"))
seen = np.zeros(len(y)); text = np.zeros(len(y))
for a_, b_, Xa, Xb in fs:
    seen[b_] = Xb["cred_seen"].values; text[b_] = Xb["text5_p"].values
s = seen > 0
trees = np.mean([lg(ld(f"r09_{m}_k10")) for m in ("lgbm", "cat", "xgb")], 0)
gru, hyb = lg(ld("r11_gru_k10")), lg(ld("r12_hyb_k5"))
print("seen share", s.mean().round(3), "AUC seen", round(roc_auc_score(y[s], final[s]), 4),
      "unseen", round(roc_auc_score(y[~s], final[~s]), 4))
print("corr trees-gru", np.corrcoef(trees, gru)[0, 1].round(3), "trees-hyb", np.corrcoef(trees, hyb)[0, 1].round(3),
      "trees-text", np.corrcoef(trees, lg(text))[0, 1].round(3))
print("text AUC", round(roc_auc_score(y, text), 4))
print("conf errors: P>.95 & flop", int(((final > .95) & (y == 0)).sum()), " P<.05 & hit", int(((final < .05) & (y == 1)).sum()))
print("profit share", (p375 / oracle).round(3), "invest share oof", (final > BREAK_EVEN).mean().round(3))
