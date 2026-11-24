"""Run 01: LightGBM with growing feature sets, to see where the signal is."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb

from src.common import report, save_oof
from src.features import AUDIO
from src.pipeline import design, run_cv

y, fs, (Xfull, Xtest) = design(seed=0)
allc = list(Xfull.columns)
sup = ["art_te", "art_seen", "artdec_te", "text_p"]
unsup = [c for c in allc if c not in sup]

def lgbm():
    return lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                              verbose=-1, random_state=0)

sets = {
    "audio": ["decade"] + AUDIO,
    "unsup": unsup,
    "unsup+artist_te": unsup + ["art_te", "art_seen", "artdec_te"],
    "unsup+text": unsup + ["text_p"],
    "all": allc,
}
for name, cols in sets.items():
    y, oof, tp = run_cv(lgbm, cols, predict_test=(name == "all"))
    report(f"r01 lgbm {name}", y, oof, f"{len(cols)} features")
    if name == "all":
        save_oof("r01_lgbm_all", oof, tp)
