import sys, time
import numpy as np
mode = sys.argv[1]
if 'torch' in mode:
    import torch
import lightgbm as lgb, catboost as cb
X = np.random.rand(9000, 87); y = (X[:, 0] + np.random.rand(9000) > 1).astype(int)
nj = -1 if 'all' in mode else 20
t = time.time(); lgb.LGBMClassifier(n_estimators=545, num_leaves=30, colsample_bytree=0.27, verbose=-1, n_jobs=nj).fit(X, y); a = time.time() - t
t = time.time(); cb.CatBoostClassifier(iterations=968, depth=7, verbose=0, thread_count=nj, allow_writing_files=False).fit(X, y); b = time.time() - t
print(mode, 'lgbm', round(a, 1), 'cat', round(b, 1))
