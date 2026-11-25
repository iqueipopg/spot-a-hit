# Spot-a-hit: which songs should a label back?

Individual assignment for *Predictive Machine Learning in Finance* (IESE Master in Finance, November 2026). Given the
audio features, artist and title of 2,000 unreleased songs, decide which ones to invest in: a hit returns +1.5M and a
flop costs 0.9M, so the rule is to invest whenever P(hit) > 0.375. Training data: 10,000 labelled songs. No external
data allowed.

**Short answer: out-of-fold AUC 0.9485, accuracy 0.904 and 6,281M of profit on the 10,000 train songs, 86.6% of what
perfect foresight would earn.** Most of the gain came from reading the data before modelling it: the test titles are
masked (eleven lowercase letters replaced by `*`), the artist almost decides the label (83.5% of artists with two or
more songs are all hits or all flops), and the `id` only encodes the decade. The final model is a logistic stack of three tuned gradient-boosting families, a
character-level BiGRU and a TF-IDF text model. The two-page write-up is in the [report](report/QueipoDeLlano_Ignacio.pdf).

![How the CV AUC was built](report/fig/progression.png)

## Results (10-fold CV grouped by title, train titles masked like test)

| Run | What changed | AUC | Accuracy | Profit |
|---|---|---|---|---|
| 01 | LightGBM, audio only | 0.850 | 0.775 | 4,930M |
| 01 | + artist and title features, no target | 0.925 | 0.856 | 5,776M |
| 01 | + artist target encoding + text model | 0.944 | 0.893 | 6,123M |
| 07 | Honest baseline: grouped CV, masked train titles | 0.945 | 0.896 | 6,170M |
| 08 | Optuna (log-loss), LightGBM + CatBoost | 0.946 | 0.896 | 6,189M |
| 10 | Title and artist in separate text spaces, 10 folds | 0.947 | 0.901 | 6,234M |
| 13 | Stack: trees + char BiGRU | 0.948 | 0.903 | 6,249M |
| **13** | **+ text model and seen-artist interactions (submitted)** | **0.9485** | **0.904** | **6,281M** |

Investing in every song earns 2,602M; perfect foresight 7,251M. On test the model invests in 1,090 of 2,000 songs.

- **0.95 was not reached, and I think that is the ceiling.** From run 10 to 14 five different ideas land between
  0.947 and 0.9486, while the bootstrap standard error of the final AUC is 0.0022.
- **What is left needs outside knowledge.** 21 hits get P < 0.05 (a Christmas song by Los del Río, PPAP) and 15 flops
  get P > 0.95. A pretrained classifier with no training scores 0.918 from artist names alone, which is exactly the
  knowledge the rules exclude.
- The full reasoning, run by run and model by model, is in the [experiment diary](docs/DIARIO.md) (Spanish).

## Repository

```
deliverable/   the submitted notebook (CRISP-DM order, self-contained, runs in 45 min) and its decisions.csv
               build_notebook.py generates the notebook from code, so it never drifts from the experiments
src/           shared code: loading and title masking, features, CV pipeline, the BiGRU
experiments/   one script per run (run10 = run09 with K=10), plus the report figures
results/       runs.csv (every CV score), best_params.json (Optuna), decisions.csv
report/        LaTeX source, figures and PDF of the two-page report
docs/          DIARIO.md, the experiment diary
```

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# put train.xlsx and test.xlsx in data/ (course material, not included)
python experiments/run13_final_stack.py   # needs the OOF files of runs 09 and 11 first
```

Or open `deliverable/QueipoDeLlano_Ignacio.ipynb` with the two Excel files next to it and run all cells. The BiGRU uses
a CUDA GPU if there is one. Use all cores but two: with `n_jobs=-1` LightGBM and CatBoost stalled on a 22-thread WSL
machine.

Course materials and data are not included.
