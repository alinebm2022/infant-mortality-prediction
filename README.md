# Infant Mortality Prediction — Rio de Janeiro, Brazil

Reproducible machine-learning study to predict infant mortality from maternal, pregnancy, delivery and newborn characteristics using SINASC birth records linked to SIM death records.

## Research design

- Outcome: `OBITO_INFANTIL` (death within the first 365 days of life, according to the script's date calculation).
- Linkage methods: deterministic linkage and a probabilistic Fellegi–Sunter-style linkage implementation.
- Models: logistic regression (RLM), multilayer perceptron (MLP), and XGBoost.
- Six main combinations: 2 linkage methods × 3 models.
- Stratified holdout test set, with a validation split for threshold selection.
- Stratified 5-fold cross-validation on the training partition.
- Metrics: ROC-AUC, PR-AUC, accuracy, balanced accuracy, precision, recall/sensitivity, F1, specificity, confusion matrix and selected threshold.
- Logistic regression outputs coefficient estimates, odds ratios, 95% confidence intervals and Wald p-values.
- MLP includes permutation feature importance.

## Input data

Place these CSV files in a local data folder:

- `DNRJ2023.csv` — SINASC birth records
- `DOINF23.csv` — SIM infant death records for 2023
- `DOINF24.csv` — SIM infant death records for 2024

Edit `INPUT_DIR` in the Python script to point to the folder containing the files. The source data are not included in this repository due to privacy, licensing, and sensitive-data considerations.


## Environment setup

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python OBITO_INFANTIL_linkage_modelos.py
```


## Output

The script writes `OBITO_INFANTIL_linkage_modelos.xlsx` with sheets for holdout performance, cross-validation folds and summaries, logistic regression OR/CI, MLP permutation importance, and linkage counts.

## Important methodological notes

- The probabilistic linkage code is an exploratory Fellegi–Sunter-style implementation with date blocking and EM-estimated comparison parameters. It needs manual validation and sensitivity analyses of the matching threshold before being considered a definitive linkage method.
- The holdout test set is reserved for final performance evaluation. Threshold selection is done on the validation split; CV metrics use a fixed threshold of 0.50 to avoid selecting a threshold and evaluating it on the same fold.
- This is a research/portfolio project, not a clinical decision-support system.
- Keep sensitive source data outside Git. Do not commit datasets containing identifiable or sensitive information.
