# Recovery Readiness Dataset Preparation

This project prepares CSV files for a future XGBoost Recovery Readiness
classification model. It reads `Fully_sorted2.xlsx`, joins member, exercise,
and health records by `Entity Number`, cleans the merged data, engineers MVP
features, creates a rule-based readiness target, and exports train,
validation, and test datasets.

The generated `readiness_label` target is produced from transparent MVP rules.
It is not clinician-validated ground truth.

## Source Workbook

Place `Fully_sorted2.xlsx` in the project root. The script expects these sheets:

- `Personal Information`
- `Exercise Data`
- `Health Data`

The script uses `Entity Number` to join the sheets. One output row represents
one exercise record joined to the matching member and health data.

## Windows Setup

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python prepare_dataset.py
python train_model.py
python predict_example.py
```

## Outputs

The script creates `prepared_data` and writes:

- `master_dataset_with_identifiers.csv`
- `model_dataset.csv`
- `train.csv`
- `validation.csv`
- `test.csv`

Identifier, name, raw date, and display-only fields are retained in the master
export for auditing where available, but excluded from model feature exports.
Splits are made by unique `Entity Number` with a 70% / 15% / 15% train,
validation, and test split using random state 42.

## Readiness Labels

Labels are encoded as:

- `0` = `REDUCE`
- `1` = `MAINTAIN`
- `2` = `PROGRESS`

`REDUCE` is assigned when strong safety signals are present, such as no
clinician clearance, contraindication flags, high pain, acute recovery stage,
low sleep, or very high RPE.

`PROGRESS` is assigned only when all available positive conditions pass, such
as clinician clearance, no contraindication, low pain, stable recovery stage,
adequate sleep, completed workout, acceptable RPE, low/moderate VO2 risk, and
HRV of at least 35 where available.

All other records are labeled `MAINTAIN`.

## Model Training

`train_model.py` trains two multiclass classifiers:

- `LogisticRegression` baseline with balanced class weights
- `XGBClassifier` main model with class-aware sample weights

Both models use the same saved preprocessing approach:

- numeric columns: median imputation, then scaling
- categorical columns: most-frequent imputation, then one-hot encoding with
  unknown categories ignored

The script compares models on the validation dataset primarily using macro F1,
then REDUCE recall, then PROGRESS recall. REDUCE recall is treated as the most
safety-critical class-level metric. The selected pipeline is evaluated once on
the test dataset.

The training script validates that model inputs do not contain identifier
columns, names, raw date columns, `readiness_text`, or `progress_score`.

## Model Outputs

`train_model.py` creates `model_outputs` and writes:

- `recovery_readiness_pipeline.joblib`
- `metrics.json`
- `classification_report.csv`
- `test_predictions.csv`
- `confusion_matrix.png`
- `model_comparison.csv`
- `feature_importance.csv`
- `feature_importance.png`

`predict_example.py` loads the saved pipeline, reads one row from
`prepared_data/test.csv`, removes `readiness_label`, predicts the readiness
class, prints all three class probabilities, and prints the actual test label
for comparison.

## Longitudinal Synthetic Model

The longitudinal workflow uses `synthetic_data/synthetic_weekly_sessions.csv`
to predict the next weekly session's readiness label:

```powershell
python prepare_longitudinal_dataset.py
python train_longitudinal_model.py
python predict_longitudinal_example.py
```

This version creates `future_readiness_label` from next-session outcomes and
removes all `next_` fields, target-helper scores, change columns, identifiers,
names, record IDs, and raw dates from model inputs. Splits are still made by
unique `Entity Number`.

Longitudinal outputs are written to:

- `longitudinal_data`
- `longitudinal_model_outputs`

This model is trained on synthetic longitudinal data and is not clinically
validated.

## Model 2: Four-Week VO2 Max Forecast

Model 2 is a regression workflow that predicts a member's `VO2 Max Estimate`
four weekly sessions into the future. It uses the synthetic longitudinal weekly
session file at `synthetic_data/synthetic_weekly_sessions.csv`.

The target column is `future_vo2_4_weeks`. It is created by sorting each
member's weekly sessions by `Entity Number`, `Record Date`, and
`Synthetic Session Number`, then applying:

```python
groupby("Entity Number")["VO2 Max Estimate"].shift(-4)
```

The final four sessions for each member do not have a four-session-ahead VO2
value, so they are removed from the model dataset. The audit dataset keeps
`future_record_date`, `future_session_number`, and `vo2_change_4_weeks` for
review, but these fields are not used as model inputs.

Model 2 differs from Model 1 in three main ways:

- Model 1 predicts the current-row rule-based recovery readiness class.
- The longitudinal readiness model predicts a next-session readiness class.
- Model 2 predicts a continuous future VO2 Max value four weekly sessions
  ahead.

Current `VO2 Max Estimate` is intentionally preserved as an input feature.
Names, identifiers, record IDs, raw dates, future-session fields, and
`vo2_change_4_weeks` are excluded from model features. Splits are made by
unique `Entity Number` using a 70% / 15% / 15% train, validation, and test
split with random state 42.

Run the Model 2 workflow with:

```powershell
python prepare_vo2_dataset.py
python train_vo2_model.py
python predict_vo2_example.py
```

Model 2 writes prepared data to `vo2_data`:

- `vo2_audit_dataset.csv`
- `model_dataset.csv`
- `train.csv`
- `validation.csv`
- `test.csv`
- `feature_list.json`
- `preparation_summary.json`

Model 2 writes trained model outputs to `vo2_model_outputs`:

- `vo2_forecast_pipeline.joblib`
- `metrics.json`
- `model_comparison.csv`
- `test_predictions.csv`
- `feature_importance.csv`
- `feature_importance.png`
- `actual_vs_predicted.png`
- `residual_distribution.png`

Regression metrics are interpreted as follows:

- MAE is the average absolute prediction error in VO2 Max units.
- RMSE is the square root of mean squared error and penalizes larger misses
  more heavily than MAE.
- R2 describes how much target variance the model explains relative to a mean
  prediction baseline.

The training script also compares the selected model with a practical
persistence baseline that predicts the future VO2 Max will equal the current
`VO2 Max Estimate`.

Model 2 is trained on synthetic longitudinal data and is not clinically
validated. Its outputs must not be treated as medical advice or clinical
evidence.
