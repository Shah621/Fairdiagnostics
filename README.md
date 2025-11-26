# Fair Diagnostics Experiment

## Overview
This repository contains a **heart disease prediction** experiment that compares three modeling approaches:

1. **Simple models** – default hyper‑parameters.
2. **JOA‑optimized models** – hyper‑parameters tuned with the Jellyfish Optimization Algorithm.
3. **Fair models** – feature selection based on **Fair Information Gain (MI with a sensitive attribute)**, extended with redundancy (mRMR) and optional feature engineering.

The goal is to evaluate how fairness‑aware feature selection impacts accuracy compared to the baseline and JOA‑optimized pipelines.

## Project Structure
```
Fair diagnostics/
├─ src/
│  ├─ data_loader.py      # Load CSV data
│  ├─ preprocessing.py    # ColumnTransformer pipelines
│  ├─ models.py           # Model definitions (no XGBoost)
│  ├─ optimization.py     # JOA hyper‑parameter search
│  ├─ fairness.py         # FairFeatureSelector (FIG + redundancy)
│  ├─ feature_engineering.py # Targeted engineered features
│  ├─ train.py            # Train simple, JOA‑optimized, and fair models
│  └─ evaluate.py         # Evaluate and print results
├─ main.py                # Orchestrates the whole workflow
├─ exp.py                 # Original experimental script (reference)
├─ requirements.txt       # Python dependencies
└─ README.md              # This file
```

## Setup
```bash
# Clone the repo (if not already)
git clone <repo-url>
cd "Fair diagnostics"

# Install dependencies
pip install -r requirements.txt
```

## Running the Experiment
```bash
python main.py
```
The script will:
1. Load and clean the heart disease CSV.
2. Split the data into train/test.
3. Train each model in three variants (simple, JOA‑optimized, fair).
4. Print accuracy for every variant and save a `model_results.csv` file.

## Fair Feature Selection Details
- **Fair Information Gain (FIG)**: `IG - λ·I(feature; sensitive)`.
- **Redundancy penalty** (mRMR style): `- β·Avg(I(feature; selected))`.
- Hyper‑parameters (`λ`, `β`, `k`) are searched on a validation split.
- Optional engineered features (age‑HR interaction, symptom complexity, etc.) are added via `src/feature_engineering.py` before selection.

## Results (example snapshot)
| Model | Simple Acc. | JOA Optimized Acc. | Fair (FIG) Acc. |
|-------|------------|--------------------|-----------------|
| Logistic Regression | 85.25% | 85.25% | 86.89% |
| Naive Bayes | 86.89% | 86.89% | 85.25% |
| SVM | 86.89% | 88.52% | 88.52% |
| KNN | 91.80% | 86.89% | 83.61% |
| Decision Tree | 75.41% | 85.25% | 81.97% |
| Random Forest | 83.61% | 86.89% | 88.52% |
| Neural Network | 86.89% | 86.89% | 86.89% |

*Fair models improve or maintain accuracy for most algorithms while explicitly accounting for the sensitive attribute (`Sex`).*

## Extending the Experiment
- Tune `λ`, `β`, and `k` more exhaustively.
- Add additional fairness metrics (e.g., demographic parity, equalized odds).
- Experiment with other sensitive attributes (e.g., `Age`).

---
*Happy experimenting!*
