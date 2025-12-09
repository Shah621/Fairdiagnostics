# Fair Diagnostics - JOA+Fair Algorithm

## Overview
This repository implements a **novel Adaptive Fair Information Gain (AFIG)** algorithm for fairness-aware heart disease prediction. The system compares three modeling approaches:

1. **Simple models** – Default hyperparameters, all features
2. **JOA-optimized models** – Hyperparameters tuned with Jellyfish Optimization Algorithm
3. **JOA+Fair models** – JOA-optimized hyperparameters + AFIG-selected features

The goal is to achieve high accuracy (~95% target) while maintaining fairness across sensitive attributes (Sex, Age).

## Project Structure
```
Fair diagnostics/
├─ src/
│  ├─ data_loader.py           # Load CSV data
│  ├─ preprocessing.py         # Model-specific preprocessing pipelines
│  ├─ models.py                # Model definitions (LR, NB, SVM, RF, NN)
│  ├─ optimization.py          # JOA hyperparameter optimization
│  ├─ fairness.py              # Novel AFIG algorithm
│  ├─ fairness_evaluation.py   # Comprehensive fairness metrics
│  └─ feature_engineering.py   # Clinical feature engineering
├─ evaluate_joa_fair.py         # Main evaluation script
├─ exp.py                       # Original experimental script (reference)
├─ joa_fair_results.csv         # Evaluation results
├─ fairness_comparison.png      # Fairness visualization
├─ requirements.txt             # Python dependencies
└─ README.md                    # This file
```

## Setup
```bash
# Clone the repo (if not already)
git clone <repo-url>
cd "Fair diagnostics"

# Install dependencies
pip install -r requirements.txt
```

## Running the Evaluation
```bash
python evaluate_joa_fair.py
```

The script will:
1. Load and engineer features from the heart disease dataset
2. Split data into train/test sets (80/20, stratified)
3. For each model (LR, NB, SVM, RF, NN):
   - Train **Simple** variant (default params, all features)
   - Train **JOA-Optimized** variant (optimized params, all features)
   - Train **JOA+Fair** variant (optimized params, AFIG-selected features)
4. Calculate accuracy metrics (accuracy, precision, recall, F1, AUC-ROC)
5. Calculate fairness metrics (demographic parity, equalized odds, disparate impact, etc.)
6. Save results to `joa_fair_results.csv` and generate `fairness_comparison.png`

## Novel AFIG Algorithm

### Methodology

**Adaptive Fair Information Gain (AFIG)** is a novel fairness-aware feature selection algorithm that improves upon traditional Fair Information Gain (FIG) through three key innovations:

**1. Dynamic Fairness Penalty (λ_adaptive)**
```
λ_adaptive = λ_base × (1 + disparity_factor)
```
Unlike static FIG, AFIG automatically increases the fairness penalty when higher group disparities are detected in the data, ensuring stronger fairness enforcement where needed.

**2. Calibrated Redundancy Penalty (β_calibrated)**
```
β_calibrated = β_base × correlation_factor
```
The redundancy penalty adapts based on feature correlation patterns, applying stronger penalties when features are highly correlated to avoid selecting redundant information.

**3. Normalized Multi-Objective Scoring**
```python
Score = (IG/max_IG) × w_feat - λ_adaptive × (MI_A/max_MI_A) - β_calibrated × redundancy
```
Where:
- `IG`: Information Gain with target variable
- `MI_A`: Mutual Information with sensitive attribute
- `redundancy`: Average MI with already-selected features

This normalization ensures stable optimization across different datasets and feature scales.

### Fairness Metrics

The system evaluates fairness using six standard metrics:
- **Demographic Parity**: Difference in positive prediction rates between groups
- **Disparate Impact**: Ratio of positive prediction rates (80% rule: ≥0.8 is fair)
- **Equal Opportunity**: Difference in True Positive Rates between groups
- **Equalized Odds**: Maximum difference in TPR and FPR between groups
- **Statistical Parity**: Same as demographic parity
- **Group-Specific Metrics**: Accuracy, precision, recall, F1 per group

## Results

### Best Performance

| Model | Variant | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|---------|----------|-----------|--------|----|---------|
| **SVM** | **JOA+Fair** | **85.25%** | **78.57%** | **100%** | **88.00%** | **90.64%** |
| **Random Forest** | **JOA+Fair** | **85.25%** | **80.00%** | **96.97%** | **87.67%** | **91.56%** |
| SVM | Simple | 80.33% | 75.61% | 93.94% | 83.78% | 85.61% |
| Random Forest | JOA-Opt | 80.33% | 75.61% | 93.94% | 83.78% | 88.53% |
| Neural Network | JOA+Fair | 80.33% | 74.42% | 96.97% | 84.21% | 88.10% |

### Fairness Metrics (Best Models)

| Model | Variant | Demographic Parity | Disparate Impact | Equalized Odds |
|-------|---------|-------------------|------------------|----------------|
| **SVM** | **JOA+Fair** | 0.2995 | 0.6653 | **0.0133** ✅ |
| **Random Forest** | **JOA+Fair** | 0.3471 | 0.6120 | 0.0588 |
| Logistic Regression | Simple | 0.1416 ✅ | **0.8078** ✅ | 0.1287 |
| Naive Bayes | All | **0.0075** ✅ | **0.9870** ✅ | 0.3162 |

**Key Findings:**
- ✅ **Best Accuracy**: 85.25% (SVM & RF with JOA+Fair)
- ✅ **Best Equalized Odds**: 0.0133 (SVM JOA+Fair) - nearly perfect equal treatment
- ✅ **Best Demographic Parity**: 0.0075 (Naive Bayes) - nearly equal outcomes
- ✅ **Meets 80% Rule**: Logistic Regression (0.808) and Naive Bayes (0.987)
- ⚠️ **Accuracy Target (95%)**: Not achieved - 10% below target

### Selected Features (AFIG)

Most commonly selected features across models:
1. **cp** (Chest Pain Type) - Strong predictor
2. **ca** (Number of major vessels) - Critical risk factor
3. **thal** (Thallium stress test) - Key diagnostic
4. **oldpeak** (ST depression) - ECG indicator
5. **slope** (ST segment slope) - Exercise test result
6. **exang** (Exercise-induced angina) - Symptom indicator

Engineered features also selected:
- `Age_MaxHR_Interaction`
- `Symptom_Complexity`
- `Oldpeak_Squared`
- `Vessel_Score`

## Extending the Experiment

### Completed
- ✅ Removed KNN and Decision Tree models
- ✅ Implemented novel AFIG algorithm with dynamic λ and calibrated β
- ✅ Added comprehensive fairness metrics (6 metrics)
- ✅ Exhaustive tuning of λ, β, and k parameters
- ✅ Evaluation with Sex as sensitive attribute
- ✅ Model-specific preprocessing (no scaling for RF/NB)

### Future Work
- [ ] Experiment with Age as sensitive attribute (binned into groups)
- [ ] Further hyperparameter tuning to approach 95% accuracy target
- [ ] Test on additional datasets (COMPAS, Adult Income, etc.)
- [ ] Compare with other fair ML libraries (FairLearn, AIF360)
- [ ] Implement post-processing fairness adjustments
- [ ] Ensemble methods combining SVM and RF predictions

## Citation

If you use this code or the AFIG algorithm, please cite:

```
@software{afig_fair_diagnostics,
  title={Adaptive Fair Information Gain for Fairness-Aware Feature Selection},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/fair-diagnostics}
}
```

---

**Note**: This implementation prioritizes fairness alongside accuracy. The 85.25% accuracy with excellent equalized odds (0.013) represents a valuable tradeoff for fair medical diagnosis applications.

---
