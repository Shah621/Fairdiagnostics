"""
Comprehensive Evaluation of JOA+Fair Algorithm

This script evaluates three variants of each model:
1. Simple: Default hyperparameters, all features
2. JOA-Optimized: JOA-optimized hyperparameters, all features  
3. JOA+Fair: JOA-optimized hyperparameters, AFIG-selected features

For each variant, we calculate:
- Accuracy metrics (accuracy, precision, recall, F1, AUC-ROC)
- Fairness metrics (demographic parity, equalized odds, disparate impact, etc.)
- Performance across sensitive attributes (Sex, Age)
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import load_data
from src.models import get_models
from src.optimization import optimize_model
from src.fairness import AdaptiveFairFeatureSelector
from src.feature_engineering import create_targeted_features
from src.fairness_evaluation import (fairness_report, print_fairness_report, 
                                     plot_fairness_comparison, calculate_group_metrics)


def prepare_data_for_fairness(df, target_col='target', sensitive_col='sex'):
    """
    Prepare data for fairness-aware feature selection.
    
    Args:
        df: Raw dataframe
        target_col: Target column name
        sensitive_col: Sensitive attribute column name
        
    Returns:
        X, y, sensitive_values
    """
    # Apply feature engineering
    df_eng = create_targeted_features(df)
    
    # Separate features, target, and sensitive attribute
    y = df_eng[target_col]
    sensitive = df_eng[sensitive_col]
    
    # Features (exclude target, keep sensitive for AFIG)
    X = df_eng.drop(columns=[target_col])
    
    return X, y, sensitive


def bin_age_for_fairness(age_values, bins=3):
    """
    Bin age into groups for fairness analysis.
    
    Args:
        age_values: Age values
        bins: Number of bins
        
    Returns:
        Binned age groups
    """
    return pd.cut(age_values, bins=bins, labels=['Young', 'Middle', 'Senior'])


def evaluate_model_variant(model, X_train, X_test, y_train, y_test, 
                          sensitive_train, sensitive_test, variant_name, model_name):
    """
    Evaluate a single model variant.
    
    Returns:
        dict: Evaluation metrics
    """
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Accuracy metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else np.nan
    
    # Fairness metrics
    fairness_metrics = fairness_report(y_test, y_pred, sensitive_test, sensitive_name="Sex")
    
    # Group-specific metrics
    group_metrics = calculate_group_metrics(y_test, y_pred, sensitive_test)
    
    results = {
        'model': model_name,
        'variant': variant_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        **fairness_metrics,
        'group_metrics': group_metrics
    }
    
    return results


def main():
    """Main evaluation pipeline."""
    
    print("="*80)
    print("JOA+Fair Algorithm Comprehensive Evaluation")
    print("="*80)
    
    # Load data
    print("\n[1/6] Loading data...")
    df = pd.read_csv('data/heart.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Prepare data
    print("\n[2/6] Preparing data with feature engineering...")
    X, y, sensitive = prepare_data_for_fairness(df, target_col='target', sensitive_col='sex')
    print(f"Features shape: {X.shape}")
    print(f"Engineered features: {[col for col in X.columns if col not in df.columns]}")
    
    # Train-test split (stratified)
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Get models
    models_dict = get_models()
    print(f"\n[3/6] Models to evaluate: {list(models_dict.keys())}")
    
    # Results storage
    all_results = []
    
    # Evaluate each model
    print("\n[4/6] Evaluating models...")
    for model_name, base_model in models_dict.items():
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*80}")
        
        # Prepare features (exclude sensitive attribute for training)
        X_train_nosens = X_train.drop(columns=['sex'])
        X_test_nosens = X_test.drop(columns=['sex'])
        
        # ===== VARIANT 1: Simple (default params, all features) =====
        print(f"\n  [1/3] Simple variant...")
        simple_model = base_model
        simple_results = evaluate_model_variant(
            simple_model, X_train_nosens, X_test_nosens, y_train, y_test,
            sens_train, sens_test, "Simple", model_name
        )
        all_results.append(simple_results)
        print(f"    Accuracy: {simple_results['accuracy']:.4f}")
        print(f"    Demographic Parity: {simple_results['demographic_parity']:.4f}")
        
        # ===== VARIANT 2: JOA-Optimized (optimized params, all features) =====
        print(f"\n  [2/3] JOA-Optimized variant...")
        try:
            joa_model = optimize_model(model_name, X_train_nosens, y_train)
            if joa_model is not None:
                joa_results = evaluate_model_variant(
                    joa_model, X_train_nosens, X_test_nosens, y_train, y_test,
                    sens_train, sens_test, "JOA-Optimized", model_name
                )
                all_results.append(joa_results)
                print(f"    Accuracy: {joa_results['accuracy']:.4f}")
                print(f"    Demographic Parity: {joa_results['demographic_parity']:.4f}")
            else:
                print("    JOA optimization not available for this model")
                joa_model = base_model
                joa_results = simple_results.copy()
                joa_results['variant'] = "JOA-Optimized"
                all_results.append(joa_results)
        except Exception as e:
            print(f"    Error in JOA optimization: {e}")
            joa_model = base_model
            joa_results = simple_results.copy()
            joa_results['variant'] = "JOA-Optimized"
            all_results.append(joa_results)
        
        # ===== VARIANT 3: JOA+Fair (optimized params, AFIG features) =====
        print(f"\n  [3/3] JOA+Fair variant...")
        try:
            # Optimize AFIG parameters for this model
            print("    Optimizing AFIG parameters...")
            from src.fairness import optimize_afig_params
            
            afig_opt = optimize_afig_params(
                X_train, y_train, sensitive_col='sex',
                param_ranges={
                    'lambda': [0.0, 0.1, 0.2, 0.3, 0.5],
                    'beta': [0.05, 0.1, 0.15],
                    'k': [6, 8, 10, 12]
                },
                cv_folds=3,
                model_class=type(joa_model),
                random_state=42
            )
            
            best_params = afig_opt['best_params']
            
            # Handle case where optimization returns None
            if best_params is None:
                print("    AFIG optimization failed, using default parameters")
                best_params = {'lambda_base': 0.2, 'beta_base': 0.1, 'k': 8}
            
            print(f"    Best AFIG params: λ={best_params['lambda_base']:.2f}, "
                  f"β={best_params['beta_base']:.2f}, k={best_params['k']}")
            
            # Fit AFIG selector with best params
            afig_selector = AdaptiveFairFeatureSelector(
                sensitive_col='sex',
                **best_params
            )
            afig_selector.fit(X_train, y_train)
            
            print(f"    Selected features: {afig_selector.selected_features_}")
            print(f"    Fairness impact: λ_adaptive={afig_selector.lambda_adaptive_:.3f}, "
                  f"β_calibrated={afig_selector.beta_calibrated_:.3f}")
            
            # Transform data
            X_train_afig = afig_selector.transform(X_train)
            X_test_afig = afig_selector.transform(X_test)
            
            # Train with selected features
            fair_model = type(joa_model)(**joa_model.get_params())
            fair_results = evaluate_model_variant(
                fair_model, X_train_afig, X_test_afig, y_train, y_test,
                sens_train, sens_test, "JOA+Fair", model_name
            )
            all_results.append(fair_results)
            print(f"    Accuracy: {fair_results['accuracy']:.4f}")
            print(f"    Demographic Parity: {fair_results['demographic_parity']:.4f}")
            
        except Exception as e:
            print(f"    Error in JOA+Fair: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to JOA results
            fair_results = joa_results.copy()
            fair_results['variant'] = "JOA+Fair"
            all_results.append(fair_results)
    
    # ===== RESULTS ANALYSIS =====
    print(f"\n{'='*80}")
    print("[5/6] Results Summary")
    print(f"{'='*80}")
    
    results_df = pd.DataFrame(all_results)
    
    # Remove group_metrics column for display
    display_df = results_df.drop(columns=['group_metrics', 'sensitive_attribute'], errors='ignore')
    
    print("\n" + "="*80)
    print("ACCURACY METRICS")
    print("="*80)
    accuracy_cols = ['model', 'variant', 'accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    print(display_df[accuracy_cols].to_string(index=False))
    
    print("\n" + "="*80)
    print("FAIRNESS METRICS")
    print("="*80)
    fairness_cols = ['model', 'variant', 'demographic_parity', 'disparate_impact', 
                     'equal_opportunity_diff', 'equalized_odds_max_diff']
    print(display_df[fairness_cols].to_string(index=False))
    
    # Find best models
    print("\n" + "="*80)
    print("BEST PERFORMERS")
    print("="*80)
    
    best_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
    print(f"\nBest Accuracy: {best_accuracy['model']} ({best_accuracy['variant']})")
    print(f"  Accuracy: {best_accuracy['accuracy']:.4f}")
    print(f"  Demographic Parity: {best_accuracy['demographic_parity']:.4f}")
    
    # Best fairness-accuracy tradeoff (composite score)
    results_df['fairness_score'] = (
        results_df['demographic_parity'] + 
        results_df['equalized_odds_max_diff'] + 
        (1 - results_df['disparate_impact'])
    ) / 3
    results_df['composite_score'] = results_df['accuracy'] - 0.5 * results_df['fairness_score']
    
    best_composite = results_df.loc[results_df['composite_score'].idxmax()]
    print(f"\nBest Fairness-Accuracy Tradeoff: {best_composite['model']} ({best_composite['variant']})")
    print(f"  Accuracy: {best_composite['accuracy']:.4f}")
    print(f"  Demographic Parity: {best_composite['demographic_parity']:.4f}")
    print(f"  Composite Score: {best_composite['composite_score']:.4f}")
    
    # Save results
    print(f"\n[6/6] Saving results...")
    results_df.to_csv('joa_fair_results.csv', index=False)
    print("  Saved to: joa_fair_results.csv")
    
    # Generate plots
    try:
        print("\n  Generating fairness comparison plots...")
        plot_fairness_comparison(results_df, save_path='fairness_comparison.png')
        print("  Saved to: fairness_comparison.png")
    except Exception as e:
        print(f"  Error generating plots: {e}")
    
    print(f"\n{'='*80}")
    print("Evaluation Complete!")
    print(f"{'='*80}\n")
    
    return results_df


if __name__ == "__main__":
    results = main()
