"""
Comprehensive fairness metrics for evaluating model fairness across sensitive attributes.

This module provides standard fairness metrics including:
- Demographic Parity
- Equalized Odds
- Disparate Impact
- Equal Opportunity Difference
- Statistical Parity Difference
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def demographic_parity(y_pred, sensitive):
    """
    Calculate demographic parity (statistical parity).
    
    Measures the difference in positive prediction rates between groups.
    DP = P(Ŷ=1|A=0) - P(Ŷ=1|A=1)
    
    A value close to 0 indicates fairness.
    
    Args:
        y_pred: Predicted labels
        sensitive: Sensitive attribute values
        
    Returns:
        float: Demographic parity difference
    """
    sensitive = np.array(sensitive)
    y_pred = np.array(y_pred)
    
    groups = np.unique(sensitive)
    if len(groups) != 2:
        # For multi-group, calculate max difference
        rates = []
        for group in groups:
            mask = sensitive == group
            if mask.sum() > 0:
                rate = y_pred[mask].mean()
                rates.append(rate)
        return max(rates) - min(rates) if rates else 0.0
    
    # Binary sensitive attribute
    group_0_mask = sensitive == groups[0]
    group_1_mask = sensitive == groups[1]
    
    rate_0 = y_pred[group_0_mask].mean() if group_0_mask.sum() > 0 else 0
    rate_1 = y_pred[group_1_mask].mean() if group_1_mask.sum() > 0 else 0
    
    return abs(rate_0 - rate_1)


def equalized_odds(y_true, y_pred, sensitive):
    """
    Calculate equalized odds metric.
    
    Measures the difference in TPR and FPR between groups.
    EO = max(|TPR_0 - TPR_1|, |FPR_0 - FPR_1|)
    
    A value close to 0 indicates fairness.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive: Sensitive attribute values
        
    Returns:
        dict: Contains 'tpr_diff', 'fpr_diff', and 'max_diff'
    """
    sensitive = np.array(sensitive)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    groups = np.unique(sensitive)
    if len(groups) != 2:
        # For multi-group, calculate max differences
        tprs, fprs = [], []
        for group in groups:
            mask = sensitive == group
            if mask.sum() > 0:
                tn, fp, fn, tp = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1]).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                tprs.append(tpr)
                fprs.append(fpr)
        
        tpr_diff = max(tprs) - min(tprs) if tprs else 0.0
        fpr_diff = max(fprs) - min(fprs) if fprs else 0.0
        return {
            'tpr_diff': tpr_diff,
            'fpr_diff': fpr_diff,
            'max_diff': max(tpr_diff, fpr_diff)
        }
    
    # Binary sensitive attribute
    group_0_mask = sensitive == groups[0]
    group_1_mask = sensitive == groups[1]
    
    # Calculate TPR and FPR for each group
    def calc_rates(mask):
        if mask.sum() == 0:
            return 0, 0
        tn, fp, fn, tp = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        return tpr, fpr
    
    tpr_0, fpr_0 = calc_rates(group_0_mask)
    tpr_1, fpr_1 = calc_rates(group_1_mask)
    
    tpr_diff = abs(tpr_0 - tpr_1)
    fpr_diff = abs(fpr_0 - fpr_1)
    
    return {
        'tpr_diff': tpr_diff,
        'fpr_diff': fpr_diff,
        'max_diff': max(tpr_diff, fpr_diff)
    }


def disparate_impact(y_pred, sensitive):
    """
    Calculate disparate impact ratio.
    
    DI = P(Ŷ=1|A=0) / P(Ŷ=1|A=1)
    
    A value close to 1 indicates fairness.
    The "80% rule" suggests DI should be >= 0.8.
    
    Args:
        y_pred: Predicted labels
        sensitive: Sensitive attribute values
        
    Returns:
        float: Disparate impact ratio
    """
    sensitive = np.array(sensitive)
    y_pred = np.array(y_pred)
    
    groups = np.unique(sensitive)
    if len(groups) != 2:
        # For multi-group, calculate min/max ratio
        rates = []
        for group in groups:
            mask = sensitive == group
            if mask.sum() > 0:
                rate = y_pred[mask].mean()
                rates.append(rate)
        if not rates or min(rates) == 0:
            return 0.0
        return min(rates) / max(rates)
    
    # Binary sensitive attribute
    group_0_mask = sensitive == groups[0]
    group_1_mask = sensitive == groups[1]
    
    rate_0 = y_pred[group_0_mask].mean() if group_0_mask.sum() > 0 else 0
    rate_1 = y_pred[group_1_mask].mean() if group_1_mask.sum() > 0 else 0
    
    if rate_1 == 0:
        return 0.0
    
    return min(rate_0, rate_1) / max(rate_0, rate_1)


def equal_opportunity_difference(y_true, y_pred, sensitive):
    """
    Calculate equal opportunity difference.
    
    Measures the difference in TPR (recall) between groups.
    EOD = |TPR_0 - TPR_1|
    
    A value close to 0 indicates fairness.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive: Sensitive attribute values
        
    Returns:
        float: Equal opportunity difference
    """
    sensitive = np.array(sensitive)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    groups = np.unique(sensitive)
    tprs = []
    
    for group in groups:
        mask = sensitive == group
        if mask.sum() > 0:
            # Only consider positive class (y_true == 1)
            pos_mask = mask & (y_true == 1)
            if pos_mask.sum() > 0:
                tpr = (y_pred[pos_mask] == 1).mean()
                tprs.append(tpr)
    
    if len(tprs) < 2:
        return 0.0
    
    return max(tprs) - min(tprs)


def statistical_parity_difference(y_pred, sensitive):
    """
    Calculate statistical parity difference (same as demographic parity).
    
    SPD = P(Ŷ=1|A=0) - P(Ŷ=1|A=1)
    
    A value close to 0 indicates fairness.
    
    Args:
        y_pred: Predicted labels
        sensitive: Sensitive attribute values
        
    Returns:
        float: Statistical parity difference
    """
    return demographic_parity(y_pred, sensitive)


def fairness_report(y_true, y_pred, sensitive, sensitive_name="Sensitive"):
    """
    Generate a comprehensive fairness report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive: Sensitive attribute values
        sensitive_name: Name of the sensitive attribute
        
    Returns:
        dict: Dictionary containing all fairness metrics
    """
    eo = equalized_odds(y_true, y_pred, sensitive)
    
    report = {
        'sensitive_attribute': sensitive_name,
        'demographic_parity': demographic_parity(y_pred, sensitive),
        'disparate_impact': disparate_impact(y_pred, sensitive),
        'equal_opportunity_diff': equal_opportunity_difference(y_true, y_pred, sensitive),
        'equalized_odds_tpr_diff': eo['tpr_diff'],
        'equalized_odds_fpr_diff': eo['fpr_diff'],
        'equalized_odds_max_diff': eo['max_diff'],
        'statistical_parity_diff': statistical_parity_difference(y_pred, sensitive)
    }
    
    return report


def print_fairness_report(report):
    """
    Print a formatted fairness report.
    
    Args:
        report: Dictionary from fairness_report()
    """
    print(f"\n{'='*60}")
    print(f"Fairness Report - {report['sensitive_attribute']}")
    print(f"{'='*60}")
    print(f"Demographic Parity:          {report['demographic_parity']:.4f}  (closer to 0 is fairer)")
    print(f"Disparate Impact:            {report['disparate_impact']:.4f}  (closer to 1 is fairer, >=0.8 is good)")
    print(f"Equal Opportunity Diff:      {report['equal_opportunity_diff']:.4f}  (closer to 0 is fairer)")
    print(f"Equalized Odds (TPR diff):   {report['equalized_odds_tpr_diff']:.4f}  (closer to 0 is fairer)")
    print(f"Equalized Odds (FPR diff):   {report['equalized_odds_fpr_diff']:.4f}  (closer to 0 is fairer)")
    print(f"Equalized Odds (max diff):   {report['equalized_odds_max_diff']:.4f}  (closer to 0 is fairer)")
    print(f"Statistical Parity Diff:     {report['statistical_parity_diff']:.4f}  (closer to 0 is fairer)")
    print(f"{'='*60}\n")


def plot_fairness_comparison(results_df, save_path=None):
    """
    Plot fairness metrics comparison across different models/variants.
    
    Args:
        results_df: DataFrame with columns: model, variant, accuracy, demographic_parity, etc.
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Accuracy vs Demographic Parity
    ax = axes[0, 0]
    for variant in results_df['variant'].unique():
        subset = results_df[results_df['variant'] == variant]
        ax.scatter(subset['demographic_parity'], subset['accuracy'], 
                  label=variant, s=100, alpha=0.7)
    ax.set_xlabel('Demographic Parity (lower is fairer)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy vs Demographic Parity', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy vs Equalized Odds
    ax = axes[0, 1]
    for variant in results_df['variant'].unique():
        subset = results_df[results_df['variant'] == variant]
        ax.scatter(subset['equalized_odds_max_diff'], subset['accuracy'], 
                  label=variant, s=100, alpha=0.7)
    ax.set_xlabel('Equalized Odds Max Diff (lower is fairer)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy vs Equalized Odds', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Disparate Impact by Model
    ax = axes[1, 0]
    pivot = results_df.pivot(index='model', columns='variant', values='disparate_impact')
    pivot.plot(kind='bar', ax=ax, width=0.8)
    ax.axhline(y=0.8, color='red', linestyle='--', label='80% Rule', linewidth=2)
    ax.axhline(y=1.0, color='green', linestyle='--', label='Perfect Fairness', linewidth=2)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Disparate Impact (closer to 1 is fairer)', fontsize=12)
    ax.set_title('Disparate Impact by Model', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 4: Fairness-Accuracy Tradeoff Heatmap
    ax = axes[1, 1]
    # Create a composite fairness score (lower is better)
    results_df['fairness_score'] = (
        results_df['demographic_parity'] + 
        results_df['equalized_odds_max_diff'] + 
        (1 - results_df['disparate_impact'])
    ) / 3
    
    pivot_heatmap = results_df.pivot(index='model', columns='variant', values='fairness_score')
    sns.heatmap(pivot_heatmap, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax, 
                cbar_kws={'label': 'Composite Fairness Score (lower is better)'})
    ax.set_title('Composite Fairness Score Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Variant', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def calculate_group_metrics(y_true, y_pred, sensitive):
    """
    Calculate performance metrics for each group in the sensitive attribute.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive: Sensitive attribute values
        
    Returns:
        DataFrame: Metrics per group
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    groups = np.unique(sensitive)
    metrics = []
    
    for group in groups:
        mask = sensitive == group
        if mask.sum() > 0:
            group_metrics = {
                'group': group,
                'count': mask.sum(),
                'accuracy': accuracy_score(y_true[mask], y_pred[mask]),
                'precision': precision_score(y_true[mask], y_pred[mask], zero_division=0),
                'recall': recall_score(y_true[mask], y_pred[mask], zero_division=0),
                'f1': f1_score(y_true[mask], y_pred[mask], zero_division=0)
            }
            metrics.append(group_metrics)
    
    return pd.DataFrame(metrics)
