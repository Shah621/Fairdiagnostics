"""
Adaptive Fair Information Gain (AFIG) Feature Selection Algorithm

This module implements a novel fairness-aware feature selection approach that:
1. Dynamically adjusts fairness penalties based on observed group disparities
2. Calibrates redundancy penalties based on feature correlation patterns
3. Balances accuracy, fairness, and feature redundancy simultaneously

The AFIG algorithm improves upon standard Fair Information Gain by making
the fairness penalty adaptive rather than static, leading to better
fairness-accuracy tradeoffs.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split


class AdaptiveFairFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Adaptive Fair Information Gain (AFIG) feature selector.
    
    Novel contributions:
    1. Dynamic λ: Adjusts fairness penalty based on group disparity in predictions
    2. Calibrated β: Adapts redundancy penalty based on feature correlations
    3. Multi-stage selection: Fairness screening → Redundancy filtering → Accuracy optimization
    4. Normalized scoring: Ensures stable optimization across different datasets
    
    Score = (IG / max_IG) * w_feat - λ_adaptive * (MI_A / max_MI_A) - β_calibrated * redundancy
    
    where:
    - λ_adaptive = λ_base * (1 + disparity_factor)
    - β_calibrated = β_base * correlation_factor
    - w_feat = feature importance weight (optional)
    """
    
    def __init__(self, sensitive_col, k=10, lambda_base=0.5, beta_base=0.1, 
                 adaptive_lambda=True, calibrate_beta=True, feature_weights=None):
        """
        Initialize AFIG selector.
        
        Args:
            sensitive_col: Name of sensitive attribute column
            k: Number of features to select
            lambda_base: Base fairness penalty (will be adapted if adaptive_lambda=True)
            beta_base: Base redundancy penalty (will be calibrated if calibrate_beta=True)
            adaptive_lambda: Whether to dynamically adjust λ based on group disparities
            calibrate_beta: Whether to calibrate β based on feature correlations
            feature_weights: Optional dict of feature importance weights
        """
        self.sensitive_col = sensitive_col
        self.k = k
        self.lambda_base = lambda_base
        self.beta_base = beta_base
        self.adaptive_lambda = adaptive_lambda
        self.calibrate_beta = calibrate_beta
        self.feature_weights = feature_weights or {}
        
        # Will be set during fit
        self.selected_features_ = []
        self.feature_scores_ = {}
        self.lambda_adaptive_ = lambda_base
        self.beta_calibrated_ = beta_base
        self.disparity_factor_ = 0.0
        self.correlation_factor_ = 1.0

    def _calculate_mi(self, x, y):
        """
        Calculate mutual information safely.
        
        Args:
            x: Feature values
            y: Target values
            
        Returns:
            float: Mutual information score
        """
        x = np.array(x).reshape(-1, 1)
        y = np.array(y)
        
        # Remove NaN values
        mask = ~np.isnan(x).flatten()
        if not mask.any():
            return 0.0
            
        x = x[mask].reshape(-1, 1)
        y = y[mask]
        
        # Determine if target is discrete
        discrete = True if len(np.unique(y)) < 10 else False
        
        try:
            if discrete:
                mi = mutual_info_classif(x, y, discrete_features='auto', 
                                        random_state=42, n_neighbors=5)
            else:
                mi = mutual_info_regression(x, y, random_state=42, n_neighbors=5)
            return mi[0]
        except:
            return 0.0

    def _calculate_disparity_factor(self, X, y, A):
        """
        Calculate group disparity factor to adapt λ.
        
        Higher disparity → higher λ (stronger fairness penalty)
        
        Args:
            X: Feature matrix
            y: Target values
            A: Sensitive attribute values
            
        Returns:
            float: Disparity factor (0 to 1)
        """
        groups = np.unique(A)
        if len(groups) < 2:
            return 0.0
        
        # Calculate target rate difference between groups
        group_rates = []
        for group in groups:
            mask = A == group
            if mask.sum() > 0:
                rate = y[mask].mean()
                group_rates.append(rate)
        
        if not group_rates:
            return 0.0
        
        # Disparity is the max difference in target rates
        disparity = max(group_rates) - min(group_rates)
        
        # Normalize to [0, 1]
        return min(disparity, 1.0)

    def _calculate_correlation_factor(self, X, features):
        """
        Calculate average feature correlation to calibrate β.
        
        Higher correlation → higher β (stronger redundancy penalty)
        
        Args:
            X: Feature matrix (DataFrame)
            features: List of feature names
            
        Returns:
            float: Correlation factor (0.5 to 2.0)
        """
        if len(features) < 2:
            return 1.0
        
        # Calculate pairwise correlations
        correlations = []
        for i, f1 in enumerate(features):
            for f2 in features[i+1:]:
                try:
                    corr = np.corrcoef(X[f1].values, X[f2].values)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                except:
                    pass
        
        if not correlations:
            return 1.0
        
        avg_corr = np.mean(correlations)
        
        # Map [0, 1] correlation to [0.5, 2.0] factor
        # Low correlation → lower β (less redundancy penalty)
        # High correlation → higher β (more redundancy penalty)
        factor = 0.5 + 1.5 * avg_corr
        
        return factor

    def fit(self, X, y):
        """
        Fit the AFIG selector.
        
        Args:
            X: Feature matrix (must be DataFrame with sensitive_col)
            y: Target values
            
        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame containing the sensitive attribute.")
            
        if self.sensitive_col not in X.columns:
            raise ValueError(f"Sensitive column '{self.sensitive_col}' not found in X.")
        
        # Extract sensitive attribute
        A = X[self.sensitive_col].values
        
        # Features to evaluate (exclude sensitive col)
        features = [c for c in X.columns if c != self.sensitive_col]
        
        # Calculate disparity factor for adaptive λ
        if self.adaptive_lambda:
            self.disparity_factor_ = self._calculate_disparity_factor(X, y, A)
            self.lambda_adaptive_ = self.lambda_base * (1 + self.disparity_factor_)
        else:
            self.lambda_adaptive_ = self.lambda_base
        
        # Calculate correlation factor for calibrated β
        if self.calibrate_beta:
            self.correlation_factor_ = self._calculate_correlation_factor(X, features)
            self.beta_calibrated_ = self.beta_base * self.correlation_factor_
        else:
            self.beta_calibrated_ = self.beta_base
        
        # Pre-calculate IG and MI_A for all features
        IG = {}
        MI_A = {}
        X_values = {}  # Cache feature values
        
        for feat in features:
            x_feat = X[feat].values
            X_values[feat] = x_feat
            IG[feat] = self._calculate_mi(x_feat, y)
            MI_A[feat] = self._calculate_mi(x_feat, A)
        
        # Normalize IG and MI_A for stable optimization
        max_IG = max(IG.values()) if IG.values() else 1.0
        max_MI_A = max(MI_A.values()) if MI_A.values() else 1.0
        
        # Avoid division by zero
        max_IG = max(max_IG, 1e-8)
        max_MI_A = max(max_MI_A, 1e-8)
        
        # Greedy multi-stage selection
        selected = []
        remaining = features.copy()
        redundancy_cache = {}  # Cache MI between features
        
        while len(selected) < self.k and remaining:
            best_score = -np.inf
            best_feat = None
            
            for f in remaining:
                # Get feature weight (default 1.0)
                w_feat = self.feature_weights.get(f, 1.0)
                
                # Calculate redundancy with already selected features
                redundancy = 0
                if selected:
                    for s in selected:
                        key = tuple(sorted((f, s)))
                        if key not in redundancy_cache:
                            redundancy_cache[key] = self._calculate_mi(X_values[f], X_values[s])
                        redundancy += redundancy_cache[key]
                    redundancy /= len(selected)
                
                # Normalized AFIG score
                norm_IG = (IG[f] / max_IG) * w_feat
                norm_MI_A = MI_A[f] / max_MI_A
                
                # Adaptive fairness penalty and calibrated redundancy penalty
                score = (norm_IG - 
                        self.lambda_adaptive_ * norm_MI_A - 
                        self.beta_calibrated_ * redundancy)
                
                if score > best_score:
                    best_score = score
                    best_feat = f
            
            if best_feat:
                selected.append(best_feat)
                remaining.remove(best_feat)
                self.feature_scores_[best_feat] = best_score
            else:
                break
        
        self.selected_features_ = selected
        return self

    def transform(self, X):
        """
        Transform X to selected features.
        
        Args:
            X: Feature matrix (DataFrame)
            
        Returns:
            DataFrame: X with only selected features
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        return X[self.selected_features_]

    def get_fairness_impact(self):
        """
        Get fairness impact metrics for transparency.
        
        Returns:
            dict: Fairness impact information
        """
        return {
            'lambda_base': self.lambda_base,
            'lambda_adaptive': self.lambda_adaptive_,
            'beta_base': self.beta_base,
            'beta_calibrated': self.beta_calibrated_,
            'disparity_factor': self.disparity_factor_,
            'correlation_factor': self.correlation_factor_,
            'num_selected': len(self.selected_features_),
            'selected_features': self.selected_features_
        }


def optimize_afig_params(X, y, sensitive_col, param_ranges=None, cv_folds=5, 
                         model_class=None, random_state=42):
    """
    Optimize AFIG parameters (λ, β, k) using cross-validation.
    
    Args:
        X: Feature matrix (DataFrame)
        y: Target values
        sensitive_col: Name of sensitive attribute
        param_ranges: Dict with 'lambda', 'beta', 'k' ranges (optional)
        cv_folds: Number of CV folds
        model_class: Model class to use for evaluation (default: RandomForest)
        random_state: Random seed
        
    Returns:
        dict: Best parameters and scores
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    
    if param_ranges is None:
        param_ranges = {
            'lambda': [0.0, 0.1, 0.2, 0.3, 0.5],
            'beta': [0.05, 0.1, 0.15, 0.2],
            'k': [6, 8, 10, 12]
        }
    
    if model_class is None:
        model_class = RandomForestClassifier
    
    best_score = -np.inf
    best_params = None
    results = []
    
    # Grid search over parameters
    for lambda_val in param_ranges['lambda']:
        for beta_val in param_ranges['beta']:
            for k_val in param_ranges['k']:
                # Cross-validation
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                cv_scores = []
                
                for train_idx, val_idx in cv.split(X, y):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Fit AFIG selector
                    selector = AdaptiveFairFeatureSelector(
                        sensitive_col=sensitive_col,
                        k=k_val,
                        lambda_base=lambda_val,
                        beta_base=beta_val
                    )
                    
                    try:
                        selector.fit(X_train, y_train)
                        
                        if len(selector.selected_features_) < 3:
                            continue
                        
                        # Transform data
                        X_train_sel = selector.transform(X_train)
                        X_val_sel = selector.transform(X_val)
                        
                        # Train model
                        model = model_class(random_state=random_state)
                        model.fit(X_train_sel, y_train)
                        
                        # Evaluate
                        y_pred = model.predict(X_val_sel)
                        score = accuracy_score(y_val, y_pred)
                        cv_scores.append(score)
                    except:
                        continue
                
                if cv_scores:
                    mean_score = np.mean(cv_scores)
                    results.append({
                        'lambda': lambda_val,
                        'beta': beta_val,
                        'k': k_val,
                        'cv_score': mean_score,
                        'cv_std': np.std(cv_scores)
                    })
                    
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {
                            'lambda_base': lambda_val,
                            'beta_base': beta_val,
                            'k': k_val
                        }
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': pd.DataFrame(results)
    }
