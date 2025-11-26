import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

class FairFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects features based on Fair Information Gain (FIG) with Redundancy penalty.
    Score = IG - lambda * I(Feature; Sensitive) - beta * Avg(I(Feature; Selected))
    """
    def __init__(self, sensitive_col, k=10, lambda_val=0.5, beta_val=0.1):
        self.sensitive_col = sensitive_col
        self.k = k
        self.lambda_val = lambda_val
        self.beta_val = beta_val
        self.selected_features_ = []
        self.feature_scores_ = {}

    def _calculate_mi(self, x, y):
        # Helper to calculate MI safely
        x = np.array(x).reshape(-1, 1)
        y = np.array(y)
        
        mask = ~np.isnan(x).flatten()
        if not mask.any():
            return 0.0
            
        x = x[mask].reshape(-1, 1)
        y = y[mask]
        
        discrete = True if len(np.unique(y)) < 10 else False
        try:
            if discrete:
                mi = mutual_info_classif(x, y, discrete_features='auto', random_state=42, n_neighbors=5)
            else:
                mi = mutual_info_regression(x, y, random_state=42, n_neighbors=5)
            return mi[0]
        except:
            return 0.0

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame containing the sensitive attribute.")
            
        if self.sensitive_col not in X.columns:
            raise ValueError(f"Sensitive column '{self.sensitive_col}' not found in X.")
            
        A = X[self.sensitive_col].values
        # Features to evaluate (exclude sensitive col)
        features = [c for c in X.columns if c != self.sensitive_col]
        
        # Pre-calculate IG and MI_A for all features
        IG = {}
        MI_A = {}
        X_values = {} # Cache feature values
        
        for feat in features:
            x_feat = X[feat].values
            X_values[feat] = x_feat
            IG[feat] = self._calculate_mi(x_feat, y)
            MI_A[feat] = self._calculate_mi(x_feat, A)
            
        # Greedy Selection (mRMR style)
        selected = []
        remaining = features.copy()
        redundancy_cache = {} # Cache MI between features
        
        while len(selected) < self.k and remaining:
            best_score = -np.inf
            best_feat = None
            
            for f in remaining:
                # Calculate Redundancy
                redundancy = 0
                if selected:
                    for s in selected:
                        key = tuple(sorted((f, s)))
                        if key not in redundancy_cache:
                            redundancy_cache[key] = self._calculate_mi(X_values[f], X_values[s])
                        redundancy += redundancy_cache[key]
                    redundancy /= len(selected)
                
                # Score
                # Normalize terms to keep lambda/beta meaningful? 
                # exp.py normalized by max_IG. Let's do simple version first.
                # Score = IG - lambda*MI_A - beta*Redundancy
                score = IG[f] - self.lambda_val * MI_A[f] - self.beta_val * redundancy
                
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
        if not isinstance(X, pd.DataFrame):
             raise ValueError("X must be a pandas DataFrame.")
        return X[self.selected_features_]
