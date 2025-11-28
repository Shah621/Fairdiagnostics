import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Import from existing modules
from src.data_loader import load_data, clean_data
from src.optimization import optimize_model
from exp import precision_mir_ig_feature_selection, get_optimized_random_forest, optimized_feature_encoding

def run_improvement_pipeline():
    print("=== STARTING FIG MODEL IMPROVEMENT PIPELINE ===")
    
    # 1. Load and Clean Data
    print("\n1. Loading and Cleaning Data...")
    try:
        df = load_data('data/heart.csv')
        df = clean_data(df)
        # Lowercase columns to match exp.py expectations
        df.columns = df.columns.str.lower()
        print(f"Data loaded. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Run FIG Feature Selection (Grid Search for best lambda/beta/k)
    print("\n2. Running FIG Feature Selection...")
    # We will use a slightly wider range or the same as exp.py to find the best features first
    lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4]
    beta_values = [0.05, 0.1, 0.15]
    k_values = [6, 8, 10, 12] # Added 12 just in case
    
    best_fig_score = -1
    best_features = []
    best_params = {}
    
    # We use the "known best" RF for selection as per the methodology
    rf_selector = get_optimized_random_forest()
    
    results = []

    for k in k_values:
        for lam in lambda_values:
            for beta in beta_values:
                try:
                    selected_features, _ = precision_mir_ig_feature_selection(
                        df, target_col='heartdisease', sensitive_col='sex', 
                        k=k, λ=lam, β=beta
                    )
                    
                    if not selected_features or len(selected_features) < 4:
                        continue
                        
                    # Evaluate this feature set with the standard RF
                    # We need to handle encoding
                    original_features = [f for f in selected_features if f in df.columns]
                    if len(original_features) < 4:
                        continue
                        
                    X = df[original_features].copy()
                    y = df['heartdisease']
                    
                    # Encoding
                    for col in X.columns:
                        if X[col].dtype == 'object':
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col].astype(str))
                            
                    # CV Score
                    cv_scores = cross_val_score(rf_selector, X, y, 
                                              cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                              scoring='accuracy')
                    mean_cv = cv_scores.mean()
                    
                    results.append({
                        'k': k, 'lambda': lam, 'beta': beta, 
                        'features': original_features, 
                        'cv_score': mean_cv
                    })
                    
                    if mean_cv > best_fig_score:
                        best_fig_score = mean_cv
                        best_features = original_features
                        best_params = {'k': k, 'lambda': lam, 'beta': beta}
                        print(f"  New Best FIG: k={k}, λ={lam}, β={beta} -> CV Acc: {mean_cv:.4f}")
                        
                except Exception as e:
                    continue
    
    print(f"\nBest FIG Parameters: {best_params}")
    print(f"Best Features ({len(best_features)}): {best_features}")
    print(f"Best CV Score (Standard RF): {best_fig_score:.4f}")
    
    # 3. Optimize Model Hyperparameters for the Selected Features
    print("\n3. Optimizing Model Hyperparameters (JOA) for Selected Features...")
    
    X_final = df[best_features].copy()
    y_final = df['heartdisease']
    
    # Encode
    for col in X_final.columns:
        if X_final[col].dtype == 'object':
            le = LabelEncoder()
            X_final[col] = le.fit_transform(X_final[col].astype(str))
            
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
    )
    
    # Optimize RF using JOA
    print("  Optimizing Random Forest...")
    optimized_rf = optimize_model("Random Forest", X_train, y_train)
    
    # Train and Evaluate
    optimized_rf.fit(X_train, y_train)
    y_pred = optimized_rf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Optimized Model Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nOptimized Hyperparameters:")
    print(optimized_rf.get_params())

if __name__ == "__main__":
    run_improvement_pipeline()
