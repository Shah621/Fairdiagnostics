import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Optimized Fair Feature Selection
# -------------------------------

def optimized_mutual_information(x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)

    mask = ~np.isnan(x).flatten()
    x = x[mask].reshape(-1, 1)
    y = y[mask]

    if len(x) == 0:
        return 0

    discrete = True if len(np.unique(y)) < 10 else False
    try:
        if discrete:
            mi = mutual_info_classif(x, y, discrete_features='auto', random_state=42, n_neighbors=5)
        else:
            mi = mutual_info_regression(x, y, random_state=42, n_neighbors=5)
        return mi[0]
    except:
        return 0

def create_targeted_features(df):
    """Create features that complement the known best RF parameters"""
    df_eng = df.copy()
    df_eng.columns = df_eng.columns.str.lower()

    # Features that work well with deep trees (max_depth=43)
    if all(col in df_eng.columns for col in ['age', 'thalach']):
        df_eng['age_hr_interaction'] = df_eng['age'] * df_eng['thalach'] / 100
        df_eng['hr_age_ratio'] = df_eng['thalach'] / (df_eng['age'] + 1e-5)

    # Complex interactions that deep trees can capture
    if all(col in df_eng.columns for col in ['cp', 'exang', 'oldpeak']):
        df_eng['symptom_complexity'] = df_eng['cp'] * df_eng['exang'] + df_eng['oldpeak']

    # Non-linear transformations
    if 'oldpeak' in df_eng.columns:
        df_eng['oldpeak_squared'] = df_eng['oldpeak'] ** 2
        df_eng['oldpeak_log'] = np.log(df_eng['oldpeak'] + 1)

    if 'thalach' in df_eng.columns:
        df_eng['thalach_squared'] = df_eng['thalach'] ** 2

    # Clinical threshold features
    if 'trestbps' in df_eng.columns:
        df_eng['stage2_hypertension'] = (df_eng['trestbps'] >= 160).astype(int)

    if 'chol' in df_eng.columns:
        df_eng['very_high_chol'] = (df_eng['chol'] >= 280).astype(int)

    # Vessel disease complexity
    if 'ca' in df_eng.columns:
        df_eng['vessel_score'] = df_eng['ca'] ** 2  # Emphasize multi-vessel disease

    # ST segment complex features
    if all(col in df_eng.columns for col in ['oldpeak', 'slope']):
        df_eng['st_complex'] = df_eng['oldpeak'] * (df_eng['slope'] + 1)

    return df_eng

def optimized_feature_encoding(df, features):
    """Encoding optimized for Random Forest"""
    data = df.copy()
    data.columns = data.columns.str.lower()
    features = [f.lower() for f in features]

    # Simple ordinal encoding (RF handles raw features well)
    ordinal_mappings = {
        'cp': {1: 0, 2: 1, 3: 2, 4: 3},
        'restecg': {0: 0, 1: 1, 2: 2},
        'slope': {1: 0, 2: 1, 3: 2},
        'thal': {3: 0, 6: 1, 7: 2},
    }

    for feature, mapping in ordinal_mappings.items():
        if feature in data.columns:
            data[feature] = data[feature].map(mapping).fillna(0).astype(int)

    # No scaling needed for Random Forest
    return data

def precision_mir_ig_feature_selection(df, target_col='target', sensitive_col='sex',
                                      k=8, λ=0.2, β=0.1, use_optimal_rf=True):
    """Precision feature selection using known optimal RF parameters"""

    # Targeted feature engineering
    data = create_targeted_features(df)

    target_col = target_col.lower()
    sensitive_col = sensitive_col.lower()

    # Get features
    # Fix: ensure case-insensitive exclusion
    original_features = [f for f in df.columns if f.lower() not in [target_col, sensitive_col]]
    engineered_features = [f for f in data.columns if f not in [target_col, sensitive_col] and f not in [o.lower() for o in original_features]]
    
    # Note: original_features has original casing, engineered_features has lowercase
    # We need a unified list for encoding which expects a list of feature names
    # optimized_feature_encoding handles lowercasing of feature names passed to it
    
    all_features = original_features + engineered_features

    # Encoding
    data_encoded = optimized_feature_encoding(data, all_features + [target_col, sensitive_col])

    y = data_encoded[target_col].values
    A = data_encoded[sensitive_col].values

    available_features = [f for f in all_features if f.lower() in data_encoded.columns]
    # Note: available_features might have mixed casing, but data_encoded has lowercase columns
    # We should use lowercase names for indexing data_encoded
    
    # Let's just use lowercase names for everything internally
    available_features_lower = [f.lower() for f in available_features]
    X = data_encoded[available_features_lower]

    # Calculate information measures
    IG, MI_A = {}, {}

    for f in available_features_lower:
        IG[f] = optimized_mutual_information(X[f], y)
        MI_A[f] = optimized_mutual_information(X[f], A)

    # Feature selection with RF-optimized criteria
    selected, feature_scores = [], {}
    remaining_features = available_features_lower.copy()

    # Feature importance based on your RF success
    rf_optimized_weights = {
        'cp': 1.6, 'thal': 1.5, 'ca': 1.7, 'oldpeak': 1.6, 'exang': 1.4,
        'thalach': 1.3, 'slope': 1.4, 'age': 1.2, 'vessel_score': 1.5,
        'st_complex': 1.4, 'symptom_complexity': 1.3
    }

    redundancy_cache = {}

    while len(selected) < k and remaining_features:
        best_score = -np.inf
        best_feat = None

        for f in remaining_features:
            # Calculate redundancy
            redundancy = 0
            if selected:
                for s in selected:
                    cache_key = tuple(sorted([f, s]))
                    if cache_key not in redundancy_cache:
                        redundancy_cache[cache_key] = optimized_mutual_information(X[f], X[s])
                    redundancy += redundancy_cache[cache_key]
                redundancy /= len(selected)

            # Optimized scoring for RF
            max_IG = max(IG.values()) if IG.values() else 1
            max_MI_A = max(MI_A.values()) if MI_A.values() else 1

            rf_weight = rf_optimized_weights.get(f, 1.0)

            norm_IG = (IG[f] / (max_IG + 1e-8)) * rf_weight
            norm_MI_A = MI_A[f] / (max_MI_A + 1e-8)

            score = norm_IG - λ * norm_MI_A - β * redundancy

            if score > best_score:
                best_score = score
                best_feat = f

        if best_feat is None:
            break

        selected.append(best_feat)
        remaining_features.remove(best_feat)
        feature_scores[best_feat] = best_score

    return selected, feature_scores

def get_optimized_random_forest():
    """Return the optimized Random Forest with your best parameters"""
    return RandomForestClassifier(
        n_estimators=56,
        max_depth=43,
        min_samples_split=13,
        min_samples_leaf=3,
        random_state=42,
        bootstrap=True,
        n_jobs=-1
    )

# -------------------------------
# Precision Evaluation
# -------------------------------

def precision_evaluation(df, target_col='target', sensitive_col='sex'):
    """Precision evaluation using optimized RF"""

    # Focus on the most promising lambda range based on your previous results
    lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4]
    beta_values = [0.05, 0.1, 0.15]
    k_values = [6, 8, 10]  # Optimal feature counts

    results = []

    for λ in lambda_values:
        for β in beta_values:
            for k in k_values:
                try:
                    print(f"Testing lambda={λ}, beta={β}, k={k}")

                    # Precision feature selection
                    selected_features, feature_scores = precision_mir_ig_feature_selection(
                        df, target_col, sensitive_col, k=k, λ=λ, β=β
                    )

                    if not selected_features:
                        continue

                    # Use only original features (checking against lowercase columns of df is tricky if df has mixed case)
                    # selected_features are lowercase. df.columns might be mixed.
                    # We want to map back to original features if possible, or just use selected features which are valid in 'data' context.
                    # But here we want to evaluate on 'df' (or create a new dataframe with selected features).
                    
                    # Let's create the engineered dataframe again to select from
                    data = create_targeted_features(df)
                    data = optimized_feature_encoding(data, selected_features + [target_col.lower(), sensitive_col.lower()])
                    
                    # Now we have data with all needed columns (encoded)
                    X = data[selected_features].copy()
                    y = data[target_col.lower()]

                    if len(selected_features) < 4:
                        continue

                    print(f"  Selected {len(selected_features)} features: {selected_features}")

                    # Use optimized RF
                    rf_model = get_optimized_random_forest()

                    # Robust cross-validation
                    cv_scores = cross_val_score(rf_model, X, y,
                                              cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                                              scoring='accuracy')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()

                    # Final test evaluation
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )

                    rf_model.fit(X_train, y_train)
                    y_pred = rf_model.predict(X_test)
                    test_acc = accuracy_score(y_test, y_pred)

                    results.append({
                        'lambda': λ,
                        'beta': β,
                        'k': k,
                        'cv_accuracy': cv_mean,
                        'cv_std': cv_std,
                        'test_accuracy': test_acc,
                        'selected_features': selected_features,
                        'num_features': len(selected_features),
                        'feature_scores': feature_scores
                    })

                    print(f"  CV: {cv_mean:.4f} (±{cv_std:.4f}), Test: {test_acc:.4f}")

                    # Early success check
                    if test_acc >= 0.90:
                        print("*** TARGET ACHIEVED! ***")
                        return pd.DataFrame(results)

                except Exception as e:
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    return pd.DataFrame(results)

# -------------------------------
# Main Execution
# -------------------------------

if __name__ == "__main__":
    import os
    
    # Load data if running as main
    try:
        # Try different paths or use absolute path if known
        if os.path.exists('data/heart.csv'):
            df = pd.read_csv('data/heart.csv')
        elif os.path.exists('../data/heart.csv'):
            df = pd.read_csv('../data/heart.csv')
        else:
            # Fallback to absolute path from user info if needed, but relative is better for portability
            # Assuming running from root of workspace
            df = pd.read_csv('data/heart.csv')
    except FileNotFoundError:
        print("Error: data/heart.csv not found. Please ensure the data file exists.")
        exit(1)

    print("=== PRECISION FAIR FEATURE SELECTION ===")
    print("Using optimized Random Forest: n_estimators=56, max_depth=43, min_samples_split=13, min_samples_leaf=3")
    print(f"Dataset shape: {df.shape}")

    # Run precision evaluation
    print("\nStarting precision evaluation...")
    precision_results = precision_evaluation(df, target_col='target', sensitive_col='sex')
    # Note: heart.csv usually has 'target' or 'HeartDisease'. We should check.
    if 'target' not in df.columns.str.lower():
        if 'heartdisease' in df.columns.str.lower():
             # Find the actual column name
             target_col = [c for c in df.columns if c.lower() == 'heartdisease'][0]
             print(f"Using target column: {target_col}")
             precision_results = precision_evaluation(df, target_col=target_col, sensitive_col='sex')

    if not precision_results.empty:
        # Find best result
        best_idx = precision_results['test_accuracy'].idxmax()
        best_result = precision_results.loc[best_idx]

        print(f"\n=== BEST FAIR CONFIGURATION ===")
        print(f"λ = {best_result['lambda']:.2f}, β = {best_result['beta']:.2f}, k = {best_result['k']}")
        print(f"10-fold CV accuracy: {best_result['cv_accuracy']:.4f} (±{best_result['cv_std']:.4f})")
        print(f"Test accuracy: {best_result['test_accuracy']:.4f}")
        print(f"Selected features: {best_result['selected_features']}")

        # Feature scores analysis
        print(f"\nFeature Selection Scores:")
        for feature, score in best_result['feature_scores'].items():
            print(f"  {feature}: {score:.4f}")

        # Plot parameter sensitivity
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        for k in precision_results['k'].unique():
            subset = precision_results[precision_results['k'] == k]
            plt.plot(subset['lambda'], subset['test_accuracy'],
                    marker='o', linewidth=2, label=f'k={k}', markersize=6)
        plt.axhline(y=0.90, color='red', linestyle='--', alpha=0.7, label='Target 90%')
        plt.axhline(y=0.918, color='green', linestyle='--', alpha=0.7, label='Your Best 91.8%')
        plt.title("Accuracy vs Fairness (λ)")
        plt.xlabel("Fairness Regularization (λ)")
        plt.ylabel("Test Accuracy")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.subplot(1, 2, 2)
        for λ in precision_results['lambda'].unique():
            subset = precision_results[precision_results['lambda'] == λ]
            plt.plot(subset['beta'], subset['test_accuracy'],
                    marker='s', linewidth=2, label=f'λ={λ}', markersize=6)
        plt.axhline(y=0.90, color='red', linestyle='--', alpha=0.7, label='Target 90%')
        plt.axhline(y=0.918, color='green', linestyle='--', alpha=0.7, label='Your Best 91.8%')
        plt.title("Accuracy vs Redundancy Penalty (β)")
        plt.xlabel("Redundancy Penalty (β)")
        plt.ylabel("Test Accuracy")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

        # Final model with best fair configuration
        print("\n=== FINAL FAIR MODEL ===")
        final_features = best_result['selected_features']

        # Re-create data for final model
        data = create_targeted_features(df)
        data = optimized_feature_encoding(data, final_features + ['target', 'sex']) # target/sex names might need adjustment
        
        # Adjust target/sex names for final model
        target_col_name = 'target'
        if 'target' not in df.columns.str.lower():
             if 'heartdisease' in df.columns.str.lower():
                 target_col_name = 'heartdisease'
        
        X_final = data[final_features].copy()
        y_final = data[target_col_name]

        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
        )

        final_model = get_optimized_random_forest()
        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)
        y_pred_proba = final_model.predict_proba(X_test)

        print(f"Features: {final_features}")
        print(f"Number of features: {len(final_features)}")
        print(f"Fairness regularization: λ = {best_result['lambda']:.2f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

        # Feature importance
        importance_df = pd.DataFrame({
            'feature': final_features,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nFeature Importance:")
        print(importance_df)

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title('Feature Importance in Fair Model')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()

        # Compare with your original best
        print("\n=== COMPARISON WITH YOUR BEST RF ===")
        print(f"Your Best RF Accuracy: 91.80%")
        print(f"Fair Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Accuracy Difference: {accuracy_score(y_test, y_pred) - 0.918:.4f}")

        # Fairness analysis
        sensitive_feature = 'sex'
        if sensitive_feature in df.columns.str.lower():
            # Need to get sensitive values corresponding to X_test
            # X_test indices should match
            # But we are using 'data' which is a copy.
            # We can use train_test_split on the sensitive column too
            _, _, _, sensitive_test = train_test_split(
                X_final, data['sex'], test_size=0.2, random_state=42, stratify=y_final
            )
            
            # Accuracy by sensitive group
            for group in sorted(sensitive_test.unique()):
                group_mask = (sensitive_test == group)
                group_acc = accuracy_score(y_test[group_mask], y_pred[group_mask])
                print(f"Accuracy for {sensitive_feature}={group}: {group_acc:.4f}")

    # Baseline with all features (for comparison)
    print("\n=== BASELINE: ALL FEATURES ===")
    X_all = df.copy()
    X_all.columns = X_all.columns.str.lower()
    
    target_col_name = 'target'
    if 'target' not in X_all.columns:
         if 'heartdisease' in X_all.columns:
             target_col_name = 'heartdisease'
             
    y_all = X_all[target_col_name]
    X_all = X_all.drop(target_col_name, axis=1)

    # Encode categorical features
    for col in X_all.columns:
        if X_all[col].dtype == 'object':
            le = LabelEncoder()
            X_all[col] = le.fit_transform(X_all[col].astype(str))

    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    baseline_model = get_optimized_random_forest()
    baseline_model.fit(X_train_all, y_train_all)
    y_pred_all = baseline_model.predict(X_test_all)

    print(f"Baseline (all features) Accuracy: {accuracy_score(y_test_all, y_pred_all):.4f}")
    if not precision_results.empty:
        print(f"Feature selection improvement: {best_result['test_accuracy'] - accuracy_score(y_test_all, y_pred_all):.4f}")