from sklearn.pipeline import Pipeline
from src.optimization import optimize_model
from src.fairness import FairFeatureSelector
import pandas as pd

def train_models(X_train, y_train, preprocessor, models, sensitive_col='Sex'):
    """
    Trains simple, JOA optimized, and Fair versions of models.
    
    Args:
        X_train (pd.DataFrame): Training features (must include sensitive col for fairness).
        y_train (pd.Series): Training target.
        preprocessor (ColumnTransformer): Preprocessing pipeline.
        models (dict): Dictionary of base models.
        sensitive_col (str): Name of the sensitive attribute.
        
    Returns:
        tuple: (simple_pipelines, optimized_pipelines, fair_pipelines)
    """
    simple_pipelines = {}
    optimized_pipelines = {}
    fair_pipelines = {}
    
    # 1. Prepare Data for Optimization (Fit preprocessor once)
    # Note: Preprocessor usually drops or encodes columns. 
    # For fairness, we need the sensitive column to be present in X_train passed to FairFeatureSelector.
    # Our current preprocessor (from src/preprocessing.py) likely transforms everything to numpy array.
    # We need to be careful. 
    # Strategy:
    # - Simple/Optimized: Use standard preprocessor (which might include sensitive col as feature).
    # - Fair: Use FairFeatureSelector FIRST (on raw DF), then Preprocessor (on selected features), then Model.
    
    # Check if sensitive col is in X_train
    # Note: X_train is the raw dataframe here.
    
    # Fit preprocessor for optimization usage
    preprocessor.fit(X_train)
    X_train_transformed = preprocessor.transform(X_train)
    
    for name, model in models.items():
        print(f"Processing {name}...")
        
        # --- 1. Simple Model ---
        print(f"  Training Simple {name}...")
        simple_clf = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', model)])
        simple_clf.fit(X_train, y_train)
        simple_pipelines[name] = simple_clf
        
        # --- 2. JOA Optimized Model ---
        print(f"  Optimizing {name} with JOA...")
        optimized_model = optimize_model(name, X_train_transformed, y_train)
        
        if optimized_model is not None:
            opt_clf = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', optimized_model)])
            opt_clf.fit(X_train, y_train)
            optimized_pipelines[name] = opt_clf
        else:
            print(f"  Skipping optimization for {name} (config not found).")
            
        # --- 3. Fair Model (Fair Feature Selection + JOA Optimized Model) ---
        print(f"  Training Fair {name}...")
        
        if sensitive_col in X_train.columns:
            # 1. Feature Engineering
            from src.feature_engineering import create_targeted_features
            X_train_eng = create_targeted_features(X_train)
            
            # 2. Optimize Fair Parameters (Lambda/Beta/K)
            # We will do a small grid search to find the best feature set for this model
            # This mimics the "Precision" approach in exp.py
            best_acc = -1
            best_fs = None
            best_model = None
            best_features = []
            
            # Search space (simplified for speed)
            lambdas = [0.1, 0.3]
            betas = [0.0, 0.1]
            ks = [8, 10, 12]
            
            # Use a validation split for this search
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            from src.preprocessing import get_preprocessor
            
            X_f_train, X_f_val, y_f_train, y_f_val = train_test_split(X_train_eng, y_train, test_size=0.2, random_state=42)
            
            # Base model to use (Optimized or Simple)
            base_model = optimized_model if optimized_model is not None else model
            
            print(f"    Searching for best Fair parameters for {name}...")
            
            for lam in lambdas:
                for beta in betas:
                    for k in ks:
                        try:
                            fs = FairFeatureSelector(sensitive_col=sensitive_col, k=k, lambda_val=lam, beta_val=beta)
                            fs.fit(X_f_train, y_f_train)
                            feats = fs.selected_features_
                            
                            if len(feats) < 2: continue
                            
                            # Train temp model
                            X_sub_train = X_f_train[feats]
                            X_sub_val = X_f_val[feats]
                            
                            prep = get_preprocessor(X_sub_train)
                            clf = Pipeline(steps=[('preprocessor', prep), ('classifier', base_model)])
                            clf.fit(X_sub_train, y_f_train)
                            
                            acc = accuracy_score(y_f_val, clf.predict(X_sub_val))
                            
                            if acc > best_acc:
                                best_acc = acc
                                best_fs = fs
                                best_features = feats
                                # We don't save the model, we retrain on full train later
                        except Exception as e:
                            # print(f"Error in search: {e}")
                            pass
            
            print(f"    Best Fair Params: Acc={best_acc:.4f}, Feats={len(best_features)}")
            
            # 3. Train Final Fair Model with Best Features
            X_train_final = X_train_eng[best_features]
            
            # Preprocessor for final features
            fair_preprocessor = get_preprocessor(X_train_final)
            
            # Model
            model_to_use = optimized_model if optimized_model is not None else model
            from sklearn.base import clone
            model_to_use = clone(model_to_use)
            
            fair_clf = Pipeline(steps=[('preprocessor', fair_preprocessor),
                                       ('classifier', model_to_use)])
            fair_clf.fit(X_train_final, y_train)
            fair_pipelines[name] = fair_clf
            
            # Store metadata
            fair_clf.selected_features_ = best_features
            fair_clf.is_engineered_ = True # Flag to tell evaluate to engineer features
        else:
            print(f"    Warning: Sensitive column '{sensitive_col}' not found. Skipping Fair model.")

    return simple_pipelines, optimized_pipelines, fair_pipelines
