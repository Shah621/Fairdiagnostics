from sklearn.metrics import accuracy_score
import pandas as pd

def evaluate_models(simple_pipelines, optimized_pipelines, fair_pipelines, X_test, y_test):
    """
    Evaluates trained models and prints results in the requested format.
    """
    
    print("\nSimple Model Results:")
    simple_results = []
    for name, pipeline in simple_pipelines.items():
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"The accuracy score achieved using {name} is: {acc*100:.2f} %")
        simple_results.append({'Model': name, 'Type': 'Simple', 'Accuracy': acc})
        
    print("\nJOA Optimized Model Results:")
    opt_results = []
    for name, pipeline in optimized_pipelines.items():
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"The accuracy score achieved using {name} is: {acc*100:.2f} %")
        opt_results.append({'Model': name, 'Type': 'JOA Optimized', 'Accuracy': acc})
        
    print("\nFair Model Results (MI with Sensitive Attribute):")
    fair_results = []
    
    # Check if we need to engineer features for test set
    # We do this lazily or once? 
    # Since different models might have different features, but the engineering function is same.
    # Let's create engineered test set once.
    from src.feature_engineering import create_targeted_features
    X_test_eng = create_targeted_features(X_test)
    
    for name, pipeline in fair_pipelines.items():
        # For fair pipelines, we need to subset X_test to the selected features
        if hasattr(pipeline, 'selected_features_'):
            # Determine if we use engineered features or raw
            if getattr(pipeline, 'is_engineered_', False):
                X_source = X_test_eng
            else:
                X_source = X_test
                
            X_test_subset = X_source[pipeline.selected_features_]
            y_pred = pipeline.predict(X_test_subset)
            acc = accuracy_score(y_test, y_pred)
            print(f"The accuracy score achieved using {name} is: {acc*100:.2f} %")
            fair_results.append({'Model': name, 'Type': 'Fair (MI)', 'Accuracy': acc})
        
    all_results = pd.DataFrame(simple_results + opt_results + fair_results)
    return all_results
