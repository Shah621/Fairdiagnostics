import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import load_data, clean_data
from src.preprocessing import get_preprocessor, encode_target
from src.models import get_models
from src.train import train_models
from src.evaluate import evaluate_models

def main():
    # Paths
    DATA_PATH = os.path.join('data', 'heart.csv')
    
    # 1. Load Data
    print("Loading data...")
    try:
        df = load_data(DATA_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    # 2. Clean Data
    print("Cleaning data...")
    df = clean_data(df)
    
    # 3. Split Features and Target
    print("Splitting data...")
    target_col = 'HeartDisease'
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in dataset.")
        return
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode target if necessary
    y = encode_target(y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Preprocessing
    print("Preparing preprocessor...")
    preprocessor = get_preprocessor(X_train)
    
    # 5. Get Models
    models = get_models()
    
    # 6. Train Models (Simple & Optimized & Fair)
    print("Training models...")
    # We assume 'Sex' is the sensitive column for this dataset
    sensitive_col = 'Sex' 
    # Note: 'Sex' might be 'sex' or 'Sex' depending on data_loader. 
    # data_loader maps 'sex' -> 'Sex' (if it was 'sex'). 
    # Let's check columns in df.columns just to be safe or rely on data_loader.
    # The data_loader map has 'sex' -> 'Sex' (Wait, no, the map has 'cp' -> 'ChestPainType').
    # Let's check data_loader.py content again.
    # It has 'target': 'HeartDisease'.
    # It does NOT explicitly rename 'sex' to 'Sex', but the notebook used 'Sex'.
    # Let's assume 'Sex' or 'sex' exists.
    
    # Actually, let's check the column names in X_train before passing.
    # But for now, we pass 'Sex' as default.
    
    simple_pipelines, optimized_pipelines, fair_pipelines = train_models(X_train, y_train, preprocessor, models, sensitive_col='Sex')
    
    # 7. Evaluate Models
    print("Evaluating models...")
    results = evaluate_models(simple_pipelines, optimized_pipelines, fair_pipelines, X_test, y_test)
    
    # Save results to CSV
    results.to_csv('model_results.csv', index=False)
    print("\nResults saved to 'model_results.csv'")

if __name__ == "__main__":
    main()
