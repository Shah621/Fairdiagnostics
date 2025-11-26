import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

def main():
    # Load data
    df = pd.read_csv(os.path.join('data', 'heart.csv'))
    
    # Rename columns to match notebook (simplified)
    rename_map = {
        'cp': 'ChestPainType',
        'trestbps': 'RestingBP',
        'chol': 'Cholesterol',
        'fbs': 'FastingBS',
        'restecg': 'RestingECG',
        'thalach': 'MaxHR',
        'exang': 'ExerciseAngina',
        'oldpeak': 'Oldpeak',
        'slope': 'ST_Slope',
        'ca': 'Ca',
        'thal': 'Thallium',
        'target': 'HeartDisease',
        'condition': 'HeartDisease'
    }
    df = df.rename(columns=rename_map)
    
    # Notebook-style Preprocessing: Label Encoding ONLY, NO Scaling
    le = LabelEncoder()
    
    # Columns to encode (categorical)
    cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    # Check if columns exist before encoding
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
            
    # Target
    target_col = 'HeartDisease'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split (Notebook used random_state=0 often, or default)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    # Train KNN (n=7)
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, y_train)
    
    # Predict
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Notebook-style KNN (No Scaling, Label Encoded) Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
