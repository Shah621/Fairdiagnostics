from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

def get_preprocessor(X):
    """
    Creates a column transformer for preprocessing numerical and categorical features.
    
    Args:
        X (pd.DataFrame): Training data to infer column types.
        
    Returns:
        ColumnTransformer: The preprocessing pipeline.
    """
    # Identify numerical and categorical columns
    # We exclude the target variable 'HeartDisease' which should be separated before calling this.
    
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Preprocessing for numerical data
    # Notebook didn't explicitly scale everything, but for SVM/KNN/NN it's crucial.
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
        
    return preprocessor

def encode_target(y):
    """
    Encodes the target variable if it's categorical.
    
    Args:
        y (pd.Series): Target variable.
        
    Returns:
        pd.Series: Encoded target.
    """
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    return y
