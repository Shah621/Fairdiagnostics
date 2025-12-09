from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

def get_preprocessor(X, use_scaling=True):
    """
    Creates a column transformer for preprocessing numerical and categorical features.
    
    Args:
        X (pd.DataFrame): Training data to infer column types.
        use_scaling (bool): Whether to apply StandardScaler to numerical features.
        
    Returns:
        ColumnTransformer: The preprocessing pipeline.
    """
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Preprocessing for numerical data
    if use_scaling:
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
    else:
        # No scaling for tree-based models
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
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


def get_preprocessor_for_model(X, model_name):
    """
    Get appropriate preprocessor based on model type.
    
    Tree-based models (Random Forest) and Naive Bayes don't need scaling.
    Distance-based and gradient-based models (SVM, LR, NN) need scaling.
    
    Args:
        X (pd.DataFrame): Training data
        model_name (str): Name of the model
        
    Returns:
        ColumnTransformer: Appropriate preprocessor
    """
    # Models that don't need scaling
    no_scaling_models = ["Random Forest", "Naive Bayes"]
    
    use_scaling = model_name not in no_scaling_models
    
    return get_preprocessor(X, use_scaling=use_scaling)


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

