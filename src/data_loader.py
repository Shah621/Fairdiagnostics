import pandas as pd
import os

def load_data(filepath):
    """
    Loads the heart disease dataset from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} was not found.")
    
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    """
    Cleans the dataframe by renaming columns to match standard conventions
    used in the analysis notebook.
    
    Args:
        df (pd.DataFrame): Raw dataframe.
        
    Returns:
        pd.DataFrame: Cleaned dataframe with standardized column names.
    """
    # Mapping based on the notebook's usage vs potential raw data differences
    # The notebook uses: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, HeartDisease
    # We will ensure the dataframe has these columns.
    
    # Common variations found in heart disease datasets
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
        'ca': 'Ca', # Notebook had 'ca' / 'Number of vessels fluro'
        'thal': 'Thallium', # Notebook had 'thal'
        'target': 'HeartDisease',
        'condition': 'HeartDisease',
        'sex': 'Sex',
        'age': 'Age'
    }
    
    # Apply renaming if columns exist
    df = df.rename(columns=rename_map)
    
    # Ensure all required columns are present (based on notebook features)
    # Note: 'Ca' and 'Thallium' were in some parts of the notebook but maybe not all datasets have them.
    # The UCI Heart Disease dataset usually has 14 columns.
    # The notebook used: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, HeartDisease
    # It seems the notebook used a dataset that might be a combination or a specific version (like the one on Kaggle).
    # We will proceed with the columns available in the loaded df.
    
    return df
