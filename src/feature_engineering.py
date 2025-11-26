import numpy as np
import pandas as pd

def create_targeted_features(df):
    """
    Create features that complement the known best RF parameters,
    based on the logic from exp.py.
    """
    df_eng = df.copy()
    # Ensure column names are lower case for consistent matching if needed, 
    # but our pipeline uses specific Capitalized names. 
    # We will map the logic to our column names:
    # Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, Ca, Thallium
    
    # Map exp.py names to our names
    # age -> Age
    # thalach -> MaxHR
    # cp -> ChestPainType
    # exang -> ExerciseAngina
    # oldpeak -> Oldpeak
    # trestbps -> RestingBP
    # chol -> Cholesterol
    # ca -> Ca
    # slope -> ST_Slope
    
    # 1. Age * MaxHR Interaction
    if 'Age' in df_eng.columns and 'MaxHR' in df_eng.columns:
        df_eng['Age_MaxHR_Interaction'] = df_eng['Age'] * df_eng['MaxHR'] / 100
        df_eng['MaxHR_Age_Ratio'] = df_eng['MaxHR'] / (df_eng['Age'] + 1e-5)

    # 2. Symptom Complexity (CP * ExAng + Oldpeak)
    # Note: ChestPainType and ExerciseAngina are categorical/ordinal.
    # We need to ensure they are numeric for this calculation.
    # In our pipeline, they might be strings before encoding.
    # We should probably apply this AFTER encoding or handle the conversion here.
    # Let's assume this runs BEFORE encoding (on raw data).
    # We need to map string values to numbers if they are strings.
    # But wait, data_loader loads them as they are. In heart.csv, they are often numeric codes.
    # Let's check data types.
    
    # If they are numeric, we can proceed.
    # If they are strings (e.g. 'TA', 'ATA'), we can't multiply directly.
    # The notebook had them as numeric (LabelEncoded).
    # We will try to convert to numeric if possible or skip.
    
    try:
        # Temporary numeric conversion for feature creation
        cp_num = pd.to_numeric(df_eng['ChestPainType'], errors='coerce').fillna(0)
        exang_num = pd.to_numeric(df_eng['ExerciseAngina'], errors='coerce').fillna(0)
        oldpeak_num = df_eng['Oldpeak']
        
        df_eng['Symptom_Complexity'] = cp_num * exang_num + oldpeak_num
    except KeyError:
        pass

    # 3. Non-linear transformations
    if 'Oldpeak' in df_eng.columns:
        df_eng['Oldpeak_Squared'] = df_eng['Oldpeak'] ** 2
        df_eng['Oldpeak_Log'] = np.log(df_eng['Oldpeak'] + 1)

    if 'MaxHR' in df_eng.columns:
        df_eng['MaxHR_Squared'] = df_eng['MaxHR'] ** 2

    # 4. Clinical Thresholds
    if 'RestingBP' in df_eng.columns:
        df_eng['Stage2_Hypertension'] = (df_eng['RestingBP'] >= 160).astype(int)

    if 'Cholesterol' in df_eng.columns:
        df_eng['Very_High_Chol'] = (df_eng['Cholesterol'] >= 280).astype(int)

    # 5. Vessel Disease Complexity
    if 'Ca' in df_eng.columns:
        df_eng['Vessel_Score'] = df_eng['Ca'] ** 2

    # 6. ST Complex
    if 'Oldpeak' in df_eng.columns and 'ST_Slope' in df_eng.columns:
        # Slope might be categorical (1,2,3 or Up,Flat,Down).
        slope_num = pd.to_numeric(df_eng['ST_Slope'], errors='coerce').fillna(0)
        df_eng['ST_Complex'] = df_eng['Oldpeak'] * (slope_num + 1)

    return df_eng
