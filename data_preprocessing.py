"""
Data preprocessing module for student dropout prediction
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def handle_missing_values(df, strategy='median'):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'drop':
        df = df.dropna()
    return df

def encode_categorical(df, categorical_cols=None):
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    encoders = {}
    df_encoded = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        # Manejar valores desconocidos asignándolos a "Unknown"
        df_encoded[col] = df_encoded[col].astype(str).replace('nan', 'Desconocido')
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le
    return df_encoded, encoders

def prepare_data(df, target_col='Abandono', test_size=0.2, random_state=42):
    df_clean = handle_missing_values(df.copy())
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    X_encoded, encoders = encode_categorical(X, categorical_cols)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Retornar objetos ajustados para reutilización en predicción
    return {
        'X_train': X_train_scaled, 'X_test': X_test_scaled,
        'y_train': y_train, 'y_test': y_test,
        'feature_names': X_encoded.columns.tolist(),
        'encoders': encoders, 'scaler': scaler,
        'categorical_cols': categorical_cols
    }
    
    return df


def encode_categorical(df, categorical_cols=None):
    """
    Encode categorical variables
    
    Args:
        df: DataFrame
        categorical_cols: List of categorical column names
    
    Returns:
        DataFrame with encoded categorical variables
        Dictionary of encoders for inverse transformation
    """
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    encoders = {}
    df_encoded = df.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
    
    return df_encoded, encoders


def scale_features(X_train, X_test=None):
    """
    Scale numerical features using StandardScaler
    
    Args:
        X_train: Training features
        X_test: Test features (optional)
    
    Returns:
        Scaled training features, scaled test features (if provided), and scaler object
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, scaler


def prepare_data(df, target_col, test_size=0.2, random_state=42):
    """
    Complete data preparation pipeline
    
    Args:
        df: DataFrame
        target_col: Name of target column
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test, feature names, encoders, scaler
    """
    # Handle missing values
    df = handle_missing_values(df)
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Get categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Encode categorical variables
    X, encoders = encode_categorical(X, categorical_cols)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    return (
        X_train_scaled, X_test_scaled, y_train, y_test,
        X.columns.tolist(), encoders, scaler
    )


if __name__ == "__main__":
    print("Data preprocessing module loaded successfully")
