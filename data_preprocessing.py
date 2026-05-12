"""
Data preprocessing module for student dropout prediction
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load data from CSV file"""
    return pd.read_csv(file_path)

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in the dataset
    Args:
        df: DataFrame
        strategy: 'mean', 'median', or 'drop'
    Returns:
        DataFrame without missing values
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'median':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
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
    df = handle_missing_values(df)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    X, encoders = encode_categorical(X, categorical_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    return (
        X_train_scaled, X_test_scaled, y_train, y_test,
        X.columns.tolist(), encoders, scaler
    )

# ✅ CORRECCIÓN APLICADA AQUÍ
if __name__ == "__main__":
    print("✅ Data preprocessing module loaded successfully")
if __name__ == "__main__":
    print("✅ Data preprocessing module loaded successfully")
