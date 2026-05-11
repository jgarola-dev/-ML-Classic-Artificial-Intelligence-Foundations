"""
Data preprocessing module for student dropout prediction
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def handle_missing_values(df, strategy='mean'):
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
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    return X_train_scaled, scaler

def prepare_data(df, target_col, test_size=0.2, random_state=42):
    df = handle_missing_values(df)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    X, encoders = encode_categorical(X, categorical_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
