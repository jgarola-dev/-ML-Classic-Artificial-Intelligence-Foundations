"""
Data preprocessing module
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def handle_missing_values(df, strategy='median'):
    num_cols = df.select_dtypes(include=[np.number]).columns
    if strategy == 'median': df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    elif strategy == 'mean': df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    return df

def encode_categorical(df, categorical_cols=None):
    if categorical_cols is None: categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    encoders = {}
    df_encoded = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
    return df_encoded, encoders

if __name__ == "__main__":
    print("✅ Data preprocessing module loaded successfully")
