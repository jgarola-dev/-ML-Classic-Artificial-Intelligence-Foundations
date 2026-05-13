"""
Data preprocessing module for student dropout prediction
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def handle_missing_values(df, strategy='median'):
    """
    Manejo de valores faltantes con estrategias configurables
    
    Args:
        df: DataFrame de pandas
        strategy: 'drop' | 'mean' | 'median' | 'mode'
    
    Returns:
        DataFrame con valores imputados o filas eliminadas
    """
    if strategy == 'drop':
        return df.dropna()
    
    elif strategy == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
    elif strategy == 'median':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
    elif strategy == 'mode':
        for col in df.columns:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])
    
    return df


def encode_categorical(df, categorical_cols=None, handle_unknown='ignore'):
    """
    Codificación de variables categóricas con LabelEncoder
    
    Args:
        df: DataFrame de pandas
        categorical_cols: Lista de columnas a codificar (auto-detecta si None)
        handle_unknown: 'ignore' | 'error' para valores no vistos
    
    Returns:
        df_encoded: DataFrame con columnas codificadas
        encoders: Diccionario {col: LabelEncoder} para inversa si es necesario
    """
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    encoders = {}
    df_encoded = df.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        # Convertir a string para evitar errores con valores mixtos
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
    
    return df_encoded, encoders


def normalize_numeric(df, columns=None, method='standard'):
    """
    Normalización de variables numéricas
    
    Args:
        df: DataFrame de pandas
        columns: Lista de columnas a normalizar (auto-detecta numéricas si None)
        method: 'standard' (z-score) | 'minmax' (0-1) | 'robust' (IQR)
    
    Returns:
        df_normalized: DataFrame con columnas normalizadas
        scalers: Diccionario {col: scaler_object} para transformación inversa
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    scalers = {}
    df_normalized = df.copy()
    
    scaler_map = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }
    
    for col in columns:
        scaler = scaler_map.get(method, StandardScaler())
        df_normalized[[col]] = scaler.fit_transform(df_normalized[[col]])
        scalers[col] = scaler
    
    return df_normalized, scalers


if __name__ == "__main__":
    """Ejecución directa para pruebas del módulo"""
    print("✅ Data preprocessing module loaded successfully")
    print("📦 Funciones disponibles:")
    print("   • handle_missing_values(df, strategy='median')")
    print("   • encode_categorical(df, categorical_cols=None)")
    print("   • normalize_numeric(df, columns=None, method='standard')")
