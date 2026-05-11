"""
Data preprocessing module for UCI Student Dropout Dataset
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def clean_uci_columns(df):
    """Normaliza nombres de columnas del dataset UCI"""
    df.columns = df.columns.str.strip().str.lower().str.replace(r'[^a-z0-9]', '_', regex=True)
    return df

def prepare_uci_data(df, target_col='target', test_size=0.2, random_state=42):
    """
    Prepara datos UCI para clasificación binaria (Dropout vs Graduate)
    Retorna diccionario con datos, escalers y encoders para reutilización
    """
    df = clean_uci_columns(df.copy())
    
    # 1. Filtrar para clasificación binaria (eliminar 'Enrolled')
    if target_col in df.columns and df[target_col].dtype == 'object':
        df = df[df[target_col].isin(['dropout', 'graduate'])]
        df[target_col] = df[target_col].map({'dropout': 0, 'graduate': 1})
    elif target_col in df.columns:
        # Si ya es numérico, asegurar binario
        df[target_col] = df[target_col].replace({1: 0, 2: 1, 'Enrolled': np.nan}).dropna()
        
    if len(df) == 0:
        raise ValueError("Dataset vacío tras filtrar clases objetivo")
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 2. Separar tipos
    num_cols = X.select_dtypes(include='number').columns.tolist()
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    
    # 3. Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 4. Codificar categóricas (guardar encoders)
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        encoders[col] = le
        
    # 5. Escalar numéricas (guardar scaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Guardar medianas/modas para rellenar features no expuestas en UI
    medians = X_train[num_cols].median()
    modes = {c: X_train[c].mode()[0] for c in cat_cols}
    
    return {
        'X_train': X_train_scaled, 'X_test': X_test_scaled,
        'y_train': y_train, 'y_test': y_test,
        'feature_names': X_train.columns.tolist(),
        'scaler': scaler, 'encoders': encoders,
        'cat_cols': cat_cols, 'num_cols': num_cols,
        'medians': medians, 'modes': modes,
        'df_train': X_train  # Para referencia de distribución
    }
