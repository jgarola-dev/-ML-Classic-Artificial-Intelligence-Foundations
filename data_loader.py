"""
Data loading and preprocessing from UCI Student Dropout Dataset
"""
import pandas as pd
import numpy as np
import streamlit as st
import requests
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

@st.cache_data(ttl=300)  # TTL reducido a 5 min para debugging
def load_uci_dataset():
    """Carga el dataset UCI con gestión de errores robusta y fallbacks múltiples"""
    # URLs alternativas en orden de prioridad
    urls = [
        "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.csv",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00697/student_dropout.csv"
    ]
    
    for url in urls:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return pd.read_csv(pd.io.common.BytesIO(response.content))
        except Exception as e:
            print(f"⚠️ Intento fallido con {url}: {e}")
            continue
    return None


def create_sample_dataset(n_samples=2000):
    """Dataset demo con columnas compatibles y target garantizado (en español)"""
    np.random.seed(42)
    return pd.DataFrame({
        'edad': np.random.randint(17, 35, n_samples),
        'gpa': np.random.uniform(1.0, 5.0, n_samples),
        'asistencia': np.random.uniform(40, 100, n_samples),
        'horas_estudio': np.random.uniform(0, 15, n_samples),
        'motivacion': np.random.choice(['Baja', 'Media', 'Alta'], n_samples, p=[0.3, 0.5, 0.2]),
        'primer_trimestre': np.random.choice(['Aprobado', 'Reprobado'], n_samples, p=[0.65, 0.35]),
        'socioeconomico': np.random.choice(['Bajo', 'Medio', 'Alto'], n_samples, p=[0.4, 0.4, 0.2]),
        'abandono': (np.random.uniform(0, 1, n_samples) > 0.7).astype(int)  # ~30% abandono
    })


def preprocess_dataset(df):
    """Limpieza segura sin eliminar columnas ni causar KeyError"""
    # 1. Normalizar nombres a snake_case (robusto a espacios y caracteres especiales)
    df.columns = df.columns.str.strip().str.lower().str.replace(r'[^a-z0-9]', '_', regex=True)
    
    # 2. Imputación segura (sin dropna agresivo)
    num_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(include='object').columns

    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    for col in cat_cols:
        mode_val = df[col].mode()
        fill_val = mode_val.iloc[0] if not mode_val.empty else 'desconocido'
        df[col] = df[col].fillna(fill_val)
    
    # 3. Crear target 'abandono' explícitamente con mapeo ROBUSTO
    target_candidates = ['target', 'status', 'Target', 'Status']
    target_found = next((c for c in target_candidates if c in df.columns), None)

    if target_found:
        # Función de mapeo case-insensitive y tolerante a variantes
        def map_target_to_binary(value):
            val_str = str(value).strip().lower()
            if val_str in ['dropout', 'abandono', 'drop out', 'deserción', '1']:
                return 1
            elif val_str in ['graduate', 'graduated', 'graduado', 'enrolled', 'matriculado', 'active', '0']:
                return 0
            else:
                # Fallback conservador: si no se reconoce, asumir 0 (no abandono)
                return 0
        
        df['abandono'] = df[target_found].apply(map_target_to_binary)
    else:
        # Fallback: generar target aleatorio balanceado (~30% abandono)
        df['abandono'] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
        
    return df


def load_dataset():
    """Orquestador de carga: intenta UCI, fallback a demo"""
    df = load_uci_dataset()
    if df is not None:
        return preprocess_dataset(df), "UCI Dataset"
    return preprocess_dataset(create_sample_dataset()), "Dataset Demo"
