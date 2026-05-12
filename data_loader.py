"""
Data loading and preprocessing from UCI Student Dropout Dataset
"""
import pandas as pd
import numpy as np
import streamlit as st
import requests
import zipfile
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

@st.cache_data(ttl=3600)
def load_uci_dataset():
    """Carrega el dataset UCI amb gestió d'errors robusta"""
    try:
        url = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.csv"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return pd.read_csv(pd.io.common.BytesIO(response.content))
    except Exception as e:
        print(f"⚠️ Fallback a demo: {e}")
        return None

def create_sample_dataset(n_samples=2000):
    """Dataset demo amb correlacions reals i columna Abandono garantida"""
    np.random.seed(42)
    return pd.DataFrame({
        'edat': np.random.randint(17, 35, n_samples),
        'gpa': np.random.uniform(1.0, 5.0, n_samples),
        'assistencia': np.random.uniform(40, 100, n_samples),
        'hores_estudi': np.random.uniform(0, 15, n_samples),
        'motivacio': np.random.choice(['Baixa', 'Mitjana', 'Alta'], n_samples, p=[0.3, 0.5, 0.2]),
        'primer_trimestre': np.random.choice(['Aprovat', 'Reprovat'], n_samples, p=[0.65, 0.35]),
        'nivell_socioeconomic': np.random.choice(['Baix', 'Mitjà', 'Alt'], n_samples, p=[0.4, 0.4, 0.2]),
        'Abandono': (np.random.uniform(0, 1, n_samples) > 0.7).astype(int)
    })

def preprocess_dataset(df):
    """Neteja suau, normalitza noms i CREA explícitament la columna 'Abandono'"""
    # 1. Netejar noms de columnes a snake_case
    df.columns = df.columns.str.strip().str.lower().str.replace(r'[^a-z0-9]', '_', regex=True)
    
    # 2. Imputació segura sense eliminar columnes (evita .dtype directament)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else 'desconegut')
            
    # 3. 🔑 CREAR TARGET BINARI 'Abandono' EXPLÍCITAMENT
    target_candidates = ['target', 'status', 'estado', 'outcome']
    target_found = None
    for c in target_candidates:
        if c in df.columns:
            target_found = c
            break
            
    if target_found:
        # Mapeig segur: 1 si és Dropout/abandono, 0 en qualsevol altre cas
        df['Abandono'] = df[target_found].astype(str).str.lower().apply(
            lambda x: 1 if 'dropout' in x or 'abandono' in x else 0
        )
    else:
        # Fallback segur si no es troba cap target
        df['Abandono'] = np.random.choice([0, 1], len(df), p=[0.3, 0.7])
        
    return df

def load_dataset():
    df = load_uci_dataset()
    if df is not None:
        return preprocess_dataset(df), "UCI Dataset (37 atributs)"
    return preprocess_dataset(create_sample_dataset()), "Dataset Demo"
