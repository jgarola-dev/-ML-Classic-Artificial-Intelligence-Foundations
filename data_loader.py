"""
Data loading and preprocessing from UCI/Kaggle Student Dropout Dataset
"""
import pandas as pd
import numpy as np
import streamlit as st
from io import StringIO
import requests
import zipfile
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

@st.cache_data(ttl=3600)
def load_kaggle_dataset(api_key=None):
    """
    Intenta cargar desde Kaggle (requiere API key opcional)
    Fallback: descarga directa si el dataset es público
    """
    try:
        # Opción A: Usar API de Kaggle (si está configurada)
        if api_key and os.environ.get('KAGGLE_USERNAME'):
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                'mahwiz/students-dropout-and-academic-success-dataset',
                path='temp_dataset',
                unzip=True
            )
            csv_files = list(Path('temp_dataset').glob('**/*.csv'))
            if csv_files:
                return pd.read_csv(csv_files[0])
        
        # Opción B: Descarga directa desde enlace público de Kaggle
        # Nota: Kaggle requiere login, pero algunos datasets permiten descarga directa
        url = "https://www.kaggle.com/api/v1/datasets/download/mahwiz/students-dropout-and-academic-success-dataset"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            # Guardar y extraer zip
            temp_dir = Path("temp_dataset")
            temp_dir.mkdir(exist_ok=True)
            zip_path = temp_dir / "kaggle_dataset.zip"
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            csv_files = list(temp_dir.glob("**/*.csv"))
            if csv_files:
                return pd.read_csv(csv_files[0])
                
    except Exception as e:
        print(f"⚠️ Kaggle no disponible: {e}")
    
    return None

@st.cache_data(ttl=3600)
def load_uci_dataset():
    """Carga desde UCI Archive (fuente oficial)"""
    try:
        url = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.csv"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))
    except Exception as e:
        print(f"⚠️ UCI no disponible: {e}")
        return None

def create_sample_dataset(n_samples=2000):
    """Dataset demo con columnas compatibles"""
    np.random.seed(42)
    return pd.DataFrame({
        'edad': np.random.randint(17, 35, n_samples),
        'gpa': np.random.uniform(1.0, 5.0, n_samples),
        'asistencia': np.random.uniform(40, 100, n_samples),
        'horas_estudio': np.random.uniform(0, 15, n_samples),
        'motivacion': np.random.choice(['Baja', 'Media', 'Alta'], n_samples, p=[0.3, 0.5, 0.2]),
        'primer_trimestre': np.random.choice(['Aprobado', 'Reprobado'], n_samples, p=[0.65, 0.35]),
        'socioeconomico': np.random.choice(['Bajo', 'Medio', 'Alto'], n_samples, p=[0.4, 0.4, 0.2]),
        'abandono': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    })

def preprocess_dataset(df):
    """Preprocesamiento seguro con mapeo de columnas UCI/Kaggle"""
    # Normalizar nombres a snake_case
    df.columns = df.columns.str.strip().str.lower().str.replace(r'[^a-z0-9]', '_', regex=True)
    
    # Mapear columnas comunes a nombres simplificados
    column_mapping = {
        'age': 'edad', 'age_at_enrollment': 'edad',
        'admission_grade': 'gpa', 'curricular_units_1st_sem_grade': 'gpa',
        'curricular_units_1st_sem_approved': 'asistencia',
        'unemployment_rate': 'horas_estudio',
        'status': 'abandono', 'target': 'abandono'
    }
    for uci_col, simple_name in column_mapping.items():
        if uci_col in df.columns and simple_name not in df.columns:
            df[simple_name] = df[uci_col]
    
    # Imputación segura
    num_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(include='object').columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for col in cat_cols:
        mode_val = df[col].mode()
        df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else 'desconocido')
    
    # Crear target binario 'abandono'
    target_candidates = ['target', 'status', 'abandono']
    target_found = next((c for c in target_candidates if c in df.columns), None)
    if target_found:
        df['abandono'] = df[target_found].astype(str).str.lower().apply(
            lambda x: 1 if 'dropout' in x or x in ['1', '0'] and x == '0' else 0
        )
    else:
        df['abandono'] = np.random.choice([0, 1], len(df), p=[0.3, 0.7])
    
    return df

def load_dataset(kaggle_api_key=None):
    """Función principal: intenta Kaggle → UCI → Demo"""
    # 1. Intentar Kaggle
    df = load_kaggle_dataset(kaggle_api_key)
    if df is not None:
        return preprocess_dataset(df), "Kaggle Dataset"
    
    # 2. Intentar UCI
    df = load_uci_dataset()
    if df is not None:
        return preprocess_dataset(df), "UCI Dataset"
    
    # 3. Fallback a demo
    return preprocess_dataset(create_sample_dataset()), "Dataset Demo"

if __name__ == "__main__":
    df, source = load_dataset()
    print(f"✅ Dataset cargado desde: {source} | Shape: {df.shape}")
