"""
Data loading and preprocessing from UCI Student Dropout Dataset
"""
import pandas as pd
import numpy as np
import requests
import zipfile
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_uci_dataset():
    """Load the UCI Student Dropout and Academic Success dataset"""
    try:
        url = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        temp_dir = Path("temp_dataset")
        temp_dir.mkdir(exist_ok=True)
        
        zip_path = temp_dir / "dataset.zip"
        with open(zip_path, 'wb') as f:
            f.write(response.content)
            
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        csv_files = list(temp_dir.glob("**/*.csv"))
        if csv_files:
            df = pd.read_csv(csv_files[0])
            return df
    except Exception as e:
        print(f"⚠️ Fallback a demo: {e}")
    return None

def create_sample_dataset():
    """Create a realistic sample dataset if UCI is unavailable"""
    np.random.seed(42)
    n_samples = 4424
    data = {
        'Admission_Grade': np.random.uniform(95, 150, n_samples),
        'Curricular_Units_1st_Sem_Grade': np.random.uniform(0, 20, n_samples),
        'Age': np.random.randint(18, 50, n_samples),
        'Unemployment_Rate': np.random.uniform(2, 12, n_samples),
        'Status': np.random.choice(['Dropout', 'Graduate', 'Enrolled'], n_samples, p=[0.3, 0.5, 0.2])
    }
    return pd.DataFrame(data)

def load_dataset():
    """Main function to load dataset"""
    df = load_uci_dataset()
    if df is not None:
        return df, "UCI Dataset"
    else:
        return create_sample_dataset(), "Sample Dataset"

def preprocess_dataset(df):
    """Basic preprocessing and target mapping"""
    # Clean missing values
    df = df.dropna(thresh=len(df) * 0.5, axis=1)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        mode_val = df[col].mode()
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val[0])
            
    # Map target to binary 'Abandono'
    target_col = None
    for col in ['Target', 'Status', 'Y', 'target', 'status']:
        if col in df.columns:
            target_col = col
            break
            
    if target_col:
        df['Abandono'] = df[target_col].apply(
            lambda x: 1 if 'dropout' in str(x).lower() else 0
        )
    elif 'Abandono' not in df.columns:
        df['Abandono'] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
        
    return df

if __name__ == "__main__":
    df, source = load_dataset()
    print(f"✅ Dataset cargado desde: {source} | Shape: {df.shape}")
