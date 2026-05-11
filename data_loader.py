"""
Data loading and preprocessing from UCI Student Dropout Dataset
"""

import pandas as pd
import numpy as np
import streamlit as st
from io import StringIO
import requests
import zipfile
import os
from pathlib import Path

# Cache the data loading
@st.cache_data
def load_uci_dataset():
    """
    Load the UCI Student Dropout and Academic Success dataset
    Source: https://archive.ics.uci.edu/dataset/697
    """
    try:
        # Define the dataset URL
        url = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
        
        # Download and extract
        print("Descargando dataset de UCI...")
        response = requests.get(url, timeout=30)
        
        # Create temporary directory for extraction
        temp_dir = Path("temp_dataset")
        temp_dir.mkdir(exist_ok=True)
        
        # Save and extract zip
        zip_path = temp_dir / "dataset.zip"
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find and load the CSV file
        csv_files = list(temp_dir.glob("**/*.csv"))
        
        if csv_files:
            df = pd.read_csv(csv_files[0])
            print(f"Dataset cargado desde: {csv_files[0]}")
            return df
        else:
            return None
            
    except Exception as e:
        print(f"Error descargando dataset: {e}")
        return None


def create_sample_dataset():
    """
    Create a sample dataset if the UCI dataset is not available
    """
    np.random.seed(42)
    n_samples = 4424  # Approximate size of original dataset
    
    data = {
        'Marital_Status': np.random.choice([1, 2, 3, 4, 5, 6], n_samples),
        'Application_Mode': np.random.choice([1, 2, 3, 4, 5, 6], n_samples),
        'Application_Order': np.random.randint(1, 10, n_samples),
        'Course': np.random.randint(1, 33, n_samples),
        'Daytime_Evening_Attendance': np.random.choice([1, 2], n_samples),
        'Previous_Qualification': np.random.randint(1, 35, n_samples),
        'Mothers_Qualification': np.random.randint(1, 16, n_samples),
        'Fathers_Qualification': np.random.randint(1, 16, n_samples),
        'Mothers_Occupation': np.random.randint(1, 22, n_samples),
        'Fathers_Occupation': np.random.randint(1, 22, n_samples),
        'Admission_Grade': np.random.uniform(95, 150, n_samples),
        'Curricular_Units_1st_Sem': np.random.randint(0, 20, n_samples),
        'Curricular_Units_1st_Sem_Approved': np.random.randint(0, 20, n_samples),
        'Curricular_Units_1st_Sem_Grade': np.random.uniform(0, 20, n_samples),
        'Curricular_Units_2nd_Sem': np.random.randint(0, 20, n_samples),
        'Curricular_Units_2nd_Sem_Approved': np.random.randint(0, 20, n_samples),
        'Curricular_Units_2nd_Sem_Grade': np.random.uniform(0, 20, n_samples),
        'Scholarship': np.random.choice([0, 1], n_samples),
        'International_Student': np.random.choice([0, 1], n_samples),
        'Age': np.random.randint(18, 50, n_samples),
        'Tuition_Fees_Paid': np.random.choice([0, 1], n_samples),
        'Inflation_Rate': np.random.uniform(-3, 3, n_samples),
        'Unemployment_Rate': np.random.uniform(2, 12, n_samples),
        'Status': np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2])  # 0: Dropout, 1: Enrolled, 2: Graduate
    }
    
    df = pd.DataFrame(data)
    df['Status_Label'] = df['Status'].map({0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'})
    
    return df


def load_dataset():
    """
    Main function to load dataset - tries UCI first, falls back to sample
    """
    try:
        df = load_uci_dataset()
        if df is not None:
            return df, "UCI Dataset"
    except:
        pass
    
    # Fallback to sample dataset
    df = create_sample_dataset()
    return df, "Sample Dataset"


def preprocess_dataset(df):
    """
    Basic preprocessing of the dataset
    """
    # Handle missing values
    df = df.dropna(thresh=len(df) * 0.5, axis=1)  # Drop columns with >50% missing
    df = df.fillna(df.mean(numeric_only=True))  # Fill numeric columns with mean
    df = df.fillna(df.mode().iloc[0])  # Fill categorical with mode
    
    # Ensure target column exists
    if 'Status' not in df.columns:
        # Try to find target column with different names
        potential_targets = ['Target', 'Y', 'target', 'status', 'outcome', 'class']
        for col in potential_targets:
            if col in df.columns:
                df['Status'] = df[col]
                break
        if 'Status' not in df.columns:
            # Create a dummy target if none exists
            df['Status'] = np.random.choice([0, 1, 2], len(df), p=[0.3, 0.5, 0.2])
    
    # Map status to labels if numeric
    if df['Status'].dtype in ['int64', 'float64']:
        unique_vals = sorted(df['Status'].unique())
        if len(unique_vals) == 2:
            df['Status_Label'] = df['Status'].map({unique_vals[0]: 'Dropout', unique_vals[1]: 'Enrolled'})
        elif len(unique_vals) == 3:
            df['Status_Label'] = df['Status'].map({
                unique_vals[0]: 'Dropout', 
                unique_vals[1]: 'Enrolled', 
                unique_vals[2]: 'Graduate'
            })
    
    return df


if __name__ == "__main__":
    df, source = load_dataset()
    print(f"Dataset cargado desde: {source}")
    print(df.head())
