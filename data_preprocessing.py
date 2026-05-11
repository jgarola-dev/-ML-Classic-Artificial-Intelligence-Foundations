"""
Data preprocessing module for student dropout prediction
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def create_realistic_dataset(n_samples=2000):
    """Genera dataset con correlaciones educativas reales para que el modelo aprenda"""
    np.random.seed(42)
    edad = np.random.randint(17, 35, n_samples)
    gpa = np.random.uniform(1.0, 5.0, n_samples)
    asistencia = np.random.uniform(40, 100, n_samples)
    horas = np.random.uniform(0, 15, n_samples)
    motivacion = np.random.choice(['Baja', 'Media', 'Alta'], n_samples, p=[0.3, 0.5, 0.2])
    trimestre1 = np.random.choice(['Aprobado', 'Reprobado'], n_samples, p=[0.65, 0.35])
    socioeconomico = np.random.choice(['Bajo', 'Medio', 'Alto'], n_samples, p=[0.4, 0.4, 0.2])
    
    # Lógica realista: bajo rendimiento + baja motivación = mayor riesgo de abandono
    risk_score = (
        (5.0 - gpa)/4.0 * 0.25 +
        (100 - asistencia)/60.0 * 0.20 +
        (15 - horas)/15.0 * 0.15 +
        np.where(motivacion=='Baja', 0.25, np.where(motivacion=='Media', 0.10, 0.0)) +
        np.where(trimestre1=='Reprobado', 0.20, 0.0) +
        np.where(socioeconomico=='Bajo', 0.10, 0.0)
    )
    # Umbral adaptativo para ~30% de abandono
    abandono = (risk_score > np.percentile(risk_score, 70)).astype(int)
    
    return pd.DataFrame({
        'Edad': edad, 'GPA': gpa, 'Asistencia': asistencia, 'Horas_Estudio': horas,
        'Motivacion': motivacion, 'Primer_Trimestre': trimestre1,
        'Socioeconomico': socioeconomico, 'Abandono': abandono
    })

def prepare_data(df, target_col='Abandono', test_size=0.2, random_state=42):
    """Prepara datos y guarda transformadores para reutilización en predicción"""
    df = df.copy()
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include='number').columns.tolist()
    
    # Codificación categórica (guardar encoders)
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        
    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Escalado numérico (guardar scaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Guardar estadísticas para rellenar features no expuestas en UI
    medians = X_train[num_cols].median().to_dict()
    modes = {c: X_train[c].mode()[0] for c in cat_cols}
    
    return {
        'X_train': X_train_scaled, 'X_test': X_test_scaled,
        'y_train': y_train, 'y_test': y_test,
        'feature_names': X.columns.tolist(),
        'encoders': encoders, 'scaler': scaler,
        'cat_cols': cat_cols, 'num_cols': num_cols,
        'medians': medians, 'modes': modes
    }
