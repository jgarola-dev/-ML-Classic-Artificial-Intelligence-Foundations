"""
Streamlit app for student dropout prediction
"""
import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 🔧 Fix para imports locales
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from data_preprocessing import handle_missing_values, encode_categorical
from model import DropoutPredictor, compare_models

st.set_page_config(
    page_title="Predicción de Abandono Escolar",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-title { color: #1f77b4; text-align: center; font-size: 2.5em; margin-bottom: 10px; }
.metric-box { background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

def create_sample_dataset(n_samples=200):
    np.random.seed(42)
    return pd.DataFrame({
        'Edad': np.random.randint(15, 25, n_samples),
        'GPA': np.random.uniform(1.5, 4.0, n_samples),
        'Asistencia': np.random.uniform(50, 100, n_samples),
        'Horas_Estudio': np.random.uniform(0, 10, n_samples),
        'Socioeconomico': np.random.choice(['Bajo', 'Medio', 'Alto'], n_samples),
        'Primer_Trimestre': np.random.choice(['Aprobado', 'Reprobado'], n_samples),
        'Motivacion': np.random.choice(['Baja', 'Media', 'Alta'], n_samples),
        'Abandono': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })

# Session state
if 'df' not in st.session_state: st.session_state.df = None
if 'results' not in st.session_state: st.session_state.results = None
if 'feature_names' not in st.session_state: st.session_state.feature_names = None
if 'scaler' not in st.session_state: st.session_state.scaler = None
if 'encoders' not in st.session_state: st.session_state.encoders = None

# Sidebar
st.sidebar.title("📋 Navegación")
page = st.sidebar.radio("Selecciona una sección:", ["🏠 Inicio", "📊 Análisis EDA", "🤖 Entrenamiento", "🔮 Predicción"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Cargar Datos")
uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV", type=['csv'])
if uploaded_file is not None:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Archivo cargado correctamente")
elif st.sidebar.button("Usar datos de demostración"):
    st.session_state.df = create_sample_dataset()
    st.sidebar.success("✅ Datos de demostración cargados")

if st.session_state.df is not None:
    st.sidebar.info(f"Dataset: {st.session_state.df.shape[0]} filas × {st.session_state.df.shape[1]} columnas")

# ==================== PAGES ====================
if page == "🏠 Inicio":
    st.markdown('<div class="main-title">🎓 Predicción de Abandono Escolar</div>', unsafe_allow_html=True)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ## 📚 Sobre el Proyecto
        Este proyecto implementa un modelo de ML clásico para predecir el riesgo de abandono escolar.
        ### 🎯 Objetivos
        - Identificar estudiantes en riesgo
        - Proporcionar recomendaciones de intervención
        - Analizar factores clave del abandono
        ### 📋 Formulación
        - **Tipo:** Supervisado | **Tarea:** Clasificación binaria
        - **Target:** Abandono (0/1) | **Métrica:** F1-Score, ROC-AUC
        """)
    with col2:
        st.markdown("""
        ## 🤖 Modelos Implementados
        1. Logistic Regression
        2. Random Forest
        3. Gradient Boosting
        4. Support Vector Machine (SVM)
        """)
    c1, c2, c3 = st.columns(3)
    c1.metric("📊 Modelos", "4", "Clasificadores")
    c2.metric("📈 Métricas", "6+", "Evaluación")
    c3.metric("🎓 Categoría", "ML Clásico", "Fundació URV")

elif page == "📊 Análisis EDA":
    st.title("📊 Análisis Exploratorio de Datos")
    if st.session_state.df is None:
        st.warning("⚠️ Carga datos primero")
    else:
        df = st.session_state.df
        with st.expander("📋 Vista previa de datos"):
            st.dataframe(df.head(10), use_container_width=True)
        st.subheader("📊 Estadísticas Descriptivas")
        st.dataframe(df.describe(), use_container_width=True)
        st.markdown("---")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Abandono' in num_cols: num_cols.remove('Abandono')
        if num_cols:
            sel = st.selectbox("Selecciona característica numérica:", num_cols)
            if 'Abandono' in df.columns:
                fig = px.box(df, x='Abandono', y=sel, color='Abandono', title=f"Distribución de {sel} por Abandono")
                st.plotly_chart(fig, use_container_width=True)
        if 'Abandono' in df.columns:
            counts = df['Abandono'].value_counts()
            fig = px.pie(values=counts.values, names=['No Abandono', 'Abandono'], title="Distribución de Clases")
            st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Entrenamiento":
    st.title("🤖 Entrenamiento de Modelos")
    if st.session_state.df is None:
        st.warning("⚠️ Carga datos primero")
    else:
        test_size = st.slider("Tamaño del conjunto de prueba (%)", 10, 40, 20) / 100
        if st.button("🚀 Entrenar Modelos", key='train'):
            with st.spinner("Entrenando modelos..."):
                try:
                    from sklearn.model_selection import train_test_split
                    from sklearn.preprocessing import StandardScaler
                    
                    df_clean = handle_missing_values(st.session_state.df.copy())
                    X = df_clean.drop(columns=['Abandono'])
                    y = df_clean['Abandono']
                    
                    X_encoded, encoders = encode_categorical(X)
                    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size, random_state=42, stratify=y)
                    
                    # ✅ GUARDAR SCALER Y ENCODERS EN SESIÓN
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    st.session_state.results = compare_models(X_train_scaled, X_test_scaled, y_train, y_test)
                    st.session_state.feature_names = X_encoded.columns.tolist()
                    st.session_state.scaler = scaler
                    st.session_state.encoders = encoders
                    
                    st.success("✅ Modelos entrenados correctamente. Pipeline guardado.")
                except Exception as e:
                    st.error(f"❌ Error al entrenar: {str(e)}")
        
        if st.session_state.results:
            st.subheader("📊 Resultados de Modelos")
            rows = []
            for model_name, result in st.session_state.results.items():
                m = result['metrics']
                rows.append({'Modelo': model_name.replace('_', ' ').title(), 'Accuracy': f"{m['accuracy']:.4f}", 'F1-Score': f"{m['f1']:.4f}", 'ROC-AUC': f"{m.get('roc_auc', 0):.4f}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            
            st.subheader("🔍 Importancia de Características")
            sel = st.selectbox("Selecciona modelo:", list(st.session_state.results.keys()))
            imp = st.session_state.results[sel]['model'].get_feature_importance(st.session_state.feature_names, top_n=10)
            if imp is not None:
                st.plotly_chart(px.bar(imp, x='importance', y='feature', orientation='h', title=f"Top 10 Características ({sel.replace('_', ' ').title()})"), use_container_width=True)

elif page == "🔮 Predicción":
    st.title("🔮 Predicción Individual")
    if st.session_state.df is None or st.session_state.results is None:
        st.warning("⚠️ Carga datos y entrena los modelos primero")
    else:
        st.subheader("📋 Ingresa los datos del estudiante")
        col1, col2 = st.columns(2)
        with col1:
            edad = st.number_input("Edad", 15, 25, 18)
            gpa = st.slider("GPA", 1.5, 4.0, 3.0)
            asistencia = st.slider("Asistencia (%)", 50, 100, 85)
            horas = st.slider("Horas de Estudio (semanales)", 0, 10, 5)
        with col2:
            socio = st.selectbox("Nivel Socioeconómico", ['Bajo', 'Medio', 'Alto'])
            trim = st.selectbox("Desempeño 1er Trimestre", ['Aprobado', 'Reprobado'])
            motiv = st.selectbox("Nivel de Motivación", ['Baja', 'Media', 'Alta'])
            
        if st.button("🔮 Realizar Predicción"):
            try:
                # 1. Crear DataFrame de entrada
                pred_df = pd.DataFrame({
                    'Edad': [edad], 'GPA': [gpa], 'Asistencia': [asistencia],
                    'Horas_Estudio': [horas], 'Socioeconomico': [socio],
                    'Primer_Trimestre': [trim], 'Motivacion': [motiv]
                })
                
                # 2. Alinear columnas EXACTAMENTE con el entrenamiento
                pred_df = pred_df.reindex(columns=st.session_state.feature_names)
                # Rellenar features faltantes (si el dataset original tiene más columnas)
                pred_df = pred_df.fillna(0)
                
                # 3. Aplicar EXACTAMENTE los mismos encoders del entrenamiento
                for col, encoder in st.session_state.encoders.items():
                    if col in pred_df.columns:
                        val = str(pred_df[col].values[0]).strip()
                        # Manejo seguro de categorías no vistas
                        if val in encoder.classes_:
                            pred_df[col] = encoder.transform([val])[0]
                        else:
                            pred_df[col] = 0  # o encoder.classes_[0]
                            
                # 4. Escalar CON EL MISMO SCALER (CRÍTICO: usar transform, NO fit_transform)
                X_sample = st.session_state.scaler.transform(pred_df)
                
                # 5. Predecir
                st.markdown("---")
                st.subheader("🎯 Resultados de Predicción")
                cols = st.columns(4)
                for idx, (model_name, result) in enumerate(st.session_state.results.items()):
                    with cols[idx]:
                        model_obj = result['model']
                        pred = model_obj.predict(X_sample)[0]
                        proba = model_obj.predict_proba(X_sample)[0]
                        
                        # ✅ Extraer probabilidad correcta para clase 1 (Abandono)
                        if 1 in model_obj.model.classes_:
                            idx_1 = np.where(model_obj.model.classes_ == 1)[0][0]
                            prob = proba[idx_1]
                        else:
                            prob = 0.0
                            
                        rec = model_obj.get_recommendation(pred, prob)
                        st.markdown(f"### {model_name.replace('_', ' ').title()}")
                        st.metric("Probabilidad de Abandono", f"{prob:.1%}")
                        st.markdown(rec)
                        
            except Exception as e:
                st.error(f"❌ Error en predicción: {str(e)}")
                st.exception(e)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9em;">
 <p>🎓 Predicción de Abandono Escolar | ML Clásico | Artificial Intelligence Foundations (Fundació URV)</p>
</div>
""", unsafe_allow_html=True)
