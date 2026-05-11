import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_preprocessing import prepare_data
from model import DropoutPredictor
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Predicción de Abandono Escolar", page_icon="🎓", layout="wide")

# ==================== SESIÓN & ESTADO ====================
if 'df' not in st.session_state: st.session_state.df = None
if 'trained_models' not in st.session_state: st.session_state.trained_models = {}
if 'scaler' not in st.session_state: st.session_state.scaler = None
if 'encoders' not in st.session_state: st.session_state.encoders = None
if 'feature_names' not in st.session_state: st.session_state.feature_names = []
if 'cat_cols' not in st.session_state: st.session_state.cat_cols = []

# ==================== DATASET DEMO REALISTA ====================
def create_realistic_sample_dataset(n_samples=500):
    np.random.seed(42)
    gpa = np.random.uniform(1.5, 4.0, n_samples)
    asistencia = np.random.uniform(50, 100, n_samples)
    horas = np.random.uniform(0, 10, n_samples)
    motivacion = np.random.choice(['Baja', 'Media', 'Alta'], n_samples, p=[0.3, 0.5, 0.2])
    primer_trim = np.random.choice(['Aprobado', 'Reprobado'], n_samples, p=[0.6, 0.4])
    socioeconomico = np.random.choice(['Bajo', 'Medio', 'Alto'], n_samples, p=[0.4, 0.4, 0.2])
    edad = np.random.randint(15, 25, n_samples)
    
    # Correlación real: bajo rendimiento + baja motivación = mayor probabilidad de abandono
    risk_score = (
        (4.0 - gpa)/2.5 * 0.3 + 
        (100 - asistencia)/50 * 0.2 + 
        (10 - horas)/10 * 0.15 +
        np.where(motivacion=='Baja', 0.3, np.where(motivacion=='Media', 0.15, 0.0)) +
        np.where(primer_trim=='Reprobado', 0.25, 0.0) +
        np.where(socioeconomico=='Bajo', 0.15, 0.0)
    )
    abandono = (risk_score > np.percentile(risk_score, 70)).astype(int)
    
    return pd.DataFrame({
        'Edad': edad, 'GPA': gpa, 'Asistencia': asistencia, 'Horas_Estudio': horas,
        'Socioeconomico': socioeconomico, 'Primer_Trimestre': primer_trim,
        'Motivacion': motivacion, 'Abandono': abandono
    })

# ==================== SIDEBAR ====================
st.sidebar.title("📋 Navegación")
page = st.sidebar.radio("Selecciona:", ["🏠 Inicio", "📊 EDA", "🤖 Entrenar", "🔮 Predecir"])

uploaded_file = st.sidebar.file_uploader("📥 Cargar CSV (requiere columna 'Abandono')", type=['csv'])
if uploaded_file:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ CSV cargado")
elif st.session_state.df is None and st.sidebar.button("Usar datos demo"):
    st.session_state.df = create_realistic_sample_dataset()
    st.sidebar.success("✅ Demo cargada (correlación real)")

if st.session_state.df is not None:
    st.sidebar.info(f"Dataset: {st.session_state.df.shape[0]} filas × {st.session_state.df.shape[1]} columnas")

# ==================== PÁGINAS ====================
if page == "🤖 Entrenar":
    st.title("🤖 Entrenamiento de Modelos")
    if st.session_state.df is None or 'Abandono' not in st.session_state.df.columns:
        st.error("❌ Sube un CSV con columna 'Abandono' o usa el demo.")
    else:
        df = st.session_state.df
        if st.button("🚀 Entrenar Modelos"):
            with st.spinner("Entrenando..."):
                data = prepare_data(df, target_col='Abandono')
                st.session_state.scaler = data['scaler']
                st.session_state.encoders = data['encoders']
                st.session_state.feature_names = data['feature_names']
                st.session_state.cat_cols = data['categorical_cols']
                
                models = ['random_forest', 'gradient_boost', 'logistic', 'svm']
                results = {}
                for m in models:
                    clf = DropoutPredictor(model_type=m)
                    clf.train(data['X_train'], data['y_train'])
                    results[m] = {
                        'model': clf,
                        'metrics': clf.evaluate(data['X_test'], data['y_test']),
                        'importance': clf.get_feature_importance(data['X_test'], data['y_test'], data['feature_names'])
                    }
                st.session_state.trained_models = results
                st.success("✅ Modelos entrenados correctamente")

        if st.session_state.trained_models:
            st.subheader("📊 Comparativa de Modelos")
            rows = []
            for name, res in st.session_state.trained_models.items():
                m = res['metrics']
                rows.append({'Modelo': name.replace('_', ' ').title(), **m})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            
            st.subheader("🔍 Importancia de Características (Random Forest)")
            imp_df = st.session_state.trained_models['random_forest']['importance']
            if imp_df is not None:
                fig = px.bar(imp_df, x='importance', y='feature', orientation='h', 
                            title="Top Features (RF)", labels={'feature':'Característica'})
                st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Predecir":
    st.title("🔮 Predicción Individual")
    if not st.session_state.trained_models:
        st.info("ℹ️ Entrena los modelos primero en la pestaña 🤖 Entrenar")
    else:
        st.subheader("📋 Datos del Estudiante")
        col1, col2 = st.columns(2)
        with col1:
            edad = st.number_input("Edad", 15, 25, 18)
            gpa = st.slider("GPA", 1.5, 4.0, 2.0)
            asistencia = st.slider("Asistencia (%)", 50, 100, 60)
            horas = st.slider("Horas Estudio", 0, 10, 2)
        with col2:
            socio = st.selectbox("Nivel Socioeconómico", ['Bajo', 'Medio', 'Alto'])
            trim = st.selectbox("1er Trimestre", ['Aprobado', 'Reprobado'])
            motiv = st.selectbox("Motivación", ['Baja', 'Media', 'Alta'])
            
        if st.button("🔮 Predecir"):
            pred_df = pd.DataFrame([{
                'Edad': edad, 'GPA': gpa, 'Asistencia': asistencia, 'Horas_Estudio': horas,
                'Socioeconomico': socio, 'Primer_Trimestre': trim, 'Motivacion': motiv
            }])
            
            # 1. Codificar categóricas con encoders entrenados
            for col in st.session_state.cat_cols:
                le = st.session_state.encoders[col]
                val = pred_df[col].values[0]
                # Manejar categorías no vistas
                if val not in le.classes_:
                    pred_df[col] = 'Desconocido'
                pred_df[col] = le.transform([val])[0]
                
            # 2. Reordenar columnas y escalar
            pred_df = pred_df[st.session_state.feature_names]
            pred_scaled = st.session_state.scaler.transform(pred_df)
            
            # 3. Predecir con todos los modelos
            st.markdown("---")
            st.subheader("🎯 Resultados")
            cols = st.columns(4)
            for idx, (name, res) in enumerate(st.session_state.trained_models.items()):
                with cols[idx]:
                    clf = res['model']
                    pred = clf.model.predict(pred_scaled)[0]
                    prob = clf.model.predict_proba(pred_scaled)[0][1] if hasattr(clf.model, 'predict_proba') else None
                    color = "🔴 ALTO RIESGO" if pred == 1 else "🟢 BAJO RIESGO"
                    st.metric(f"{name.replace('_',' ').title()}", color, delta=f"{prob:.1%}" if prob else None)
