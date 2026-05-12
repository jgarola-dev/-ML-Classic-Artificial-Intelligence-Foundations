"""
Streamlit app for student dropout prediction
"""
import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# 🔧 Fix para imports en Streamlit Cloud
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from data_preprocessing import handle_missing_values, encode_categorical
from model import DropoutPredictor, compare_models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==================== CONFIGURACIÓN ====================
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
.success { color: #00c853; font-weight: bold; }
.warning { color: #ff9100; font-weight: bold; }
.danger { color: #d32f2f; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==================== FUNCIONES AUXILIARES ====================
def create_sample_dataset(n_samples=500):
    """Genera dataset demo con columnas compatibles y correlaciones realistas"""
    np.random.seed(42)
    edad = np.random.randint(17, 35, n_samples)
    gpa = np.random.uniform(1.5, 5.0, n_samples)
    asistencia = np.random.uniform(50, 100, n_samples)
    horas_estudio = np.random.uniform(0, 15, n_samples)
    socioeconomico = np.random.choice(['Bajo', 'Medio', 'Alto'], n_samples, p=[0.4, 0.4, 0.2])
    primer_trimestre = np.random.choice(['Aprobado', 'Reprobado'], n_samples, p=[0.65, 0.35])
    motivacion = np.random.choice(['Baja', 'Media', 'Alta'], n_samples, p=[0.3, 0.5, 0.2])
    
    # Lógica realista para target
    risk = (5.0 - gpa)/3.5 * 0.3 + (100 - asistencia)/50 * 0.25 + (15 - horas_estudio)/15 * 0.15 + \
           np.where(motivacion=='Baja', 0.25, np.where(motivacion=='Media', 0.10, 0.0)) + \
           np.where(primer_trimestre=='Reprobado', 0.20, 0.0) + \
           np.where(socioeconomico=='Bajo', 0.10, 0.0)
    abandono = (risk > np.percentile(risk, 70)).astype(int)
    
    return pd.DataFrame({
        'Edad': edad, 'GPA': gpa, 'Asistencia': asistencia, 'Horas_Estudio': horas_estudio,
        'Socioeconomico': socioeconomico, 'Primer_Trimestre': primer_trimestre,
        'Motivacion': motivacion, 'Abandono': abandono
    })

# ==================== ESTADO DE SESIÓN ====================
if 'df' not in st.session_state: st.session_state.df = None
if 'results' not in st.session_state: st.session_state.results = None
if 'feature_names' not in st.session_state: st.session_state.feature_names = None
if 'scaler' not in st.session_state: st.session_state.scaler = None
if 'encoders' not in st.session_state: st.session_state.encoders = None

# ==================== SIDEBAR ====================
st.sidebar.title("📋 Navegación")
page = st.sidebar.radio("Selecciona una sección:", 
    ["🏠 Inicio", "📊 Análisis EDA", "🤖 Entrenamiento", "🔮 Predicción"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Cargar Datos")

# 🔘 Cargador de archivos (siempre visible)
uploaded_file = st.sidebar.file_uploader("📁 Subir CSV personalizado", type=['csv'])
if uploaded_file is not None:
    try:
        st.session_state.df = pd.read_csv(uploaded_file)
        # Normalizar target si es necesario
        cols_lower = {c.lower(): c for c in st.session_state.df.columns}
        if 'abandono' not in cols_lower:
            st.session_state.df['Abandono'] = np.random.choice([0, 1], len(st.session_state.df), p=[0.7, 0.3])
        st.sidebar.success("✅ CSV cargado correctamente")
    except Exception as e:
        st.sidebar.error(f"❌ Error al leer CSV: {str(e)}")

# 🔘 Botones de carga alternativos (siempre visibles e independientes)
if st.sidebar.button("🌐 Cargar Dataset UCI Oficial"):
    with st.spinner("Descargando y procesando..."):
        try:
            from data_loader import load_dataset, preprocess_dataset
            df_uci, source = load_dataset()
            df_uci = preprocess_dataset(df_uci)
            if 'Status' in df_uci.columns:
                df_uci['Abandono'] = df_uci['Status'].apply(lambda x: 1 if 'dropout' in str(x).lower() else 0)
            st.session_state.df = df_uci
            st.sidebar.success(f"✅ {source} cargado ({len(df_uci)} registros)")
        except Exception as e:
            st.sidebar.error(f"❌ Error UCI: {str(e)}")

if st.sidebar.button("🎲 Usar datos de demostración"):
    st.session_state.df = create_sample_dataset()
    st.sidebar.success("✅ Datos de demostración cargados (500 registros)")

if st.session_state.df is not None:
    st.sidebar.info(f"📊 Dataset activo: {len(st.session_state.df)} filas × {len(st.session_state.df.columns)} columnas")

# ==================== PÁGINAS ====================
if page == "🏠 Inicio":
    st.markdown('<div class="main-title">🎓 Predicción de Abandono Escolar</div>', unsafe_allow_html=True)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ## 📚 Sobre el Proyecto
        Implementa un pipeline de **Machine Learning clásico** para predecir el riesgo de abandono escolar.
        ### 🎯 Objetivos
        - ✅ Identificar estudiantes en riesgo
        - 📋 Facilitar intervención temprana basada en datos
        - 🔍 Analizar factores académicos y socioeconómicos
        ### 📋 Formulación
        - **Tipo:** Supervisado | **Tarea:** Clasificación Binaria
        - **Target:** `Abandono` (0: No, 1: Sí)
        - **Métrica principal:** F1-Score, ROC-AUC
        """)
    with col2:
        st.markdown("""
        ## 🤖 Modelos Implementados
        1. **Logistic Regression** 📈 (Lineal, interpretable)
        2. **Random Forest** 🌲 (Ensemble, robusto)
        3. **Gradient Boosting** 🚀 (Secuencial, alta precisión)
        4. **Support Vector Machine (SVM)** 🎯 (Kernel RBF)
        """)
    m1, m2, m3 = st.columns(3)
    m1.metric("🤖 Modelos", "4", "Clasificadores")
    m2.metric("📈 Métricas", "6+", "Evaluación")
    m3.metric("🎓 Categoría", "ML Clásico", "Fundació URV")

elif page == "📊 Análisis EDA":
    st.title("📊 Análisis Exploratorio de Datos")
    if st.session_state.df is None:
        st.warning("⚠️ Carga datos primero en la barra lateral")
    else:
        df = st.session_state.df
        if 'Abandono' not in df.columns:
            st.error("❌ El dataset debe contener la columna `Abandono`")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Registros", f"{len(df):,}")
            c2.metric("Features", len(df.columns)-1)
            c3.metric("Faltantes", df.isnull().sum().sum())
            c4.metric("Tasa Abandono", f"{df['Abandono'].mean()*100:.1f}%")
            st.markdown("---")
            
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Abandono' in num_cols: num_cols.remove('Abandono')
            if num_cols:
                st.subheader("📊 Estadísticas Descriptivas")
                st.dataframe(df[num_cols].describe().T.round(3), use_container_width=True)
                
            st.subheader("🥧 Distribución de Clases")
            counts = df['Abandono'].value_counts()
            fig = px.pie(values=counts.values, names=['No Abandona (0)', 'Abandona (1)'],
                        title="Proporción de Clases", hole=0.4, color_discrete_map={0: '#00c853', 1: '#d32f2f'})
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("🔗 Matriz de Correlación")
            if len(num_cols) >= 2:
                corr = df[num_cols].corr()
                st.plotly_chart(px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                                        title="Correlaciones Pearson"), use_container_width=True)

elif page == "🤖 Entrenamiento":
    st.title("🤖 Entrenamiento de Modelos")
    if st.session_state.df is None or 'Abandono' not in st.session_state.df.columns:
        st.warning("⚠️ Carga un dataset con columna `Abandono`")
    else:
        df = st.session_state.df
        test_size = st.slider("📊 Tamaño del conjunto de prueba (%)", 10, 40, 20) / 100
        train_size = 1.0 - test_size
        n_train, n_test = int(len(df) * train_size), len(df) - int(len(df) * train_size)
        st.info(f"📋 **División:** ✅ Train: {n_train} ({train_size*100:.0f}%) | 🔍 Test: {n_test} ({test_size*100:.0f}%)")
        
        if st.button("🚀 Entrenar 4 Modelos", type="primary"):
            with st.spinner("Entrenando (~15s)..."):
                try:
                    df_clean = handle_missing_values(df.copy())
                    X = df_clean.drop(columns=['Abandono'])
                    y = df_clean['Abandono']
                    X_enc, encoders = encode_categorical(X)
                    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=test_size, random_state=42, stratify=y)
                    
                    st.session_state.scaler = StandardScaler()
                    X_train_s = st.session_state.scaler.fit_transform(X_train)
                    X_test_s = st.session_state.scaler.transform(X_test)
                    
                    st.session_state.results = compare_models(X_train_s, X_test_s, y_train, y_test)
                    st.session_state.feature_names = X_enc.columns.tolist()
                    st.session_state.encoders = encoders
                    st.success("✅ Modelos entrenados correctamente")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    
        if st.session_state.results:
            st.subheader("📊 Comparativa de Modelos")
            rows = []
            for name, res in st.session_state.results.items():
                m = res['metrics']
                rows.append({'Modelo': name.replace('_',' ').title(), 
                            'Accuracy': f"{m['accuracy']:.3f}", 'Precision': f"{m['precision']:.3f}",
                            'Recall': f"{m['recall']:.3f}", 'F1': f"{m['f1']:.3f}", 'ROC-AUC': f"{m.get('roc_auc', 0):.3f}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            
            st.subheader("🔍 Importancia de Características")
            sel_model = st.selectbox("Selecciona modelo:", list(st.session_state.results.keys()))
            imp_df = st.session_state.results[sel_model]['model'].get_feature_importance(st.session_state.feature_names, top_n=10)
            
            # 🔧 FIX: Recortar valores negativos y forzar eje X en 0
            if imp_df is not None and not imp_df.empty:
                imp_df['importance'] = imp_df['importance'].clip(lower=0)
                fig = px.bar(imp_df, x='importance', y='feature', orientation='h',
                            title=f"Top Features ({sel_model.replace('_',' ').title()})")
                fig.update_xaxes(range=[0, None])  # Fuerza origen en 0
                st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Predicción":
    st.title("🔮 Predicción Individual")
    if st.session_state.results is None:
        st.info("ℹ️ Entrena modelos primero en la sección 🤖 Entrenamiento")
    else:
        col1, col2 = st.columns(2)
        with col1:
            edad = st.number_input("Edad", 15, 50, 19)
            gpa = st.slider("GPA / Nota Promedio", 0.0, 5.0, 3.0)
            asistencia = st.slider("Asistencia (%)", 0, 100, 85)
            horas = st.slider("Horas Estudio/Semana", 0, 20, 5)
        with col2:
            socio = st.selectbox("Nivel Socioeconómico", ['Bajo', 'Medio', 'Alto'])
            trim = st.selectbox("1er Trimestre", ['Aprobado', 'Reprobado'])
            motiv = st.selectbox("Nivel Motivación", ['Baja', 'Media', 'Alta'])
            
        if st.button("🔮 Realizar Predicción", type="primary"):
            try:
                pred_df = pd.DataFrame({'Edad': [edad], 'GPA': [gpa], 'Asistencia': [asistencia], 
                                      'Horas_Estudio': [horas], 'Socioeconomico': [socio],
                                      'Primer_Trimestre': [trim], 'Motivacion': [motiv]})
                for col, le in st.session_state.encoders.items():
                    if col in pred_df.columns:
                        val = str(pred_df[col].values[0])
                        pred_df[col] = le.transform([val])[0] if val in le.classes_ else 0
                for feat in st.session_state.feature_names:
                    if feat not in pred_df.columns: pred_df[feat] = 0
                pred_df = pred_df[st.session_state.feature_names]
                X_scaled = st.session_state.scaler.transform(pred_df)
                
                st.markdown("---")
                st.subheader("🎯 Resultados")
                cols = st.columns(4)
                for i, (name, res) in enumerate(st.session_state.results.items()):
                    with cols[i]:
                        p = res['model'].predict(X_scaled)[0]
                        prob = res['model'].predict_proba(X_scaled)[0][1] if hasattr(res['model'].model, 'predict_proba') else 0.5
                        risk = "🔴 ALTO RIESGO" if p == 1 else "🟢 BAJO RIESGO"
                        st.metric(name.replace('_',' ').title(), risk, delta=f"{prob:.1%}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.85rem;'>
    <p>🎓 ML Clásico - Predicción de Abandono Escolar | Artificial Intelligence Foundations | Fundació URV</p>
    <p>Desarrollado con Streamlit, Scikit-learn y Plotly | Abril 2026</p>
</div>
""", unsafe_allow_html=True)
