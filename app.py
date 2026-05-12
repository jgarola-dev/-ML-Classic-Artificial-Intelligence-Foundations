"""
Streamlit App: ML Clásico - Predicción de Abandono Escolar
Artificial Intelligence Foundations | Fundació URV | Abril 2026
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

from data_preprocessing import handle_missing_values, encode_categorical, prepare_data
from model import DropoutPredictor, compare_models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==================== CONFIGURACIÓN ====================
st.set_page_config(
    page_title="🎓 Predicción de Abandono Escolar",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-title { color: #1f77b4; text-align: center; font-size: 2.5rem; margin-bottom: 1rem; }
.metric-box { background: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e9ecef; }
</style>
""", unsafe_allow_html=True)

# ==================== ESTADO DE SESIÓN ====================
if 'df' not in st.session_state: st.session_state.df = None
if 'results' not in st.session_state: st.session_state.results = None
if 'feature_names' not in st.session_state: st.session_state.feature_names = None
if 'scaler' not in st.session_state: st.session_state.scaler = None
if 'encoders' not in st.session_state: st.session_state.encoders = None

# ==================== SIDEBAR (MENÚ + CARGA CSV) ====================
st.sidebar.title("📋 Navegación")
page = st.sidebar.radio("Selecciona una sección:", 
    ["🏠 Inicio", "📊 Análisis EDA", "🤖 Entrenamiento", "🔮 Predicción"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Cargar Datos")

uploaded_file = st.sidebar.file_uploader("📁 Subir archivo CSV", type=['csv'])
if uploaded_file is not None:
    try:
        st.session_state.df = pd.read_csv(uploaded_file)
        # Normalizar nombre de target si es necesario
        cols_lower = {c.lower(): c for c in st.session_state.df.columns}
        if 'abandono' not in cols_lower and 'estado' in cols_lower:
            st.session_state.df.rename(columns={cols_lower['estado']: 'Abandono'}, inplace=True)
        st.sidebar.success("✅ CSV cargado correctamente")
    except Exception as e:
        st.sidebar.error(f"❌ Error al leer CSV: {str(e)}")
else:
    if st.sidebar.button("🌐 Cargar Dataset UCI Oficial"):
        with st.spinner("Descargando y procesando..."):
            try:
                from data_loader import load_dataset, preprocess_dataset
                df_uci, source = load_dataset()
                df_uci = preprocess_dataset(df_uci)
                # Asegurar columna Abandono
                if 'Status' in df_uci.columns:
                    df_uci['Abandono'] = df_uci['Status'].apply(lambda x: 1 if 'dropout' in str(x).lower() else 0)
                st.session_state.df = df_uci
                st.sidebar.success(f"✅ {source} cargado ({len(df_uci)} registros)")
            except Exception as e:
                st.sidebar.error(f"❌ Error UCI: {str(e)}")
    
    if st.sidebar.button("🎲 Usar Datos Demo"):
        np.random.seed(42)
        st.session_state.df = pd.DataFrame({
            'Edad': np.random.randint(17, 35, 1000),
            'GPA': np.random.uniform(1.5, 5.0, 1000),
            'Asistencia': np.random.uniform(50, 100, 1000),
            'Horas_Estudio': np.random.uniform(0, 15, 1000),
            'Socioeconomico': np.random.choice(['Bajo', 'Medio', 'Alto'], 1000),
            'Primer_Trimestre': np.random.choice(['Aprobado', 'Reprobado'], 1000),
            'Motivacion': np.random.choice(['Baja', 'Media', 'Alta'], 1000),
            'Abandono': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
        })
        st.sidebar.success("✅ Datos demo generados")

if st.session_state.df is not None:
    st.sidebar.info(f"📊 Dataset activo: {len(st.session_state.df)} filas × {len(st.session_state.df.columns)} columnas")

# ==================== FUNCIONES AUXILIARES ====================
def get_target_col(df):
    candidates = ['Abandono', 'Target', 'Estado', 'Status', 'target', 'estado']
    for c in candidates:
        if c in df.columns: return c
    return None

# ==================== PÁGINAS ====================
if page == "🏠 Inicio":
    st.markdown('<div class="main-title">🎓 Predicción de Abandono Escolar</div>', unsafe_allow_html=True)
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
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
    with c2:
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
        target = get_target_col(df)
        
        if target is None:
            st.error("❌ No se encontró columna objetivo ('Abandono', 'Target', 'Status').")
        else:
            # KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("📊 Total Registros", f"{len(df):,}")
            k2.metric("📋 Características", len(df.columns))
            k3.metric("❌ Valores Faltantes", df.isnull().sum().sum())
            if df[target].dtype in ['int64', 'float64']:
                tasa = (df[target] == 1).mean() * 100
            else:
                tasa = df[target].astype(str).str.lower().isin(['dropout', 'abandono', '1']).mean() * 100
            k4.metric("📉 Tasa de Abandono", f"{tasa:.1f}%")
            
            st.markdown("---")
            
            # 📊 ESTADÍSTICAS DESCRIPTIVAS
            st.subheader("📊 Estadísticas Descriptivas")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target in num_cols: num_cols.remove(target)
            
            if num_cols:
                st.dataframe(df[num_cols].describe().round(3), use_container_width=True)
                st.download_button("📥 Descargar Estadísticas (CSV)", 
                                  df[num_cols].describe().to_csv(), "estadisticas_descriptivas.csv", "text/csv")
            else:
                st.info("ℹ️ No hay variables numéricas disponibles.")
            
            st.markdown("---")
            
            # 🥧 PORCENTAJE EN CIRCUNFERENCIA (DONUT CHART)
            st.subheader("🥧 Distribución de Abandonos (%)")
            counts = df[target].value_counts()
            if df[target].dtype in ['int64', 'float64']:
                labels = ['No Abandono' if x==0 else 'Abandono' for x in counts.index]
            else:
                labels = counts.index.astype(str)
                
            fig_pie = px.pie(
                values=counts.values, 
                names=labels, 
                title="Proporción de Clases",
                hole=0.4,
                color_discrete_map={'Abandono': '#d32f2f', 'No Abandono': '#00c853'}
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # 📈 VISUALIZACIONES ADICIONALES
            st.markdown("---")
            st.subheader("📈 Visualizaciones")
            c1, c2 = st.columns(2)
            with c1:
                if num_cols:
                    sel = st.selectbox("Selecciona variable numérica:", num_cols)
                    fig_box = px.box(df, x=target, y=sel, color=target,
                                    title=f"Distribución de {sel} por Clase",
                                    color_discrete_map={1: '#d32f2f', 0: '#00c853'})
                    st.plotly_chart(fig_box, use_container_width=True)
            with c2:
                if len(num_cols) > 1:
                    corr = df[num_cols].corr()
                    fig_corr = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                                        title="Matriz de Correlación (Pearson)")
                    st.plotly_chart(fig_corr, use_container_width=True)

elif page == "🤖 Entrenamiento":
    st.title("🤖 Entrenamiento de Modelos")
    if st.session_state.df is None:
        st.warning("⚠️ Carga datos primero en la barra lateral")
    else:
        df = st.session_state.df
        target = get_target_col(df)
        
        if target is None:
            st.error("❌ Columna objetivo no encontrada")
        else:
            # ⚙️ CONFIGURACIÓN TAMAÑO TEST
            st.subheader("⚙️ Configuración del Entrenamiento")
            col1, col2 = st.columns(2)
            with col1:
                test_pct = st.slider("📊 Tamaño del conjunto de prueba (%)", 10, 40, 20, step=5)
                test_size = test_pct / 100.0
            with col2:
                train_size = 1.0 - test_size
                n_train = int(len(df) * train_size)
                n_test = len(df) - n_train
                st.info(f"📋 **División de Datos:**\n✅ Entrenamiento: **{n_train}** registros ({train_size*100:.0f}%)\n🔍 Prueba: **{n_test}** registros ({test_size*100:.0f}%)")
            
            if st.button("🚀 Entrenar 4 Modelos", type="primary"):
                with st.spinner("Preparando datos y entrenando... (~10s)"):
                    try:
                        df_clean = handle_missing_values(df.copy())
                        X = df_clean.drop(columns=[target])
                        y = df_clean[target]
                        X_encoded, encoders = encode_categorical(X)
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_encoded, y, test_size=test_size, random_state=42, stratify=y
                        )
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        st.session_state.results = compare_models(X_train_scaled, X_test_scaled, y_train, y_test)
                        st.session_state.feature_names = X_encoded.columns.tolist()
                        st.session_state.scaler = scaler
                        st.session_state.encoders = encoders
                        st.success("✅ Modelos entrenados y métricas calculadas")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        
            # 📊 RESULTADOS
            if st.session_state.results is not None:
                st.markdown("---")
                st.subheader("📊 Comparativa de Modelos")
                rows = []
                for name, res in st.session_state.results.items():
                    m = res['metrics']
                    rows.append({'Modelo': name.replace('_', ' ').title(), 
                                'Accuracy': f"{m['accuracy']:.3f}", 'Precision': f"{m['precision']:.3f}",
                                'Recall': f"{m['recall']:.3f}", 'F1': f"{m['f1']:.3f}", 
                                'ROC-AUC': f"{m.get('roc_auc', 0):.3f}"})
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
                
                st.subheader("🔍 Importancia de Características")
                sel_model = st.selectbox("Selecciona modelo:", list(st.session_state.results.keys()))
                imp_df = st.session_state.results[sel_model]['model'].get_feature_importance(
                    st.session_state.feature_names, top_n=10
                )
                if imp_df is not None and not imp_df.empty:
                    st.plotly_chart(px.bar(imp_df, x='importance', y='feature', orientation='h',
                                        title=f"Top Features ({sel_model.replace('_',' ').title()})"),
                                   use_container_width=True)

elif page == "🔮 Predicción":
    st.title("🔮 Predicción Individual")
    if st.session_state.results is None:
        st.info("ℹ️ Entrena modelos primero en la sección 🤖 Entrenamiento")
    else:
        st.subheader("📋 Ingresa los datos del estudiante")
        c1, c2 = st.columns(2)
        with c1:
            edad = st.number_input("Edad", 15, 50, 19)
            gpa = st.slider("GPA / Nota Promedio", 0.0, 5.0, 3.0)
            asistencia = st.slider("Asistencia (%)", 0, 100, 85)
            horas = st.slider("Horas Estudio/Semana", 0, 20, 5)
        with c2:
            socio = st.selectbox("Nivel Socioeconómico", ['Bajo', 'Medio', 'Alto'])
            trim = st.selectbox("1er Trimestre", ['Aprobado', 'Reprobado'])
            motiv = st.selectbox("Nivel Motivación", ['Baja', 'Media', 'Alta'])
            
        if st.button("🔮 Realizar Predicción", type="primary"):
            try:
                pred_df = pd.DataFrame({
                    'Edad': [edad], 'GPA': [gpa], 'Asistencia': [asistencia], 
                    'Horas_Estudio': [horas], 'Socioeconomico': [socio],
                    'Primer_Trimestre': [trim], 'Motivacion': [motiv]
                })
                # Codificar con los mismos encoders del entrenamiento
                for col, le in st.session_state.encoders.items():
                    if col in pred_df.columns:
                        val = str(pred_df[col].values[0])
                        pred_df[col] = le.transform([val])[0] if val in le.classes_ else 0
                # Rellenar faltantes y escalar
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
                        st.metric(name.replace('_', ' ').title(), risk, delta=f"{prob:.1%}")
            except Exception as e:
                st.error(f"❌ Error en predicción: {str(e)}")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.85rem;'>
    <p>🎓 ML Clásico - Predicción de Abandono Escolar | Artificial Intelligence Foundations | Fundació URV</p>
    <p>Desarrollado con Streamlit, Scikit-learn y Plotly | Abril 2026</p>
</div>
""", unsafe_allow_html=True)
