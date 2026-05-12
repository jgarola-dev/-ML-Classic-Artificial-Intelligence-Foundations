"""
Streamlit app for student dropout prediction
"""
import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_preprocessing import prepare_data, handle_missing_values, encode_categorical
from model import DropoutPredictor, compare_models
import io
import warnings
warnings.filterwarnings('ignore')

# 🔧 Fix para imports locales en Streamlit Cloud
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

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

def create_sample_dataset(n_samples=500):
    """Create a sample dataset for demonstration"""
    np.random.seed(42)
    data = {
        'Edad': np.random.randint(17, 35, n_samples),
        'GPA': np.random.uniform(1.5, 5.0, n_samples),
        'Asistencia': np.random.uniform(40, 100, n_samples),
        'Horas_Estudio': np.random.uniform(0, 20, n_samples),
        'Socioeconomico': np.random.choice(['Bajo', 'Medio', 'Alto'], n_samples),
        'Primer_Trimestre': np.random.choice(['Aprobado', 'Reprobado'], n_samples),
        'Motivacion': np.random.choice(['Baja', 'Media', 'Alta'], n_samples),
        'Abandono': (np.random.uniform(0, 1, n_samples) > 0.7).astype(int)
    }
    return pd.DataFrame(data)

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

uploaded_file = st.sidebar.file_uploader("📁 Sube un archivo CSV", type=['csv'])
if uploaded_file is not None:
    try:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ Archivo cargado correctamente")
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")
elif st.sidebar.button("🎲 Usar datos de demostración"):
    st.session_state.df = create_sample_dataset()
    st.sidebar.success("✅ Datos de demostración cargados")

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
        Este proyecto implementa un **modelo de Machine Learning clásico** para predecir el riesgo de abandono escolar.
        ### 🎯 Objetivos
        - ✅ Identificar estudiantes en riesgo
        - 📋 Facilitar intervención temprana
        - 🔍 Analizar factores académicos y socioeconómicos
        ### 📋 Formulación
        - **Tipo:** Supervisado | **Tarea:** Clasificación Binaria
        - **Target:** `Abandono` (0: No, 1: Sí)
        """)
    with col2:
        st.markdown("""
        ## 🤖 Modelos Implementados
        1. **Logistic Regression** 📈
        2. **Random Forest** 🌲
        3. **Gradient Boosting** 🚀
        4. **Support Vector Machine (SVM)** 🎯
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
        
        if 'Abandono' not in df.columns and 'abandono' in df.columns:
            df.rename(columns={'abandono': 'Abandono'}, inplace=True)
            
        target_col = 'Abandono' if 'Abandono' in df.columns else None
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Registros", f"{len(df):,}")
        col2.metric("Características", df.shape[1])
        col3.metric("Valores Faltantes", df.isnull().sum().sum())
        if target_col:
            col4.metric("Tasa de Abandono", f"{df[target_col].mean()*100:.1f}%")
        
        st.markdown("---")
        st.subheader("📈 Visualizaciones Interactivas")
        
        vis_var = st.selectbox(
            "Selecciona una característica para visualizar:", 
            ["Edad", "GPA", "Asistencia", "Horas_Estudio"]
        )
        
        if vis_var in df.columns:
            c1, c2 = st.columns(2)
            
            with c1:
                if target_col and df[target_col].nunique() == 2:
                    fig_box = px.box(df, x=target_col, y=vis_var, color=target_col,
                                    title=f"Distribución de {vis_var} por Abandono",
                                    color_discrete_map={0: '#00c853', 1: '#d32f2f'})
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    fig_hist = px.histogram(df, x=vis_var, nbins=30, title=f"Distribución de {vis_var}")
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
            with c2:
                if target_col:
                    counts = df[target_col].value_counts()
                    fig_pie = px.pie(values=counts.values, names=['No Abandono', 'Abandono'],
                                    title="Distribución de Clases", color=counts.index,
                                    color_discrete_map={0: '#00c853', 1: '#d32f2f'})
                    st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("🔗 Matriz de Correlación")
        numeric_df = df.select_dtypes(include=[np.number])
        if target_col and target_col in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[target_col])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            fig_corr = px.imshow(corr_matrix, color_continuous_scale='RdBu',
                               title="Matriz de Correlación", labels=dict(color="Correlación"))
            st.plotly_chart(fig_corr, use_container_width=True)

elif page == "🤖 Entrenamiento":
    st.title("🤖 Entrenamiento de Modelos")
    if st.session_state.df is None or 'Abandono' not in st.session_state.df.columns:
        st.warning("⚠️ El dataset debe contener la columna 'Abandono'")
    else:
        test_size = st.slider("Tamaño del conjunto de prueba (%)", 10, 40, 20) / 100
        train_size = 1.0 - test_size
        n_train = int(len(st.session_state.df) * train_size)
        n_test = len(st.session_state.df) - n_train
        st.info(f"📋 **División:** ✅ Train: {n_train} ({train_size*100:.0f}%) | 🔍 Test: {n_test} ({test_size*100:.0f}%)")
        
        if st.button("🚀 Entrenar Modelos", key='train'):
            with st.spinner("Entrenando modelos... (~15s)"):
                try:
                    df_clean = handle_missing_values(st.session_state.df)
                    X = df_clean.drop(columns=['Abandono'])
                    y = df_clean['Abandono']
                    X_encoded, encoders = encode_categorical(X)
                    
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_encoded, y, test_size=test_size, random_state=42, stratify=y
                    )
                    
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    st.session_state.results = compare_models(X_train_scaled, X_test_scaled, y_train, y_test)
                    st.session_state.feature_names = X_encoded.columns.tolist()
                    st.session_state.scaler = scaler
                    st.session_state.encoders = encoders
                    st.success("✅ Modelos entrenados correctamente")
                except Exception as e:
                    st.error(f"❌ Error al entrenar: {str(e)}")
                    
        if st.session_state.results:
            st.subheader("📊 Resultados de Modelos")
            results_data = []
            for model_name, result in st.session_state.results.items():
                m = result['metrics']
                results_data.append({
                    'Modelo': model_name.replace('_', ' ').title(),
                    'Accuracy': f"{m['accuracy']:.3f}",
                    'Precision': f"{m['precision']:.3f}",
                    'Recall': f"{m['recall']:.3f}",
                    'F1-Score': f"{m['f1']:.3f}",
                    'ROC-AUC': f"{m.get('roc_auc', 0):.3f}"
                })
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # Gráfico Comparativo de Modelos
            st.subheader("📈 Comparación de Modelos")
            plot_df = results_df.copy()
            for c in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
                plot_df[c] = plot_df[c].astype(float)
            
            fig_comp = px.bar(
                plot_df.melt(id_vars='Modelo', var_name='Métrica', value_name='Valor'),
                x='Modelo', y='Valor', color='Métrica', barmode='group',
                title="Comparación de Rendimiento (Accuracy, F1, Precision, Recall, ROC-AUC)"
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Importancia de Características
            st.subheader("🔍 Importancia de Características")
            selected_model = st.selectbox("Selecciona un modelo:", list(st.session_state.results.keys()))
            model_obj = st.session_state.results[selected_model]['model']
            
            importance_df = model_obj.get_feature_importance(st.session_state.feature_names, top_n=10)
            if importance_df is not None and not importance_df.empty:
                # Mapear nombres si es necesario
                importance_df['feature'] = importance_df['feature'].apply(lambda x: x.replace('_', ' ').title())
                fig_imp = px.bar(importance_df, x='importance', y='feature', orientation='h',
                            title=f"Top 10 Características ({selected_model.replace('_', ' ').title()})")
                fig_imp.update_xaxes(range=[0, None])
                st.plotly_chart(fig_imp, use_container_width=True)

elif page == "🔮 Predicción":
    st.title("🔮 Predicción Individual")
    if st.session_state.results is None:
        st.info("ℹ️ Entrena modelos primero en la sección 🤖 Entrenamiento")
    else:
        st.subheader("📋 Ingresa los datos del estudiante")
        col1, col2 = st.columns(2)
        
        with col1:
            edad = st.number_input("Edad", min_value=15, max_value=100, value=20)
            gpa = st.slider("GPA / Nota Promedio", 0.0, 5.0, 3.0)
            asistencia = st.slider("Asistencia (%)", 0, 100, 85)
        with col2:
            horas = st.number_input("Horas de Estudio (semanales)", 0, 50, 10)
            socioeconomico = st.selectbox("Nivel Socioeconómico", ['Bajo', 'Medio', 'Alto'])
            primer_trimestre = st.selectbox("Desempeño Primer Trimestre", ['Aprobado', 'Reprobado'])
            motivacion = st.selectbox("Nivel de Motivación", ['Baja', 'Media', 'Alta'])
            
        if st.button("🔮 Realizar Predicción", type="primary"):
            try:
                pred_data = pd.DataFrame({
                    'Edad': [edad], 'GPA': [gpa], 'Asistencia': [asistencia],
                    'Horas_Estudio': [horas], 'Socioeconomico': [socioeconomico],
                    'Primer_Trimestre': [primer_trimestre], 'Motivacion': [motivacion]
                })
                
                pred_encoded, _ = encode_categorical(pred_data)
                for col in st.session_state.feature_names:
                    if col not in pred_encoded.columns:
                        pred_encoded[col] = 0
                pred_encoded = pred_encoded[st.session_state.feature_names]
                
                X_sample = st.session_state.scaler.transform(pred_encoded)
                
                st.markdown("---")
                st.subheader("🎯 Resultados de Predicción")
                cols = st.columns(4)
                for idx, (model_name, result) in enumerate(st.session_state.results.items()):
                    with cols[idx]:
                        model_obj = result['model']
                        pred = model_obj.predict(X_sample)[0]
                        prob = model_obj.predict_proba(X_sample)[0][1] if hasattr(model_obj.model, 'predict_proba') else 0.5
                        recommendation = model_obj.get_recommendation(pred, prob)
                        st.markdown(f"### {model_name.replace('_', ' ').title()}")
                        st.metric("Probabilidad", f"{prob:.1%}")
                        st.markdown(recommendation)
            except Exception as e:
                st.error(f"❌ Error en predicción: {str(e)}")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.85rem;'>
    <p>🎓 ML Clásico - Predicción de Abandono Escolar | Artificial Intelligence Foundations | Fundació URV</p>
</div>
""", unsafe_allow_html=True)
