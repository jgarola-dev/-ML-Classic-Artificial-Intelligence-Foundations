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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# 🔧 Fix para imports locales en Streamlit Cloud
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from data_loader import load_dataset, preprocess_dataset
from data_preprocessing import prepare_data, handle_missing_values, encode_categorical, scale_features
from model import DropoutPredictor, compare_models

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
.success { color: #28a745; font-weight: bold; }
.warning { color: #ffc107; font-weight: bold; }
.danger { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==================== ESTADO DE SESIÓN ====================
if 'df' not in st.session_state: st.session_state.df = None
if 'prep_data' not in st.session_state: st.session_state.prep_data = None
if 'models' not in st.session_state: st.session_state.models = {}
if 'tuning_results' not in st.session_state: st.session_state.tuning_results = None

# ==================== SIDEBAR ====================
st.sidebar.title("📋 Navegación")
page = st.sidebar.radio("Selecciona una sección:", 
    ["🏠 Inicio", "📊 EDA", "🤖 Entrenamiento", "🔮 Predicción"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Cargar Datos")
if st.sidebar.button("🌐 Cargar Dataset UCI Oficial"):
    with st.spinner("Descargando y procesando..."):
        try:
            df, source = load_dataset()
            df = preprocess_dataset(df)
            # Normalizar target a binario (0: Dropout, 1: Graduate/Activo)
            if 'Status' in df.columns:
                df['Abandono'] = df['Status'].apply(lambda x: 0 if 'dropout' in str(x).lower() else 1)
            elif 'Status_Label' in df.columns:
                df['Abandono'] = df['Status_Label'].apply(lambda x: 0 if 'dropout' in str(x).lower() else 1)
            st.session_state.df = df
            st.sidebar.success(f"✅ {source} cargado ({len(df)} registros)")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")

uploaded_file = st.sidebar.file_uploader("📁 Subir CSV personalizado", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'Abandono' not in df.columns and 'Status' in df.columns:
        df['Abandono'] = df['Status'].apply(lambda x: 0 if 'dropout' in str(x).lower() else 1)
    st.session_state.df = df
    st.sidebar.success("✅ CSV cargado correctamente")

if st.session_state.df is not None:
    st.sidebar.info(f"📊 Dataset: {len(st.session_state.df)} filas × {len(st.session_state.df.columns)} columnas")

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
        - ✅ Identificar estudiantes en riesgo de abandono
        - 📋 Proporcionar recomendaciones de intervención temprana
        - 🔍 Analizar factores académicos y socioeconómicos clave
        
        ### 📋 Formulación del Problema
        | Aspecto | Detalle |
        |---------|---------|
        | **Tipo de aprendizaje** | Supervisado |
        | **Tarea** | Clasificación binaria |
        | **Variable objetivo** | `Abandono` (0: No, 1: Sí) |
        | **Métrica principal** | F1-Score (macro) |
        """)
    with col2:
        st.markdown("""
        ## 🤖 Modelos Implementados
        Se entrenan y comparan 4 clasificadores de `scikit-learn`:
        
        1. **Logistic Regression** 📈
        2. **Random Forest** 🌲
        3. **Gradient Boosting** 🚀
        4. **Support Vector Machine (SVM)** 🎯
        """)
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("🤖 Modelos", "4", "Clasificadores")
    m2.metric("📈 Métricas", "6+", "Evaluación")
    m3.metric("🎓 Categoría", "ML Clásico", "Fundació URV")
    st.markdown("---")
    st.info("📌 **Flujo de trabajo:** Cargar datos → Análisis EDA → Entrenar con Tuning → Predecir en tiempo real")

elif page == "📊 EDA":
    st.title("📊 Análisis Exploratorio de Datos")
    if st.session_state.df is None:
        st.warning("⚠️ Carga datos primero en la barra lateral")
    else:
        df = st.session_state.df
        target = 'Abandono' if 'Abandono' in df.columns else None
        
        # KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("📊 Registros", f"{len(df):,}")
        kpi2.metric("📋 Features", len(df.columns) - (1 if target else 0))
        kpi3.metric("❌ Valores Faltantes", df.isnull().sum().sum())
        if target:
            kpi4.metric("📉 Tasa Abandono", f"{df[target].mean()*100:.1f}%")
            
        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["🔍 Distribuciones", "🔗 Correlaciones", "⚖️ Desequilibrio"])
        
        with tab1:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if target in num_cols: num_cols.remove(target)
            if num_cols:
                sel_col = st.selectbox("Selecciona variable:", num_cols)
                if sel_col:
                    if target and df[target].nunique() == 2:
                        fig = px.box(df, x=target, y=sel_col, color=target,
                                    title=f"{sel_col} por Clase de Abandono",
                                    color_discrete_map={0: '#28a745', 1: '#dc3545'})
                    else:
                        fig = px.histogram(df, x=sel_col, nbins=30, title=f"Distribución: {sel_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    
        with tab2:
            if len(num_cols) >= 2:
                corr = df[num_cols].corr()
                fig_heat = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                                    title="Matriz de Correlación (Pearson)")
                st.plotly_chart(fig_heat, use_container_width=True)
                
                # Top correlaciones
                pairs = []
                for i in range(len(corr.columns)):
                    for j in range(i+1, len(corr.columns)):
                        pairs.append({'Var1': corr.columns[i], 'Var2': corr.columns[j], 'Corr': corr.iloc[i,j]})
                top = pd.DataFrame(pairs).sort_values('Corr', key=abs, ascending=False).head(5)
                st.markdown("🏆 **Top 5 Correlaciones más fuertes:**")
                st.dataframe(top.style.format({'Corr': '{:.3f}'}), use_container_width=True)
                
        with tab3:
            if target:
                counts = df[target].value_counts()
                fig_pie = px.pie(values=counts.values, names=counts.index.astype(str),
                                title="Distribución de Clases", color=counts.index,
                                color_discrete_map={0: '#28a745', 1: '#dc3545'})
                st.plotly_chart(fig_pie, use_container_width=True)
                st.info(f"💡 Ratio desequilibrio: {counts.max()/counts.min():.2f}:1")

elif page == "🤖 Entrenamiento":
    st.title("🤖 Entrenamiento & Ajuste de Hiperparámetros")
    if st.session_state.df is None or 'Abandono' not in st.session_state.df.columns:
        st.warning("⚠️ Carga un dataset con columna `Abandono`")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("⚙️ Configuración")
            test_size = st.slider("Tamaño Test (%)", 10, 40, 20) / 100
            tune_model = st.radio("Modelo para Tuning:", ['random_forest', 'gradient_boosting'])
            
            if st.button("🚀 Entrenar & Optimizar"):
                with st.spinner("Preparando datos y entrenando..."):
                    try:
                        # 1. Preprocessing
                        df_clean = handle_missing_values(st.session_state.df.copy())
                        prep = prepare_data(df_clean, target_col='Abandono', test_size=test_size)
                        st.session_state.prep_data = {
                            'X_train': prep[0], 'X_test': prep[1],
                            'y_train': prep[2], 'y_test': prep[3],
                            'feature_names': prep[4], 'encoders': prep[5], 'scaler': prep[6]
                        }
                        
                        # 2. Entrenar modelos base
                        results = compare_models(prep[0], prep[1], prep[2], prep[3])
                        st.session_state.models = results
                        
                        # 3. Hyperparameter Tuning (Requisito URV)
                        with st.spinner(f"Ajustando hiperparámetros ({tune_model})..."):
                            X_tr, y_tr = prep[0], prep[2]
                            if tune_model == 'random_forest':
                                model = DropoutPredictor('random_forest').model
                                param_grid = {
                                    'n_estimators': [100, 200],
                                    'max_depth': [10, 20, None],
                                    'min_samples_split': [2, 5],
                                    'class_weight': ['balanced']
                                }
                            else:
                                model = DropoutPredictor('gradient_boosting').model
                                param_grid = {
                                    'n_estimators': [100, 200],
                                    'learning_rate': [0.01, 0.1],
                                    'max_depth': [3, 5],
                                    'subsample': [0.8, 1.0]
                                }
                                
                            grid = GridSearchCV(model, param_grid, scoring='f1', cv=5, n_jobs=-1)
                            grid.fit(X_tr, y_tr)
                            
                            st.session_state.tuning_results = {
                                'best_params': grid.best_params_,
                                'best_f1': grid.best_score_,
                                'model_name': tune_model
                            }
                            
                        st.success("✅ Entrenamiento y optimización completados")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        
        with col2:
            if st.session_state.models:
                st.subheader("📊 Comparativa de Modelos")
                rows = []
                for name, res in st.session_state.models.items():
                    m = res['metrics']
                    rows.append({
                        'Modelo': name.replace('_', ' ').title(),
                        'Accuracy': f"{m['accuracy']:.3f}",
                        'Precision': f"{m['precision']:.3f}",
                        'Recall': f"{m['recall']:.3f}",
                        'F1-Score': f"{m['f1']:.3f}",
                        'ROC-AUC': f"{m.get('roc_auc', 0):.3f}"
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
                
                st.markdown("---")
                st.subheader("🔧 Ajuste de Hiperparámetros")
                if st.session_state.tuning_results:
                    t = st.session_state.tuning_results
                    st.success(f"🏆 Mejores params: `{t['best_params']}` | F1 optimizado: `{t['best_f1']:.3f}`")
                    
                    # Gráfico importancia
                    best_model = st.session_state.models[t['model_name']]['model']
                    imp_df = best_model.get_feature_importance(st.session_state.prep_data['feature_names'], top_n=8)
                    if imp_df is not None and not imp_df.empty:
                        st.plotly_chart(px.bar(imp_df, x='importance', y='feature', orientation='h',
                                            title=f"Top Features ({t['model_name'].replace('_',' ').title()})"),
                                       use_container_width=True)

elif page == "🔮 Predicción":
    st.title("🔮 Predicción Individual")
    if not st.session_state.models:
        st.info("ℹ️ Entrena modelos primero en la sección 🤖 Entrenamiento")
    else:
        st.markdown("📌 *Introduce los datos del estudiante. El modelo reutiliza los transformadores del entrenamiento.*")
        
        col1, col2 = st.columns(2)
        with col1:
            edad = st.number_input("Edad", 15, 50, 19)
            gpa = st.slider("GPA / Nota Promedio", 0.0, 5.0, 3.0)
            asistencia = st.slider("Asistencia (%)", 0, 100, 85)
            horas = st.slider("Horas Estudio Semanal", 0, 20, 5)
        with col2:
            socioeconomico = st.selectbox("Nivel Socioeconómico", ['Bajo', 'Medio', 'Alto'])
            trim1 = st.selectbox("Desempeño 1er Trimestre", ['Aprobado', 'Reprobado'])
            motivacion = st.selectbox("Nivel Motivación", ['Baja', 'Media', 'Alta'])
            
        if st.button("🔮 Realizar Predicción", type="primary"):
            try:
                # 1. Crear DataFrame de entrada
                input_data = pd.DataFrame([{
                    'Edad': edad, 'GPA': gpa, 'Asistencia': asistencia, 'Horas_Estudio': horas,
                    'Socioeconomico': socioeconomico, 'Primer_Trimestre': trim1, 'Motivacion': motivacion
                }])
                
                # 2. Aplicar EXACTAMENTE los mismos encoders del entrenamiento
                prep = st.session_state.prep_data
                for col in prep['encoders'].keys():
                    le = prep['encoders'][col]
                    if col in input_data.columns:
                        val = str(input_data[col].values[0])
                        input_data[col] = le.transform([val])[0] if val in le.classes_ else 0
                        
                # 3. Rellenar columnas faltantes con medianas/modas del entrenamiento (si hay más features)
                for feat in prep['feature_names']:
                    if feat not in input_data.columns:
                        input_data[feat] = 0  # Fallback seguro
                        
                # 4. Reordenar y escalar
                input_data = input_data[prep['feature_names']]
                X_scaled = prep['scaler'].transform(input_data)
                
                # 5. Predecir
                st.markdown("---")
                st.subheader("🎯 Resultados")
                cols = st.columns(4)
                for idx, (name, res) in enumerate(st.session_state.models.items()):
                    with cols[idx]:
                        model_obj = res['model']
                        pred = model_obj.predict(X_scaled)[0]
                        prob = model_obj.predict_proba(X_scaled)[0][1] if hasattr(model_obj.model, 'predict_proba') else 0.5
                        risk = "🔴 ALTO RIESGO" if pred == 1 else "🟢 BAJO RIESGO"
                        st.metric(name.replace('_', ' ').title(), risk, delta=f"{prob:.1%}")
                        
            except Exception as e:
                st.error(f"❌ Error en predicción: {str(e)}")
                st.exception(e)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.85rem;'>
    <p>🎓 ML Clásico - Predicción de Abandono Escolar | Artificial Intelligence Foundations | Fundació URV</p>
    <p>Desarrollado con Streamlit, Scikit-learn y Plotly | Abril 2026</p>
</div>
""", unsafe_allow_html=True)
