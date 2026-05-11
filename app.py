"""
Streamlit App: ML Clásico - Predicción de Abandono Escolar
Artificial Intelligence Foundations | Fundació URV | 2026
"""
import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_preprocessing import create_realistic_dataset, prepare_data
from model import compare_and_tune
import warnings
warnings.filterwarnings('ignore')

# 🔧 Fix para imports en Streamlit Cloud
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

st.set_page_config(page_title="🎓 Predicción de Abandono", page_icon="🎓", layout="wide")

# Estado de sesión
if 'df' not in st.session_state: st.session_state.df = None
if 'prep' not in st.session_state: st.session_state.prep = None
if 'models' not in st.session_state: st.session_state.models = None
if 'tuning' not in st.session_state: st.session_state.tuning = None

# Sidebar
st.sidebar.title("📋 Navegación")
page = st.sidebar.radio("Selecciona:", ["🏠 Inicio", "📊 EDA", "🤖 Entrenar", "🔮 Predecir"])
st.sidebar.markdown("---")
if st.sidebar.button("🌐 Cargar Dataset Realista"):
    st.session_state.df = create_realistic_dataset()
    st.sidebar.success("✅ Dataset cargado con correlaciones reales")

if st.session_state.df is not None:
    st.sidebar.info(f"📊 Dataset: {len(st.session_state.df)} filas × {len(st.session_state.df.columns)} columnas")

# ==================== PÁGINAS ====================
if page == "🏠 Inicio":
    st.title("🎓 Predicción de Abandono Escolar")
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        ## 📚 Sobre el Proyecto
        Pipeline de **ML Clásico** para predecir riesgo de abandono escolar.
        ### 🎯 Objetivos
        - Identificar estudiantes en riesgo
        - Analizar factores académicos y socioeconómicos
        - Facilitar intervención temprana basada en datos
        
        ### 📋 Formulación
        - **Tipo:** Supervisado | **Tarea:** Clasificación Binaria
        - **Target:** `Abandono` (0: No, 1: Sí)
        - **Métrica:** F1-Score, ROC-AUC
        """)
    with c2:
        st.markdown("""
        ## 🤖 Modelos Implementados
        1. **Logistic Regression** (Lineal, interpretable)
        2. **Random Forest** (Ensemble, robusto)
        3. **Gradient Boosting** (Secuencial, alta precisión)
        4. **SVM** (Kernel RBF, óptimo en alta dimensión)
        """)
    st.metric("📊 Modelos", "4"); st.metric("📈 Métricas", "6+"); st.metric("🎓 Categoría", "ML Clásico")

elif page == "📊 EDA":
    st.title("📊 Análisis Exploratorio de Datos")
    if st.session_state.df is None:
        st.warning("⚠️ Carga datos primero en la barra lateral")
    else:
        df = st.session_state.df
        target = 'Abandono'
        
        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Registros", f"{len(df):,}")
        c2.metric("Features", len(df.columns)-1)
        c3.metric("Faltantes", df.isnull().sum().sum())
        c4.metric("Tasa Abandono", f"{df[target].mean()*100:.1f}%")
        st.markdown("---")
        
        # Matriz de Correlación
        st.subheader("🔗 Matriz de Correlación (Pearson)")
        num_cols = df.select_dtypes('number').columns
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                           title="Correlaciones entre Variables Numéricas")
            st.plotly_chart(fig, use_container_width=True)
            
            # Top correlaciones
            pairs = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    pairs.append({'Var1': corr.columns[i], 'Var2': corr.columns[j], 'r': corr.iloc[i,j]})
            top = pd.DataFrame(pairs).sort_values('r', key=abs, ascending=False).head(5)
            st.dataframe(top.style.format({'r': '{:.3f}'}), use_container_width=True)
            
        # Distribuciones
        st.subheader("📈 Distribución de Variables Clave")
        sel = st.selectbox("Selecciona variable:", num_cols)
        if sel:
            fig = px.box(df, x=target, y=sel, color=target,
                        title=f"{sel} por Clase de Abandono",
                        color_discrete_map={0: '#28a745', 1: '#dc3545'})
            st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Entrenar":
    st.title("🤖 Entrenamiento & Ajuste de Hiperparámetros")
    if st.session_state.df is None:
        st.warning("⚠️ Carga datos primero")
    elif st.button("🚀 Entrenar 4 Modelos + GridSearchCV"):
        with st.spinner("Entrenando y optimizando... (~15s)"):
            try:
                prep = prepare_data(st.session_state.df, target_col='Abandono')
                st.session_state.prep = prep
                
                results, tuning = compare_and_tune(
                    prep['X_train'], prep['y_train'],
                    prep['X_test'], prep['y_test'],
                    prep['feature_names']
                )
                st.session_state.models = results
                st.session_state.tuning = tuning
                st.success("✅ Entrenamiento y optimización completados")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                
    if st.session_state.models:
        st.subheader("📊 Comparativa de Modelos")
        rows = []
        for name, res in st.session_state.models.items():
            m = res['metrics']
            rows.append({'Modelo': name.replace('_',' ').title(), 
                        'Accuracy': f"{m['accuracy']:.3f}", 'Precision': f"{m['precision']:.3f}",
                        'Recall': f"{m['recall']:.3f}", 'F1': f"{m['f1']:.3f}", 'ROC-AUC': f"{m['roc_auc']:.3f}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        
        st.markdown("---")
        st.subheader("🔧 Ajuste de Hiperparámetros (Random Forest)")
        if st.session_state.tuning:
            t = st.session_state.tuning
            st.info(f"🏆 F1 Baseline: `{t['baseline_f1']:.3f}` → Optimizado: `{t['best_f1']:.3f}` | **Mejora: `{t['improvement']:.1f}%`**")
            st.code(f"Mejores parámetros: {t['best_params']}", language="python")
            
        st.subheader("🔍 Importancia de Características")
        sel_model = st.selectbox("Modelo:", list(st.session_state.models.keys()))
        imp_df = st.session_state.models[sel_model]['importance']
        if imp_df is not None and not imp_df.empty:
            st.plotly_chart(px.bar(imp_df, x='importance', y='feature', orientation='h',
                                title=f"Top Features ({sel_model.replace('_',' ').title()})"),
                           use_container_width=True)

elif page == "🔮 Predecir":
    st.title("🔮 Predicción Individual")
    if not st.session_state.models:
        st.info("ℹ️ Entrena modelos primero en 🤖 Entrenar")
    else:
        prep = st.session_state.prep
        st.markdown("📌 *Introduce valores extremos para verificar sensibilidad del modelo*")
        
        c1, c2 = st.columns(2)
        with c1:
            gpa = st.slider("GPA (0-5)", 0.0, 5.0, 2.0)
            asistencia = st.slider("Asistencia (%)", 0, 100, 60)
            horas = st.slider("Horas Estudio/Semana", 0, 15, 3)
        with c2:
            motiv = st.selectbox("Motivación", ['Baja', 'Media', 'Alta'])
            trim = st.selectbox("1er Trimestre", ['Aprobado', 'Reprobado'])
            socio = st.selectbox("Nivel Socioeconómico", ['Bajo', 'Medio', 'Alto'])
            
        if st.button("🔮 Predecir Trayectoria", type="primary"):
            try:
                # 1. Crear input
                pred_dict = {'GPA': gpa, 'Asistencia': asistencia, 'Horas_Estudio': horas,
                             'Motivacion': motiv, 'Primer_Trimestre': trim, 'Socioeconomico': socio}
                
                # 2. Rellenar columnas faltantes con medianas/modas del entrenamiento
                for feat in prep['feature_names']:
                    if feat not in pred_dict:
                        pred_dict[feat] = prep['medians'].get(feat, 0) if feat in prep['num_cols'] else prep['modes'].get(feat, 0)
                        
                pred_df = pd.DataFrame([pred_dict])
                
                # 3. Aplicar EXACTAMENTE los mismos encoders del entrenamiento
                for col in prep['cat_cols']:
                    le = prep['encoders'][col]
                    val = str(pred_df[col].values[0])
                    pred_df[col] = le.transform([val])[0] if val in le.classes_ else 0
                    
                # 4. Reordenar y escalar
                pred_df = pred_df[prep['feature_names']]
                X_scaled = prep['scaler'].transform(pred_df)
                
                # 5. Predecir y mostrar
                st.markdown("---")
                st.subheader("🎯 Resultados")
                cols = st.columns(4)
                for i, (name, res) in enumerate(st.session_state.models.items()):
                    with cols[i]:
                        p = res['model'].model.predict(X_scaled)[0]
                        prob = res['model'].model.predict_proba(X_scaled)[0][1] if hasattr(res['model'].model, 'predict_proba') else 0.5
                        risk = "🔴 ALTO RIESGO" if p == 1 else "🟢 BAJO RIESGO"
                        st.metric(name.replace('_',' ').title(), risk, delta=f"{prob:.1%}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
