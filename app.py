"""
Streamlit app for student dropout prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys
from data_loader import load_and_prepare_data
from data_preprocessing import prepare_data
from model import DropoutPredictor
import warnings
warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(
    page_title="🎓 ML Clásico - Predicción de Abandono Escolar",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estado de sesión
if 'df' not in st.session_state: st.session_state.df = None
if 'prep_data' not in st.session_state: st.session_state.prep_data = None
if 'models' not in st.session_state: st.session_state.models = {}

# Sidebar
st.sidebar.title("📋 Navegación")
page = st.sidebar.radio("Selecciona:", ["🏠 Inicio", "📊 EDA", "🤖 Entrenar", "🔮 Predecir"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Cargar Datos")
load_btn = st.sidebar.button("🌐 Cargar Dataset UCI Oficial")
demo_btn = st.sidebar.button("🎲 Usar Datos Demo")

if load_btn:
    df, msg = load_and_prepare_data(source='uci')
    st.session_state.df = df
    st.sidebar.success(msg)
elif demo_btn:
    df, msg = load_and_prepare_data(source='demo')
    st.session_state.df = df
    st.sidebar.success(msg)

if st.session_state.df is not None:
    st.sidebar.info(f"📊 Dataset: {len(st.session_state.df)} filas × {len(st.session_state.df.columns)} columnas")

# ==================== PÁGINAS ====================
if page == "🏠 Inicio":
    st.markdown('<div class="main-title">🎓 Predicción de Abandono Escolar</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Renderizar README.md dinámicamente
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning("⚠️ `README.md` no encontrado en la carpeta raíz. Asegúrate de subirlo al repositorio.")
        st.markdown("""
        ## 📋 Proyecto ML Clásico
        - **Tipo**: Clasificación binaria supervisada
        - **Dataset**: UCI Student Dropout (≥500 registros)
        - **Modelos**: Logistic Regression, Random Forest, Gradient Boosting, SVM
        - **Métrica principal**: F1-Score (macro)
        """)

elif page == "📊 EDA":
    st.title("📊 Análisis Exploratorio de Datos")
    if st.session_state.df is None:
        st.warning("⚠️ Carga datos primero en la barra lateral")
    else:
        df = st.session_state.df
        # Buscar columna target
        target_candidates = ['target', 'status', 'abandono', 'y']
        target_col = next((c for c in target_candidates if c in df.columns), None)
        
        if target_col is None:
            st.error("❌ No se encontró columna objetivo. El dataset debe contener 'target', 'status' o similar.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Registros", f"{len(df):,}")
            c2.metric("Features", f"{len(df.columns)-1}")
            c3.metric("Dropout Rate", f"{(df[target_col].astype(str).str.lower().str.contains('dropout')).mean()*100:.1f}%")
            
            st.bar_chart(df[target_col].value_counts(), use_container_width=True)
            
            if df.select_dtypes('number').shape[1] > 2:
                st.subheader("🔗 Top Correlaciones Numéricas")
                corr = df.select_dtypes('number').corr().abs()
                top = corr[corr.columns[0]].sort_values(ascending=False).head(10).drop(corr.columns[0], errors='ignore')
                st.bar_chart(top, use_container_width=True)

elif page == "🤖 Entrenar":
    st.title("🤖 Entrenamiento de Modelos")
    if st.session_state.df is None:
        st.warning("⚠️ Carga datos primero")
    else:
        df = st.session_state.df
        target_col = next((c for c in ['target', 'status', 'abandono'] if c in df.columns), None)
        
        if not target_col:
            st.error("❌ Columna objetivo no encontrada")
        elif st.button("🚀 Entrenar 4 Modelos"):
            with st.spinner("Entrenando... (esto toma ~15s)"):
                try:
                    data = prepare_data(df, target_col=target_col)
                    st.session_state.prep_data = data
                    
                    models_list = ['logistic_regression', 'random_forest', 'gradient_boosting', 'svm']
                    for m in models_list:
                        clf = DropoutPredictor(model_type=m)
                        clf.train(data['X_train'], data['y_train'])
                        st.session_state.models[m] = clf
                        
                    st.success("✅ Modelos entrenados y transformadores guardados")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    
        if st.session_state.prep_
            st.subheader("📊 Comparativa de Modelos")
            rows = []
            for name, clf in st.session_state.models.items():
                m = clf.evaluate(st.session_state.prep_data['X_test'], st.session_state.prep_data['y_test'])
                rows.append({'Modelo': name.replace('_', ' ').title(), **m})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            
            st.subheader("🔍 Importancia de Características")
            sel = st.selectbox("Modelo:", list(st.session_state.models.keys()))
            imp = st.session_state.models[sel].get_feature_importance(
                st.session_state.prep_data['feature_names'], top_n=10
            )
            if imp is not None and not imp.empty:
                st.plotly_chart(px.bar(imp, x='importance', y='feature', orientation='h'), use_container_width=True)

elif page == "🔮 Predecir":
    st.title("🔮 Predicción Individual")
    if not st.session_state.models:
        st.info("ℹ️ Entrena modelos primero en 🤖 Entrenar")
    else:
        data = st.session_state.prep_data
        st.markdown("📌 *Introduce valores para ver la sensibilidad del modelo*")
        
        # Inputs dinámicos según features disponibles
        col1, col2 = st.columns(2)
        with col1:
            # Features clave UCI (si existen, sino sliders genéricos)
            adm = st.slider("📝 Nota admisión", 0, 200, 100)
            g1 = st.slider("📚 Nota 1er sem", 0.0, 20.0, 10.0)
            app1 = st.slider("✅ Aprobadas 1er sem", 0, 15, 7)
            age = st.number_input("🎂 Edad", 17, 50, 19)
        with col2:
            bec = st.selectbox("🎁 Beca", [0, 1], format_func=lambda x: "Sí" if x else "No")
            desp = st.slider("📉 Desempleo (%)", 0.0, 30.0, 10.0)
            infl = st.slider("📈 Inflación (%)", -5.0, 15.0, 2.0)
            att = st.selectbox("🕒 Horario", [1, 2], format_func=lambda x: "Mañana" if x==1 else "Tarde")
            
        if st.button("🔮 Predecir Trayectoria", type="primary"):
            try:
                # 1. Crear input con TODAS las columnas del entrenamiento
                pred_dict = {
                    'admission_grade': adm,
                    'curricular_units_1st_sem_grade': g1,
                    'curricular_units_1st_sem_approved': app1,
                    'age_at_enrollment': age,
                    'scholarship_holder': bec,
                    'unemployment_rate': desp,
                    'inflation_rate': infl,
                    'daytime_evening_attendance': att
                }
                # Rellenar faltantes con medianas/modas del entrenamiento
                for col in data['feature_names']:
                    if col not in pred_dict:
                        pred_dict[col] = data['medians'].get(col, 0) if col in data['num_cols'] else data['modes'].get(col, 0)
                        
                pred_df = pd.DataFrame([pred_dict])
                
                # 2. Aplicar EXACTAMENTE los mismos encoders del entrenamiento
                for col in data['cat_cols']:
                    le = data['encoders'][col]
                    val = str(pred_df[col].values[0])
                    pred_df[col] = le.transform([val])[0] if val in le.classes_ else 0
                    
                # 3. Escalar y predecir
                pred_df = pred_df[data['feature_names']]
                pred_scaled = data['scaler'].transform(pred_df)
                
                st.markdown("---")
                st.subheader("🎯 Resultados")
                cols = st.columns(4)
                for i, (name, clf) in enumerate(st.session_state.models.items()):
                    with cols[i]:
                        p = clf.model.predict(pred_scaled)[0]
                        prob = clf.model.predict_proba(pred_scaled)[0][1] if hasattr(clf.model, 'predict_proba') else 0
                        risk = "🔴 ALTO RIESGO" if p == 1 else "🟢 BAJO RIESGO"
                        st.metric(name.replace('_', ' ').title(), risk, delta=f"{prob:.1%}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
