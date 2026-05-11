import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from data_preprocessing import clean_uci_columns, prepare_uci_data
from model import DropoutPredictor
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="🎓 Predicción Abandono Escolar", page_icon="🎓", layout="wide")

# ==================== ESTADO INICIAL ====================
if 'df' not in st.session_state: st.session_state.df = None
if 'prep_data' not in st.session_state: st.session_state.prep_data = None
if 'models' not in st.session_state: st.session_state.models = {}

# ==================== SIDEBAR ====================
st.sidebar.title("📋 Navegación")
page = st.sidebar.radio("Selecciona:", ["🏠 Inicio", "📊 EDA", "🤖 Entrenar", "🔮 Predecir"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Cargar Datos")
uploaded_file = st.sidebar.file_uploader("Subir CSV (columna 'Target' requerida)", type=['csv'])
if uploaded_file:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.session_state.df = clean_uci_columns(st.session_state.df)
    st.sidebar.success("✅ CSV cargado y limpiado")
elif st.sidebar.button("Cargar Dataset UCI Oficial"):
    try:
        # Descarga directa desde UCI (alternativa: descarga manual)
        url = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.csv"
        st.session_state.df = pd.read_csv(url)
        st.session_state.df = clean_uci_columns(st.session_state.df)
        st.sidebar.success("✅ Dataset UCI cargado")
    except Exception as e:
        st.sidebar.error(f"❌ Error cargando UCI: {str(e)}")

if st.session_state.df is not None:
    st.sidebar.info(f"📊 Dataset: {len(st.session_state.df)} filas × {len(st.session_state.df.columns)} columnas")

# ==================== PÁGINAS ====================
if page == "🏠 Inicio":
    st.title("🎓 ML Clásico - Predicción de Abandono Escolar")
    st.markdown("Proyecto para *Artificial Intelligence Foundations* | Fundació URV")
    st.info("✅ Pipeline completo: UCI Dataset (36 cols) → Preprocessing → 4 Modelos → Streamlit")
    st.markdown("---")
    st.markdown("📌 **Características clave:**\n"
                "- Clasificación binaria: `Dropout` vs `Graduate`\n"
                "- Métrica principal: F1-Score (macro)\n"
                "- Modelos: Random Forest, Gradient Boost, Logistic, SVM\n"
                "- Predicción robusta con reutilización de transformadores")

elif page == "📊 EDA":
    st.title("📊 Análisis Exploratorio")
    if st.session_state.df is None:
        st.warning("⚠️ Carga datos primero")
    else:
        df = st.session_state.df
        if 'target' not in df.columns:
            st.error("❌ El dataset debe contener columna 'target'")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total", len(df))
            c2.metric("Features", len(df.columns)-1)
            c3.metric("Dropout Rate", f"{(df['target']=='dropout').mean()*100:.1f}%")
            
            fig = px.histogram(df, x='target', title="Distribución Clases")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("🔗 Correlación Top 10 con Target")
            num_cols = df.select_dtypes('number').columns
            corr = df[num_cols].corr()['target'].abs().sort_values(ascending=False).head(10)
            st.bar_chart(corr.drop('target', errors='ignore'))

elif page == "🤖 Entrenar":
    st.title("🤖 Entrenamiento de Modelos")
    if st.session_state.df is None or 'target' not in st.session_state.df.columns:
        st.warning("⚠️ Carga dataset con columna 'target'")
    elif st.button("🚀 Entrenar 4 Modelos"):
        with st.spinner("Entrenando... (esto toma ~10-20s)"):
            try:
                data = prepare_uci_data(st.session_state.df, target_col='target')
                st.session_state.prep_data = data
                
                models_to_train = ['random_forest', 'gradient_boost', 'logistic', 'svm']
                for m in models_to_train:
                    clf = DropoutPredictor(m)
                    clf.train(data['X_train'], data['y_train'])
                    st.session_state.models[m] = clf
                    
                st.success("✅ Modelos entrenados y transformadores guardados en memoria")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                
    if st.session_state.prep_data:
        st.subheader("📊 Resultados")
        res = []
        for name, clf in st.session_state.models.items():
            m = clf.evaluate(st.session_state.prep_data['X_test'], st.session_state.prep_data['y_test'])
            res.append({'Modelo': name.title().replace('_',' '), **m})
        st.dataframe(pd.DataFrame(res), use_container_width=True)
        
        st.subheader("🔍 Importancia de Características")
        sel_model = st.selectbox("Modelo:", list(st.session_state.models.keys()))
        imp_df = st.session_state.models[sel_model].get_feature_importance(
            st.session_state.prep_data['X_test'],
            st.session_state.prep_data['y_test'],
            st.session_state.prep_data['feature_names']
        )
        if imp_df is not None and not imp_df.empty:
            st.plotly_chart(px.bar(imp_df, x='importance', y='feature', orientation='h'), use_container_width=True)

elif page == "🔮 Predecir":
    st.title("🔮 Predicción Individual")
    if not st.session_state.models:
        st.info("ℹ️ Entrena modelos primero en la pestaña 🤖")
    else:
        data = st.session_state.prep_data
        st.markdown("📌 *Introduce valores extremos para verificar sensibilidad del modelo*")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            adm_grade = st.slider("📝 Nota admisión (0-200)", 0, 200, 95)
            sem1_grade = st.slider("📚 Nota 1er sem (0-20)", 0.0, 20.0, 9.5)
            age = st.number_input("🎂 Edad", 17, 50, 19)
        with col2:
            app_units = st.slider("✅ Asignaturas aprobadas 1er sem", 0, 10, 5)
            unemployment = st.slider("📉 Desempleo (%)", 0, 30, 10)
            inflation = st.slider("📈 Inflación (%)", -5, 15, 2)
        with col3:
            scholarship = st.selectbox("🎁 Beca", [0, 1], format_func=lambda x: "Sí" if x else "No")
            gender = st.selectbox("⚧ Género", [1, 2], format_func=lambda x: "Hombre" if x==1 else "Mujer")
            attendance = st.selectbox("🕒 Horario", [1, 2], format_func=lambda x: "Mañana" if x==1 else "Tarde")
            
        if st.button("🔮 Predecir Trayectoria", type="primary"):
            try:
                # 1. Crear DataFrame con las columnas exactas del entrenamiento
                pred_dict = {
                    'admission_grade': adm_grade,
                    'curricular_units_1st_sem__grade_': sem1_grade,
                    'curricular_units_1st_sem__approved_': app_units,
                    'age_at_enrollment': age,
                    'unemployment_rate': unemployment,
                    'inflation_rate': inflation,
                    'scholarship_holder': scholarship,
                    'gender': gender,
                    'daytime_evening_attendance': attendance
                }
                
                # 2. Rellenar las 36 columnas faltantes con medianas/modas del entrenamiento
                for col in data['feature_names']:
                    if col not in pred_dict:
                        if col in data['num_cols']:
                            pred_dict[col] = data['medians'][col]
                        elif col in data['cat_cols']:
                            pred_dict[col] = data['modes'][col]
                            
                pred_df = pd.DataFrame([pred_dict])
                
                # 3. Aplicar EXACTAMENTE los mismos encoders del entrenamiento
                for col in data['cat_cols']:
                    le = data['encoders'][col]
                    val = str(pred_df[col].values[0])
                    # Manejar categorías no vistas
                    if val not in le.classes_:
                        val = le.classes_[0]
                    pred_df[col] = le.transform([val])[0]
                    
                # 4. Reordenar columnas y escalar
                pred_df = pred_df[data['feature_names']]
                pred_scaled = data['scaler'].transform(pred_df)
                
                # 5. Predecir
                st.markdown("---")
                st.subheader("🎯 Resultados")
                cols = st.columns(4)
                for i, (name, clf) in enumerate(st.session_state.models.items()):
                    with cols[i]:
                        p = clf.model.predict(pred_scaled)[0]
                        prob = clf.model.predict_proba(pred_scaled)[0][1]
                        risk = "🔴 ALTO RIESGO" if p == 1 else "🟢 BAJO RIESGO"
                        st.metric(name.title().replace('_',' '), risk, delta=f"{prob:.1%}")
                        
            except Exception as e:
                st.error(f"❌ Error en predicción: {str(e)}")
                st.exception(e)
