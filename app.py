"""
Streamlit app for student dropout prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_preprocessing import prepare_data, handle_missing_values, encode_categorical
from model import DropoutPredictor, compare_models

st.set_page_config(
    page_title="Predicción de Abandono Escolar",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'df' not in st.session_state: st.session_state.df = None
if 'results' not in st.session_state: st.session_state.results = {}

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

# Sidebar
st.sidebar.title("📋 Navegación")
page = st.sidebar.radio("Selecciona:", ["🏠 Inicio", "📊 EDA", "🤖 Entrenamiento", "🔮 Predicción"])
st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Cargar Datos")
uploaded_file = st.sidebar.file_uploader("Sube un CSV", type=['csv'])
if uploaded_file:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ CSV cargado")
elif st.sidebar.button("Usar datos demo"):
    st.session_state.df = create_sample_dataset()
    st.sidebar.success("✅ Datos demo cargados")

if st.session_state.df is not None:
    st.sidebar.info(f"Tamaño: {st.session_state.df.shape[0]} filas × {st.session_state.df.shape[1]} columnas")

# ==================== PAGES ====================
elif page == "🏠 Inicio":
    st.markdown('<h1 style="text-align:center; color:#1f77b4;">🎓 Predicción de Abandono Escolar</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Columnas principales: Proyecto vs Modelos
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ## 📚 Sobre el Proyecto
        Este proyecto implementa un **modelo de Machine Learning clásico** para predecir el riesgo de abandono escolar en estudiantes.

        ### 🎯 Objetivos
        - ✅ Identificar estudiantes en riesgo de abandono
        - 📋 Proporcionar recomendaciones de intervención temprana
        - 🔍 Analizar factores académicos y socioeconómicos clave

        ### 📋 Formulación del Problema
        | Aspecto | Detalle |
        |---------|---------|
        | **Tipo de aprendizaje** | Supervisado |
        | **Tarea** | Clasificación binaria |
        | **Variable objetivo** | Abandono (0/1) |
        | **Métrica de éxito** | F1-Score, ROC-AUC |
        """)

    with col2:
        st.markdown("""
        ## 🤖 Modelos Implementados
        Se utilizan 4 clasificadores clásicos de `scikit-learn`:

        1. **Logistic Regression** 📈
           - Modelo lineal, rápido y altamente interpretable

        2. **Random Forest** 🌲
           - Ensemble robusto, resistente a overfitting

        3. **Gradient Boosting** 🚀
           - Boosting secuencial para máxima precisión

        4. **Support Vector Machine (SVM)** 🎯
           - Óptimo para espacios de alta dimensionalidad
        """)

    st.markdown("---")

    # Métricas técnicas
    st.subheader("📊 Resumen Técnico")
    m1, m2, m3 = st.columns(3)
    m1.metric("🤖 Modelos", "4", "Clasificadores")
    m2.metric("📈 Métricas", "6+", "Evaluación")
    m3.metric("🎓 Categoría", "ML Clásico", "Fundació URV")

    st.markdown("---")

    # Características del Dataset (tabla interactiva)
    st.subheader("📊 Características del Dataset")
    st.info("El modelo utiliza las siguientes variables para realizar la predicción:")
    
    features_df = pd.DataFrame({
        "Característica": ["Edad", "GPA", "Asistencia", "Horas_Estudio", "Socioeconomico", "Primer_Trimestre", "Motivacion"],
        "Descripción": [
            "Edad del estudiante",
            "Promedio de calificaciones (0-5 o 0-20)",
            "Porcentaje de asistencia a clases",
            "Horas dedicadas al estudio semanal",
            "Nivel socioeconómico familiar",
            "Desempeño en el primer trimestre",
            "Nivel de motivación académica"
        ],
        "Tipo": ["Numérica", "Numérica", "Numérica", "Numérica", "Categórica", "Categórica", "Categórica"]
    })
    st.dataframe(features_df, use_container_width=True, hide_index=True)

elif page == "📊 EDA":
    st.title("📊 Análisis Exploratorio")
    if st.session_state.df is None:
        st.warning("⚠️ Carga datos primero")
    else:
        df = st.session_state.df
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Registros", len(df))
        c2.metric("Features", df.shape[1])
        c3.metric("Faltantes", df.isnull().sum().sum())
        if 'Abandono' in df.columns:
            c4.metric("Tasa Abandono", f"{df['Abandono'].mean()*100:.1f}%")
        
        st.dataframe(df.head(10), use_container_width=True)
        st.dataframe(df.describe(), use_container_width=True)
        
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Abandono' in num_cols: num_cols.remove('Abandono')
        if num_cols:
            fig = px.box(df, x='Abandono' if 'Abandono' in df.columns else None, y=num_cols[0], title=f"Distribución: {num_cols[0]}")
            st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Entrenamiento":
    st.title("🤖 Entrenamiento de Modelos")
    if st.session_state.df is None:
        st.warning("⚠️ Carga datos primero")
    elif 'Abandono' not in st.session_state.df.columns:
        st.error("❌ El dataset debe tener columna 'Abandono'")
    else:
        if st.button("🚀 Entrenar 4 Modelos"):
            with st.spinner("Entrenando..."):
                try:
                    df_clean = handle_missing_values(st.session_state.df)
                    X = df_clean.drop(columns=['Abandono'])
                    y = df_clean['Abandono']
                    X_encoded, encoders = encode_categorical(X)
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    
                    st.session_state.results = compare_models(X_train, X_test, y_train, y_test)
                    st.session_state.feature_names = X_encoded.columns.tolist()
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.success("✅ Modelos entrenados")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        if st.session_state.results:
            rows = []
            for name, res in st.session_state.results.items():
                m = res['metrics']
                rows.append({'Modelo': name.replace('_',' ').title(), 'Accuracy': f"{m['accuracy']:.4f}", 'Precision': f"{m['precision']:.4f}", 'Recall': f"{m['recall']:.4f}", 'F1': f"{m['f1']:.4f}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            
            sel = st.selectbox("Modelo para importancia:", list(st.session_state.results.keys()))
            imp = st.session_state.results[sel]['model'].get_feature_importance(st.session_state.feature_names)
            if imp is not None:
                st.plotly_chart(px.bar(imp, x='importance', y='feature', orientation='h'), use_container_width=True)

elif page == "🔮 Predicción":
    st.title("🔮 Predicción Individual")
    if not st.session_state.results:
        st.info("ℹ️ Entrena modelos primero")
    else:
        col1, col2 = st.columns(2)
        with col1:
            edad = st.number_input("Edad", 15, 25, 18)
            gpa = st.slider("GPA", 1.5, 4.0, 3.0)
            asistencia = st.slider("Asistencia (%)", 50, 100, 85)
            horas = st.slider("Horas Estudio", 0, 10, 5)
        with col2:
            socio = st.selectbox("Socioeconómico", ['Bajo', 'Medio', 'Alto'])
            trim = st.selectbox("1er Trimestre", ['Aprobado', 'Reprobado'])
            motiv = st.selectbox("Motivación", ['Baja', 'Media', 'Alta'])
            
        if st.button("🔮 Predecir"):
            try:
                pred = pd.DataFrame({'Edad': [edad], 'GPA': [gpa], 'Asistencia': [asistencia], 'Horas_Estudio': [horas], 'Socioeconomico': [socio], 'Primer_Trimestre': [trim], 'Motivacion': [motiv]})
                pred_enc, _ = encode_categorical(pred)
                from sklearn.preprocessing import StandardScaler
                pred_scaled = StandardScaler().fit_transform(pred_enc)
                
                st.subheader("🎯 Resultados")
                cols = st.columns(4)
                for i, (name, res) in enumerate(st.session_state.results.items()):
                    with cols[i]:
                        p = res['model'].predict(pred_scaled)[0]
                        prob = res['model'].predict_proba(pred_scaled)[0][1] if hasattr(res['model'].model, 'predict_proba') else 0
                        st.metric(name.replace('_',' ').title(), "🔴 ALTO" if p==1 else "🟢 BAJO", f"{prob:.1%}")
            except Exception as e:
                st.error(f"❌ Error: {e}")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#888;'>🎓 ML Clásico | Artificial Intelligence Foundations | Fundació URV</p>", unsafe_allow_html=True)
