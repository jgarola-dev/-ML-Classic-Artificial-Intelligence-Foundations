"""
Streamlit app for student dropout prediction
"""

import sys
import os

# Añadir el directorio src al path para importar módulos locales
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Ahora sí, importar tus módulos
from data_preprocessing import prepare_data
from model import DropoutPredictor
import pandas as pd

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_preprocessing import prepare_data, handle_missing_values, encode_categorical
from model import DropoutPredictor, compare_models
import io

# Page configuration
st.set_page_config(
    page_title="Predicción de Abandono Escolar",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        color: #1f77b4;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 10px;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success {
        color: #00c853;
        font-weight: bold;
    }
    .warning {
        color: #ff9100;
        font-weight: bold;
    }
    .danger {
        color: #d32f2f;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Create sample dataset
def create_sample_dataset(n_samples=200):
    """Create a sample dataset for demonstration"""
    np.random.seed(42)
    
    data = {
        'Edad': np.random.randint(15, 25, n_samples),
        'GPA': np.random.uniform(1.5, 4.0, n_samples),
        'Asistencia': np.random.uniform(50, 100, n_samples),
        'Horas_Estudio': np.random.uniform(0, 10, n_samples),
        'Socioeconomico': np.random.choice(['Bajo', 'Medio', 'Alto'], n_samples),
        'Primer_Trimestre': np.random.choice(['Aprobado', 'Reprobado'], n_samples),
        'Motivacion': np.random.choice(['Baja', 'Media', 'Alta'], n_samples),
        'Abandono': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    return pd.DataFrame(data)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None

# Sidebar
st.sidebar.title("📋 Navegación")
page = st.sidebar.radio("Selecciona una sección:", 
    ["🏠 Inicio", "📊 Análisis EDA", "🤖 Entrenamiento", "🔮 Predicción"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Cargar Datos")
uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV", type=['csv'])

if uploaded_file is not None:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Archivo cargado correctamente")
else:
    if st.sidebar.button("Usar datos de demostración"):
        st.session_state.df = create_sample_dataset()
        st.sidebar.success("✅ Datos de demostración cargados")

if st.session_state.df is not None:
    st.sidebar.markdown(f"**Tamaño del dataset:** {st.session_state.df.shape[0]} filas, {st.session_state.df.shape[1]} columnas")

# PAGE: INICIO
if page == "🏠 Inicio":
    st.markdown('<div class="main-title">🎓 Predicción de Abandono Escolar</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## 📚 Sobre el Proyecto
        
        Este proyecto implementa un **modelo de Machine Learning clásico** para predecir 
        el riesgo de abandono escolar en estudiantes.
        
        ### 🎯 Objetivos
        - Identificar estudiantes en riesgo de abandono
        - Proporcionar recomendaciones de intervención
        - Analizar factores clave del abandono escolar
        
        ### 📋 Formulación del Problema
        - **Tipo de aprendizaje:** Supervisado
        - **Tarea:** Clasificación binaria
        - **Variable objetivo:** Abandono (0/1)
        - **Métrica de éxito:** F1-Score, ROC-AUC
        """)
    
    with col2:
        st.markdown("""
        ## 🤖 Modelos Implementados
        
        Se utilizan 4 modelos ML clásicos:
        
        1. **Logistic Regression**
           - Modelo lineal simple y interpretable
        
        2. **Random Forest**
           - Ensemble de árboles de decisión
        
        3. **Gradient Boosting**
           - Boosting secuencial de árboles
        
        4. **Support Vector Machine (SVM)**
           - Máquinas de vectores de soporte
        """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Modelos", "4", "Clasificadores")
    with col2:
        st.metric("📈 Métricas", "6+", "Evaluación")
    with col3:
        st.metric("🎓 Categoría", "ML Clásico", "Fundació URV")
    
    st.markdown("---")
    st.markdown("""
    ### 📊 Características del Dataset
    
    El modelo utiliza las siguientes características:
    - **Edad:** Edad del estudiante
    - **GPA:** Promedio de calificaciones
    - **Asistencia:** Porcentaje de asistencia
    - **Horas_Estudio:** Horas de estudio semanales
    - **Socioeconomico:** Nivel socioeconómico
    - **Primer_Trimestre:** Desempeño primer trimestre
    - **Motivacion:** Nivel de motivación
    """)

# PAGE: ANÁLISIS EDA
elif page == "📊 Análisis EDA":
    st.title("📊 Análisis Exploratorio de Datos")
    
    if st.session_state.df is None:
        st.warning("⚠️ Por favor, carga datos primero en la barra lateral")
    else:
        df = st.session_state.df
        
        # Dataset overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Registros", len(df))
        with col2:
            st.metric("Características", df.shape[1])
        with col3:
            st.metric("Valores Faltantes", df.isnull().sum().sum())
        with col4:
            if 'Abandono' in df.columns:
                abandono_pct = (df['Abandono'].sum() / len(df) * 100)
                st.metric("Tasa de Abandono", f"{abandono_pct:.1f}%")
        
        st.markdown("---")
        
        # Data preview
        with st.expander("📋 Vista previa de datos"):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Statistics
        st.subheader("📊 Estadísticas Descriptivas")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.markdown("---")
        
        # Visualizations
        st.subheader("📈 Visualizaciones")
        
        col1, col2 = st.columns(2)
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Abandono' in numeric_cols:
            numeric_cols.remove('Abandono')
        
        with col1:
            if numeric_cols:
                selected_col = st.selectbox("Selecciona una característica numérica:", numeric_cols)
                
                if 'Abandono' in df.columns:
                    fig = px.box(df, x='Abandono', y=selected_col, 
                                title=f"Distribución de {selected_col} por Abandono",
                                color='Abandono')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.histogram(df, x=selected_col, nbins=30,
                                      title=f"Distribución de {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Abandono' in df.columns:
                abandono_counts = df['Abandono'].value_counts()
                fig = px.pie(values=abandono_counts.values, 
                            names=['No Abandono', 'Abandono'],
                            title="Distribución de Abandonos",
                            color_discrete_sequence=['#00c853', '#d32f2f'])
                st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("🔗 Matriz de Correlación")
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            fig = px.imshow(corr_matrix, color_continuous_scale='RdBu',
                           title="Matriz de Correlación",
                           labels=dict(color="Correlación"))
            st.plotly_chart(fig, use_container_width=True)

# PAGE: ENTRENAMIENTO
elif page == "🤖 Entrenamiento":
    st.title("🤖 Entrenamiento de Modelos")
    
    if st.session_state.df is None:
        st.warning("⚠️ Por favor, carga datos primero en la barra lateral")
    else:
        df = st.session_state.df
        
        # Check if target column exists
        if 'Abandono' not in df.columns:
            st.error("❌ La columna 'Abandono' no existe en el dataset")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("⚙️ Configuración")
                test_size = st.slider("Tamaño del conjunto de prueba (%)", 10, 40, 20) / 100
            
            with col2:
                st.subheader("📊 División de Datos")
                st.info(f"Training: {int((1-test_size)*100)}% | Test: {int(test_size*100)}%")
            
            if st.button("🚀 Entrenar Modelos", key='train'):
                with st.spinner("Entrenando modelos... (esto puede tomar un momento)"):
                    try:
                        # Prepare data
                        from sklearn.model_selection import train_test_split
                        
                        df_clean = handle_missing_values(df)
                        X = df_clean.drop(columns=['Abandono'])
                        y = df_clean['Abandono']
                        
                        X_encoded, encoders = encode_categorical(X)
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_encoded, y, test_size=test_size, random_state=42, stratify=y
                        )
                        
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                        
                        # Train and compare models
                        results = compare_models(X_train, X_test, y_train, y_test)
                        st.session_state.results = results
                        st.session_state.feature_names = X_encoded.columns.tolist()
                        st.session_state.X_test = X_test
                        st.session_state.y_test = y_test
                        
                        st.success("✅ Modelos entrenados correctamente")
                        
                    except Exception as e:
                        st.error(f"❌ Error al entrenar: {str(e)}")
            
            # Display results
            if hasattr(st.session_state, 'results'):
                st.markdown("---")
                st.subheader("📊 Resultados de Modelos")
                
                results_data = []
                for model_name, result in st.session_state.results.items():
                    metrics = result['metrics']
                    results_data.append({
                        'Modelo': model_name.replace('_', ' ').title(),
                        'Accuracy': f"{metrics['accuracy']:.4f}",
                        'Precision': f"{metrics['precision']:.4f}",
                        'Recall': f"{metrics['recall']:.4f}",
                        'F1-Score': f"{metrics['f1']:.4f}",
                        'ROC-AUC': f"{metrics.get('roc_auc', 'N/A')}"
                    })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Model comparison chart
                metrics_for_plot = []
                for model_name, result in st.session_state.results.items():
                    metrics = result['metrics']
                    metrics_for_plot.append({
                        'Modelo': model_name.replace('_', ' ').title(),
                        'Accuracy': metrics['accuracy'],
                        'F1-Score': metrics['f1'],
                        'ROC-AUC': metrics.get('roc_auc', 0)
                    })
                
                plot_df = pd.DataFrame(metrics_for_plot)
                fig = px.bar(plot_df, x='Modelo', y=['Accuracy', 'F1-Score', 'ROC-AUC'],
                            barmode='group', title="Comparación de Modelos")
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance
                st.markdown("---")
                st.subheader("🔍 Importancia de Características")
                
                selected_model = st.selectbox("Selecciona un modelo:", 
                    list(st.session_state.results.keys()))
                
                model_obj = st.session_state.results[selected_model]['model']
                importance_df = model_obj.get_feature_importance(
                    st.session_state.feature_names, top_n=10
                )
                
                if importance_df is not None:
                    fig = px.bar(importance_df, x='importance', y='feature',
                                orientation='h', title="Top 10 Características",
                                labels={'importance': 'Importancia', 'feature': 'Característica'})
                    st.plotly_chart(fig, use_container_width=True)

# PAGE: PREDICCIÓN
elif page == "🔮 Predicción":
    st.title("🔮 Predicción Individual")
    
    if st.session_state.df is None:
        st.warning("⚠️ Por favor, carga datos primero en la barra lateral")
    else:
        if not hasattr(st.session_state, 'results'):
            st.info("ℹ️ Primero debes entrenar los modelos en la sección de Entrenamiento")
        else:
            st.subheader("📋 Ingresa los datos del estudiante")
            
            col1, col2 = st.columns(2)
            
            with col1:
                edad = st.number_input("Edad", min_value=15, max_value=25, value=18)
                gpa = st.slider("GPA", 1.5, 4.0, 3.0)
                asistencia = st.slider("Asistencia (%)", 50, 100, 85)
                horas_estudio = st.slider("Horas de Estudio (semanales)", 0, 10, 5)
            
            with col2:
                socioeconomico = st.selectbox("Nivel Socioeconómico", 
                    ['Bajo', 'Medio', 'Alto'])
                primer_trimestre = st.selectbox("Desempeño Primer Trimestre", 
                    ['Aprobado', 'Reprobado'])
                motivacion = st.selectbox("Nivel de Motivación", 
                    ['Baja', 'Media', 'Alta'])
            
            if st.button("🔮 Realizar Predicción"):
                try:
                    # Create prediction data
                    pred_data = pd.DataFrame({
                        'Edad': [edad],
                        'GPA': [gpa],
                        'Asistencia': [asistencia],
                        'Horas_Estudio': [horas_estudio],
                        'Socioeconomico': [socioeconomico],
                        'Primer_Trimestre': [primer_trimestre],
                        'Motivacion': [motivacion]
                    })
                    
                    # Encode and scale
                    from data_preprocessing import encode_categorical
                    pred_encoded, _ = encode_categorical(pred_data)
                    
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_sample = scaler.fit_transform(pred_encoded)
                    
                    # Make predictions
                    st.markdown("---")
                    st.subheader("🎯 Resultados de Predicción")
                    
                    results_cols = st.columns(len(st.session_state.results))
                    
                    for idx, (model_name, result) in enumerate(st.session_state.results.items()):
                        with results_cols[idx]:
                            model_obj = result['model']
                            pred = model_obj.predict(X_sample)[0]
                            
                            if hasattr(model_obj.model, 'predict_proba'):
                                prob = model_obj.model.predict_proba(X_sample)[0][1]
                            else:
                                prob = None
                            
                            recommendation = model_obj.get_recommendation(pred, prob)
                            
                            st.markdown(f"### {model_name.replace('_', ' ').title()}")
                            if prob:
                                st.metric("Probabilidad de Abandono", f"{prob:.1%}")
                            st.markdown(recommendation)
                
                except Exception as e:
                    st.error(f"❌ Error en predicción: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9em;">
    <p>🎓 Predicción de Abandono Escolar | ML Clásico | Artificial Intelligence Foundations (Fundació URV)</p>
    <p>Desarrollado con Streamlit, Scikit-learn y Plotly</p>
</div>
""", unsafe_allow_html=True)
