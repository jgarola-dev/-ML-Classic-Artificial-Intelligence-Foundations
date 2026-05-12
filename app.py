"""
Streamlit app for student dropout prediction
"""
import sys
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# 🔧 Fix para importar módulos locales
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from data_preprocessing import prepare_data, handle_missing_values, encode_categorical
from model import DropoutPredictor, compare_models

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
.main-title { color: #1f77b4; text-align: center; font-size: 2.5em; margin-bottom: 10px; }
.metric-box { background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0; }
.success { color: #00c853; font-weight: bold; }
.warning { color: #ff9100; font-weight: bold; }
.danger { color: #d32f2f; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

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
if 'df' not in st.session_state: st.session_state.df = None
if 'results' not in st.session_state: st.session_state.results = None
if 'feature_names' not in st.session_state: st.session_state.feature_names = None
if 'scaler' not in st.session_state: st.session_state.scaler = None
if 'encoders' not in st.session_state: st.session_state.encoders = None

# ==================== SIDEBAR ====================
st.sidebar.title("📋 Navegación")
page = st.sidebar.radio("Selecciona una sección:", ["🏠 Inicio", "📊 Análisis EDA", "🤖 Entrenamiento", "🔮 Predicción"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Cargar Datos")

uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV", type=['csv'])
if uploaded_file is not None:
    try:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ Archivo cargado correctamente")
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")
else:
    if st.sidebar.button("Usar datos de demostración"):
        st.session_state.df = create_sample_dataset()
        st.sidebar.success("✅ Datos de demostración cargados")

if st.session_state.df is not None:
    st.sidebar.markdown(f"📊 Dataset activo: {len(st.session_state.df)} filas")

# ==================== PÁGINAS ====================

# 🏠 INICIO
if page == "🏠 Inicio":
    st.markdown('<div class="main-title">🎓 Predicción de Abandono Escolar</div>', unsafe_allow_html=True)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ## 📚 Sobre el Proyecto
        Este proyecto implementa un **modelo de Machine Learning clásico** para predecir el riesgo de abandono escolar.
        ### 🎯 Objetivos
        - Identificar estudiantes en riesgo
        - Proporcionar recomendaciones
        - Analizar factores clave
        ### 📋 Formulación
        - **Tipo:** Supervisado (Clasificación Binaria)
        - **Target:** Abandono (0/1)
        """)
    with col2:
        st.markdown("""
        ## 🤖 Modelos Implementados
        1. **Logistic Regression**
        2. **Random Forest**
        3. **Gradient Boosting**
        4. **Support Vector Machine (SVM)**
        """)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("📊 Modelos", "4")
    c2.metric("📈 Métricas", "6+")
    c3.metric("🎓 Categoría", "ML Clásico")
    
    st.markdown("---")
    st.markdown("""
    ### 📊 Características del Dataset
    - **Edad:** Edad del estudiante
    - **GPA:** Promedio de calificaciones
    - **Asistencia:** Porcentaje de asistencia
    - **Horas_Estudio:** Horas de estudio semanales
    - **Socioeconomico:** Nivel socioeconómico
    - **Primer_Trimestre:** Desempeño primer trimestre
    - **Motivacion:** Nivel de motivación
    """)

# 📊 EDA
elif page == "📊 Análisis EDA":
    st.title("📊 Análisis Exploratorio de Datos")
    if st.session_state.df is None:
        st.warning("⚠️ Por favor, carga datos primero")
    else:
        df = st.session_state.df
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Registros", len(df))
        col2.metric("Características", df.shape[1])
        col3.metric("Valores Faltantes", df.isnull().sum().sum())
        if 'Abandono' in df.columns:
            rate = (df['Abandono'].sum() / len(df)) * 100
            col4.metric("Tasa de Abandono", f"{rate:.1f}%")

        with st.expander("📋 Vista previa de datos"):
            st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("📊 Estadísticas Descriptivas")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.markdown("---")
        st.subheader("📈 Visualizaciones")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Abandono' in numeric_cols: numeric_cols.remove('Abandono')
        
        if numeric_cols:
            selected_col = st.selectbox("Selecciona una característica numérica:", numeric_cols)
            if 'Abandono' in df.columns:
                fig = px.box(df, x='Abandono', y=selected_col, color='Abandono', title=f"Distribución de {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("🔗 Matriz de Correlación")
            corr_matrix = df[numeric_cols + ['Abandono']].corr() if 'Abandono' in df.columns else df[numeric_cols].corr()
            fig_corr = px.imshow(corr_matrix, color_continuous_scale='RdBu', title="Matriz de Correlación")
            st.plotly_chart(fig_corr, use_container_width=True)

# 🤖 ENTRENAMIENTO
elif page == "🤖 Entrenamiento":
    st.title("🤖 Entrenamiento de Modelos")
    if st.session_state.df is None:
        st.warning("⚠️ Carga datos primero")
    elif 'Abandono' not in st.session_state.df.columns:
        st.error("❌ El dataset debe contener la columna 'Abandono'")
    else:
        test_size = st.slider("Tamaño del conjunto de prueba (%)", 10, 40, 20) / 100
        st.info(f"División: Training {(1-test_size)*100:.0f}% | Test {test_size*100:.0f}%")
        
        if st.button("🚀 Entrenar Modelos", key='train'):
            with st.spinner("Entrenando..."):
                try:
                    from sklearn.model_selection import train_test_split
                    from sklearn.preprocessing import StandardScaler
                    
                    # Preprocesamiento
                    df_clean = handle_missing_values(st.session_state.df)
                    X = df_clean.drop(columns=['Abandono'])
                    y = df_clean['Abandono']
                    
                    # 1. Codificar y GUARDAR encoders
                    X_encoded, encoders = encode_categorical(X)
                    
                    # 2. Dividir datos
                    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size, random_state=42, stratify=y)
                    
                    # 3. Escalar y GUARDAR scaler
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Entrenar
                    results = compare_models(X_train_scaled, X_test_scaled, y_train, y_test)
                    
                    # Guardar todo en sesión para usarlo en Predicción
                    st.session_state.results = results
                    st.session_state.feature_names = X_encoded.columns.tolist()
                    st.session_state.scaler = scaler       # <--- FIX CRÍTICO
                    st.session_state.encoders = encoders   # <--- FIX CRÍTICO
                    
                    st.success("✅ Modelos entrenados y pipeline guardado")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        if st.session_state.results:
            st.subheader("📊 Resultados de Modelos")
            rows = []
            for name, res in st.session_state.results.items():
                m = res['metrics']
                rows.append({'Modelo': name.replace('_', ' ').title(), 'Accuracy': m['accuracy'], 'F1-Score': m['f1'], 'ROC-AUC': m.get('roc_auc', 0)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            
            # Gráfico comparativo
            plot_df = pd.DataFrame(rows).melt(id_vars='Modelo', var_name='Métrica', value_name='Valor')
            st.plotly_chart(px.bar(plot_df, x='Modelo', y='Valor', color='Métrica', barmode='group', title="Comparación de Rendimiento"), use_container_width=True)
            
            # Importancia
            st.subheader("🔍 Importancia de Características")
            sel_model = st.selectbox("Selecciona modelo:", list(st.session_state.results.keys()))
            model_obj = st.session_state.results[sel_model]['model']
            imp_df = model_obj.get_feature_importance(st.session_state.feature_names, top_n=10)
            if imp_df is not None:
                st.plotly_chart(px.bar(imp_df, x='importance', y='feature', orientation='h', title="Top Features"), use_container_width=True)

# 🔮 PREDICCIÓN
elif page == "🔮 Predicción":
    st.title("🔮 Predicción Individual")
    if not st.session_state.results:
        st.info("ℹ️ Primero debes entrenar los modelos en la sección de Entrenamiento")
    else:
        st.subheader("📋 Ingresa los datos del estudiante")
        col1, col2 = st.columns(2)
        
        with col1:
            edad = st.number_input("Edad", min_value=15, max_value=100, value=18)
            gpa = st.slider("GPA", 0.0, 5.0, 3.0)
            asistencia = st.slider("Asistencia (%)", 0, 100, 85)
            horas = st.slider("Horas de Estudio (semanales)", 0, 20, 5)
        with col2:
            socioeconomico = st.selectbox("Nivel Socioeconómico", ['Bajo', 'Medio', 'Alto'])
            primer_trimestre = st.selectbox("Desempeño Primer Trimestre", ['Aprobado', 'Reprobado'])
            motivacion = st.selectbox("Nivel de Motivación", ['Baja', 'Media', 'Alta'])
            
        if st.button("🔮 Realizar Predicción"):
            try:
                # 1. Crear DataFrame con los inputs
                input_data = pd.DataFrame({
                    'Edad': [edad],
                    'GPA': [gpa],
                    'Asistencia': [asistencia],
                    'Horas_Estudio': [horas],
                    'Socioeconomico': [socioeconomico],
                    'Primer_Trimestre': [primer_trimestre],
                    'Motivacion': [motivacion]
                })
                
                # 2. Codificar usando los ENCODERS GUARDADOS del entrenamiento
                processed_data = input_data.copy()
                if 'encoders' in st.session_state:
                    for col, le in st.session_state.encoders.items():
                        if col in processed_data.columns:
                            val = processed_data[col].values[0]
                            if val in le.classes_:
                                processed_data[col] = le.transform([val])[0]
                            else:
                                st.warning(f"Valor '{val}' no visto en {col}. Usando valor por defecto.")
                                processed_data[col] = le.transform([le.classes_[0]])[0]
                
                # 3. Asegurar que las columnas estén en el orden correcto
                if 'feature_names' in st.session_state:
                    # Rellenar columnas faltantes si las hubiera (por seguridad)
                    for col in st.session_state.feature_names:
                        if col not in processed_data.columns:
                            processed_data[col] = 0
                    # Reordenar
                    processed_data = processed_data[st.session_state.feature_names]
                
                # 4. Escalar usando el SCALER GUARDADO del entrenamiento (NO fit_transform)
                if 'scaler' in st.session_state:
                    X_sample = st.session_state.scaler.transform(processed_data)
                else:
                    st.error("Error: No se encontró el escalador. Entrena los modelos primero.")
                    st.stop()

                # Expander para ver los datos procesados (Debug)
                with st.expander("🔍 Ver datos procesados enviados al modelo"):
                    st.write("Valores Numéricos Escalados (Media ~0, Varianza ~1):")
                    st.dataframe(pd.DataFrame(X_sample, columns=st.session_state.feature_names).T, use_container_width=True)
                    st.info("Si los valores son correctos, el modelo responderá adecuadamente a la sensibilidad.")

                # 5. Predecir
                st.markdown("---")
                st.subheader("🎯 Resultados de Predicción")
                cols = st.columns(4)
                
                for idx, (model_name, result) in enumerate(st.session_state.results.items()):
                    with cols[idx]:
                        model_obj = result['model']
                        pred = model_obj.predict(X_sample)[0]
                        
                        prob = 0.0
                        if hasattr(model_obj.model, 'predict_proba'):
                            prob = model_obj.model.predict_proba(X_sample)[0][1]
                        
                        # Lógica de recomendación mejorada
                        if prob > 0.60:
                            rec = f"🔴 ALTO RIESGO ({prob:.1%})"
                        elif prob > 0.40:
                            rec = f"🟡 RIESGO MEDIO ({prob:.1%})"
                        else:
                            rec = f"🟢 BAJO RIESGO ({prob:.1%})"
                            
                        st.markdown(f"**{model_name.replace('_', ' ').title()}**")
                        st.success(rec)
            
            except Exception as e:
                st.error(f"❌ Error en predicción: {str(e)}")
                st.exception(e) # Muestra el error detallado para depurar

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9em;">
 <p>🎓 Predicción de Abandono Escolar | ML Clásico | Artificial Intelligence Foundations (Fundació URV)</p>
 <p>Desarrollado con Streamlit, Scikit-learn y Plotly</p>
</div>
""", unsafe_allow_html=True)
