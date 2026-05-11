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

elif page == "📊 Análisis EDA":
    st.title("📊 Análisis Exploratorio de Datos")
    
    if st.session_state.df is None:
        st.warning("⚠️ Por favor, carga datos primero en la barra lateral")
    else:
        df = st.session_state.df.copy()
        
        # ==================== KPIs PRINCIPALES ====================
        st.subheader("📈 Indicadores Clave")
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        
        with kpi1:
            st.metric("📊 Registros", f"{len(df):,}")
        with kpi2:
            st.metric("📋 Columnas", len(df.columns))
        with kpi3:
            st.metric("❌ Valores Faltantes", df.isnull().sum().sum())
        with kpi4:
            numeric_cols = df.select_dtypes(include=np.number).columns
            st.metric("🔢 Variables Numéricas", len(numeric_cols))
        with kpi5:
            if 'Abandono' in df.columns:
                rate = df['Abandono'].mean() * 100
                st.metric("📉 Tasa de Abandono", f"{rate:.1f}%")
            else:
                st.metric("🎯 Target", "No detectado")
        
        st.markdown("---")
        
        # ==================== VISTA PREVIA + TIPOS ====================
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.expander("📋 Vista Previa de Datos (primeras 10 filas)", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            with st.expander("📝 Tipos de Datos"):
                dtype_df = pd.DataFrame({
                    'Columna': df.columns,
                    'Tipo': df.dtypes.values,
                    'Únicos': [df[c].nunique() for c in df.columns]
                })
                st.dataframe(dtype_df, use_container_width=True, hide_index=True)
        
        # ==================== ESTADÍSTICAS DESCRIPTIVAS ====================
        st.subheader("📊 Estadísticas Descriptivas")
        
        # Pestañas para organizar mejor
        tab1, tab2, tab3 = st.tabs(["🔢 Resumen Numérico", "📈 Distribuciones", "🔗 Correlaciones"])
        
        with tab1:
            if numeric_cols:
                stats_df = df[numeric_cols].describe().T
                stats_df['CV (%)'] = (stats_df['std'] / stats_df['mean'] * 100).round(2)
                st.dataframe(stats_df.round(3), use_container_width=True)
                
                # Botón de descarga
                csv = stats_df.round(3).to_csv()
                st.download_button(
                    label="📥 Descargar Estadísticas (CSV)",
                    data=csv,
                    file_name="estadisticas_eda.csv",
                    mime="text/csv"
                )
        
        # ==================== DISTRIBUCIONES VISUALES ====================
        with tab2:
            st.markdown("### 🔍 Visualización de Distribuciones")
            
            col_dist1, col_dist2 = st.columns(2)
            
            with col_dist1:
                # Selector de variable para histograma + boxplot
                selected_var = st.selectbox("Selecciona variable para análisis:", numeric_cols)
                
                if selected_var:
                    # Histograma con curva de densidad
                    fig_hist = px.histogram(df, x=selected_var, nbins=30, 
                                           title=f"Distribución: {selected_var}",
                                           marginal="box", color_discrete_sequence=['#636EFA'])
                    st.plotly_chart(fig_hist, use_container_width=True)
            
            with col_dist2:
                # Si hay target, comparar distribuciones por clase
                if 'Abandono' in df.columns and selected_var:
                    fig_compare = px.box(df, x='Abandono', y=selected_var,
                                        title=f"{selected_var} por Clase de Abandono",
                                        color='Abandono', color_discrete_map={0: '#00c853', 1: '#d32f2f'})
                    st.plotly_chart(fig_compare, use_container_width=True)
            
            # Gráfico de barras para variables categóricas
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            if cat_cols:
                st.markdown("#### 📊 Distribución de Variables Categóricas")
                selected_cat = st.selectbox("Variable categórica:", cat_cols)
                
                if selected_cat:
                    counts = df[selected_cat].value_counts()
                    fig_cat = px.bar(x=counts.index, y=counts.values,
                                    title=f"Frecuencia: {selected_cat}",
                                    labels={'x': selected_cat, 'y': 'Frecuencia'},
                                    color=counts.values, color_continuous_scale='Viridis')
                    st.plotly_chart(fig_cat, use_container_width=True)
        
        # ==================== MATRIZ DE CORRELACIÓN MEJORADA ====================
        with tab3:
            st.markdown("### 🔗 Matriz de Correlación Interactiva")
            
            if len(numeric_cols) >= 2:
                # Opciones de visualización
                corr_method = st.radio("Método de correlación:", 
                                      ["Pearson (lineal)", "Spearman (rangos)"], 
                                      horizontal=True)
                method = 'pearson' if 'Pearson' in corr_method else 'spearman'
                
                # Calcular matriz
                corr_matrix = df[numeric_cols].corr(method=method)
                
                # Heatmap interactivo con Plotly
                fig_heatmap = px.imshow(corr_matrix, 
                                       text_auto='.2f',
                                       aspect='auto',
                                       color_continuous_scale='RdBu_r',
                                       title=f"Matriz de Correlación ({method.title()})",
                                       labels={'x': 'Variable', 'y': 'Variable', 'color': 'Correlación'})
                fig_heatmap.update_layout(height=600, xaxis_tickangle=-45)
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Top correlaciones positivas y negativas
                st.markdown("#### 🏆 Correlaciones Más Fuertes")
                
                # Extraer pares únicos
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_pairs.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlación': corr_matrix.iloc[i, j]
                        })
                
                corr_df = pd.DataFrame(corr_pairs)
                corr_df['Abs_Corr'] = corr_df['Correlación'].abs()
                
                col_pos, col_neg = st.columns(2)
                
                with col_pos:
                    st.markdown("✅ **Top 5 Correlaciones Positivas**")
                    top_pos = corr_df.nlargest(5, 'Correlación')[['Variable 1', 'Variable 2', 'Correlación']]
                    st.dataframe(top_pos.style.format({'Correlación': '{:.3f}'}), use_container_width=True)
                
                with col_neg:
                    st.markdown("❌ **Top 5 Correlaciones Negativas**")
                    top_neg = corr_df.nsmallest(5, 'Correlación')[['Variable 1', 'Variable 2', 'Correlación']]
                    st.dataframe(top_neg.style.format({'Correlación': '{:.3f}'}), use_container_width=True)
                
                # Gráfico de dispersión para correlación seleccionada
                st.markdown("#### 🔍 Explorar Relación entre Dos Variables")
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    var_x = st.selectbox("Variable X:", numeric_cols, key='corr_x')
                with col_v2:
                    var_y = st.selectbox("Variable Y:", numeric_cols, index=1 if len(numeric_cols)>1 else 0, key='corr_y')
                
                if var_x and var_y and var_x != var_y:
                    # Scatter con línea de tendencia
                    fig_scatter = px.scatter(df, x=var_x, y=var_y, 
                                            trendline='ols',
                                            title=f"{var_x} vs {var_y}",
                                            color='Abandono' if 'Abandono' in df.columns else None,
                                            color_discrete_map={0: '#00c853', 1: '#d32f2f'} if 'Abandono' in df.columns else None)
                    
                    # Calcular correlación específica
                    r = df[var_x].corr(df[var_y], method=method)
                    fig_scatter.update_layout(subtitle_text=f"r = {r:.3f}")
                    st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("ℹ️ Se necesitan al menos 2 variables numéricas para calcular correlaciones")
        
        # ==================== ANÁLISIS ADICIONAL ====================
        st.markdown("---")
        st.subheader("🔎 Análisis Adicional")
        
        exp_outliers, exp_missing = st.columns(2)
        
        with exp_outliers:
            with st.expander("🚨 Detección de Outliers (Método IQR)"):
                if numeric_cols:
                    outliers_summary = []
                    for col in numeric_cols[:10]:  # Limitar a 10 para rendimiento
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                        if n_outliers > 0:
                            outliers_summary.append({
                                'Variable': col,
                                'Outliers': int(n_outliers),
                                '%': f"{n_outliers/len(df)*100:.1f}%"
                            })
                    
                    if outliers_summary:
                        st.dataframe(pd.DataFrame(outliers_summary), use_container_width=True)
                    else:
                        st.success("✅ No se detectaron outliers significativos")
        
        with exp_missing:
            with st.expander("❌ Análisis de Valores Faltantes"):
                missing = df.isnull().sum()
                missing = missing[missing > 0]
                
                if len(missing) > 0:
                    fig_missing = px.bar(x=missing.index, y=missing.values,
                                        title="Valores Faltantes por Variable",
                                        labels={'x': 'Variable', 'y': 'Cantidad'})
                    st.plotly_chart(fig_missing, use_container_width=True)
                else:
                    st.success("✅ Dataset completo: sin valores faltantes")

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
