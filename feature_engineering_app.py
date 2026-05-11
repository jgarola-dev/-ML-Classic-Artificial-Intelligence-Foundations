"""
Feature Engineering Analysis Application
Interactive Streamlit application for comprehensive data analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_loader import load_dataset, preprocess_dataset
from feature_engineering import FeatureEngineeringPipeline
import warnings
warnings.filterwarnings('ignore')


# Configure page
st.set_page_config(
    page_title="Feature Engineering - Análisis de Abandono Escolar",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    h1, h2, h3 {
        color: #1f77b4;
    }
    .success {
        color: #00cc00;
    }
    .warning {
        color: #ff6600;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🎓 Feature Engineering - Predicción de Abandono Escolar")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    st.markdown("### 📊 Opciones de Análisis")
    
    analysis_type = st.radio(
        "Selecciona el tipo de análisis:",
        [
            "📋 Descripción General",
            "🔢 Estadísticas Descriptivas",
            "📈 Variables Categóricas",
            "📉 Trazado de Atributos",
            "🔗 Variables Múltiples",
            "🎯 Matriz de Dispersión",
            "🌡️ Matriz de Correlación",
            "⚖️ Desequilibrio de Clases"
        ]
    )

# Load data
@st.cache_data
def load_and_preprocess():
    df, source = load_dataset()
    df = preprocess_dataset(df)
    return df, source

df, data_source = load_and_preprocess()
pipeline = FeatureEngineeringPipeline(df)

# Main content
if analysis_type == "📋 Descripción General":
    st.header("📋 Descripción General del Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Registros", f"{len(df):,}")
    with col2:
        st.metric("📋 Columnas", f"{len(df.columns)}")
    with col3:
        st.metric("🔢 Numéricas", f"{len(pipeline.numeric_cols)}")
    with col4:
        st.metric("📂 Categóricas", f"{len(pipeline.categorical_cols)}")
    
    st.markdown("---")
    
    # Dataset source
    st.info(f"📥 Fuente de datos: {data_source}")
    
    # Data preview
    st.subheader("📊 Previsualización de Datos")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Missing values
    st.subheader("❌ Análisis de Valores Faltantes")
    missing_plot = pipeline.plot_missing_data()
    if missing_plot:
        st.plotly_chart(missing_plot, use_container_width=True)
    else:
        st.success("✅ No hay valores faltantes en el dataset")
    
    # Data types
    st.subheader("📋 Tipos de Datos")
    dtype_counts = df.dtypes.value_counts()
    st.bar_chart(dtype_counts)


elif analysis_type == "🔢 Estadísticas Descriptivas":
    st.header("🔢 Estadísticas Descriptivas - Variables Numéricas")
    
    stats_df = pipeline.get_descriptive_statistics()
    
    st.subheader("📊 Resumen Estadístico")
    st.dataframe(stats_df.round(2), use_container_width=True)
    
    # Download statistics
    csv = stats_df.round(2).to_csv()
    st.download_button(
        label="📥 Descargar Estadísticas (CSV)",
        data=csv,
        file_name="estadisticas_descriptivas.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Interpret statistics
    st.subheader("📈 Interpretación de Estadísticas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Asimetría (Skewness):**")
        st.markdown("""
        - **≈ 0**: Distribución simétrica
        - **> 0**: Asimetría a la derecha
        - **< 0**: Asimetría a la izquierda
        """)
    
    with col2:
        st.markdown("**Curtosis (Kurtosis):**")
        st.markdown("""
        - **≈ 0**: Normal (mesokúrtica)
        - **> 0**: Colas pesadas (leptokúrtica)
        - **< 0**: Colas ligeras (platikúrtica)
        """)


elif analysis_type == "📈 Variables Categóricas":
    st.header("📈 Análisis de Variables Categóricas")
    
    cat_stats = pipeline.get_categorical_statistics()
    
    for col, stats in cat_stats.items():
        with st.expander(f"📊 {col} (Frecuencia)"):
            st.dataframe(
                pd.DataFrame({
                    'Categoría': stats['counts'].index,
                    'Frecuencia': stats['counts'].values,
                    'Porcentaje': stats['percentages'].values
                }),
                use_container_width=True
            )
            
            # Visualizations
            fig_bar = px.bar(
                x=stats['counts'].index.astype(str),
                y=stats['counts'].values,
                title=f"Distribución: {col}",
                color=stats['counts'].values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_bar, use_container_width=True)


elif analysis_type == "📉 Trazado de Atributos":
    st.header("📉 Trazado de Estadísticas de Atributos Individuales")
    
    selected_var = st.selectbox("Selecciona una variable:", pipeline.numeric_cols)
    
    if selected_var:
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig_hist = px.histogram(
                df,
                x=selected_var,
                nbins=50,
                title=f"Histograma: {selected_var}",
                color_discrete_sequence=['#636EFA']
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot
            fig_box = px.box(
                df,
                y=selected_var,
                title=f"Box Plot: {selected_var}",
                color_discrete_sequence=['#636EFA']
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Statistics
        st.subheader("📊 Estadísticas Detalladas")
        stats_dict = {
            'Media': df[selected_var].mean(),
            'Mediana': df[selected_var].median(),
            'Desv. Est.': df[selected_var].std(),
            'Mín': df[selected_var].min(),
            'Máx': df[selected_var].max(),
            'Q1': df[selected_var].quantile(0.25),
            'Q3': df[selected_var].quantile(0.75),
            'Asimetría': df[selected_var].skew(),
            'Curtosis': df[selected_var].kurtosis()
        }
        st.dataframe(pd.DataFrame(stats_dict.items(), columns=['Métrica', 'Valor']), use_container_width=True)


elif analysis_type == "🔗 Variables Múltiples":
    st.header("🔗 Relación entre Variables Múltiples")
    
    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox("Selecciona Variable 1:", pipeline.numeric_cols)
    with col2:
        var2 = st.selectbox("Selecciona Variable 2:", pipeline.numeric_cols, index=1 if len(pipeline.numeric_cols) > 1 else 0)
    
    if var1 and var2 and var1 != var2:
        fig = pipeline.plot_pairwise_relationships(var1, var2)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation
        corr = df[var1].corr(df[var2])
        st.info(f"📊 Correlación de Pearson: {corr:.4f}")


elif analysis_type == "🎯 Matriz de Dispersión":
    st.header("🎯 Matriz de Dispersión (Scatter Matrix)")
    
    st.info("📊 Mostrando matriz de dispersión de las variables con mayor varianza (máx 8 variables)")
    
    fig = pipeline.plot_scatter_matrix()
    if fig:
        st.plotly_chart(fig, use_container_width=True)


elif analysis_type == "🌡️ Matriz de Correlación":
    st.header("🌡️ Matriz de Correlación")
    
    # Heatmap
    st.subheader("🔥 Heatmap de Correlación")
    fig_heatmap = pipeline.plot_correlation_heatmap()
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown("---")
    
    # Top correlations
    st.subheader("🏆 Top 15 Correlaciones Más Fuertes")
    top_corr = pipeline.get_top_correlations(top_n=15)
    st.dataframe(top_corr.round(4), use_container_width=True)
    
    # Download
    csv = top_corr.to_csv(index=False)
    st.download_button(
        label="📥 Descargar Correlaciones (CSV)",
        data=csv,
        file_name="correlaciones_top15.csv",
        mime="text/csv"
    )


elif analysis_type == "⚖️ Desequilibrio de Clases":
    st.header("⚖️ Análisis de Desequilibrio de Clases")
    
    if pipeline.target_col:
        # Class distribution
        fig_bar, fig_pie = pipeline.plot_class_imbalance()
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_bar, use_container_width=True)
        with col2:
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("---")
        
        # Statistics
        stats_df, recommendations = pipeline.get_imbalance_statistics()
        
        st.subheader("📊 Estadísticas de Clases")
        st.dataframe(stats_df, use_container_width=True)
        
        # Imbalance analysis
        imbalance_info = pipeline.analyze_class_imbalance()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⚠️ Ratio Desequilibrio", f"{imbalance_info['imbalance_ratio']:.2f}:1")
        with col2:
            st.metric("👥 Clase Mayoritaria", f"{imbalance_info['majority_count']:,}")
        with col3:
            st.metric("👥 Clase Minoritaria", f"{imbalance_info['minority_count']:,}")
        
        st.markdown("---")
        
        # Recommendations
        st.subheader("💡 Recomendaciones")
        for rec in recommendations:
            st.markdown(f"- {rec}")
    else:
        st.warning("⚠️ No se encontró columna de target en el dataset")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🎓 Feature Engineering - Artificial Intelligence Foundations | Fundació URV</p>
    <p>Dataset: UCI Student Dropout and Academic Success</p>
</div>
""", unsafe_allow_html=True)
