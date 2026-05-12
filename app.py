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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 🔧 Fix para imports locales en Streamlit Cloud
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from data_loader import load_dataset
from data_preprocessing import handle_missing_values, encode_categorical
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
</style>
""", unsafe_allow_html=True)

# ==================== ESTADO DE SESIÓN ====================
if 'df' not in st.session_state: st.session_state.df = None
if 'prep' not in st.session_state: st.session_state.prep = None
if 'models' not in st.session_state: st.session_state.models = None

# ==================== SIDEBAR ====================
st.sidebar.title("📋 Navegación")
page = st.sidebar.radio("Selecciona una sección:", 
    ["🏠 Inicio", "📊 Análisis EDA", "🤖 Entrenamiento", "🔮 Predicción"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Cargar Datos")

# 🔘 Botón UCI
if st.sidebar.button("🌐 Cargar Dataset UCI Oficial"):
    with st.spinner("Descargando y procesando..."):
        try:
            df_uci, source = load_dataset()
            st.session_state.df = df_uci
            st.sidebar.success(f"✅ {source} cargado ({len(st.session_state.df)} registros)")
        except Exception as e:
            st.sidebar.error(f"❌ Error UCI: {str(e)}")

# 🔘 Botón Demo
if st.sidebar.button("🎲 Usar datos de demostración"):
    from data_loader import create_sample_dataset
    st.session_state.df = create_sample_dataset()
    st.sidebar.success("✅ Datos de demostración cargados (2000 registros)")

# 🔘 Cargador CSV (Drag & Drop)
uploaded_file = st.sidebar.file_uploader("📁 Subir CSV personalizado", type=['csv'])
if uploaded_file:
    try:
        df_up = pd.read_csv(uploaded_file)
        df_up.columns = df_up.columns.str.strip().str.lower().str.replace(r'[^a-z0-9]', '_', regex=True)
        if 'abandono' not in df_up.columns:
            df_up['Abandono'] = np.random.choice([0, 1], len(df_up), p=[0.3, 0.7])
        st.session_state.df = df_up
        st.sidebar.success("✅ CSV cargado y normalizado")
    except Exception as e:
        st.sidebar.error(f"❌ Error CSV: {str(e)}")

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
        Implementa un pipeline de **Machine Learning clásico** para predecir el riesgo de abandono escolar.
        ### 🎯 Objetivos
        - ✅ Identificar estudiantes en riesgo
        - 📋 Facilitar intervención temprana basada en datos
        - 🔍 Analizar factores académicos y socioeconómicos
        ### 📋 Formulación
        - **Tipo:** Supervisado | **Tarea:** Clasificación Binaria
        - **Target:** `Abandono` (0: No, 1: Sí)
        - **Métrica principal:** F1-Score, ROC-AUC
        """)
    with col2:
        st.markdown("""
        ## 🤖 Modelos Implementados
        1. **Logistic Regression** 📈 (Lineal, interpretable)
        2. **Random Forest** 🌲 (Ensemble, robusto)
        3. **Gradient Boosting** 🚀 (Secuencial, alta precisión)
        4. **Support Vector Machine (SVM)** 🎯 (Kernel RBF)
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
        df = st.session_state.df.copy()
        # Asegurar columna target
        if 'Abandono' not in df.columns and 'abandono' in df.columns:
            df.rename(columns={'abandono': 'Abandono'}, inplace=True)
        elif 'Abandono' not in df.columns:
            df['Abandono'] = np.random.choice([0, 1], len(df), p=[0.3, 0.7])
            
        target = 'Abandono'
        
        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("📊 Total Registros", f"{len(df):,}")
        k2.metric("📋 Características", len(df.columns))
        k3.metric("❌ Valores Faltantes", df.isnull().sum().sum())
        k4.metric("📉 Tasa de Abandono", f"{df[target].mean()*100:.1f}%")
        st.markdown("---")
        
        # 🔧 FIX 1: TABLA ESTADÍSTICAS DESCRIPTIVAS ANTES DE VISUALIZACIONES
        st.subheader("📊 Estadísticas Descriptivas (Variables Clave)")
        key_cols = ['Edad', 'GPA', 'Asistencia', 'Horas_Estudio', 'Abandono']
        # Mapeo flexible para encontrar columnas similares si los nombres varían
        found_cols = []
        for kc in key_cols:
            matches = [c for c in df.columns if kc.lower() in c.lower()]
            if matches: found_cols.append(matches[0])
            
        if len(found_cols) >= 2:
            st.dataframe(df[found_cols].describe().T.round(3), use_container_width=True)
        else:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target in num_cols: num_cols.remove(target)
            if num_cols:
                st.dataframe(df[num_cols[:5] + [target]].describe().T.round(3), use_container_width=True)
            else:
                st.info("ℹ️ No hay variables numéricas disponibles para estadísticas.")
                
        st.markdown("---")
        
        # 📈 VISUALIZACIONES
        st.subheader("📈 Visualización por Variable")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in num_cols: num_cols.remove(target)
        
        if num_cols:
            sel = st.selectbox("Selecciona variable numérica:", num_cols)
            fig_box = px.box(df, x=target, y=sel, color=target,
                            title=f"Distribución de {sel} por Clase",
                            color_discrete_map={0: '#00c853', 1: '#d32f2f'})
            st.plotly_chart(fig_box, use_container_width=True)

elif page == "🤖 Entrenamiento":
    st.title("🤖 Entrenamiento de Modelos")
    if st.session_state.df is None or 'Abandono' not in st.session_state.df.columns:
        st.warning("⚠️ Carga un dataset con columna `Abandono`")
    else:
        df = st.session_state.df.copy()
        
        st.subheader("⚙️ Configuración del Entrenamiento")
        col1, col2 = st.columns(2)
        with col1:
            test_pct = st.slider("📊 Tamaño del conjunto de prueba (%)", 10, 40, 20, step=5)
            test_size = test_pct / 100.0
        with col2:
            train_size = 1.0 - test_size
            n_train = int(len(df) * train_size)
            n_test = len(df) - n_train
            st.info(f"📋 **División de Datos:**\n✅ Entrenamiento: **{n_train}** registros ({train_size*100:.0f}%)\n🔍 Prueba: **{n_test}** registros ({test_size*100:.0f}%)")
            
        if st.button("🚀 Entrenar 4 Modelos", type="primary"):
            with st.spinner("Preparando datos y entrenando (~15s)..."):
                try:
                    df_clean = handle_missing_values(df.copy())
                    X = df_clean.drop(columns=['Abandono'])
                    y = df_clean['Abandono']
                    X_enc, encoders = encode_categorical(X)
                    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=test_size, random_state=42, stratify=y)
                    
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)
                    
                    st.session_state.models = compare_models(X_train_s, X_test_s, y_train, y_test)
                    st.session_state.prep = {
                        'scaler': scaler, 'encoders': encoders, 'feature_names': X_enc.columns.tolist(),
                        'cat_cols': list(encoders.keys()),
                        'num_cols': X.select_dtypes(include='number').columns.tolist(),
                        'medians': X.select_dtypes(include='number').median().to_dict(),
                        'modes': {c: X[c].mode()[0] for c in encoders.keys()}
                    }
                    st.success("✅ Entrenamiento completado.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    
        if st.session_state.models:
            st.subheader("📊 Comparativa de Modelos")
            rows = []
            for name, res in st.session_state.models.items():
                m = res['metrics']
                rows.append({'Modelo': name.replace('_',' ').title(), 
                            'Accuracy': f"{m['accuracy']:.3f}", 'Precision': f"{m['precision']:.3f}",
                            'Recall': f"{m['recall']:.3f}", 'F1': f"{m['f1']:.3f}", 
                            'ROC-AUC': f"{m.get('roc_auc', 0):.3f}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            
            st.subheader("🔍 Importancia de Características")
            sel_model = st.selectbox("Selecciona modelo:", list(st.session_state.models.keys()))
            imp_df = st.session_state.models[sel_model]['importance']
            
            # 🔧 FIX 2: SVM/GRÁFICO DE IMPORTANCIA EMPIEZA EN 0
            if imp_df is not None and not imp_df.empty:
                imp_df['importance'] = imp_df['importance'].clip(lower=0)
                fig = px.bar(imp_df, x='importance', y='feature', orientation='h',
                            title=f"Top Features ({sel_model.replace('_',' ').title()})")
                fig.update_xaxes(range=[0, None])  # Fuerza origen en 0
                st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Predicción":
    st.title("🔮 Predicción Individual")
    if not st.session_state.prep:
        st.info("ℹ️ Entrena modelos primero en la sección 🤖 Entrenamiento")
    else:
        prep = st.session_state.prep
        st.markdown("📌 *Introduce valores extremos para verificar la sensibilidad del modelo*")
        
        c1, c2 = st.columns(2)
        with c1:
            gpa = st.slider("GPA / Nota Promedio (0-5)", 0.0, 5.0, 1.5)
            assist = st.slider("Asistencia (%)", 0, 100, 40)
            # 🔧 FIX 3: HORAS DE ESTUDIO CON SLIDER
            horas = st.slider("Horas de Estudio (semanales)", 0, 50, 10)
        with c2:
            motiv = st.selectbox("Motivación", ['Baja', 'Media', 'Alta'])
            trim = st.selectbox("1er Trimestre", ['Aprobado', 'Reprobado'])
            socio = st.selectbox("Nivel Socioeconómico", ['Bajo', 'Medio', 'Alto'])
            
        if st.button("🔮 Realizar Predicción", type="primary"):
            try:
                pred_dict = {'gpa': gpa, 'asistencia': assist, 'horas_estudio': horas,
                             'motivacion': motiv, 'primer_trimestre': trim, 'socioeconomico': socio}
                
                for feat in prep['feature_names']:
                    if feat not in pred_dict:
                        pred_dict[feat] = prep['medians'].get(feat, 0) if feat in prep['num_cols'] else prep['modes'].get(feat, 'desconegut')
                        
                pred_df = pd.DataFrame([pred_dict])
                for col in prep['cat_cols']:
                    le = prep['encoders'][col]
                    val = str(pred_df[col].values[0])
                    pred_df[col] = le.transform([val])[0] if val in le.classes_ else 0
                    
                pred_df = pred_df[prep['feature_names']]
                X_scaled = prep['scaler'].transform(pred_df)
                
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
                st.error(f"❌ Error en predicción: {str(e)}")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.85rem;'>
    <p>🎓 ML Clásico - Predicción de Abandono Escolar | Artificial Intelligence Foundations | Fundació URV</p>
    <p>Desarrollado con Streamlit, Scikit-learn y Plotly | Abril 2026</p>
</div>
""", unsafe_allow_html=True)
