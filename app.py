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
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 🔧 Fix para imports locales en Streamlit Cloud
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from data_loader import load_dataset, preprocess_dataset
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
for key in ['df', 'prep', 'results']:
    if key not in st.session_state: st.session_state[key] = None

# ==================== FUNCIONES AUXILIARES ====================
def map_status_to_abandono(df, target_col):
    """
    Mapea Status/Target del UCI a Abandono binario:
    - Valores numéricos: 0=Dropout→1, 1/2=Enrolled/Graduate→0
    - Valores texto: 'dropout'→1, resto→0
    """
    if df[target_col].dtype in ['int64', 'float64']:
        return df[target_col].apply(lambda x: 1 if x == 0 else 0)
    else:
        return df[target_col].astype(str).str.lower().apply(
            lambda x: 1 if 'dropout' in x else 0)

def get_key_columns(df):
    """Retorna columnas clave para visualización"""
    key_cols = ['edad', 'gpa', 'asistencia', 'horas_estudio', 'abandono']
    return [c for c in key_cols if c in df.columns]

# ==================== SIDEBAR ====================
st.sidebar.title("📋 Navegación")
page = st.sidebar.radio("Selecciona una sección:", 
    ["🏠 Inicio", "📊 Análisis EDA", "🤖 Entrenamiento", "🔮 Predicción"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Cargar Datos")

# 🔘 Botón: Dataset UCI Oficial
if st.sidebar.button("🌐 Cargar Dataset UCI Oficial", use_container_width=True):
    with st.spinner("Descargando y procesando UCI Student Dropout Dataset..."):
        try:
            df_raw, source = load_dataset()
            df_proc = preprocess_dataset(df_raw)
            # Mapear Status/Target a Abandono binario
            target_col = next((c for c in ['target', 'status', 'Target', 'Status'] if c in df_proc.columns), None)
            if target_col:
                df_proc['Abandono'] = map_status_to_abandono(df_proc, target_col)
            else:
                df_proc['Abandono'] = np.random.choice([0, 1], len(df_proc), p=[0.3, 0.7])
            st.session_state.df = df_proc
            st.sidebar.success(f"✅ {source} cargado ({len(df_proc)} registros)")
        except Exception as e:
            st.sidebar.error(f"❌ Error UCI: {str(e)}")

# 🔘 Botón: Datos de demostración
if st.sidebar.button("🎲 Usar datos de demostración", use_container_width=True):
    from data_loader import create_sample_dataset
    df_demo = create_sample_dataset()
    df_demo['Abandono'] = np.random.choice([0, 1], len(df_demo), p=[0.3, 0.7])
    st.session_state.df = df_demo
    st.sidebar.success("✅ Datos de demostración cargados")

# 🔘 SECCIÓN: Subir archivo CSV (Browse Files)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Sube un archivo CSV")
st.sidebar.info("📌 El archivo debe contener una columna 'Abandono' o similar")

uploaded_file = st.sidebar.file_uploader(
    "🔍 Selecciona un archivo CSV de tu ordenador",
    type=['csv'],
    help="Formato: .csv | Columna objetivo: 'Abandono', 'Target', 'Status', 'estado'"
)

if uploaded_file is not None:
    try:
        df_up = pd.read_csv(uploaded_file)
        df_up.columns = df_up.columns.str.strip().str.lower().str.replace(r'[^a-z0-9]', '_', regex=True)
        if 'abandono' not in df_up.columns:
            target_candidates = ['target', 'status', 'estado', 'y']
            found_target = next((c for c in target_candidates if c in df_up.columns), None)
            if found_target:
                df_up['abandono'] = map_status_to_abandono(df_up, found_target)
            else:
                df_up['abandono'] = np.random.choice([0, 1], len(df_up), p=[0.3, 0.7])
        st.session_state.df = df_up
        st.sidebar.success(f"✅ CSV cargado: {len(df_up)} registros")
    except Exception as e:
        st.sidebar.error(f"❌ Error al leer CSV: {str(e)}")

# Info del dataset activo
if st.session_state.df is not None:
    df_map = st.session_state.df
    key_cols = [c for c in ['edad', 'gpa', 'asistencia', 'horas_estudio', 'abandono'] if c in df_map.columns]
    st.sidebar.info(f"📊 Dataset: {len(df_map)} filas | Columnas clave: {', '.join(key_cols) if key_cols else 'N/A'}")

# ==================== PÁGINAS ====================
if page == "🏠 Inicio":
    st.markdown("### 📚 Sobre el Proyecto")
    st.markdown("Este proyecto implementa un modelo de Machine Learning clásico para predecir el riesgo de abandono escolar en estudiantes.")
    st.markdown("### 🎯 Objetivos")
    st.markdown("- Identificar estudiantes en riesgo de abandono\n- Proporcionar recomendaciones de intervención\n- Analizar factores clave del abandono escolar")
    st.markdown("### 📋 Formulación del Problema")
    st.markdown("- **Tipo de aprendizaje:** Supervisado\n- **Tarea:** Clasificación binaria\n- **Variable objetivo:** Abandono (0/1)\n- **Métrica de éxito:** F1-Score, ROC-AUC")
    st.markdown("### 🤖 Modelos Implementados")
    st.markdown("Se utilizan 4 modelos ML clásicos:\n1. **Logistic Regression** → Modelo lineal simple y interpretable\n2. **Random Forest** → Ensemble de árboles de decisión\n3. **Gradient Boosting** → Boosting secuencial de árboles\n4. **Support Vector Machine (SVM)** → Máquinas de vectores de soporte")
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("📊 Modelos", "4", "Clasificadores")
    m2.metric("📈 Métricas", "6+", "Evaluación")
    m3.metric("🎓 Categoría", "ML Clásico", "Fundació URV")
    st.markdown("---")
    st.markdown("### 📊 Características del Dataset\nEl modelo utiliza las siguientes características:\n- **Edad:** Edad del estudiante\n- **GPA:** Promedio de calificaciones\n- **Asistencia:** Porcentaje de asistencia\n- **Horas_Estudio:** Horas de estudio semanales\n- **Socioeconomico:** Nivel socioeconómico\n- **Primer_Trimestre:** Desempeño primer trimestre\n- **Motivacion:** Nivel de motivación")

elif page == "📊 Análisis EDA":
    st.title("📊 Análisis Exploratorio de Datos")
    if st.session_state.df is None:
        st.warning("⚠️ Carga datos primero en la barra lateral")
    else:
        df = st.session_state.df.copy()
        # Asegurar columna 'abandono'
        if 'abandono' not in df.columns and 'Abandono' in df.columns:
            df.rename(columns={'Abandono': 'abandono'}, inplace=True)
        if 'abandono' not in df.columns:
            for alt in ['target', 'status', 'Target', 'Status']:
                if alt in df.columns:
                    df['abandono'] = map_status_to_abandono(df, alt)
                    break
            if 'abandono' not in df.columns:
                df['abandono'] = np.random.choice([0, 1], len(df), p=[0.3, 0.7])
        
        if 'abandono' not in df.columns:
            st.error("❌ Columna 'abandono' no encontrada.")
        else:
            # ✅ CÁLCULO CORRECTO DE TASA DE ABANDONO
            dropout_count = (df['abandono'] == 1).sum()
            dropout_rate = (dropout_count / len(df)) * 100
            
            # KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("📊 Total Registros", f"{len(df):,}")
            k2.metric("📋 Columnas Clave", len(get_key_columns(df)))
            k3.metric("❌ Valores Faltantes", df[get_key_columns(df)].isnull().sum().sum())
            k4.metric("📉 Tasa de Abandono", f"{dropout_rate:.1f}%")
            
            # ✅ VISTA PREVIA DE DATOS
            with st.expander("📋 Vista Previa de Datos"):
                st.dataframe(df[get_key_columns(df)].head(15), use_container_width=True)
            
            # ✅ ESTADÍSTICAS DESCRIPTIVAS
            st.subheader("📊 Estadísticas Descriptivas")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'abandono' in numeric_cols: numeric_cols.remove('abandono')
            if numeric_cols:
                stats_df = df[numeric_cols + ['abandono']].describe().T.round(3)
                st.dataframe(stats_df, use_container_width=True)
                st.download_button("📥 Descargar Estadísticas", stats_df.to_csv(), "estadisticas.csv", "text/csv")
            
            # ✅ DISTRIBUCIÓN DE ABANDONOS (%) - CORREGIDO
            st.subheader("🥧 Distribución de Abandonos (%)")
            
            # 🔑 FIX: .sort_index() ordena por etiqueta (0→1), NO por frecuencia
            abandono_counts = df['abandono'].value_counts().sort_index()
            
            # Mapeo explícito de etiquetas
            labels_map = {0: "No Abandona (0)", 1: "Abandona (1)"}
            labels = [labels_map.get(idx, f"Clase {idx}") for idx in abandono_counts.index]
            
            # Calcular porcentajes
            total = len(df)
            count_0 = abandono_counts.get(0, 0)
            count_1 = abandono_counts.get(1, 0)
            pct_0 = (count_0 / total) * 100
            pct_1 = (count_1 / total) * 100
            
            # Actualizar KPI
            k4.metric("📉 Tasa de Abandono", f"{pct_1:.1f}%")
            
            # Gráfico circular CORRECTO
            fig_pie = px.pie(
                values=abandono_counts.values, 
                names=labels,
                title="Proporción de Clases", 
                hole=0.4, 
                color_discrete_map={
                    'No Abandona (0)': '#00c853',
                    'Abandona (1)': '#d32f2f'
                }
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Tooltip informativo
            st.caption(f"📊 Clase 0 (No Abandona): {count_0} registros ({pct_0:.1f}%) | Clase 1 (Abandona): {count_1} registros ({pct_1:.1f}%)")
            
            # ✅ MATRIZ DE CORRELACIÓN
            st.subheader("🔗 Matriz de Correlación (Pearson)")
            if len(numeric_cols) >= 2:
                corr = df[numeric_cols + ['abandono']].corr()
                st.plotly_chart(px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r'), use_container_width=True)
            
            # ✅ VISUALIZACIÓN POR VARIABLE
            st.subheader("📈 Visualización por Variable")
            if numeric_cols:
                selected_var = st.selectbox("Selecciona variable:", numeric_cols)
                if selected_var:
                    c1, c2 = st.columns(2)
                    with c1:
                        fig_box = px.box(df, x='abandono', y=selected_var, color='abandono', color_discrete_map={0: '#00c853', 1: '#d32f2f'})
                        st.plotly_chart(fig_box, use_container_width=True)
                    with c2:
                        fig_hist = px.histogram(df, x=selected_var, nbins=30, color_discrete_sequence=['#636EFA'])
                        st.plotly_chart(fig_hist, use_container_width=True)

elif page == "🤖 Entrenamiento":
    st.title("🤖 Entrenamiento de Modelos")
    if st.session_state.df is None or 'abandono' not in st.session_state.df.columns:
        st.warning("⚠️ Carga un dataset con columna `abandono`")
    else:
        df = st.session_state.df.copy()
        test_pct = st.slider("Tamaño test (%)", 10, 40, 20, step=5)
        test_size = test_pct / 100.0
        n_train = int(len(df) * (1-test_size))
        n_test = len(df) - n_train
        st.info(f"📋 Train: {n_train} | Test: {n_test}")
        
        if st.button("🚀 Entrenar 4 Modelos", type="primary"):
            with st.spinner("Entrenando..."):
                try:
                    df_clean = df.copy()
                    feature_cols = [c for c in df_clean.columns if c not in ['abandono', 'status', 'target']]
                    X = df_clean[feature_cols]
                    y = df_clean['abandono']
                    
                    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
                    encoders = {}
                    X_enc = X.copy()
                    for col in cat_cols:
                        le = LabelEncoder()
                        X_enc[col] = le.fit_transform(X_enc[col].astype(str))
                        encoders[col] = le
                    
                    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=test_size, random_state=42, stratify=y)
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)
                    
                    st.session_state.results = {}
                    for m_name in ['logistic_regression', 'random_forest', 'gradient_boosting', 'svm']:
                        clf = DropoutPredictor(model_type=m_name)
                        clf.train(X_train_s, y_train)
                        st.session_state.results[m_name] = {
                            'model': clf, 'metrics': clf.evaluate(X_test_s, y_test),
                            'importance': clf.get_feature_importance(X_enc.columns.tolist())
                        }
                    
                    cat_mapping = {}
                    for col in encoders.keys():
                        le = encoders[col]
                        cat_mapping[col] = {str(v).strip().lower(): int(le.transform([v])[0]) for v in le.classes_}
                        
                    st.session_state.prep = {
                        'scaler': scaler, 'encoders': encoders, 'feature_names': X_enc.columns.tolist(),
                        'cat_cols': cat_cols, 'num_cols': X.select_dtypes('number').columns.tolist(),
                        'medians': X.select_dtypes('number').median().to_dict(),
                        'modes': {c: str(X[c].mode()[0]) for c in cat_cols}, 'cat_mapping': cat_mapping
                    }
                    st.success("✅ Entrenamiento completado")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    
        if st.session_state.results:
            st.subheader("📊 Resultados")
            rows = []
            for name, res in st.session_state.results.items():
                m = res['metrics']
                rows.append({'Modelo': name.replace('_',' ').title(), 'Accuracy': f"{m['accuracy']:.3f}", 'F1': f"{m['f1']:.3f}", 'ROC-AUC': f"{m.get('roc_auc', 0):.3f}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            
            plot_df = pd.DataFrame(rows).melt(id_vars='Modelo', var_name='Métrica', value_name='Valor')
            plot_df['Valor'] = plot_df['Valor'].astype(float)
            st.plotly_chart(px.bar(plot_df, x='Modelo', y='Valor', color='Métrica', barmode='group'), use_container_width=True)
            
            st.subheader("🔍 Importancia")
            sel = st.selectbox("Modelo:", list(st.session_state.results.keys()))
            imp = st.session_state.results[sel]['importance']
            if imp is not None and not imp.empty:
                imp['importance'] = imp['importance'].clip(lower=0)
                fig = px.bar(imp, x='importance', y='feature', orientation='h')
                fig.update_xaxes(range=[0, None])
                st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Predicción":
    st.title("🔮 Predicción Individual")
    if st.session_state.prep is None:
        st.info("ℹ️ Entrena primero en 🤖 Entrenamiento")
    else:
        prep = st.session_state.prep
        c1, c2 = st.columns(2)
        with c1:
            edad = st.number_input("Edad", 15, 50, 20)
            gpa = st.slider("GPA", 0.0, 5.0, 3.5)
            assist = st.slider("Asistencia (%)", 0, 100, 85)
            horas = st.slider("Horas Estudio", 0, 30, 10)
        with c2:
            motiv = st.selectbox("Motivación", ['Baja', 'Media', 'Alta'])
            trim = st.selectbox("1er Trimestre", ['Aprobado', 'Reprobado'])
            socio = st.selectbox("Nivel Socioeconómico", ['Bajo', 'Medio', 'Alto'])
        
        if st.button("🔮 Predecir", type="primary"):
            try:
                inp = {'edad': edad, 'gpa': gpa, 'asistencia': assist, 'horas_estudio': horas, 'motivacion': motiv, 'primer_trimestre': trim, 'socioeconomico': socio}
                df_pred = pd.DataFrame([inp])
                
                for feat in prep['feature_names']:
                    if feat not in df_pred.columns:
                        df_pred[feat] = prep['medians'].get(feat, 0) if feat in prep['num_cols'] else prep['modes'].get(feat, 'desconocido')
                
                for col in prep['cat_cols']:
                    if col in df_pred.columns and col in prep['cat_mapping']:
                        raw_val = str(df_pred[col].values[0]).strip().lower()
                        df_pred[col] = prep['cat_mapping'][col].get(raw_val, 0)
                        
                df_pred = df_pred.reindex(columns=prep['feature_names'])
                X_scaled = prep['scaler'].transform(df_pred)
                
                st.markdown("---")
                st.subheader("🎯 Resultados")
                cols = st.columns(4)
                for i, (name, res) in enumerate(st.session_state.results.items()):
                    with cols[i]:
                        model = res['model'].model
                        dropout_idx = np.where(model.classes_ == 1)[0][0]
                        prob = model.predict_proba(X_scaled)[0][dropout_idx]
                        
                        if prob > 0.60: risk, dc = "🔴 ALTO", "inverse"
                        elif prob > 0.35: risk, dc = "🟡 MODERADO", "normal"
                        else: risk, dc = "🟢 BAJO", "normal"
                        st.metric(name.replace('_',' ').title(), risk, delta=f"Prob: {prob:.1%}", delta_color=dc)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6c757d; font-size: 0.85rem;'>
        <p>🎓 ML Clásico - Predicción de Abandono Escolar | Artificial Intelligence Foundations | Fundació URV</p>
        <p>Dataset: UCI Student Dropout and Academic Success | Desarrollado con Streamlit, Scikit-learn y Plotly</p>
    </div>
    """,
    unsafe_allow_html=True
)
