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

from data_loader import load_dataset, preprocess_dataset
from data_preprocessing import handle_missing_values, encode_categorical
from model import DropoutPredictor

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

# ==================== SIDEBAR ====================
st.sidebar.title("📋 Navegación")
page = st.sidebar.radio("Selecciona una sección:", 
    ["🏠 Inicio", "📊 Análisis EDA", "🤖 Entrenamiento", "🔮 Predicción"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Cargar Datos")

if st.sidebar.button("🌐 Cargar Dataset UCI Oficial"):
    with st.spinner("Descargando y procesando..."):
        try:
            df_raw, source = load_dataset()
            df_proc = preprocess_dataset(df_raw)
            target_col = next((c for c in ['target', 'status', 'Target', 'Status'] if c in df_proc.columns), None)
            if target_col:
                df_proc['Abandono'] = df_proc[target_col].apply(lambda x: 1 if 'dropout' in str(x).lower() else 0)
            else:
                df_proc['Abandono'] = np.random.choice([0, 1], len(df_proc), p=[0.3, 0.7])
            st.session_state.df = df_proc
            st.sidebar.success(f"✅ {source} cargado ({len(df_proc)} registros)")
        except Exception as e:
            st.sidebar.error(f"❌ Error UCI: {str(e)}")

if st.sidebar.button("🎲 Usar datos de demostración"):
    from data_loader import create_sample_dataset
    df_demo = create_sample_dataset()
    df_demo['Abandono'] = np.random.choice([0, 1], len(df_demo), p=[0.3, 0.7])
    st.session_state.df = df_demo
    st.sidebar.success("✅ Datos de demostración cargados")

uploaded_file = st.sidebar.file_uploader("📁 Subir CSV personalizado (Browse Files)", type=['csv'])
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
        if 'abandono' in df.columns and 'Abandono' not in df.columns:
            df.rename(columns={'abandono': 'Abandono'}, inplace=True)
        if 'Abandono' not in df.columns:
            df['Abandono'] = np.random.choice([0, 1], len(df))
            
        target = 'Abandono'
        
        # ✅ CÁLCULO CORRECTO DE TASA DE ABANDONO
        if df[target].dtype == 'object':
            dropout_count = df[target].astype(str).str.lower().isin(['dropout', 'abandono', '1', 'si']).sum()
        else:
            unique = df[target].unique()
            if 0 in unique and 1 in unique:
                # Si hay más 0s que 1s, probablemente 0=Abandono (invertido)
                dropout_count = (df[target] == 0).sum() if (df[target]==0).sum() > (df[target]==1).sum() else (df[target] == 1).sum()
            else:
                dropout_count = (df[target] == 1).sum()
        dropout_rate = (dropout_count / len(df)) * 100
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("📊 Total Registros", f"{len(df):,}")
        k2.metric("📋 Características", len(df.columns))
        k3.metric("❌ Valores Faltantes", df.isnull().sum().sum())
        k4.metric("📉 Tasa de Abandono", f"{dropout_rate:.1f}%")
        
        with st.expander("📋 Vista Previa de Datos"):
            st.dataframe(df.head(15), use_container_width=True)
            
        st.subheader("📊 Estadísticas Descriptivas")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in num_cols: num_cols.remove(target)
        st.dataframe(df[num_cols].describe().T.round(3), use_container_width=True)
        
        st.subheader("🥧 Distribución de Abandonos (%)")
        counts = df[target].value_counts()
        fig_pie = px.pie(values=counts.values, names=['No Abandona (0)', 'Abandona (1)'],
                        title="Proporción de Clases", hole=0.4, color_discrete_map={0: '#00c853', 1: '#d32f2f'})
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.subheader("🔗 Matriz de Correlación (Pearson)")
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            st.plotly_chart(px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', title="Correlaciones"), use_container_width=True)
            
        st.subheader("📈 Visualización por Variable")
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
        test_pct = st.slider("📊 Tamaño del conjunto de prueba (%)", 10, 40, 20, step=5)
        test_size = test_pct / 100.0
        n_train = int(len(df) * (1-test_size))
        n_test = len(df) - n_train
        st.info(f"📋 **División de Datos:**\n✅ Entrenamiento: **{n_train}** registros ({(1-test_size)*100:.0f}%)\n🔍 Prueba: **{n_test}** registros ({test_size*100:.0f}%)")
        
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
                    
                    st.session_state.results = {}
                    for m_name in ['logistic_regression', 'random_forest', 'gradient_boosting', 'svm']:
                        clf = DropoutPredictor(model_type=m_name)
                        clf.train(X_train_s, y_train)
                        st.session_state.results[m_name] = {
                            'model': clf,
                            'metrics': clf.evaluate(X_test_s, y_test),
                            'importance': clf.get_feature_importance(X_enc.columns.tolist())
                        }
                    
                    # 🔑 PIPELINE ROBUSTO: Guardar mapeo seguro de categóricas
                    cat_mapping = {}
                    for col in encoders.keys():
                        le = encoders[col]
                        cat_mapping[col] = {str(v).strip().lower(): int(le.transform([v])[0]) for v in le.classes_}
                        
                    st.session_state.prep = {
                        'scaler': scaler, 'encoders': encoders, 'feature_names': X_enc.columns.tolist(),
                        'cat_cols': list(encoders.keys()), 'num_cols': X.select_dtypes('number').columns.tolist(),
                        'medians': X.select_dtypes('number').median().to_dict(),
                        'modes': {c: str(X[c].mode()[0]) for c in encoders.keys()},
                        'cat_mapping': cat_mapping, 'X_test': X_test_s, 'y_test': y_test
                    }
                    st.success("✅ Entrenamiento completado. Pipeline guardado para predicción.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    
        if st.session_state.results:
            st.subheader("📊 Resultados de Modelos")
            rows = []
            for name, res in st.session_state.results.items():
                m = res['metrics']
                rows.append({'Modelo': name.replace('_',' ').title(), 
                            'Accuracy': f"{m['accuracy']:.3f}", 'Precision': f"{m['precision']:.3f}",
                            'Recall': f"{m['recall']:.3f}", 'F1-Score': f"{m['f1']:.3f}", 
                            'ROC-AUC': f"{m.get('roc_auc', 0):.3f}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            
            # 📈 GRÁFICO COMPARATIVO
            plot_df = pd.DataFrame(rows).melt(id_vars='Modelo', var_name='Métrica', value_name='Valor')
            plot_df['Valor'] = plot_df['Valor'].astype(float)
            st.plotly_chart(px.bar(plot_df, x='Modelo', y='Valor', color='Métrica', barmode='group', title="Comparación de Rendimiento"), use_container_width=True)
            
            st.subheader("🔍 Importancia de Características")
            sel_model = st.selectbox("Selecciona modelo:", list(st.session_state.results.keys()))
            imp_df = st.session_state.results[sel_model]['importance']
            if imp_df is not None and not imp_df.empty:
                imp_df['importance'] = imp_df['importance'].clip(lower=0)
                fig = px.bar(imp_df, x='importance', y='feature', orientation='h',
                            title=f"Top Features ({sel_model.replace('_',' ').title()})")
                fig.update_xaxes(range=[0, None]) # ✅ Fuerza origen en 0
                st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Predicción":
    st.title("🔮 Predicción Individual")
    if st.session_state.prep is None:
        st.info("ℹ️ Entrena primero en la sección 🤖 Entrenamiento para guardar escaladores y codificadores.")
    else:
        prep = st.session_state.prep
        st.markdown("📌 *Ajusta los valores para ver cómo cambia la probabilidad en tiempo real*")
        
        c1, c2 = st.columns(2)
        with c1:
            edad = st.number_input("🎂 Edad", 15, 50, 20)
            gpa = st.slider("📚 GPA / Nota Promedio (0-5)", 0.0, 5.0, 3.5)
            assist = st.slider("📅 Asistencia (%)", 0, 100, 85)
            horas = st.slider("⏱️ Horas de Estudio (semanales)", 0, 30, 10)
        with c2:
            motiv = st.selectbox("💡 Motivación", ['Baja', 'Media', 'Alta'])
            trim = st.selectbox("📋 Desempeño 1er Trimestre", ['Aprobado', 'Reprobado'])
            socio = st.selectbox("🏠 Nivel Socioeconómico", ['Bajo', 'Medio', 'Alto'])
            
        if st.button("🔮 Realizar Predicción", type="primary"):
            try:
                inp = {'edad': edad, 'gpa': gpa, 'asistencia': assist, 'horas_estudio': horas,
                       'motivacion': motiv, 'primer_trimestre': trim, 'socioeconomico': socio}
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
                st.subheader("🎯 Resultados de Probabilidad")
                cols = st.columns(4)
                for i, (name, res) in enumerate(st.session_state.results.items()):
                    with cols[i]:
                        model = res['model'].model
                        # ✅ FIX CRÍTICO: Detectar índice correcto de la clase "Abandono" (1)
                        dropout_idx = np.where(model.classes_ == 1)[0][0]
                        prob = model.predict_proba(X_scaled)[0][dropout_idx]
                        
                        # ✅ Umbrales realistas para datos desequilibrados (~30% dropout)
                        if prob > 0.60:
                            risk = "🔴 ALTO RIESGO"
                            delta_color = "inverse"
                        elif prob > 0.35:
                            risk = "🟡 RIESGO MODERADO"
                            delta_color = "normal"
                        else:
                            risk = "🟢 BAJO RIESGO"
                            delta_color = "normal"
                            
                        st.metric(name.replace('_',' ').title(), risk, delta=f"Prob: {prob:.1%}", delta_color=delta_color)
                        
            except Exception as e:
                st.error(f"❌ Error en predicción: {str(e)}")
                st.exception(e)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.85rem;'>
    <p>🎓 ML Clásico - Predicción de Abandono Escolar | Artificial Intelligence Foundations | Fundació URV</p>
    <p>Desarrollado con Streamlit, Scikit-learn y Plotly | Abril 2026</p>
</div>
""", unsafe_allow_html=True)
