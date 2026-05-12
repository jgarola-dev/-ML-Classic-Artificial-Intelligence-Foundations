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

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from data_loader import load_dataset
from data_preprocessing import handle_missing_values, encode_categorical
from model import DropoutPredictor

# ==================== CONFIGURACIÓN ====================
st.set_page_config(page_title="🎓 Predicción de Abandono Escolar", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

# ==================== ESTADO DE SESIÓN ====================
for k in ['df', 'prep', 'results']:
    if k not in st.session_state: st.session_state[k] = None

# ==================== SIDEBAR ====================
st.sidebar.title("📋 Navegación")
page = st.sidebar.radio("Selecciona una sección:", ["🏠 Inicio", "📊 Análisis EDA", "🤖 Entrenamiento", "🔮 Predicción"])
st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Cargar Datos")

if st.sidebar.button("🌐 Cargar Dataset UCI Oficial"):
    with st.spinner("Descargando y procesando..."):
        try:
            df, src = load_dataset()
            st.session_state.df = df
            st.sidebar.success(f"✅ {src} cargado ({len(df)} registros)")
        except Exception as e:
            st.sidebar.error(f"❌ Error UCI: {str(e)}")

if st.sidebar.button("🎲 Usar datos de demostración"):
    from data_loader import create_sample_dataset
    st.session_state.df = create_sample_dataset()
    st.sidebar.success("✅ Demo cargado")

uploaded_file = st.sidebar.file_uploader("📁 Subir CSV", type=['csv'])
if uploaded_file:
    try:
        df_up = pd.read_csv(uploaded_file)
        df_up.columns = df_up.columns.str.strip().str.lower().str.replace(r'[^a-z0-9]', '_', regex=True)
        if 'abandono' not in df_up.columns:
            df_up['abandono'] = np.random.choice([0, 1], len(df_up), p=[0.3, 0.7])
        st.session_state.df = df_up
        st.sidebar.success("✅ CSV cargado")
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")

if st.session_state.df is not None:
    st.sidebar.info(f"📊 Dataset: {len(st.session_state.df)} filas × {len(st.session_state.df.columns)} columnas")

# ==================== PÁGINAS ====================
if page == "🏠 Inicio":
    st.title("🎓 Predicción de Abandono Escolar")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 📚 Sobre el Proyecto
        Este proyecto implementa un modelo de Machine Learning clásico para predecir el riesgo de abandono escolar en estudiantes.
        
        ### 🎯 Objetivos
        - Identificar estudiantes en riesgo de abandono
        - Proporcionar recomendaciones de intervención
        - Analizar factores clave del abandono escolar
        """)
    with col2:
        st.markdown("""
        ### 📋 Formulación del Problema
        - **Tipo de aprendizaje:** Supervisado
        - **Tarea:** Clasificación binaria
        - **Variable objetivo:** Abandono (0/1)
        - **Métrica de éxito:** F1-Score, ROC-AUC
        """)
    st.markdown("""
    ### 🤖 Modelos Implementados
    Se utilizan 4 modelos ML clásicos:
    1. **Logistic Regression** → Modelo lineal simple y interpretable
    2. **Random Forest** → Ensemble de árboles de decisión
    3. **Gradient Boosting** → Boosting secuencial de árboles
    4. **Support Vector Machine (SVM)** → Máquinas de vectores de soporte
    """)
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("📊 Modelos", "4", "Clasificadores")
    m2.metric("📈 Métricas", "6+", "Evaluación")
    m3.metric("🎓 Categoría", "ML Clásico", "Fundació URV")
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

elif page == "📊 Análisis EDA":
    st.title("📊 Análisis Exploratorio de Datos")
    if st.session_state.df is None:
        st.warning("⚠️ Carga datos primero")
    else:
        df = st.session_state.df.copy()
        if 'abandono' in df.columns and 'Abandono' not in df.columns:
            df.rename(columns={'abandono': 'Abandono'}, inplace=True)
        target = 'Abandono'
        
        # ✅ CÁLCULO CORRECTO DE TASA (clase minoritaria = abandono)
        if df[target].dtype == 'object':
            drop_count = df[target].astype(str).str.lower().isin(['dropout', 'abandono', '1', 'si']).sum()
        else:
            c0, c1 = (df[target]==0).sum(), (df[target]==1).sum()
            drop_count = c0 if c0 < c1 else c1
        drop_rate = (drop_count / len(df)) * 100
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("📊 Total Registros", f"{len(df):,}")
        k2.metric("📋 Características", len(df.columns))
        k3.metric("❌ Valores Faltantes", df.isnull().sum().sum())
        k4.metric("📉 Tasa de Abandono", f"{drop_rate:.1f}%")
        
        with st.expander("📋 Vista Previa de Datos"):
            st.dataframe(df.head(15), use_container_width=True)
            
        st.subheader("📊 Estadísticas Descriptivas")
        num_cols = df.select_dtypes('number').columns.tolist()
        if target in num_cols: num_cols.remove(target)
        st.dataframe(df[num_cols].describe().T.round(3), use_container_width=True)
        
        st.subheader("🥧 Distribución de Abandonos (%)")
        counts = df[target].value_counts()
        labels = ["No Abandona (0)" if i==0 else "Abandona (1)" for i in counts.index]
        fig_pie = px.pie(values=counts.values, names=labels, title="Proporción de Clases", hole=0.4, color_discrete_map={'No Abandona (0)':'#00c853', 'Abandona (1)':'#d32f2f'})
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.subheader("🔗 Matriz de Correlación")
        if len(num_cols) >= 2:
            st.plotly_chart(px.imshow(df[num_cols].corr(), text_auto='.2f', color_continuous_scale='RdBu_r'), use_container_width=True)
            
        st.subheader("📈 Visualización por Variable")
        if num_cols:
            sel = st.selectbox("Selecciona variable:", num_cols)
            st.plotly_chart(px.box(df, x=target, y=sel, color=target, color_discrete_map={0:'#00c853', 1:'#d32f2f'}), use_container_width=True)

elif page == "🤖 Entrenamiento":
    st.title("🤖 Entrenamiento de Modelos")
    if st.session_state.df is None or 'Abandono' not in st.session_state.df.columns:
        st.warning("⚠️ Carga dataset con columna `Abandono`")
    else:
        df = st.session_state.df.copy()
        test_pct = st.slider("📊 Tamaño del conjunto de prueba (%)", 10, 40, 20)/100
        n_train, n_test = int(len(df)*(1-test_pct)), len(df) - int(len(df)*(1-test_pct))
        st.info(f"📋 **División:** ✅ Train: {n_train} | 🔍 Test: {n_test}")
        
        if st.button("🚀 Entrenar 4 Modelos", type="primary"):
            with st.spinner("Entrenando..."):
                try:
                    df_c = handle_missing_values(df.copy())
                    X, y = df_c.drop(columns=['Abandono']), df_c['Abandono']
                    X_enc, enc = encode_categorical(X)
                    X_tr, X_te, y_tr, y_te = train_test_split(X_enc, y, test_size=test_pct, random_state=42, stratify=y)
                    sc = StandardScaler()
                    X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)
                    
                    st.session_state.results = {}
                    for m in ['logistic_regression', 'random_forest', 'gradient_boosting', 'svm']:
                        clf = DropoutPredictor(model_type=m)
                        clf.train(X_tr_s, y_tr)
                        st.session_state.results[m] = {'model': clf, 'metrics': clf.evaluate(X_te_s, y_te), 'importance': clf.get_feature_importance(X_enc.columns.tolist())}
                    
                    cat_map = {col: {str(v).strip().lower(): int(enc[col].transform([v])[0]) for v in enc[col].classes_} for col in enc.keys()}
                    st.session_state.prep = {'scaler': sc, 'encoders': enc, 'feature_names': X_enc.columns.tolist(), 'cat_cols': list(enc.keys()), 'num_cols': X.select_dtypes('number').columns.tolist(), 'medians': X.select_dtypes('number').median().to_dict(), 'modes': {c: str(X[c].mode()[0]) for c in enc.keys()}, 'cat_mapping': cat_map}
                    st.success("✅ Entrenamiento completado")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    
        if st.session_state.results:
            st.subheader("📊 Resultados de Modelos")
            rows = [{'Modelo': n.replace('_',' ').title(), 'Accuracy': f"{m['accuracy']:.3f}", 'Precision': f"{m['precision']:.3f}", 'Recall': f"{m['recall']:.3f}", 'F1-Score': f"{m['f1']:.3f}", 'ROC-AUC': f"{m.get('roc_auc', 0):.3f}"} for n, r in st.session_state.results.items() for m in [r['metrics']]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            
            st.subheader("📈 Comparación de Rendimiento")
            plot_df = pd.DataFrame(rows).melt(id_vars='Modelo', var_name='Métrica', value_name='Valor')
            plot_df['Valor'] = plot_df['Valor'].astype(float)
            st.plotly_chart(px.bar(plot_df, x='Modelo', y='Valor', color='Métrica', barmode='group'), use_container_width=True)
            
            st.subheader("🔍 Importancia de Características")
            sel = st.selectbox("Selecciona modelo:", list(st.session_state.results.keys()))
            imp = st.session_state.results[sel]['importance']
            if imp is not None and not imp.empty:
                imp['importance'] = imp['importance'].clip(lower=0)
                fig = px.bar(imp, x='importance', y='feature', orientation='h', title=f"Top Features ({sel.replace('_',' ').title()})")
                fig.update_xaxes(range=[0, None])
                st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Predicción":
    st.title("🔮 Predicción Individual")
    if st.session_state.prep is None:
        st.info("ℹ️ Entrena primero en 🤖 Entrenamiento")
    else:
        prep = st.session_state.prep
        st.markdown("📌 *Ajusta los valores para ver cómo cambia la probabilidad*")
        c1, c2 = st.columns(2)
        with c1:
            edad = st.number_input("🎂 Edad", 15, 50, 20)
            gpa = st.slider("📚 GPA (0-5)", 0.0, 5.0, 3.5)
            assist = st.slider("📅 Asistencia (%)", 0, 100, 85)
            horas = st.slider("⏱️ Horas Estudio/Semana", 0, 30, 10)
        with c2:
            motiv = st.selectbox("💡 Motivación", ['Baja', 'Media', 'Alta'])
            trim = st.selectbox("📋 1er Trimestre", ['Aprobado', 'Reprobado'])
            socio = st.selectbox("🏠 Nivel Socioeconómico", ['Bajo', 'Medio', 'Alto'])
            
        if st.button("🔮 Realizar Predicción", type="primary"):
            try:
                inp = {'edad': edad, 'gpa': gpa, 'asistencia': assist, 'horas_estudio': horas, 'motivacion': motiv, 'primer_trimestre': trim, 'socioeconomico': socio}
                df_pred = pd.DataFrame([inp])
                for f in prep['feature_names']:
                    if f not in df_pred.columns:
                        df_pred[f] = prep['medians'].get(f, 0) if f in prep['num_cols'] else prep['modes'].get(f, 'desconocido')
                for col in prep['cat_cols']:
                    if col in df_pred.columns and col in prep['cat_mapping']:
                        raw = str(df_pred[col].values[0]).strip().lower()
                        df_pred[col] = prep['cat_mapping'][col].get(raw, 0)
                df_pred = df_pred.reindex(columns=prep['feature_names'])
                X_s = prep['scaler'].transform(df_pred)
                
                st.markdown("---")
                st.subheader("🎯 Resultados de Probabilidad de Abandono")
                cols = st.columns(4)
                for i, (n, r) in enumerate(st.session_state.results.items()):
                    with cols[i]:
                        model = r['model'].model
                        # ✅ FIX CRÍTICO: Detectar índice correcto de la clase "Abandono" (1)
                        classes = list(model.classes_)
                        idx = classes.index(1) if 1 in classes else (classes.index(0) if 0 in classes else 0)
                        prob = model.predict_proba(X_s)[0][idx]
                        
                        # ✅ Umbrales realistes per a dades desequilibrades (~30% dropout)
                        if prob > 0.60: risk, dc = "🔴 ALTO RIESGO", "inverse"
                        elif prob > 0.35: risk, dc = "🟡 RIESGO MODERADO", "normal"
                        else: risk, dc = "🟢 BAJO RIESGO", "normal"
                        st.metric(n.replace('_',' ').title(), risk, delta=f"Prob: {prob:.1%}", delta_color=dc)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

st.markdown("---")
st.markdown("<div style='text-align:center; color:#6c757d; font-size:0.85rem;'>🎓 ML Clásico - Predicción de Abandono Escolar | Artificial Intelligence Foundations | Fundació URV | Abril 2026</div>", unsafe_allow_html=True)
