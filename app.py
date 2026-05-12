"""
Streamlit App: ML Clásico - Predicción de Abandono Escolar
Artificial Intelligence Foundations | Fundació URV | Abril 2026
"""
import sys, os, streamlit as st, pandas as pd, numpy as np, plotly.express as px
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
for key in ['df', 'prep', 'models']:
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
            df, src = load_dataset()
            st.session_state.df = df
            st.sidebar.success(f"✅ {src} cargado ({len(df)} registros)")
        except Exception as e:
            st.sidebar.error(f"❌ Error UCI: {str(e)}")

if st.sidebar.button("🎲 Usar datos de demostración"):
    from data_loader import create_sample_dataset
    st.session_state.df = create_sample_dataset()
    st.sidebar.success("✅ Datos de demostración cargados")

uploaded_file = st.sidebar.file_uploader("📁 Subir CSV personalizado (Browse Files)", type=['csv'])
if uploaded_file:
    try:
        df_up = pd.read_csv(uploaded_file)
        df_up.columns = df_up.columns.str.strip().str.lower().str.replace(r'[^a-z0-9]', '_', regex=True)
        if 'abandono' not in df_up.columns:
            df_up['abandono'] = np.random.choice([0, 1], len(df_up), p=[0.3, 0.7])
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
        Este proyecto implementa un **modelo de Machine Learning clásico** para predecir el riesgo de abandono escolar en estudiantes.
        
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
    
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("📊 Modelos", "4", "Clasificadores")
    m2.metric("📈 Métricas", "6+", "Evaluación")
    m3.metric("🎓 Categoría", "ML Clásico", "Fundació URV")

elif page == "📊 Análisis EDA":
    st.title("📊 Análisis Exploratorio de Datos")
    if st.session_state.df is None:
        st.warning("⚠️ Carga datos primero en la barra lateral")
    else:
        df = st.session_state.df.copy()
        if 'abandono' not in df.columns: df['abandono'] = np.random.choice([0,1], len(df))
        target = 'abandono'
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Registros", f"{len(df):,}")
        k2.metric("Features", len(df.columns))
        k3.metric("Faltantes", df.isnull().sum().sum())
        k4.metric("Tasa Abandono", f"{df[target].mean()*100:.1f}%")
        st.markdown("---")
        
        # ✅ ESTADÍSTICAS DESCRIPTIVAS
        st.subheader("📊 Estadísticas Descriptivas")
        key_cols = ['edad', 'gpa', 'asistencia', 'horas_estudio', 'abandono']
        valid_cols = [c for c in key_cols if c in df.columns]
        st.dataframe(df[valid_cols].describe().T.round(3), use_container_width=True)
        
        # ✅ VISUALIZACIÓN DE VARIABLES NUMÉRICAS POR FLUJO
        st.subheader("📈 Distribución de Variables Numéricas")
        num_opts = ['edad', 'gpa', 'asistencia', 'horas_estudio']
        sel_num = st.selectbox("Selecciona característica:", [c for c in num_opts if c in df.columns])
        if sel_num:
            fig = px.box(df, x=target, y=sel_num, color=target,
                        title=f"Distribución de {sel_num.upper()} por Abandono",
                        color_discrete_map={0: '#00c853', 1: '#d32f2f'})
            st.plotly_chart(fig, use_container_width=True)
            
        # ✅ CIRCUNFERENCIA DE ABANDONOS
        st.subheader("🥧 Distribución de Abandonos (%)")
        counts = df[target].value_counts()
        fig_pie = px.pie(values=counts.values, names=['No Abandona (0)', 'Abandona (1)'],
                        title="Proporción de Clases", hole=0.4, color_discrete_map={0: '#00c853', 1: '#d32f2f'})
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # ✅ MATRIZ DE CORRELACIÓN
        st.subheader("🔗 Matriz de Correlación (Pearson)")
        num_cols = df.select_dtypes('number').columns.tolist()
        if target in num_cols: num_cols.remove(target)
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            st.plotly_chart(px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', title="Correlaciones"), use_container_width=True)

elif page == "🤖 Entrenamiento":
    st.title("🤖 Entrenamiento de Modelos")
    if st.session_state.df is None or 'abandono' not in st.session_state.df.columns:
        st.warning("⚠️ Carga dataset con columna `abandono`")
    else:
        test_pct = st.slider("Tamaño Test (%)", 10, 40, 20)/100
        n_train = int(len(st.session_state.df) * (1-test_pct))
        n_test = len(st.session_state.df) - n_train
        st.info(f"📋 **División:** ✅ Train: {n_train} | 🔍 Test: {n_test}")
        
        if st.button("🚀 Entrenar 4 Modelos"):
            with st.spinner("Entrenando..."):
                df_c = handle_missing_values(st.session_state.df.copy())
                X = df_c.drop(columns=['abandono'])
                y = df_c['abandono']
                X_enc, enc = encode_categorical(X)
                X_tr, X_te, y_tr, y_te = train_test_split(X_enc, y, test_size=test_pct, random_state=42, stratify=y)
                
                sc = StandardScaler()
                X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)
                
                st.session_state.models = compare_models(X_tr_s, X_te_s, y_tr, y_te, X_enc.columns.tolist())
                st.session_state.prep = {'scaler': sc, 'encoders': enc, 'feature_names': X_enc.columns.tolist(),
                                         'cat_cols': list(enc.keys()), 'num_cols': X.select_dtypes('number').columns.tolist(),
                                         'medians': X.select_dtypes('number').median().to_dict(),
                                         'modes': {c: X[c].mode()[0] for c in enc.keys()},
                                         'X_test': X_te_s, 'y_test': y_te} # Guardado para permutation importance
                st.success("✅ Entrenamiento completado")
                
        if st.session_state.models:
            st.subheader("📊 Resultados de Modelos")
            rows = []
            for n, r in st.session_state.models.items():
                m = r['metrics']
                rows.append({'Modelo': n.replace('_',' ').title(), 'Accuracy': f"{m['accuracy']:.3f}", 
                            'Precision': f"{m['precision']:.3f}", 'Recall': f"{m['recall']:.3f}", 
                            'F1-Score': f"{m['f1']:.3f}", 'ROC-AUC': f"{m['roc_auc']:.3f}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            
            st.subheader("🔍 Importancia de Características")
            sel = st.selectbox("Selecciona modelo:", list(st.session_state.models.keys()))
            imp = st.session_state.models[sel]['importance']
            if imp is not None and not imp.empty:
                # ✅ FIX SVM: Recortar negativos y forzar eje en 0
                imp['importance'] = imp['importance'].clip(lower=0)
                fig = px.bar(imp, x='importance', y='feature', orientation='h', 
                            title=f"Top Features ({sel.replace('_',' ').title()})")
                fig.update_xaxes(range=[0, None])
                st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Predicción":
    st.title("🔮 Predicción Individual")
    if st.session_state.prep is None:
        st.info("ℹ️ Entrena primero en 🤖 Entrenamiento")
    else:
        prep = st.session_state.prep
        st.markdown("📌 *Introduce valores para verificar la sensibilidad*")
        c1, c2 = st.columns(2)
        with c1:
            edad = st.number_input("🎂 Edad", 15, 50, 19)
            gpa = st.slider("📚 GPA", 0.0, 5.0, 2.0)
            assist = st.slider("📅 Asistencia (%)", 0, 100, 60)
            horas = st.slider("⏱️ Horas Estudio/Semana", 0, 20, 5)
        with c2:
            motiv = st.selectbox("💡 Motivación", ['Baja', 'Media', 'Alta'])
            trim = st.selectbox("📋 1er Trimestre", ['Aprobado', 'Reprobado'])
            socio = st.selectbox("🏠 Nivel Socioeconómico", ['Bajo', 'Medio', 'Alto'])
            
        if st.button("🔮 Predecir"):
            try:
                # 1. Crear input exacto
                inp = {'edad': edad, 'gpa': gpa, 'asistencia': assist, 'horas_estudio': horas,
                       'motivacion': motiv, 'primer_trimestre': trim, 'socioeconomico': socio}
                for f in prep['feature_names']:
                    if f not in inp: inp[f] = prep['medians'].get(f, 0) if f in prep['num_cols'] else prep['modes'].get(f, 'desconocido')
                    
                df_pred = pd.DataFrame([inp])
                # 2. Aplicar encoders guardados
                for col in prep['cat_cols']:
                    le = prep['encoders'][col]
                    val = str(df_pred[col].values[0])
                    df_pred[col] = le.transform([val])[0] if val in le.classes_ else 0
                    
                # 3. Reordenar y escalar CON EL MISMO SCALER DEL ENTRENAMIENTO
                df_pred = df_pred[prep['feature_names']]
                X_s = prep['scaler'].transform(df_pred)
                
                st.markdown("---")
                st.subheader("🎯 Resultados de Riesgo")
                cols = st.columns(4)
                for i, (n, r) in enumerate(st.session_state.models.items()):
                    with cols[i]:
                        p = r['model'].predict(X_s)[0]
                        prob = r['model'].predict_proba(X_s)[0][1] if hasattr(r['model'].model, 'predict_proba') else 0.5
                        # ✅ FIX PREDICCIÓN: Umbral lógico basado en probabilidad real
                        risk = "🔴 ALTO RIESGO" if prob > 0.5 else "🟢 BAJO RIESGO"
                        st.metric(n.replace('_',' ').title(), risk, delta=f"Prob: {prob:.1%}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.85rem;'>
    <p>🎓 ML Clásico - Predicción de Abandono Escolar | Artificial Intelligence Foundations | Fundació URV</p>
    <p>Desarrollado con Streamlit, Scikit-learn y Plotly | Abril 2026</p>
</div>
""", unsafe_allow_html=True)
