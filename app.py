"""
Streamlit App: ML Clásico - Predicción de Abandono Escolar
Artificial Intelligence Foundations | Fundació URV | Abril 2026
"""
import sys, os, streamlit as st, pandas as pd, numpy as np, plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.insert(0, current_dir)

from data_loader import load_dataset, preprocess_dataset
from data_preprocessing import handle_missing_values, encode_categorical
from model import DropoutPredictor

st.set_page_config(page_title="🎓 Predicción de Abandono Escolar", page_icon="🎓", layout="wide")

# ==================== ESTADO DE SESIÓN ====================
for k in ['df', 'prep', 'models']: 
    if k not in st.session_state: st.session_state[k] = None

# ==================== SIDEBAR ====================
st.sidebar.title("📋 Navegación")
page = st.sidebar.radio("Selecciona:", ["🏠 Inicio", "📊 Análisis EDA", "🤖 Entrenamiento", "🔮 Predicción"])
st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Cargar Datos")

if st.sidebar.button("🌐 Cargar Dataset UCI"):
    with st.spinner("Descargando..."):
        df, src = load_dataset()
        st.session_state.df = preprocess_dataset(df)
        st.session_state.df_source = src
        st.sidebar.success(f"✅ {src} cargado")

if st.sidebar.button("🎲 Datos Demo"):
    from data_loader import create_sample_dataset
    st.session_state.df = create_sample_dataset()
    st.session_state.df_source = "Demo Dataset"
    st.sidebar.success("✅ Demo cargado")

if st.sidebar.file_uploader("📁 Subir CSV", type=['csv'], key='uploader') is not None:
    st.session_state.df = pd.read_csv(st.session_state['uploader'])
    st.session_state.df.columns = st.session_state.df.columns.str.strip().str.lower().str.replace(r'[^a-z0-9]', '_', regex=True)
    if 'abandono' not in st.session_state.df.columns: st.session_state.df['abandono'] = 0
    st.session_state.df_source = "Archivo CSV"
    st.sidebar.success("✅ CSV cargado")

if st.session_state.df is not None:
    st.sidebar.info(f"📊 Fuente: {getattr(st.session_state, 'df_source', 'Desconocida')} | {len(st.session_state.df)} filas")

# ==================== PÁGINAS ====================
if page == "🏠 Inicio":
    st.title("🎓 Predicción de Abandono Escolar")
    
    # ✅ CARGAR README.MD DINÁMICAMENTE
    readme_path = os.path.join(current_dir, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.markdown("""
        ## 📚 Sobre el Proyecto
        Este proyecto implementa un **modelo de Machine Learning clásico** para predecir el riesgo de abandono escolar.
        ### 📋 Formulación del Problema
        - **Tipo:** Supervisado | **Tarea:** Clasificación binaria
        - **Target:** `Abandono` (0: No, 1: Sí)
        - **Métrica:** F1-Score, ROC-AUC
        ### 🤖 Modelos Implementados
        1. Logistic Regression | 2. Random Forest | 3. Gradient Boosting | 4. SVM
        ### 📊 Características del Dataset
        Edad, GPA, Asistencia, Horas_Estudio, Nivel Socioeconómico, 1er Trimestre, Motivación.
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
        # ✅ VERIFICACIÓN DE DATOS CARGADOS
        st.success(f"✅ Datos activos: {len(df)} registros | {len(df.columns)} columnas")
        if 'abandono' in df.columns:
            st.info(f"📉 Tasa de abandono en dataset actual: `{(df['abandono'].mean()*100):.1f}%`")
        else:
            df['abandono'] = np.random.choice([0,1], len(df))
            
        st.markdown("---")
        
        # ✅ PESTAÑA ACORDEÓN PARA VISTA PREVIA
        with st.expander("📋 Vista Previa de Datos"):
            st.dataframe(df.head(15), use_container_width=True)
            
        st.subheader("📊 Estadísticas Descriptivas")
        num_cols = df.select_dtypes('number').columns.tolist()
        if 'abandono' in num_cols: num_cols.remove('abandono')
        st.dataframe(df[num_cols].describe().T.round(3), use_container_width=True)
        
        st.subheader("📈 Visualizaciones")
        sel = st.selectbox("Selecciona variable:", [c for c in num_cols if c in df.columns])
        if sel:
            st.plotly_chart(px.box(df, x='abandono', y=sel, color='abandono', color_discrete_map={0:'#00c853',1:'#d32f2f'}), use_container_width=True)
        st.plotly_chart(px.pie(values=df['abandono'].value_counts().values, names=['No Abandona','Abandona'], hole=0.4), use_container_width=True)
        if len(num_cols)>=2: st.plotly_chart(px.imshow(df[num_cols].corr(), text_auto='.2f'), use_container_width=True)

elif page == "🤖 Entrenamiento":
    st.title("🤖 Entrenamiento de Modelos")
    if st.session_state.df is None or 'abandono' not in st.session_state.df.columns:
        st.warning("⚠️ Carga dataset con columna `abandono`")
    else:
        test_pct = st.slider("Tamaño Test (%)", 10, 40, 20)/100
        if st.button("🚀 Entrenar 4 Modelos"):
            with st.spinner("Entrenando..."):
                df_c = handle_missing_values(st.session_state.df.copy())
                X, y = df_c.drop(columns=['abandono']), df_c['abandono']
                X_enc, enc = encode_categorical(X)
                X_tr, X_te, y_tr, y_te = train_test_split(X_enc, y, test_size=test_pct, random_state=42, stratify=y)
                sc = StandardScaler()
                X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)
                
                # Entrenar y guardar pipeline completo
                st.session_state.models = {
                    m: {'model': DropoutPredictor(m), 'metrics': {}, 'importance': None} 
                    for m in ['logistic_regression', 'random_forest', 'gradient_boosting', 'svm']
                }
                for m, obj in st.session_state.models.items():
                    obj['model'].train(X_tr_s, y_tr)
                    obj['metrics'] = obj['model'].evaluate(X_te_s, y_te)
                    obj['importance'] = obj['model'].get_feature_importance(X_enc.columns.tolist())
                    
                st.session_state.prep = {
                    'scaler': sc, 'encoders': enc, 'feature_names': X_enc.columns.tolist(),
                    'cat_cols': list(enc.keys()), 'num_cols': X.select_dtypes('number').columns.tolist(),
                    'medians': X.select_dtypes('number').median().to_dict(),
                    'modes': {c: X[c].mode()[0] for c in enc.keys()}
                }
                st.success("✅ Entrenamiento completado. Pipeline guardado.")
                
        if st.session_state.models:
            st.subheader("📊 Resultados de Modelos")
            rows = []
            for n, r in st.session_state.models.items():
                m = r['metrics']
                rows.append({'Modelo': n.replace('_',' ').title(), 'Accuracy': m['accuracy'], 'F1-Score': m['f1'], 'ROC-AUC': m['roc_auc']})
            st.dataframe(pd.DataFrame(rows).style.format({'Accuracy': '{:.3f}', 'F1-Score': '{:.3f}', 'ROC-AUC': '{:.3f}'}), use_container_width=True)
            
            st.subheader("🔍 Importancia de Características")
            sel = st.selectbox("Selecciona modelo:", list(st.session_state.models.keys()))
            imp = st.session_state.models[sel]['importance']
            if imp is not None and not imp.empty:
                imp['feature'] = imp['feature'].str.replace('_', ' ').str.title()
                fig = px.bar(imp, x='importance', y='feature', orientation='h', title=f"Top Features ({sel.replace('_',' ').title()})")
                fig.update_xaxes(range=[0, None])
                st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Predicción":
    st.title("🔮 Predicción Individual")
    if st.session_state.prep is None: st.info("ℹ️ Entrena primero en 🤖 Entrenamiento")
    else:
        prep = st.session_state.prep
        c1, c2 = st.columns(2)
        with c1:
            edad = st.number_input("Edad", 15, 50, 19)
            gpa = st.slider("GPA", 0.0, 5.0, 2.0)
            assist = st.slider("Asistencia (%)", 0, 100, 60)
            horas = st.slider("Horas Estudio/Semana", 0, 20, 3)
        with c2:
            motiv = st.selectbox("Motivación", ['Baja', 'Media', 'Alta'])
            trim = st.selectbox("1er Trimestre", ['Aprobado', 'Reprobado'])
            socio = st.selectbox("Nivel Socioeconómico", ['Bajo', 'Medio', 'Alto'])
            
        if st.button("🔮 Predecir"):
            try:
                # 1. Input exacto
                inp = {'edad': edad, 'gpa': gpa, 'asistencia': assist, 'horas_estudio': horas,
                       'motivacion': motiv, 'primer_trimestre': trim, 'socioeconomico': socio}
                df_pred = pd.DataFrame([inp])
                
                # 2. Rellenar columnas faltantes
                for f in prep['feature_names']:
                    if f not in df_pred.columns:
                        df_pred[f] = prep['medians'].get(f, 0) if f in prep['num_cols'] else prep['modes'].get(f, 'desconocido')
                        
                # 3. Codificar con encoders guardados del entrenamiento
                for col in prep['cat_cols']:
                    le = prep['encoders'][col]
                    val = str(df_pred[col].values[0])
                    df_pred[col] = le.transform([val])[0] if val in le.classes_ else 0
                    
                # 4. Reordenar y escalar CON EL MISMO SCALER
                df_pred = df_pred[prep['feature_names']]
                X_s = prep['scaler'].transform(df_pred)
                
                st.markdown("---")
                st.subheader("🎯 Resultados de Probabilidad")
                cols = st.columns(4)
                for i, (n, r) in enumerate(st.session_state.models.items()):
                    with cols[i]:
                        prob = r['model'].model.predict_proba(X_s)[0][1]
                        risk = "🔴 ALTO RIESGO" if prob > 0.5 else "🟢 BAJO RIESGO"
                        st.metric(n.replace('_',' ').title(), risk, delta=f"Prob: {prob:.1%}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
