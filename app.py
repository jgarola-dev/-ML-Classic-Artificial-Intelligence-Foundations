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

from data_loader import load_dataset
from data_preprocessing import handle_missing_values, encode_categorical
from model import DropoutPredictor, compare_models

st.set_page_config(page_title="🎓 Predicción de Abandono Escolar", page_icon="🎓", layout="wide")

# Estado de sesión
for key in ['df', 'prep', 'results']:
    if key not in st.session_state: st.session_state[key] = None

# ==================== SIDEBAR ====================
st.sidebar.title("📋 Navegación")
page = st.sidebar.radio("Selecciona:", ["🏠 Inicio", "📊 Análisis EDA", "🤖 Entrenamiento", "🔮 Predicción"])
st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Cargar Datos")

# 🔘 Botones siempre visibles
if st.sidebar.button("🌐 Cargar Dataset UCI Oficial"):
    with st.spinner("Descargando..."):
        df, src = load_dataset()
        st.session_state.df = df
        st.sidebar.success(f"✅ {src} cargado ({len(df)} registros)")

if st.sidebar.button("🎲 Usar datos de demostración"):
    from data_loader import create_sample_dataset
    st.session_state.df = create_sample_dataset()
    st.sidebar.success("✅ Demo cargado (2000 registros)")

# 🔘 Browse Files (siempre visible e independiente)
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
    st.title("🎓 ML Clásico - Predicción de Abandono Escolar")
    st.markdown("""
    Este proyecto implementa un pipeline de **Machine Learning clásico** para identificar estudiantes en riesgo.
    Selecciona una sección en la barra lateral para comenzar el análisis.
    """)
    c1, c2, c3 = st.columns(3)
    c1.metric("🤖 Modelos", "4", "Clasificadores")
    c2.metric("📈 Métricas", "6+", "Evaluación")
    c3.metric("🎓 Categoría", "ML Clásico", "Fundació URV")

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
        
        st.subheader("📊 Estadísticas Descriptivas (Variables Clave)")
        key_cols = ['edad', 'gpa', 'asistencia', 'horas_estudio', 'abandono']
        valid_cols = [c for c in key_cols if c in df.columns]
        if valid_cols:
            st.dataframe(df[valid_cols].describe().T.round(3), use_container_width=True)
            
        st.subheader("🥧 Distribución de Clases")
        counts = df[target].value_counts()
        fig = px.pie(values=counts.values, names=['No Abandona (0)', 'Abandona (1)'],
                    title="Proporción", hole=0.4, color_discrete_map={0:'#00c853', 1:'#d32f2f'})
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🔗 Matriz de Correlación")
        num_cols = df.select_dtypes('number').columns.tolist()
        if target in num_cols: num_cols.remove(target)
        if len(num_cols) >= 2:
            st.plotly_chart(px.imshow(df[num_cols].corr(), text_auto='.2f', color_continuous_scale='RdBu_r', title="Correlaciones"), use_container_width=True)

elif page == "🤖 Entrenamiento":
    st.title("🤖 Entrenamiento de Modelos")
    if st.session_state.df is None:
        st.warning("⚠️ Carga datos primero")
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
                st.session_state.results = compare_models(sc.fit_transform(X_tr), sc.transform(X_te), y_tr, y_te, X_enc.columns.tolist())
                st.session_state.prep = {'scaler': sc, 'encoders': enc, 'feature_names': X_enc.columns.tolist(),
                                         'cat_cols': list(enc.keys()), 'num_cols': X.select_dtypes('number').columns.tolist(),
                                         'medians': X.select_dtypes('number').median().to_dict(),
                                         'modes': {c: X[c].mode()[0] for c in enc.keys()}}
                st.success("✅ Entrenamiento completado")
                
        if st.session_state.results is not None:
            st.subheader("📊 Resultados de Modelos")
            rows = []
            for n, r in st.session_state.results.items():
                m = r['metrics']
                rows.append({'Modelo': n.replace('_',' ').title(), 'Accuracy': f"{m['accuracy']:.3f}", 
                            'Precision': f"{m['precision']:.3f}", 'Recall': f"{m['recall']:.3f}", 
                            'F1-Score': f"{m['f1']:.3f}", 'ROC-AUC': f"{m['roc_auc']:.3f}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            
            st.subheader("🔍 Importancia de Características")
            sel = st.selectbox("Selecciona modelo:", list(st.session_state.results.keys()))
            imp = st.session_state.results[sel]['importance']
            if imp is not None and not imp.empty:
                st.plotly_chart(px.bar(imp, x='importance', y='feature', orientation='h', title=f"Top Features ({sel})"), use_container_width=True)

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
            gpa = st.slider("📚 GPA / Nota Promedio", 0.0, 5.0, 2.0)
            assist = st.slider("📅 Asistencia (%)", 0, 100, 60)
        with c2:
            horas = st.slider("⏱️ Horas Estudio/Semana", 0, 20, 5)
            motiv = st.selectbox("💡 Motivación", ['Baja', 'Media', 'Alta'])
            trim = st.selectbox("📋 Desempeño 1er Trimestre", ['Aprobado', 'Reprobado'])
            socio = st.selectbox("🏠 Nivel Socioeconómico", ['Bajo', 'Medio', 'Alto'])
            
        if st.button("🔮 Predecir"):
            try:
                inp = {'edad': edad, 'gpa': gpa, 'asistencia': assist, 'horas_estudio': horas,
                       'motivacion': motiv, 'primer_trimestre': trim, 'nivell_socioeconomic': socio}
                for f in prep['feature_names']:
                    if f not in inp: inp[f] = prep['medians'].get(f, 0) if f in prep['num_cols'] else prep['modes'].get(f, 'desconocido')
                    
                df_pred = pd.DataFrame([inp])
                for col in prep['cat_cols']:
                    le = prep['encoders'][col]
                    val = str(df_pred[col].values[0])
                    df_pred[col] = le.transform([val])[0] if val in le.classes_ else 0
                    
                X_s = prep['scaler'].transform(df_pred[prep['feature_names']])
                st.markdown("---")
                st.subheader("🎯 Resultados")
                cols = st.columns(4)
                for i, (n, r) in enumerate(st.session_state.results.items()):
                    with cols[i]:
                        p = r['model'].model.predict(X_s)[0]
                        prob = r['model'].model.predict_proba(X_s)[0][1] if hasattr(r['model'].model, 'predict_proba') else 0.5
                        risk = "🔴 ALTO RIESGO" if prob > 0.5 else "🟢 BAJO RIESGO"
                        st.metric(n.replace('_',' ').title(), risk, delta=f"Prob: {prob:.1%}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
