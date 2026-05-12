"""
Streamlit App: ML Clásico - Predicción de Abandono Escolar
"""
import sys, os, streamlit as st, pandas as pd, numpy as np, plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_preprocessing import handle_missing_values, encode_categorical
from model import DropoutPredictor, compare_models
import warnings; warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.insert(0, current_dir)

st.set_page_config(page_title="🎓 Predicción de Abandono Escolar", page_icon="🎓", layout="wide")

if 'df' not in st.session_state: st.session_state.df = None
if 'prep' not in st.session_state: st.session_state.prep = None
if 'results' not in st.session_state: st.session_state.results = None

def create_demo_data():
    np.random.seed(42)
    return pd.DataFrame({
        'Edad': np.random.randint(17, 35, 500), 'GPA': np.random.uniform(1.5, 5.0, 500),
        'Asistencia': np.random.uniform(40, 100, 500), 'Horas_Estudio': np.random.uniform(0, 15, 500),
        'Socioeconomico': np.random.choice(['Bajo', 'Medio', 'Alto'], 500),
        'Primer_Trimestre': np.random.choice(['Aprobado', 'Reprobado'], 500),
        'Motivacion': np.random.choice(['Baja', 'Media', 'Alta'], 500),
        'Abandono': (np.random.uniform(0, 1, 500) > 0.65).astype(int)
    })

# SIDEBAR
st.sidebar.title("📋 Navegación")
page = st.sidebar.radio("Selecciona:", ["🏠 Inicio", "📊 Análisis EDA", "🤖 Entrenamiento", "🔮 Predicción"])
st.sidebar.markdown("---")
if st.sidebar.button("🌐 Cargar Dataset UCI"):
    with st.spinner("Descargando..."):
        from data_loader import load_dataset, preprocess_dataset
        df, src = load_dataset()
        st.session_state.df = preprocess_dataset(df)
        if 'abandono' not in st.session_state.df.columns: st.session_state.df['Abandono'] = 0
        st.sidebar.success(f"✅ {src} cargado")
if st.sidebar.button("🎲 Datos Demo"):
    st.session_state.df = create_demo_data()
    st.sidebar.success("✅ Demo cargado")

# PÁGINAS
if page == "🏠 Inicio":
    st.title("🎓 ML Clásico - Predicción de Abandono Escolar")
    st.info("Selecciona una sección en la barra lateral para comenzar.")

elif page == "📊 Análisis EDA":
    st.title("📊 Análisis Exploratorio de Datos")
    if st.session_state.df is None: st.warning("⚠️ Carga datos primero")
    else:
        df = st.session_state.df.copy()
        if 'abandono' in df.columns: df.rename(columns={'abandono': 'Abandono'}, inplace=True)
        
        # ✅ ESTADÍSTICAS DESCRIPTIVAS EXPLÍCITAS
        st.subheader("📊 Estadísticas Descriptivas (Variables Clave)")
        key_cols = ['Edad', 'GPA', 'Asistencia', 'Horas_Estudio', 'Abandono']
        valid_cols = [c for c in key_cols if c in df.columns]
        st.dataframe(df[valid_cols].describe().T.round(3), use_container_width=True)
        
        st.markdown("---")
        st.subheader("🥧 Distribución de Abandonos (%)")
        if 'Abandono' in df.columns:
            counts = df['Abandono'].value_counts()
            fig_pie = px.pie(values=counts.values, names=['No Abandona (0)', 'Abandona (1)'],
                            title="Proporción de Clases", hole=0.4, color_discrete_map={0:'#00c853', 1:'#d32f2f'})
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
        st.subheader("🔗 Matriz de Correlación (Pearson)")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if 'Abandono' in num_cols: num_cols.remove('Abandono')
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            st.plotly_chart(px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', title="Correlaciones"), use_container_width=True)

elif page == "🤖 Entrenamiento":
    st.title("🤖 Entrenamiento de Modelos")
    if st.session_state.df is None or 'Abandono' not in st.session_state.df.columns:
        st.warning("⚠️ Carga dataset con columna `Abandono`")
    else:
        test_pct = st.slider("Tamaño Test (%)", 10, 40, 20)/100
        if st.button("🚀 Entrenar 4 Modelos"):
            with st.spinner("Entrenando..."):
                df_c = handle_missing_values(st.session_state.df.copy())
                X, y = df_c.drop(columns=['Abandono']), df_c['Abandono']
                X_enc, enc = encode_categorical(X)
                X_tr, X_te, y_tr, y_te = train_test_split(X_enc, y, test_size=test_pct, random_state=42, stratify=y)
                sc = StandardScaler()
                X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)
                
                st.session_state.results = compare_models(X_tr_s, X_te_s, y_tr, y_te, X_enc.columns.tolist())
                st.session_state.prep = {'scaler': sc, 'encoders': enc, 'feature_names': X_enc.columns.tolist(),
                                         'cat_cols': list(enc.keys()), 'num_cols': X.select_dtypes('number').columns.tolist(),
                                         'medians': X.select_dtypes('number').median().to_dict(),
                                         'modes': {c: X[c].mode()[0] for c in enc.keys()}}
                st.success("✅ Entrenamiento completado")
                
        if st.session_state.results:
            st.subheader("📊 Comparativa de Modelos")
            rows = []
            for n, r in st.session_state.results.items():
                m = r['metrics']
                rows.append({'Modelo': n.replace('_',' ').title(), 'Accuracy': m['accuracy'], 'F1-Score': m['f1'], 'ROC-AUC': m['roc_auc']})
            plot_df = pd.DataFrame(rows).melt(id_vars='Modelo', var_name='Métrica', value_name='Valor')
            st.plotly_chart(px.bar(plot_df, x='Modelo', y='Valor', color='Métrica', barmode='group', title="Rendimiento por Modelo"), use_container_width=True)
            
            st.subheader("🔍 Importancia de Características")
            sel = st.selectbox("Modelo:", list(st.session_state.results.keys()))
            imp = st.session_state.results[sel]['importance']
            if imp is not None and not imp.empty:
                imp['feature'] = imp['feature'].str.replace('_', ' ').str.title()
                st.plotly_chart(px.bar(imp, x='importance', y='feature', orientation='h', title=f"Top Features ({sel.replace('_',' ').title()})"), use_container_width=True)

elif page == "🔮 Predicción":
    st.title("🔮 Predicción Individual")
    if not st.session_state.prep: st.info("ℹ️ Entrena primero en 🤖 Entrenamiento")
    else:
        prep = st.session_state.prep
        c1, c2 = st.columns(2)
        with c1:
            gpa = st.slider("GPA (0-5)", 0.0, 5.0, 2.0)
            assist = st.slider("Asistencia (%)", 0, 100, 50)
            horas = st.slider("Horas Estudio/Semana", 0, 20, 3)
        with c2:
            motiv = st.selectbox("Motivación", ['Baja', 'Media', 'Alta'])
            socio = st.selectbox("Nivel Socioeconómico", ['Bajo', 'Medio', 'Alto'])
            
        if st.button("🔮 Predecir"):
            try:
                inp = {'gpa': gpa, 'asistencia': assist, 'horas_estudio': horas, 'motivacion': motiv, 'socioeconomico': socio}
                for f in prep['feature_names']:
                    if f not in inp: inp[f] = prep['medians'].get(f, 0) if f in prep['num_cols'] else prep['modes'].get(f, 'desconocido')
                df_pred = pd.DataFrame([inp])
                for col in prep['cat_cols']:
                    le = prep['encoders'][col]
                    val = str(df_pred[col].values[0])
                    df_pred[col] = le.transform([val])[0] if val in le.classes_ else 0
                df_pred = df_pred[prep['feature_names']]
                X_s = prep['scaler'].transform(df_pred)
                
                st.markdown("---")
                st.subheader("🎯 Resultados")
                cols = st.columns(4)
                for i, (n, r) in enumerate(st.session_state.results.items()):
                    with cols[i]:
                        p = r['model'].predict(X_s)[0]
                        prob = r['model'].predict_proba(X_s)[0][1] if hasattr(r['model'].model, 'predict_proba') else 0.5
                        risk = "🔴 ALTO RIESGO" if prob > 0.5 else "🟢 BAJO RIESGO"
                        st.metric(n.replace('_',' ').title(), risk, delta=f"Probabilidad: {prob:.1%}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
