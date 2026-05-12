"""
Streamlit App: ML Clásico - Predicción de Abandono Escolar
"""
import sys, os, streamlit as st, pandas as pd, numpy as np, plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.insert(0, current_dir)

from data_loader import load_dataset, preprocess_dataset
from data_preprocessing import handle_missing_values, encode_categorical
from model import DropoutPredictor, compare_models

st.set_page_config(page_title="🎓 Predicción de Abandono Escolar", page_icon="🎓", layout="wide")

# Estado de sesión
for k in ['df', 'prep', 'results']: 
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
        st.sidebar.success(f"✅ {src} cargado")

if st.sidebar.button("🎲 Datos Demo"):
    from data_loader import create_sample_dataset
    st.session_state.df = create_sample_dataset()
    st.sidebar.success("✅ Demo cargado")

if st.sidebar.file_uploader("📁 Subir CSV", type=['csv'], key='uploader') is not None:
    st.session_state.df = pd.read_csv(st.session_state['uploader'])
    st.sidebar.success("✅ CSV cargado")

if st.session_state.df is not None:
    st.sidebar.info(f"📊 Dataset: {len(st.session_state.df)} filas × {len(st.session_state.df.columns)} columnas")

# ==================== PÁGINAS ====================
if page == "🏠 Inicio":
    st.title("🎓 ML Clásico - Predicción de Abandono Escolar")
    st.info("Selecciona una sección en la barra lateral para comenzar.")

elif page == "📊 Análisis EDA":
    st.title("📊 Análisis Exploratorio de Datos")
    if st.session_state.df is None: st.warning("⚠️ Carga datos primero")
    else:
        df = st.session_state.df.copy()
        if 'abandono' not in df.columns: df['abandono'] = np.random.choice([0,1], len(df))
        target = 'abandono'
        
        # ✅ PESTAÑA VISTA PREVIA
        st.subheader("📋 Vista Previa de Datos")
        st.dataframe(df.head(10), use_container_width=True)
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Registros", f"{len(df):,}"); k2.metric("Features", len(df.columns))
        k3.metric("Faltantes", df.isnull().sum().sum()); k4.metric("Tasa Abandono", f"{df[target].mean()*100:.1f}%")
        
        st.subheader("📊 Estadísticas Descriptivas")
        num_cols = df.select_dtypes('number').columns.tolist()
        if target in num_cols: num_cols.remove(target)
        st.dataframe(df[num_cols].describe().T.round(3), use_container_width=True)
        
        st.subheader("📈 Visualizaciones")
        sel = st.selectbox("Selecciona variable:", [c for c in num_cols if c in df.columns])
        if sel:
            st.plotly_chart(px.box(df, x=target, y=sel, color=target, color_discrete_map={0:'#00c853',1:'#d32f2f'}), use_container_width=True)
        st.plotly_chart(px.pie(values=df[target].value_counts().values, names=['No Abandona','Abandona'], hole=0.4), use_container_width=True)
        if len(num_cols)>=2: st.plotly_chart(px.imshow(df[num_cols].corr(), text_auto='.2f'), use_container_width=True)

elif page == "🤖 Entrenamiento":
    st.title("🤖 Entrenamiento de Modelos")
    if st.session_state.df is None or 'abandono' not in st.session_state.df.columns:
        st.warning("⚠️ Carga dataset con columna `abandono`")
    else:
        test_pct = st.slider("Tamaño Test (%)", 10, 40, 20)/100
        n_train, n_test = int(len(st.session_state.df)*(1-test_pct)), len(st.session_state.df) - int(len(st.session_state.df)*(1-test_pct))
        st.info(f"📋 **División:** ✅ Train: {n_train} | 🔍 Test: {n_test}")
        
        if st.button("🚀 Entrenar 4 Modelos"):
            with st.spinner("Entrenando..."):
                df_c = handle_missing_values(st.session_state.df.copy())
                X, y = df_c.drop(columns=['abandono']), df_c['abandono']
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
            st.subheader("📊 Resultados de Modelos")
            rows = []
            for n, r in st.session_state.results.items():
                m = r['metrics']
                rows.append({'Modelo': n.replace('_',' ').title(), 'Accuracy': m['accuracy'], 'F1-Score': m['f1'], 'ROC-AUC': m['roc_auc']})
            
            st.dataframe(pd.DataFrame(rows).style.format({'Accuracy': '{:.3f}', 'F1-Score': '{:.3f}', 'ROC-AUC': '{:.3f}'}), use_container_width=True)
            
            # ✅ GRÁFICO COMPARATIVO EN COLUMNAS
            plot_df = pd.DataFrame(rows).melt(id_vars='Modelo', var_name='Métrica', value_name='Valor')
            st.plotly_chart(px.bar(plot_df, x='Modelo', y='Valor', color='Métrica', barmode='group', title="Comparación de Rendimiento"), use_container_width=True)
            
            # ✅ IMPORTANCIA DE CARACTERÍSTICAS (CORREGIDO)
            st.subheader("🔍 Importancia de Características")
            sel = st.selectbox("Selecciona modelo:", list(st.session_state.results.keys()))
            imp = st.session_state.results[sel]['importance']
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
            horas = st.slider("Horas Estudio/Semana", 0, 20, 5)
        with c2:
            motiv = st.selectbox("Motivación", ['Baja', 'Media', 'Alta'])
            trim = st.selectbox("1er Trimestre", ['Aprobado', 'Reprobado'])
            socio = st.selectbox("Nivel Socioeconómico", ['Bajo', 'Medio', 'Alto'])
            
        if st.button("🔮 Predecir"):
            try:
                # 1. Crear input
                inp = {'edad': edad, 'gpa': gpa, 'asistencia': assist, 'horas_estudio': horas,
                       'motivacion': motiv, 'primer_trimestre': trim, 'socioeconomico': socio}
                
                # ✅ FIX 7 vs 8 FEATURES: Rellenar exactamente con feature_names del entrenamiento
                df_pred = pd.DataFrame([inp])
                for f in prep['feature_names']:
                    if f not in df_pred.columns:
                        df_pred[f] = prep['medians'].get(f, 0) if f in prep['num_cols'] else 0
                        
                # 2. Codificar con encoders guardados
                for col in prep['cat_cols']:
                    le = prep['encoders'][col]
                    val = str(df_pred[col].values[0])
                    df_pred[col] = le.transform([val])[0] if val in le.classes_ else 0
                    
                # 3. Reordenar y escalar
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
                        st.metric(n.replace('_',' ').title(), risk, delta=f"Prob: {prob:.1%}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
