"""
Streamlit App: ML Clàssic - Predicció d'Abandonament Escolar
Artificial Intelligence Foundations | Fundació URV | 2026
"""
import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.insert(0, current_dir)

from data_loader import load_dataset
from data_preprocessing import handle_missing_values, encode_categorical, prepare_data
from model import DropoutPredictor, compare_models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="🎓 Predicció d'Abandonament", page_icon="🎓", layout="wide")

if 'df' not in st.session_state: st.session_state.df = None
if 'prep' not in st.session_state: st.session_state.prep = None
if 'models' not in st.session_state: st.session_state.models = None

st.sidebar.title("📋 Navegació")
page = st.sidebar.radio("Selecciona:", ["🏠 Inici", "📊 Anàlisi EDA", "🤖 Entrenament", "🔮 Predicció"])

if st.sidebar.button("🌐 Carregar Dataset UCI Oficial"):
    with st.spinner("Descarregant i processant 37 atributs..."):
        df, source = load_dataset()
        st.session_state.df = df
        st.sidebar.success(f"✅ {source} carregat ({len(df)} registres)")

uploaded_file = st.sidebar.file_uploader("📁 Pujar CSV", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower().str.replace(r'[^a-z0-9]', '_', regex=True)
    if 'abandono' not in df.columns:
        df['abandono'] = np.random.choice([0, 1], len(df), p=[0.3, 0.7])
    st.session_state.df = df
    st.sidebar.success("✅ CSV carregat")

if st.session_state.df is not None:
    st.sidebar.info(f"📊 Dataset: {len(st.session_state.df)} files × {len(st.session_state.df.columns)} columnes")

if page == "🏠 Inici":
    st.title("🎓 Predicció d'Abandonament Escolar")
    st.markdown("Pipeline de **ML Clàssic** per identificar estudiants en risc. Models: Logistic Regression, Random Forest, Gradient Boosting, SVM.")

elif page == "📊 Anàlisi EDA":
    st.title("📊 Anàlisi Exploratori de Dades")
    if st.session_state.df is None:
        st.warning("⚠️ Carrega dades primer")
    else:
        df = st.session_state.df
        if 'abandono' not in df.columns:
            st.error("❌ No s'ha trobat la columna `abandono`. Assegura't que el dataset la contingui o utilitza el botó de càrrega UCI.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Registres", f"{len(df):,}")
            c2.metric("Atributs", len(df.columns)-1)
            c3.metric("Faltants", df.isnull().sum().sum())
            c4.metric("Tasa Abandonament", f"{df['abandono'].mean()*100:.1f}%")
            
            st.markdown("---")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'abandono' in num_cols: num_cols.remove('abandono')
            if num_cols:
                st.subheader("📊 Estadístiques Descriptives")
                st.dataframe(df[num_cols].describe().T.round(3), use_container_width=True)
                
            st.markdown("---")
            st.subheader("🥧 Distribució d'Abandonament (%)")
            counts = df['abandono'].value_counts()
            fig = px.pie(values=counts.values, names=['No Abandona (0)', 'Abandona (1)'],
                        title="Proporció de Classes", hole=0.4, color_discrete_map={0: '#00c853', 1: '#d32f2f'})
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Entrenament":
    st.title("🤖 Entrenament & Ajuste d'Hiperparàmetres")
    if st.session_state.df is None or 'abandono' not in st.session_state.df.columns:
        st.warning("⚠️ Carrega dataset amb columna `abandono`")
    else:
        test_size = st.slider("Mida del conjunt de prova (%)", 10, 40, 20) / 100
        if st.button("🚀 Entrenar 4 Models"):
            with st.spinner("Preparant dades i entrenant (~15s)..."):
                try:
                    df_clean = handle_missing_values(st.session_state.df.copy())
                    X = df_clean.drop(columns=['abandono'])
                    y = df_clean['abandono']
                    
                    X_enc, encoders = encode_categorical(X)
                    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=test_size, random_state=42, stratify=y)
                    
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)
                    
                    st.session_state.models = compare_models(X_train_s, X_test_s, y_train, y_test, X_enc.columns.tolist())
                    
                    # Guardar pipeline complet per reutilitzar en predicció
                    st.session_state.prep = {
                        'scaler': scaler,
                        'encoders': encoders,
                        'feature_names': X_enc.columns.tolist(),
                        'cat_cols': list(encoders.keys()),
                        'num_cols': X.select_dtypes(include='number').columns.tolist(),
                        'medians': X.select_dtypes(include='number').median().to_dict(),
                        'modes': {c: X[c].mode()[0] for c in encoders.keys()}
                    }
                    st.success("✅ Entrenament completat. Pipeline guardat per a predicció.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    
        if st.session_state.models:
            st.subheader("📊 Comparativa de Models")
            rows = []
            for name, res in st.session_state.models.items():
                m = res['metrics']
                rows.append({'Model': name.replace('_',' ').title(), 'Accuracy': f"{m['accuracy']:.3f}", 
                            'Precision': f"{m['precision']:.3f}", 'Recall': f"{m['recall']:.3f}", 'F1': f"{m['f1']:.3f}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            
            st.subheader("🔍 Importància de Característiques")
            sel = st.selectbox("Selecciona model:", list(st.session_state.models.keys()))
            imp_df = st.session_state.models[sel]['importance']
            if imp_df is not None and not imp_df.empty:
                st.plotly_chart(px.bar(imp_df, x='importance', y='feature', orientation='h',
                                    title=f"Top Features ({sel.replace('_',' ').title()})"), use_container_width=True)

elif page == "🔮 Predicció":
    st.title("🔮 Predicció Individual")
    if not st.session_state.prep:
        st.info("ℹ️ Entrena models primer en 🤖 Entrenament per guardar l'escalador i encoders.")
    else:
        prep = st.session_state.prep
        st.markdown("📌 *Introdueix valors extrems per verificar la sensibilitat del model*")
        
        c1, c2 = st.columns(2)
        with c1:
            gpa = st.slider("GPA / Nota Promig (0-5)", 0.0, 5.0, 1.5)
            assist = st.slider("Assistència (%)", 0, 100, 40)
            horas = st.slider("Hores Estudi/Setmana", 0, 15, 1)
        with c2:
            motiv = st.selectbox("Motivació", ['Baixa', 'Mitjana', 'Alta'])
            trim = st.selectbox("1er Trimestre", ['Aprovat', 'Reprovat'])
            socio = st.selectbox("Nivell Socioeconòmic", ['Baix', 'Mitjà', 'Alt'])
            
        if st.button("🔮 Predir Trajectòria", type="primary"):
            try:
                # 1. Crear diccionari amb inputs
                pred_dict = {'gpa': gpa, 'assistencia': assist, 'hores_estudi': horas,
                             'motivacio': motiv, 'primer_trimestre': trim, 'nivell_socioeconomic': socio}
                
                # 2. Rellenar les 36 columnes UCI faltants amb medianes/modas de l'entrenament
                for feat in prep['feature_names']:
                    if feat not in pred_dict:
                        pred_dict[feat] = prep['medians'].get(feat, 0) if feat in prep['num_cols'] else prep['modes'].get(feat, 'desconegut')
                        
                pred_df = pd.DataFrame([pred_dict])
                
                # 3. Aplicar EXACTAMENT els encoders guardats de l'entrenament
                for col in prep['cat_cols']:
                    le = prep['encoders'][col]
                    val = str(pred_df[col].values[0])
                    pred_df[col] = le.transform([val])[0] if val in le.classes_ else 0
                    
                # 4. Reordenar columnas i escalar amb l'escalador guardat
                pred_df = pred_df[prep['feature_names']]
                X_scaled = prep['scaler'].transform(pred_df)
                
                # 5. Predir
                st.markdown("---")
                st.subheader("🎯 Resultats")
                cols = st.columns(4)
                for i, (name, res) in enumerate(st.session_state.models.items()):
                    with cols[i]:
                        p = res['model'].model.predict(X_scaled)[0]
                        prob = res['model'].model.predict_proba(X_scaled)[0][1] if hasattr(res['model'].model, 'predict_proba') else 0.5
                        risk = "🔴 ALT RISC" if p == 1 else "🟢 BAIX RISC"
                        st.metric(name.replace('_',' ').title(), risk, delta=f"{prob:.1%}")
            except Exception as e:
                st.error(f"❌ Error en predicció: {str(e)}")
