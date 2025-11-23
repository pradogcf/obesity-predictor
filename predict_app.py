import os
import pandas as pd
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------
# CONFIGURAÇÃO DO APLICATIVO
# ---------------------------------------------------
st.set_page_config(page_title="Previsor de Obesidade", page_icon="🩺", layout="wide")
st.title("🩺 Sistema Preditivo de Obesidade")

# Caminhos padrão
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "artifacts_rf/model.pkl")
DEFAULT_DATA_PATH = os.getenv("DATA_PATH", "Obesity.csv")

# ---------------------------------------------------
# FUNÇÕES
# ---------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_or_train_model(model_path: str, data_path: str):
    """
    Carrega o model.pkl. Se não existir, treina automaticamente usando o Obesity.csv.
    """
    # 1) Se já existe model.pkl, só carrega
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model, getattr(model, "classes_", None)

    # 2) Se não tem model.pkl, tenta treinar a partir do CSV (fallback)
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Modelo não encontrado em '{model_path}' e dataset '{data_path}' inexistente."
        )

    # Leitura do dataset
    df = pd.read_csv(data_path)
    if "Obesity" not in df.columns:
        raise ValueError("A coluna alvo 'Obesity' não foi encontrada no dataset.")

    y = df["Obesity"].astype(str)
    X = df.drop(columns=["Obesity"])

    # Colunas categóricas e numéricas
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    # Pré-processamento
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # ⚠️ AQUI ESTÁ A CORREÇÃO IMPORTANTE: sparse_output=False
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    # Modelo RandomForest ajustado para forte desempenho
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline([("pre", pre), ("clf", rf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    pipe.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipe, model_path)

    return pipe, getattr(pipe, "classes_", None)


def predict_input(model, form_dict: dict):
    df_input = pd.DataFrame([form_dict])
    proba = model.predict_proba(df_input)[0]
    pred = model.predict(df_input)[0]

    table = pd.DataFrame({
        "Classe": model.classes_,
        "Probabilidade": proba
    }).sort_values("Probabilidade", ascending=False)

    return pred, table


# ---------------------------------------------------
# CARREGAMENTO DO MODELO
# ---------------------------------------------------
try:
    model, classes_ = load_or_train_model(DEFAULT_MODEL_PATH, DEFAULT_DATA_PATH)
    st.success("Modelo carregado com sucesso.")
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")
    st.stop()


# ---------------------------------------------------
# INTERFACE — TABS
# ---------------------------------------------------
tab_pred, tab_lote, tab_info = st.tabs(
    ["🔍 Predição Individual", "📦 Predição em Lote", "ℹ️ Sobre o Modelo"]
)

with tab_pred:
    st.subheader("Preencha os dados do paciente:")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=5, max_value=100, value=25)
        height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=250, value=70)
        family_history = st.selectbox("family_history (Histórico familiar de obesidade?)", ["yes", "no"])
        favc = st.selectbox("FAVC (Alimentos muito calóricos?)", ["yes", "no"])
        fcvc = st.slider("FCVC (vegetais 1–3)", 1.0, 3.0, 2.0)
        ncp = st.slider("NCP (refeições principais/dia)", 1, 4, 3)

    with col2:
        caec = st.selectbox("CAEC (come entre refeições?)", ["no", "Sometimes", "Frequently", "Always"])
        smoke = st.selectbox("SMOKE (fuma?)", ["yes", "no"])
        ch2o = st.slider("CH2O (litros de água/dia)", 1.0, 3.0, 2.0)
        scc = st.selectbox("SCC (controla calorias?)", ["yes", "no"])
        faf = st.slider("FAF (atividade física 0–3)", 0.0, 3.0, 1.0)
        tue = st.slider("TUE (tempo em telas 0–2)", 0.0, 2.0, 1.0)
        calc = st.selectbox("CALC (consumo de álcool)", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox("MTRANS (transporte)", ["Walking", "Bike", "Motorbike", "Public_Transportation", "Automobile"])

    if st.button("🔍 Prever"):
        form = {
            "Gender": gender,
            "Age": age,
            "Height": height,
            "Weight": weight,
            "family_history": family_history,
            "FAVC": favc,
            "FCVC": fcvc,
            "NCP": ncp,
            "CAEC": caec,
            "SMOKE": smoke,
            "CH2O": ch2o,
            "SCC": scc,
            "FAF": faf,
            "TUE": tue,
            "CALC": calc,
            "MTRANS": mtrans,
        }

        pred, table = predict_input(model, form)

        st.success(f"### Resultado previsto: **{pred}**")
        st.write("### Probabilidades por classe:")
        st.dataframe(table.reset_index(drop=True), use_container_width=True)

with tab_lote:
    st.subheader("Upload de CSV para previsão em lote")

    file = st.file_uploader("Escolha um arquivo .csv", type=["csv"])

    if file:
        try:
            df_up = pd.read_csv(file)
            if "Obesity" in df_up.columns:
                df_up = df_up.drop(columns=["Obesity"])

            preds = model.predict(df_up)
            out = df_up.copy()
            out["predicted_Obesity"] = preds

            st.success("Previsão em lote concluída!")
            st.dataframe(out.head(50), use_container_width=True)

            st.download_button(
                "⬇️ Baixar CSV com previsões",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Erro ao processar CSV: {e}")

with tab_info:
    st.subheader("ℹ️ Informações sobre o modelo")
    st.markdown("""
    **Modelo:** RandomForestClassifier  
    **Acurácia esperada:** ~93%  
    **Dataset:** Obesity.csv  
    **Objetivo:** Apoiar a equipe médica na avaliação do nível de obesidade.  
    """)

    st.code(f"MODEL_PATH = {DEFAULT_MODEL_PATH}\nDATA_PATH = {DEFAULT_DATA_PATH}", language="bash")


