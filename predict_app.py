import os
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

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
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model, getattr(model, "classes_", None)

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Modelo não encontrado em '{model_path}' e dataset '{data_path}' inexistente."
        )

    # Leitura do dataset
    df = pd.read_csv(data_path)
    if "Obesity_level" in df.columns and "Obesity" not in df.columns:
        df = df.rename(columns={"Obesity_level": "Obesity"})

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

    # NÃO usar 'sparse' nem 'sparse_output' → compatível com várias versões do sklearn
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
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


@st.cache_data(show_spinner=False)
def load_dataset(path: str):
    df = pd.read_csv(path)
    if "Obesity_level" in df.columns and "Obesity" not in df.columns:
        df = df.rename(columns={"Obesity_level": "Obesity"})
    # cria BMI se possível
    if "Weight" in df.columns and "Height" in df.columns:
        df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    return df


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
tab_pred, tab_lote, tab_dash, tab_info = st.tabs([
    "🔍 Predição Individual",
    "📦 Predição em Lote",
    "📊 Dashboard Analítico",
    "ℹ️ Sobre o Modelo"
])

# ---------------------------------------------------
# TABELA 1 — PREDIÇÃO INDIVIDUAL
# ---------------------------------------------------
with tab_pred:
    st.subheader("Preencha os dados do paciente:")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=5, max_value=100, value=25)
        height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=250, value=70)
        family_history = st.selectbox("Family History", ["yes", "no"])
        favc = st.selectbox("FAVC (Alimentos calóricos?)", ["yes", "no"])
        fcvc = st.slider("FCVC (vegetais 1–3)", 1.0, 3.0, 2.0)
        ncp = st.slider("NCP (refeições principais por dia)", 1, 4, 3)

    with col2:
        caec = st.selectbox("CAEC (come entre refeições?)", ["no", "Sometimes", "Frequently", "Always"])
        smoke = st.selectbox("SMOKE", ["yes", "no"])
        ch2o = st.slider("CH2O (litros/dia)", 1.0, 3.0, 2.0)
        scc = st.selectbox("SCC (controla calorias?)", ["yes", "no"])
        faf = st.slider("FAF (atividade física 0–3)", 0.0, 3.0, 1.0)
        tue = st.slider("TUE (tempo em telas 0–2)", 0.0, 2.0, 1.0)
        calc = st.selectbox("CALC (álcool)", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox("MTRANS", ["Walking", "Bike", "Motorbike", "Public_Transportation", "Automobile"])

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
        st.caption("Este sistema é um apoio à decisão. O diagnóstico final é sempre clínico.")

# ---------------------------------------------------
# TABELA 2 — PREDIÇÃO EM LOTE
# ---------------------------------------------------
with tab_lote:
    st.subheader("Upload de CSV para previsão em lote")
    st.caption("O CSV deve conter as mesmas colunas do dataset original (exceto 'Obesity').")

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


# ---------------------------------------------------
# TABELA 3 — DASHBOARD ANALÍTICO
# ---------------------------------------------------
with tab_dash:
    st.subheader("📊 Visão Analítica — Estudo de Obesidade")

    try:
        df = load_dataset(DEFAULT_DATA_PATH)
    except Exception as e:
        st.error(f"Não foi possível carregar o dataset '{DEFAULT_DATA_PATH}': {e}")
        st.stop()

    # Métricas gerais
    c1, c2, c3 = st.columns(3)
    c1.metric("Total de registros", f"{len(df):,}")
    c2.metric("Níveis de obesidade", df["Obesity"].nunique())
    if "BMI" in df.columns:
        c3.metric("IMC médio (geral)", f"{df['BMI'].mean():.1f}")
    else:
        c3.metric("IMC médio (geral)", "N/A")

    st.markdown("### Distribuição de classes (Obesity)")
    class_counts = df["Obesity"].value_counts().sort_index()
    st.bar_chart(class_counts)

    # Perfil médio por classe
    st.markdown("### Perfil médio por nível de obesidade")
    cols_profile = [c for c in ["Age", "Weight", "Height", "BMI"] if c in df.columns]
    if cols_profile:
        prof = df.groupby("Obesity")[cols_profile].mean().round(1)
        st.dataframe(prof, use_container_width=True)
    else:
        st.info("Colunas de perfil (Age, Weight, Height, BMI) não disponíveis.")

    # Hábitos: FAVC, FCVC, FAF, CH2O
    st.markdown("### Hábitos e comportamentos por nível de obesidade")

    colA, colB = st.columns(2)

    with colA:
        if "FAVC" in df.columns:
            st.markdown("**Consumo frequente de alimentos calóricos (FAVC)**")
            favc_tab = pd.crosstab(df["FAVC"], df["Obesity"], normalize="index") * 100
            favc_tab = favc_tab.round(1)
            st.dataframe(favc_tab, use_container_width=True)

        if "FCVC" in df.columns:
            st.markdown("**Consumo de vegetais (FCVC)** — média por classe")
            fcvc_mean = df.groupby("Obesity")["FCVC"].mean().round(2)
            st.bar_chart(fcvc_mean)

    with colB:
        if "FAF" in df.columns:
            st.markdown("**Atividade física (FAF)** — média por classe")
            faf_mean = df.groupby("Obesity")["FAF"].mean().round(2)
            st.bar_chart(faf_mean)

        if "CH2O" in df.columns:
            st.markdown("**Consumo de água (CH2O)** — média por classe (litros/dia)")
            ch2o_mean = df.groupby("Obesity")["CH2O"].mean().round(2)
            st.bar_chart(ch2o_mean)

    # Insights textuais automáticos
    st.markdown("### 🧠 Insights automáticos para a equipe médica")

    bullets = []

    # % FAVC = yes em obesos vs normal
    if "FAVC" in df.columns:
        obese_mask = df["Obesity"].str.contains("Obesity", case=False, regex=False)
        favc_obese = df.loc[obese_mask, "FAVC"].astype(str).str.lower().eq("yes").mean() * 100
        if "Normal_Weight" in df["Obesity"].unique():
            favc_norm = df.loc[df["Obesity"] == "Normal_Weight", "FAVC"].astype(str).str.lower().eq("yes").mean() * 100
            bullets.append(
                f"- Entre os pacientes com **algum nível de obesidade**, ~**{favc_obese:.1f}%** relatam consumo frequente de alimentos muito calóricos (FAVC = yes), "
                f"contra **{favc_norm:.1f}%** entre pacientes com peso normal."
            )

    if all(c in df.columns for c in ["FCVC", "Obesity"]):
        fcvc_obese = df.loc[obese_mask, "FCVC"].mean()
        if "Normal_Weight" in df["Obesity"].unique():
            fcvc_norm = df.loc[df["Obesity"] == "Normal_Weight", "FCVC"].mean()
            bullets.append(
                f"- O **consumo de vegetais (FCVC)** tende a ser menor nos grupos com obesidade (média {fcvc_obese:.2f}) do que em pacientes com peso normal (média {fcvc_norm:.2f})."
            )

    if "FAF" in df.columns:
        faf_obese = df.loc[obese_mask, "FAF"].mean()
        if "Normal_Weight" in df["Obesity"].unique():
            faf_norm = df.loc[df["Obesity"] == "Normal_Weight", "FAF"].mean()
            bullets.append(
                f"- A **frequência de atividade física (FAF)** é mais baixa entre pacientes com obesidade (média {faf_obese:.2f}) em comparação com peso normal (média {faf_norm:.2f})."
            )

    if "CH2O" in df.columns:
        ch2o_obese = df.loc[obese_mask, "CH2O"].mean()
        if "Normal_Weight" in df["Obesity"].unique():
            ch2o_norm = df.loc[df["Obesity"] == "Normal_Weight", "CH2O"].mean()
            bullets.append(
                f"- Pacientes com obesidade tendem a ter **menor consumo de água (CH2O)** ({ch2o_obese:.2f} L/dia) do que pacientes com peso normal ({ch2o_norm:.2f} L/dia)."
            )

    if not bullets:
        st.write("- Não foi possível gerar insights automáticos com as colunas disponíveis.")
    else:
        for b in bullets:
            st.write(b)

    st.caption(
        "Os insights são baseados em associações observadas na base de dados e **não representam causalidade**. "
        "Devem sempre ser interpretados no contexto clínico."
    )


# ---------------------------------------------------
# TABELA 4 — SOBRE O MODELO
# ---------------------------------------------------
with tab_info:
    st.subheader("ℹ️ Informações sobre o modelo")
    st.markdown("""
    **Modelo:** RandomForestClassifier  
    **Acurácia de referência (offline):** ~93%  
    **Dataset:** Obesity.csv  
    **Objetivo:** Apoiar a equipe médica na avaliação do nível de obesidade a partir de fatores clínicos e comportamentais.  
    """)
    st.code(f"MODEL_PATH = {DEFAULT_MODEL_PATH}\nDATA_PATH = {DEFAULT_DATA_PATH}", language="bash")
