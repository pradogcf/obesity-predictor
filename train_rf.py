#!/usr/bin/env python3
# Treina RandomForest e salva model.pkl para o app do Streamlit

import os, json
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Caminho do CSV (estamos assumindo que Obesity.csv está na mesma pasta do script)
DATA_PATH = "Obesity.csv"
OUT_DIR   = Path("artifacts_rf")
OUT_DIR.mkdir(exist_ok=True, parents=True)

MODEL_PKL = OUT_DIR / "model.pkl"
METRICS   = OUT_DIR / "metrics.json"

print(f"Lendo dataset em: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Alvo e features
y = df["Obesity"].astype(str)
X = df.drop(columns=["Obesity"])

# Identifica colunas categóricas e numéricas
cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_cols = X.select_dtypes(include=["number"]).columns.tolist()

# Pipelines de pré-processamento
num_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

# Modelo RandomForest (config que você já validou: ~93,6% acc)
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=20,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

pipe = Pipeline([
    ("pre", pre),
    ("clf", rf)
])

print("Treinando modelo RandomForest...")
pipe.fit(X_train, y_train)

# Avaliação
y_pred = pipe.predict(X_test)
proba  = pipe.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)

classes_sorted = sorted(y.unique().tolist())
y_bin = label_binarize(y_test, classes=classes_sorted)
auc_macro = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")

cm = confusion_matrix(y_test, y_pred)
rep = classification_report(y_test, y_pred)

print("\n===== MÉTRICAS =====")
print(f"Acurácia (teste): {acc:.4f}")
print(f"ROC AUC macro:   {auc_macro:.4f}")
print("\nMatriz de confusão:\n", cm)
print("\nRelatório de classificação:\n", rep)

# Salvar modelo e métricas
joblib.dump(pipe, MODEL_PKL)
with open(METRICS, "w") as f:
    json.dump({
        "accuracy_test": float(acc),
        "roc_auc_macro": float(auc_macro),
        "classes": classes_sorted,
        "confusion_matrix": cm.tolist()
    }, f, indent=2)

print(f"\n✅ Modelo salvo em: {MODEL_PKL}")
print(f"📄 Métricas salvas em: {METRICS}")
