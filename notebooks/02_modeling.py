"""
PASO 2 — Modelado: Clasificación de Sentimiento
===================================================
Compara dos enfoques de NLP:
  A) Clásico: TF-IDF + Regresión Logística (rápido, interpretable)
  B) Moderno: DistilBERT pre-entrenado vía Hugging Face Transformers
     (entiende contexto y orden de palabras, no solo frecuencia)

Objetivo: mostrar cuánto mejora (o no) usar un modelo de lenguaje
pre-entrenado frente a un enfoque clásico de bolsa de palabras.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUTS_DIR = SCRIPT_DIR.parent / "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ---------------------------------------------------------------
# 1. Cargar datos limpios (generados por 01_eda.py)
# ---------------------------------------------------------------
df = pd.read_csv(DATA_DIR / "imdb_reviews_clean.csv")
df["label"] = (df["sentiment"] == "positive").astype(int)

# Para que el proyecto corra rápido en cualquier computador (incluyendo
# el modelo transformer sobre CPU), usamos una muestra de 4,000 reseñas
# para el set de TEST donde comparamos ambos modelos cara a cara.
# El baseline clásico SÍ se entrena con el dataset completo (es barato).
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    df["review_clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

print(f"Train (baseline clásico): {len(X_train_full)} reseñas")
print(f"Test: {len(X_test_full)} reseñas")

# Muestra más pequeña de test para evaluar el transformer (CPU es lento)
X_test_sample = X_test_full.sample(n=min(500, len(X_test_full)), random_state=42)
y_test_sample = y_test_full.loc[X_test_sample.index]
print(f"Muestra de test para el transformer (CPU): {len(X_test_sample)} reseñas")


def evaluar(y_true, y_pred, nombre):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    print(f"\n{'=' * 55}\n{nombre}\n{'=' * 55}")
    print(f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
    return {"modelo": nombre, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

resultados = []

# ---------------------------------------------------------------
# 2. Modelo A — TF-IDF + Regresión Logística (enfoque clásico)
# ---------------------------------------------------------------
print("\nEntrenando modelo clásico (TF-IDF + Regresión Logística)...")
t0 = time.time()

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=5)
X_train_tfidf = vectorizer.fit_transform(X_train_full)
X_test_tfidf_full = vectorizer.transform(X_test_full)
X_test_tfidf_sample = vectorizer.transform(X_test_sample)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_tfidf, y_train_full)

tiempo_entrenamiento_clasico = time.time() - t0

# Evaluación sobre el test COMPLETO (es barato, no hay razón para muestrear)
pred_full = clf.predict(X_test_tfidf_full)
resultados.append(evaluar(y_test_full, pred_full, "TF-IDF + Regresión Logística (test completo, 10,000 reseñas)"))

# También evaluamos sobre la MISMA muestra pequeña que usará el transformer,
# para que la comparación cabeza a cabeza sea justa (mismos ejemplos)
pred_sample = clf.predict(X_test_tfidf_sample)
resultado_clasico_muestra = evaluar(y_test_sample, pred_sample, "TF-IDF + Regresión Logística (muestra de 500, para comparar con transformer)")
resultados.append(resultado_clasico_muestra)

print(f"\nTiempo de entrenamiento (clásico): {tiempo_entrenamiento_clasico:.1f}s")

# ---------------------------------------------------------------
# 3. Modelo B — DistilBERT pre-entrenado (Transformers / Hugging Face)
# ---------------------------------------------------------------
print("\nCargando modelo DistilBERT pre-entrenado (esto puede tardar la primera vez)...")
from transformers import pipeline

t0 = time.time()
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=512,
)
tiempo_carga_modelo = time.time() - t0
print(f"Modelo cargado en {tiempo_carga_modelo:.1f}s")

print(f"\nPrediciendo sobre {len(X_test_sample)} reseñas con DistilBERT (CPU, puede tardar unos minutos)...")
t0 = time.time()

textos = X_test_sample.tolist()
preds_transformer = []
BATCH_SIZE = 16
for i in range(0, len(textos), BATCH_SIZE):
    batch = textos[i:i + BATCH_SIZE]
    outputs = sentiment_pipeline(batch)
    preds_transformer.extend([1 if o["label"] == "POSITIVE" else 0 for o in outputs])
    if (i // BATCH_SIZE) % 5 == 0:
        print(f"  Procesadas {i + len(batch)}/{len(textos)} reseñas...")

tiempo_inferencia_transformer = time.time() - t0
print(f"Inferencia completa en {tiempo_inferencia_transformer:.1f}s "
      f"({tiempo_inferencia_transformer / len(textos):.3f}s por reseña)")

resultado_transformer = evaluar(y_test_sample, preds_transformer, "DistilBERT pre-entrenado (misma muestra de 500)")
resultados.append(resultado_transformer)

# ---------------------------------------------------------------
# 4. Comparación de errores: ¿en qué casos difieren los dos modelos?
# ---------------------------------------------------------------
comparacion = pd.DataFrame({
    "review": X_test_sample.values,
    "real": y_test_sample.values,
    "pred_clasico": pred_sample,
    "pred_transformer": preds_transformer,
})
comparacion["clasico_correcto"] = comparacion["real"] == comparacion["pred_clasico"]
comparacion["transformer_correcto"] = comparacion["real"] == comparacion["pred_transformer"]

# Casos donde el transformer acierta y el clásico falla (y viceversa)
transformer_gana = comparacion[(comparacion["transformer_correcto"]) & (~comparacion["clasico_correcto"])]
clasico_gana = comparacion[(comparacion["clasico_correcto"]) & (~comparacion["transformer_correcto"])]

print(f"\nCasos donde el transformer acierta y el clásico falla: {len(transformer_gana)}")
print(f"Casos donde el clásico acierta y el transformer falla: {len(clasico_gana)}")

if len(transformer_gana) > 0:
    print("\nEjemplo donde el transformer entendió el contexto mejor:")
    ej = transformer_gana.iloc[0]
    print(f"  Reseña: {ej['review'][:200]}...")
    print(f"  Real: {'positive' if ej['real']==1 else 'negative'} | "
          f"Clásico predijo: {'positive' if ej['pred_clasico']==1 else 'negative'} | "
          f"Transformer predijo: {'positive' if ej['pred_transformer']==1 else 'negative'}")

comparacion.to_csv(OUTPUTS_DIR / "comparacion_errores.csv", index=False)

# ---------------------------------------------------------------
# 5. Gráfica comparativa
# ---------------------------------------------------------------
resultados_df = pd.DataFrame(resultados)
# Solo comparamos en la misma muestra de 500 para que sea justo
comparables = resultados_df[resultados_df["modelo"].str.contains("muestra")]

fig, ax = plt.subplots(figsize=(8, 5))
metrics = ["accuracy", "precision", "recall", "f1"]
x = np.arange(len(metrics))
width = 0.35

for i, (_, row) in enumerate(comparables.iterrows()):
    valores = [row[m] for m in metrics]
    label = "TF-IDF + LogReg" if "Regresión" in row["modelo"] else "DistilBERT"
    ax.bar(x + i * width, valores, width, label=label)

ax.set_xticks(x + width / 2)
ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
ax.set_ylim(0, 1)
ax.set_title("Comparación: TF-IDF clásico vs. DistilBERT (misma muestra de 500 reseñas)")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "04_comparacion_modelos.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 6. Resumen final
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(resultados_df.to_string(index=False))
resultados_df.to_csv(OUTPUTS_DIR / "resultados_modelos.csv", index=False)

print(f"\n Modelado completo. Gráficas y resultados guardados en {OUTPUTS_DIR}/")
