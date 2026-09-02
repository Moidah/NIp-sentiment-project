"""
PASO 1 — Análisis Exploratorio de Datos (EDA)
================================================
Dataset: IMDB Movie Reviews — 50,000 reseñas reales de películas,
etiquetadas como positivas o negativas (25,000 de cada clase).
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import re

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUTS_DIR = SCRIPT_DIR.parent / "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# --- Cargar datos ---
df = pd.read_csv(DATA_DIR / "imdb_reviews.csv")

print(f"Filas: {len(df)}")
print(f"\nBalance de clases:")
print(df["sentiment"].value_counts())

# --- Limpieza básica: el texto trae tags HTML (<br /><br />) ---
def clean_text(text):
    text = re.sub(r"<br\s*/?>", " ", text)  # quitar tags <br />
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["review_clean"] = df["review"].apply(clean_text)

print("\nEjemplo antes/después de limpieza:")
print("ANTES:", df["review"].iloc[0][:150])
print("DESPUÉS:", df["review_clean"].iloc[0][:150])

# --- Longitud de las reseñas ---
df["word_count"] = df["review_clean"].str.split().str.len()
print(f"\nLongitud promedio: {df['word_count'].mean():.0f} palabras")
print(f"Longitud mediana: {df['word_count'].median():.0f} palabras")
print(f"Longitud máxima: {df['word_count'].max()} palabras")

fig, ax = plt.subplots(figsize=(8, 4))
df["word_count"].clip(upper=800).hist(bins=50, ax=ax, color="steelblue")
ax.set_title("Distribución de longitud de reseñas (palabras)")
ax.set_xlabel("Número de palabras")
ax.set_ylabel("Frecuencia")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "01_longitud_resenas.png", dpi=120)
plt.close()

# --- Longitud por sentimiento ---
fig, ax = plt.subplots(figsize=(7, 4))
df.boxplot(column="word_count", by="sentiment", ax=ax)
ax.set_ylim(0, 600)
ax.set_title("Longitud de reseña por sentimiento")
plt.suptitle("")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "02_longitud_por_sentimiento.png", dpi=120)
plt.close()

# --- Palabras más frecuentes por clase (simple, sin librerías extra) ---
from collections import Counter

STOPWORDS = set("""the a an and or but is are was were be been being to of in on at
for with as by this that it its it's i you he she they we my your his her their our
not no so if then than too very just movie film one all
have has had do does did will would could should can may might must shall
who whom which what when where why how there here from about out some more
even also only such into over after before again further once here there
them us me him her its our your ther those these am""".split())

def top_words(texts, n=15):
    words = []
    for t in texts:
        words.extend([w.lower().strip(".,!?\"'()") for w in t.split()])
    words = [w for w in words if w and w not in STOPWORDS and len(w) > 2]
    return Counter(words).most_common(n)

pos_words = top_words(df[df["sentiment"] == "positive"]["review_clean"].sample(5000, random_state=42))
neg_words = top_words(df[df["sentiment"] == "negative"]["review_clean"].sample(5000, random_state=42))

print("\nTop 15 palabras en reseñas POSITIVAS (muestra de 5,000):")
print(pos_words)
print("\nTop 15 palabras en reseñas NEGATIVAS (muestra de 5,000):")
print(neg_words)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
words_p, counts_p = zip(*pos_words)
axes[0].barh(words_p[::-1], counts_p[::-1], color="seagreen")
axes[0].set_title("Top palabras — reseñas positivas")

words_n, counts_n = zip(*neg_words)
axes[1].barh(words_n[::-1], counts_n[::-1], color="indianred")
axes[1].set_title("Top palabras — reseñas negativas")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "03_palabras_frecuentes.png", dpi=120)
plt.close()

# Guardar el dataset limpio para el siguiente paso
df[["review_clean", "sentiment"]].to_csv(DATA_DIR / "imdb_reviews_clean.csv", index=False)

print(f"\n EDA completo. Gráficas guardadas en {OUTPUTS_DIR}/")
