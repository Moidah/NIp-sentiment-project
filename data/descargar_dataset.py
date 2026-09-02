"""Descargar el dataset real de reseñas de IMDB (50,000 reseñas, ~64MB)."""
import urllib.request

URL = "https://raw.githubusercontent.com/SK7here/Movie-Review-Sentiment-Analysis/master/IMDB-Dataset.csv"
print("Descargando dataset de IMDB (~64MB, puede tardar un minuto)...")
urllib.request.urlretrieve(URL, "imdb_reviews.csv")
print("Descarga completa: imdb_reviews.csv")
