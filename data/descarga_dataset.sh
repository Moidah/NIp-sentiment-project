#!/bin/bash
# Descargar el dataset real de reseñas de IMDB (50,000 reseñas, ~64MB)
curl -L -o imdb_reviews.csv "https://raw.githubusercontent.com/SK7here/Movie-Review-Sentiment-Analysis/master/IMDB-Dataset.csv"
echo "Descarga completa: imdb_reviews.csv"