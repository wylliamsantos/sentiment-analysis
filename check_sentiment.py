import pandas as pd

# Carregar os dados processados
df = pd.read_csv("data/tweets_cleaned.csv")

# Verificar valores únicos na coluna 'sentiment'
print("Valores únicos em 'sentiment':", df["sentiment"].unique())

# Verificar distribuição dos sentimentos
print(df["sentiment"].value_counts())
