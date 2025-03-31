import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub(r"http\S+|www.\S+", "", text)  # Remove links
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove caracteres especiais
    text = " ".join([word.lower() for word in text.split() if word.lower() not in stop_words])
    return text

column_names = ["id", "game", "sentiment", "text"]

df = pd.read_csv("data/twitter_training.csv", names=column_names, header=None, sep=",", engine="python")

df["text"] = df["text"].astype(str)  # Converte para string
df["text"] = df["text"].fillna("")  # Substitui valores NaN por string vazia
df = df.dropna(subset=["text"])  # Remove linhas onde a coluna "text" está vazia
df = df[df["text"].str.strip() != ""]  # Remove textos vazios ou espaços em branco

df["clean_text"] = df["text"].apply(clean_text)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
df["tokens"] = df["clean_text"].apply(lambda x: tokenizer.encode(x, truncation=True, padding="max_length", max_length=128))

df.to_csv("data/tweets_cleaned.csv", index=False)