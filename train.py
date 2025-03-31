import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset

df = pd.read_csv("data/tweets_cleaned.csv")

# 📌 2️⃣ Remover valores NaN antes da separação
df = df.dropna(subset=["clean_text", "sentiment"])

# 📌 2️⃣.1️⃣ Remover tweets irrelevantes, se não quiser treiná-los
df = df[df["sentiment"] != "Irrelevant"]
df = df[df["sentiment"].isin(["Positive", "Negative", "Neutral"])]  # Apenas as 3 classes

# 📌 3️⃣ Mapear sentimentos para números
sentiment_mapping = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}
df["label"] = df["sentiment"].map(sentiment_mapping)

# 📌 4️⃣ Garantir que os textos são strings
df["clean_text"] = df["clean_text"].astype(str)

# 📌 5️⃣ Separar treino e teste
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["clean_text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

# 📌 6️⃣ Corrigir qualquer diferença de tamanho entre textos e labels
min_train_size = min(len(train_texts), len(train_labels))
train_texts = train_texts[:min_train_size]
train_labels = train_labels[:min_train_size]

min_test_size = min(len(test_texts), len(test_labels))
test_texts = test_texts[:min_test_size]
test_labels = test_labels[:min_test_size]

# 📌 7️⃣ Carregar Tokenizer
model_name = "bert-base-uncased"  # Pode trocar por "ProsusAI/finbert" se for finanças
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 📌 8️⃣ Tokenizar os textos
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# 📌 9️⃣ Garantir que o número de tokens e labels é o mesmo
assert len(train_encodings["input_ids"]) == len(train_labels), "Erro: Tamanhos diferentes após tokenização!"
assert len(test_encodings["input_ids"]) == len(test_labels), "Erro: Tamanhos diferentes após tokenização!"

# 🔟 Criar datasets Hugging Face
train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": train_labels
})

test_dataset = Dataset.from_dict({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "labels": test_labels
})

# 📌 1️⃣1️⃣ Carregar Modelo pré-treinado
# Mapeamento correto das labels
id2label = {v: k for k, v in sentiment_mapping.items()}  # {0: "Negative", 1: "Neutral", 2: "Positive"}
label2id = sentiment_mapping  # {"Negative": 0, "Neutral": 1, "Positive": 2}

# Criar o modelo definindo os mapeamentos
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(sentiment_mapping),
    id2label=id2label,
    label2id=label2id
)

# 📌 1️⃣2️⃣ Definir hiperparâmetros de treinamento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,  # Reduzir se precisar
    per_device_eval_batch_size=8,
    learning_rate=2e-5,  # Diminua se necessário (ex.: 3e-5 ou 2e-5)
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    gradient_accumulation_steps=2,  # Ajuda se os batches forem pequenos
    max_grad_norm=1.0,  # Gradiente clipping
)

# 📌 1️⃣3️⃣ Criar o Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# 📌 1️⃣4️⃣ Treinar o Modelo
trainer.train()

# 📌 1️⃣5️⃣ Salvar o modelo treinado
model.save_pretrained("models/sentiment_model")
tokenizer.save_pretrained("models/sentiment_model")

# 🔥 Opcional: Salvar modelo como `trained_model.pt`
torch.save(model.state_dict(), "models_old/trained_model.pt")

print("✅ Modelo treinado e salvo com sucesso!")
