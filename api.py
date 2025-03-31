from fastapi import FastAPI
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import numpy as np

app = FastAPI()

USE_TRAINED_MODEL = True

HF_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
LOCAL_MODEL_PATH = "models/sentiment_model"

if USE_TRAINED_MODEL:
    print(f"🔹 Usando o modelo treinado localmente: {LOCAL_MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

else:
    print(f"🔹 Usando modelo Hugging Face pré-treinado: {HF_MODEL_NAME}")

    classifier = pipeline("sentiment-analysis", model=HF_MODEL_NAME)

@app.get("/predict/")
def predict(text: str):
    try:
        result = classifier(text)[0]
        print(result)
        if USE_TRAINED_MODEL:
            sentiment = result["label"]

        # **Validar o score para evitar NaN ou infinito**
        score = result.get("score", 0.0)  # Caso não exista "score", define como 0.0
        if not np.isfinite(score):  # Se for NaN ou infinito, definir como 0.0
            score = 0.0

        return {
            "text": text,
            "sentiment": sentiment,
            "score": score
        }

    except Exception as e:
        return {"error": f"Erro ao processar a previsão: {str(e)}"}
