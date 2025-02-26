from fastapi import FastAPI
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch

app = FastAPI()

USE_TRAINED_MODEL = False

HF_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

LOCAL_MODEL_PATH = "models/sentiment_model"
label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}  # Mapeamento das labels do modelo treinado

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
    result = classifier(text)[0]

    if USE_TRAINED_MODEL:
        if "LABEL" in result["label"]:
            label_id = int(result["label"].split("_")[-1])
            sentiment = label_mapping.get(label_id, "Unknown")
        else:
            sentiment = result["label"]

    else:
        sentiment = result["label"].capitalize()

    return {
        "text": text,
        "sentiment": sentiment,
        "score": result["score"]
    }
