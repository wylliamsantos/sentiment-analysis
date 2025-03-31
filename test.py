from transformers import AutoModelForSequenceClassification

model_path = "models/sentiment_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)

print(model.config.id2label) 