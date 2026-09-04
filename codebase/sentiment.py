from transformers import pipeline
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    token=HF_TOKEN
)

def predict_sentiment(text):

    result = classifier(
        text,
        candidate_labels=[
            "positive",
            "negative",
            "neutral"
        ]
    )

    return {
        "sentiment": result["labels"][0],
        "confidence": round(result["scores"][0], 4)
    }