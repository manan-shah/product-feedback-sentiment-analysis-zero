import pandas as pd
from sentiment import predict_sentiment

df = pd.read_csv(
    "data/product_feedback.csv"
)

results = df["Feedback"].apply(
    predict_sentiment
)

df["Sentiment"] = results.apply(
    lambda x: x["sentiment"]
)

df["Confidence"] = results.apply(
    lambda x: x["confidence"]
)

print(df.head())