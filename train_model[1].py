import os, pandas as pd, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.preprocessing import clean_text

df = pd.read_csv("dataset/student_feedback.csv")
df["clean_feedback"] = df["feedback"].apply(clean_text)
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_feedback"], df["sentiment"], test_size=0.25,
    random_state=42, stratify=df["sentiment"])
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)),
    ("classifier", LogisticRegression(max_iter=1000))
])
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test,pred)*100,2), "%")
print(classification_report(y_test,pred))
os.makedirs("model",exist_ok=True)
joblib.dump(pipeline,"model/sentiment_pipeline.pkl")
print("Saved: model/sentiment_pipeline.pkl")
