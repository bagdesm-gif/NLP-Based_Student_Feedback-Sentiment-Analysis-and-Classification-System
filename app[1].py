import pandas as pd, streamlit as st, joblib
from src.preprocessing import clean_text

st.set_page_config(page_title="Student Feedback Analyzer", page_icon="🎓", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("model/sentiment_pipeline.pkl")

st.title("🎓 Student Feedback Sentiment Analyzer")
st.write("NLP-based system for classifying student feedback as Positive, Neutral, or Negative.")
model=load_model()

tab1,tab2=st.tabs(["🔍 Analyze Feedback","📊 Batch Analysis"])

with tab1:
    feedback=st.text_area("Enter student feedback:",height=140,
        placeholder="Example: The teaching method is very good and easy to understand.")
    if st.button("Analyze Sentiment",type="primary"):
        if not feedback.strip():
            st.warning("Please enter some feedback.")
        else:
            pred=model.predict([clean_text(feedback)])[0]
            conf=model.predict_proba([clean_text(feedback)])[0].max()*100
            if pred=="Positive": st.success("😊 Sentiment: Positive")
            elif pred=="Negative": st.error("😞 Sentiment: Negative")
            else: st.info("😐 Sentiment: Neutral")
            st.metric("Model Confidence",f"{conf:.1f}%")

with tab2:
    uploaded=st.file_uploader("Upload CSV with a 'feedback' column",type=["csv"])
    if uploaded:
        df=pd.read_csv(uploaded)
        if "feedback" not in df.columns:
            st.error("CSV must contain a column named 'feedback'.")
        else:
            df["sentiment"]=model.predict(df["feedback"].fillna("").apply(clean_text))
            st.dataframe(df,use_container_width=True)
            st.subheader("Sentiment Distribution")
            st.bar_chart(df["sentiment"].value_counts())
            st.download_button("Download Analyzed CSV",
                df.to_csv(index=False).encode("utf-8"),
                "analyzed_feedback.csv","text/csv")
