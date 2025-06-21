import streamlit as st
import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from preprocessing import preprocess_news



with open('news_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# --- App Title ---
st.title("📰 News Category Predictor")
st.write("Enter a news article text below, and I'll predict its category!")

# --- Input Box ---
news_input = st.text_area("Paste your news article here:", height=200)

# --- Predict Button ---
if st.button("Predict Category"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Predict using pipeline
        pred_label = pipeline.predict(pd.Series([news_input]))[0]
        pred_category = label_encoder.inverse_transform([pred_label])[0]

        # Display Result
        st.success(f"**Predicted Category:** `{pred_category}`")