"""
app.py
------
Simple Streamlit web app for the Fake News Detection project.

Run with:
    streamlit run src/app.py
"""

import streamlit as st

from predict import load_artifacts, predict

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection")
st.write(
    "Paste a news headline or article text below and the model will "
    "predict whether it looks REAL or FAKE, based on writing style and "
    "language patterns learned during training."
)

try:
    model, vectorizer = load_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

text = st.text_area("Article text", height=200, placeholder="Paste article text here...")

if st.button("Analyze", type="primary"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        label, confidence = predict(text, model, vectorizer)
        if label == "FAKE":
            st.error(f"### Prediction: 🚨 FAKE NEWS")
        else:
            st.success(f"### Prediction: ✅ REAL NEWS")
        if confidence is not None:
            st.caption(f"Confidence score: {confidence:.3f}")

st.divider()
st.caption(
    "Note: This model is trained for demonstration purposes. For production-grade "
    "accuracy, train on a larger real-world dataset (see README)."
)
