"""
predict.py
----------
Load the trained model + vectorizer and classify new article text
as REAL or FAKE.

Usage:
    python src/predict.py --text "Some news headline or article text here"
    python src/predict.py   # then type text interactively
"""

import argparse
import os

import joblib

from preprocess import clean_text

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fake_news_model.pkl")
VEC_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "tfidf_vectorizer.pkl")


def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH)):
        raise FileNotFoundError(
            "Trained model not found. Run `python src/train.py` first."
        )
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    return model, vectorizer


def predict(text: str, model=None, vectorizer=None):
    if model is None or vectorizer is None:
        model, vectorizer = load_artifacts()

    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    label = model.predict(vec)[0]

    # Confidence score when the model supports it
    confidence = None
    if hasattr(model, "decision_function"):
        score = model.decision_function(vec)
        confidence = float(abs(score[0])) if hasattr(score, "__len__") else float(abs(score))
    elif hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        confidence = float(max(proba))

    return label, confidence


def main():
    parser = argparse.ArgumentParser(description="Predict if a news article is REAL or FAKE")
    parser.add_argument("--text", help="Article text to classify")
    args = parser.parse_args()

    model, vectorizer = load_artifacts()

    if args.text:
        label, confidence = predict(args.text, model, vectorizer)
        print(f"Prediction: {label}" + (f" (confidence score: {confidence:.3f})" if confidence else ""))
    else:
        print("Enter article text (Ctrl+C to exit):")
        try:
            while True:
                text = input("\n> ")
                if not text.strip():
                    continue
                label, confidence = predict(text, model, vectorizer)
                print(f"Prediction: {label}" + (f" (confidence score: {confidence:.3f})" if confidence else ""))
        except KeyboardInterrupt:
            print("\nExiting.")


if __name__ == "__main__":
    main()
