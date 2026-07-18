"""
train.py
--------
Trains a Fake News Detection model using TF-IDF features and a
Passive Aggressive Classifier (a strong, fast baseline for text
classification), then evaluates it and saves the trained model
+ vectorizer to disk.

Usage:
    python src/train.py
    python src/train.py --data data/news_dataset.csv --model passive_aggressive
"""

import argparse
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from preprocess import clean_dataframe

MODELS = {
    # Passive-Aggressive behaviour, implemented via SGDClassifier
    # (scikit-learn's current recommended equivalent).
    "passive_aggressive": SGDClassifier(
        loss="hinge", penalty=None, learning_rate="pa1", eta0=1.0, random_state=42
    ),
    "logistic_regression": LogisticRegression(max_iter=200, random_state=42),
}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["label"])
    return df


def main():
    parser = argparse.ArgumentParser(description="Train the fake news detector")
    parser.add_argument("--data", default="data/news_dataset.csv", help="Path to CSV dataset")
    parser.add_argument(
        "--model",
        default="passive_aggressive",
        choices=list(MODELS.keys()),
        help="Which classifier to train",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()

    print(f"Loading dataset from {args.data} ...")
    df = load_data(args.data)
    print(f"Loaded {len(df)} rows. Label distribution:\n{df['label'].value_counts()}")

    print("Cleaning text ...")
    df = clean_dataframe(df)

    X = df["content_clean"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    print("Vectorizing text with TF-IDF ...")
    vectorizer = TfidfVectorizer(
        max_df=0.9, min_df=2, ngram_range=(1, 2), stop_words="english"
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"Training {args.model} model ...")
    model = MODELS[args.model]
    model.fit(X_train_vec, y_train)

    print("Evaluating ...")
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc * 100:.2f}%\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix (rows=true, cols=pred), labels =", sorted(y.unique()))
    print(confusion_matrix(y_test, y_pred, labels=sorted(y.unique())))

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "fake_news_model.pkl")
    vec_path = os.path.join(args.out_dir, "tfidf_vectorizer.pkl")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"\nSaved model -> {model_path}")
    print(f"Saved vectorizer -> {vec_path}")


if __name__ == "__main__":
    main()
