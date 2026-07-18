"""
preprocess.py
-------------
Text cleaning utilities for the Fake News Detection project.
"""

import re
import string

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False


def _ensure_nltk_data():
    """Download required NLTK data quietly if not already present."""
    for pkg, path in [
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


if _NLTK_AVAILABLE:
    _ensure_nltk_data()
    _STOPWORDS = set(stopwords.words("english"))
    _LEMMATIZER = WordNetLemmatizer()
else:
    # Minimal fallback stopword list if nltk isn't installed
    _STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
        "in", "on", "at", "to", "for", "of", "with", "by", "this", "that",
        "it", "as", "be", "has", "have", "had", "not", "will", "would",
    }
    _LEMMATIZER = None


def clean_text(text: str) -> str:
    """
    Lowercase, strip URLs/HTML/punctuation/digits, remove stopwords,
    and lemmatize (if nltk is available). Returns cleaned text.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)      # URLs
    text = re.sub(r"<.*?>", " ", text)                        # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                     # punctuation/digits
    text = re.sub(r"\s+", " ", text).strip()                  # extra whitespace

    tokens = text.split()
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]

    if _LEMMATIZER is not None:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def clean_dataframe(df, text_columns=("title", "text")):
    """
    Apply clean_text to the given columns of a pandas DataFrame,
    combine them into a single 'content' column, and return the df.
    """
    for col in text_columns:
        if col not in df.columns:
            df[col] = ""

    df["content"] = (df[text_columns[0]].fillna("") + " " + df[text_columns[1]].fillna(""))
    df["content_clean"] = df["content"].apply(clean_text)
    return df


if __name__ == "__main__":
    sample = "BREAKING!!! You WON'T believe what officials found at http://example.com <b>today</b> 12345"
    print("Original:", sample)
    print("Cleaned :", clean_text(sample))
