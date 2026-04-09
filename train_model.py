"""
train_model.py — FakeNewsAI
Trains a PassiveAggressiveClassifier on the Fake/True news dataset
and saves the model + vectorizer for later use by predict.py and app.py.
"""

import os
import re
import string
import joblib
import numpy as np
import pandas as pd
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# NLTK data (download only if needed)
# ---------------------------------------------------------------------------
for pkg in ["stopwords", "punkt", "wordnet"]:
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg != "punkt" else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

FAKE_CSV  = os.path.join(DATA_DIR, "Fake.csv")
TRUE_CSV  = os.path.join(DATA_DIR, "True.csv")
MODEL_OUT = os.path.join(MODEL_DIR, "fake_news_model.pkl")
VEC_OUT   = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/digits/stopwords, then lemmatize."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"\[.*?\]", " ", text)                          # brackets
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [
        LEMMATIZER.lemmatize(w)
        for w in text.split()
        if w not in STOP_WORDS and len(w) > 2
    ]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Load & label data
# ---------------------------------------------------------------------------
def load_data():
    print("[INFO] Loading CSV files ...")
    fake = pd.read_csv(FAKE_CSV)
    true = pd.read_csv(TRUE_CSV)

    fake["label"] = 0   # FAKE  → 0
    true["label"] = 1   # REAL  → 1

    df = pd.concat([fake, true], ignore_index=True)

    # Combine title + text if both columns exist
    if "title" in df.columns and "text" in df.columns:
        df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    elif "text" in df.columns:
        df["content"] = df["text"].fillna("")
    else:
        raise ValueError("Dataset must contain at least a 'text' column.")

    df = df[["content", "label"]].dropna()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"  [+] Total samples  : {len(df):,}")
    print(f"  [+] Fake articles  : {(df['label']==0).sum():,}")
    print(f"  [+] Real articles  : {(df['label']==1).sum():,}")
    return df


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train():
    df = load_data()

    print("\n[INFO] Cleaning text ...")
    df["content"] = df["content"].apply(clean_text)

    X = df["content"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  [+] Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

    # ---- TF-IDF vectorisation ----
    print("\n[INFO] Fitting TF-IDF vectorizer ...")
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    # ---- Model training ----
    print("\n[INFO] Training SGDClassifier (PA-1 mode) ...")
    # loss='hinge', learning_rate='pa1' mimics PassiveAggressiveClassifier
    model = SGDClassifier(
        loss='hinge', 
        penalty=None, 
        learning_rate='pa1', 
        eta0=1.0, 
        max_iter=1000, 
        tol=1e-3, 
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)

    # ---- Evaluation ----
    y_pred  = model.predict(X_test_tfidf)
    acc     = accuracy_score(y_test, y_pred)

    print(f"\n[RESULT] Test Accuracy : {acc * 100:.2f}%")
    print("\n[REPORT] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))
    print("[REPORT] Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # ---- Persist ----
    joblib.dump(model,      MODEL_OUT)
    joblib.dump(vectorizer, VEC_OUT)
    print(f"\n[SAVED] Model      -> {MODEL_OUT}")
    print(f"[SAVED] Vectorizer -> {VEC_OUT}")

    return acc


if __name__ == "__main__":
    train()
