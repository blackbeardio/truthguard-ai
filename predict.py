"""
predict.py — FakeNewsAI
CLI utility to predict whether a news article is FAKE or REAL.

Usage:
    python predict.py --text "Your news article text here …"
    python predict.py --file path/to/article.txt
    python predict.py  (interactive mode)
"""

import os
import re
import string
import argparse
import joblib
import nltk

for pkg in ["stopwords", "punkt", "wordnet"]:
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg != "punkt" else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")
VEC_PATH   = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [
        LEMMATIZER.lemmatize(w)
        for w in text.split()
        if w not in STOP_WORDS and len(w) > 2
    ]
    return " ".join(tokens)


def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
        raise FileNotFoundError(
            "❌  Model not found! Run  python train_model.py  first."
        )
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    return model, vectorizer


def predict(text: str) -> dict:
    model, vectorizer = load_artifacts()
    cleaned   = clean_text(text)
    tfidf_vec = vectorizer.transform([cleaned])

    pred       = model.predict(tfidf_vec)[0]
    # decision_function gives the raw score (distance from boundary)
    score      = model.decision_function(tfidf_vec)[0]

    # Convert score to a 0-1 confidence proxy via sigmoid
    import math
    confidence = 1 / (1 + math.exp(-abs(score)))

    label = "REAL" if pred == 1 else "FAKE"
    emoji = "✅" if pred == 1 else "🚨"

    return {
        "label":      label,
        "emoji":      emoji,
        "confidence": round(confidence * 100, 2),
        "raw_score":  round(float(score), 4),
    }


def print_result(result: dict, text_preview: str = ""):
    bar_len = 40
    filled  = int(bar_len * result["confidence"] / 100)
    bar     = "#" * filled + "-" * (bar_len - filled)
    color   = "\033[92m" if result["label"] == "REAL" else "\033[91m"
    reset   = "\033[0m"

    print("\n" + "=" * 55)
    if text_preview:
        preview = text_preview[:120] + ("..." if len(text_preview) > 120 else "")
        print(f"  [ARTICLE] {preview}")
        print("-" * 55)
    print(f"  Verdict    : {color}[{result['label']}]{reset}")
    print(f"  Confidence : {result['confidence']:.1f}%  [{bar}]")
    print(f"  Raw Score  : {result['raw_score']}")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="FakeNewsAI — Predict if a news article is FAKE or REAL"
    )
    parser.add_argument("--text", type=str, help="News article text (quoted string)")
    parser.add_argument("--file", type=str, help="Path to a .txt file containing the article")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
        result = predict(text)
        print_result(result, text)

    elif args.text:
        result = predict(args.text)
        print_result(result, args.text)

    else:
        # Interactive mode
        print("\nFakeNewsAI -- Interactive Prediction Mode")
        print("   Type or paste a news article. Submit with two blank lines, or type 'quit' to exit.\n")
        while True:
            print("Enter article text (or 'quit'):")
            lines = []
            blank_count = 0
            while True:
                line = input()
                if line.strip().lower() == "quit":
                    print(">> Goodbye!")
                    return
                if line == "":
                    blank_count += 1
                    if blank_count >= 2:
                        break
                else:
                    blank_count = 0
                    lines.append(line)

            text = "\n".join(lines).strip()
            if not text:
                print("⚠️   Empty input — please try again.\n")
                continue

            result = predict(text)
            print_result(result, text)


if __name__ == "__main__":
    main()
