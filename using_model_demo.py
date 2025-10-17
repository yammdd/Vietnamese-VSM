# -*- coding: utf-8 -*-
"""Train/load TF-IDF, then query similar documents by cosine similarity."""
import argparse, os, sys, joblib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import re

def _try_import_tokenizers():
    tok = None
    try:
        from underthesea import word_tokenize as uts_tokenize
        tok = lambda s: uts_tokenize(s, format="text").split()
        return tok
    except Exception:
        pass
    try:
        from pyvi.ViTokenizer import tokenize as pyvi_tokenize
        tok = lambda s: pyvi_tokenize(s).split()
        return tok
    except Exception:
        pass
    def simple_tok(s: str):
        s = s.lower()
        tokens = re.findall(r"[0-9a-zA-ZÀ-ỹ]+", s)
        return tokens
    return simple_tok

VN_TOKENIZE = _try_import_tokenizers()

def normalize_text(s: str) -> str:
    s = s.replace("\u00a0", " ").strip()
    return s

def tokenize_vn(s):
    """Global tokenizer for Vietnamese text"""
    return VN_TOKENIZE(s)

def identity_func(s):
    """Identity preprocessor (do nothing)"""
    return s

def ensure_corpus(corpus_path="data/corpus.txt"):
    if not os.path.exists(corpus_path):
        print("Corpus not found. Run: python make_data.py --input data/datatrain.txt", file=sys.stderr)
        sys.exit(1)
    with open(corpus_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f]
    return texts

def fit_tfidf(corpus, ngram=(1,2), max_features=50000):
    vec = TfidfVectorizer(
        tokenizer=tokenize_vn,
        preprocessor=identity_func,
        ngram_range=ngram,
        max_features=max_features
    )
    X = vec.fit_transform(corpus)
    return vec, X

def load_meta(path="./data/meta.csv"):
    """Load metadata (id, title) if any"""
    import pandas as pd, os
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return pd.DataFrame({"id": [], "title": []})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true", help="fit TF‑IDF and save into models/")
    ap.add_argument("--query", type=str, default=None, help="text to search similar docs")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    os.makedirs("models", exist_ok=True)

    corpus = ensure_corpus()
    model_path = "models/tfidf_vectorizer.joblib"
    matrix_path = "models/tfidf_matrix.svd50.npz"

    if args.fit or not os.path.exists(model_path):
        vec, X = fit_tfidf(corpus)
        joblib.dump(vec, model_path)
        # store a reduced doc-matrix (SVD 50) to reuse in visualize
        svd = TruncatedSVD(n_components=min(50, max(2, min(X.shape)-1)))
        X50 = svd.fit_transform(X)
        np.savez_compressed(matrix_path, X50=X50)
        print(f"Saved TF‑IDF vectorizer to {model_path} and SVD50 doc matrix to {matrix_path}.")
    else:
        vec = joblib.load(model_path)

    if args.query:
        q_vec = vec.transform([args.query])
        D = vec.transform(corpus)
        sims = cosine_similarity(q_vec, D)[0]
        top_idx = np.argsort(-sims)[:args.topk]
        meta = load_meta()
        for rank, i in enumerate(top_idx, 1):
            title = meta.title.iloc[i] if i < len(meta) else f"doc_{i}"
            print(f"#{rank}  idx={i}  sim={sims[i]:.4f}  title={title}")

if __name__ == "__main__":
    main()
