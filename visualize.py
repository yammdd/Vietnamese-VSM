# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

def plot_scatter(points, labels=None, title="t-SNE"):
    plt.figure(figsize=(8,6))
    x, y = points[:,0], points[:,1]
    plt.scatter(x, y, s=8)
    if labels is not None:
        for i, txt in enumerate(labels):
            plt.annotate(str(txt), (x[i], y[i]), fontsize=7, alpha=0.8)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def visualize_docs(limit=500):
    # Use precomputed SVD50 doc matrix if available, else recompute
    mat_path = "models/tfidf_matrix.svd50.npz"
    if os.path.exists(mat_path):
        X50 = np.load(mat_path)["X50"]
    else:
        # recompute from vectorizer + corpus
        from sklearn.feature_extraction.text import TfidfVectorizer
        corpus = [line.strip() for line in open("data/corpus.txt","r",encoding="utf-8")]
        vec = TfidfVectorizer(tokenizer=lambda s: s.split(), preprocessor=lambda s: s)
        X = vec.fit_transform(corpus)
        svd = TruncatedSVD(n_components=min(50, max(2, min(X.shape)-1)))
        X50 = svd.fit_transform(X)
    meta = pd.read_csv("data/meta.csv") if os.path.exists("data/meta.csv") else None
    if limit and X50.shape[0] > limit:
        X50 = X50[:limit]
        meta = meta.iloc[:limit] if meta is not None else None
    ts = TSNE(n_components=2, init="pca", perplexity=30, learning_rate="auto")
    pts = ts.fit_transform(X50)
    labels = meta["title"].tolist() if meta is not None else None
    plot_scatter(pts, labels=labels, title="t-SNE documents (TF-IDF -> SVD50)")

def visualize_words(model="word2vec", topk=500):
    labels = []
    vecs = []
    if model == "word2vec":
        from gensim.models import Word2Vec
        m = Word2Vec.load("models/word2vec.model")
        for w in list(m.wv.index_to_key)[:topk]:
            labels.append(w); vecs.append(m.wv[w])
    else:
        try:
            import fasttext
        except Exception:
            raise SystemExit("fasttext missing. Install fasttext or use --model word2vec")
        m = fasttext.load_model("models/fasttext.bin")
        vocab = []
        with open("data/corpus.txt","r",encoding="utf-8") as f:
            for line in f:
                for w in line.strip().split():
                    vocab.append(w)
        from collections import Counter
        for w, _ in Counter(vocab).most_common(topk):
            labels.append(w); vecs.append(m.get_word_vector(w))
    X = np.array(vecs)
    ts = TSNE(n_components=2, init="pca", perplexity=30, learning_rate="auto")
    pts = ts.fit_transform(X)
    plot_scatter(pts, labels=labels, title=f"t-SNE words ({model})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["docs","words"], required=True)
    ap.add_argument("--model", choices=["word2vec","fasttext"], default="word2vec")
    ap.add_argument("--limit", type=int, default=500, help="max docs when mode=docs")
    ap.add_argument("--topk", type=int, default=500, help="max words when mode=words")
    args = ap.parse_args()

    if args.mode == "docs":
        visualize_docs(limit=args.limit)
    else:
        visualize_words(model=args.model, topk=args.topk)

if __name__ == "__main__":
    main()
