# -*- coding: utf-8 -*-
import argparse, os
from gensim.models import Word2Vec

def sentences_from_corpus(path="data/corpus.txt"):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield line.strip().split()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vector_size", type=int, default=100)
    ap.add_argument("--min_count", type=int, default=2)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--sg", type=int, default=1, help="1=skipgram, 0=CBOW")
    ap.add_argument("--model_path", default="models/word2vec.model")
    args = ap.parse_args()

    os.makedirs("models", exist_ok=True)
    sents = list(sentences_from_corpus())
    model = Word2Vec(sentences=sents, vector_size=args.vector_size, window=args.window, min_count=args.min_count, workers=4, sg=args.sg, epochs=args.epochs)
    model.save(args.model_path)
    print(f"Saved Word2Vec to {args.model_path} (vocab={len(model.wv)}).")

if __name__ == "__main__":
    main()
