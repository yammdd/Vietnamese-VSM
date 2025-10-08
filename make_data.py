# -*- coding: utf-8 -*-
"""Create tokenized corpus from data/datatrain.txt (id\ttitle\ttext)."""
import argparse, csv, os, pandas as pd

import re

def _try_import_tokenizers():
    tok = None
    # underthesea
    try:
        from underthesea import word_tokenize as uts_tokenize  # type: ignore
        tok = lambda s: uts_tokenize(s, format="text").split()
        return tok
    except Exception:
        pass
    # pyvi
    try:
        from pyvi.ViTokenizer import tokenize as pyvi_tokenize  # type: ignore
        tok = lambda s: pyvi_tokenize(s).split()
        return tok
    except Exception:
        pass

    # fallback: simple regex tokenizer
    def simple_tok(s: str):
        s = s.lower()
        # keep Vietnamese letters; coarse fallback
        tokens = re.findall(r"[0-9a-zA-ZÀ-ỹ]+", s)
        return tokens
    return simple_tok

VN_TOKENIZE = _try_import_tokenizers()

def normalize_text(s: str) -> str:
    s = s.replace("\\u00a0", " ").strip()
    return s


def load_stopwords(path: str):
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set([w.strip() for w in f if w.strip()])

def preprocess(text: str, stopwords: set):
    text = normalize_text(text)
    tokens = [t for t in VN_TOKENIZE(text) if t and t not in stopwords]
    return tokens

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./data/wikipedia_vi.csv", help="CSV with columns id, title, text")
    ap.add_argument("--stopwords", default="./data/stopwords.txt")
    ap.add_argument("--out_corpus", default="./data/corpus.txt")
    ap.add_argument("--out_meta", default="./data/meta.csv")
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)
    sw = load_stopwords(args.stopwords)

    df = pd.read_csv(args.input, sep=",", dtype=str)
    df = df.fillna("")
    corpus_lines = []
    meta_rows = []

    for _, row in df.iterrows():
        doc_id, title, text = row.get("id",""), row.get("title",""), row.get("text","")
        tokens = preprocess(text, sw)
        corpus_lines.append(" ".join(tokens))
        meta_rows.append({"id": doc_id, "title": title})

    with open(args.out_corpus, "w", encoding="utf-8") as f:
        for line in corpus_lines:
            f.write(line.strip()+"\n")
    pd.DataFrame(meta_rows).to_csv(args.out_meta, index=False, encoding="utf-8")
    print(f"Wrote {len(corpus_lines)} lines to {args.out_corpus} and meta to {args.out_meta}.")

if __name__ == "__main__":
    main()
