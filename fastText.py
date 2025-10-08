# -*- coding: utf-8 -*-
import argparse, os, sys
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/corpus.txt")
    ap.add_argument("--model_path", default="models/fasttext.bin")
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--epoch", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.1)
    args = ap.parse_args()

    try:
        import fasttext
    except Exception as e:
        print("fasttext is not installed. Try: pip install fasttext", file=sys.stderr)
        sys.exit(1)

    os.makedirs("models", exist_ok=True)
    model = fasttext.train_unsupervised(args.input, model="skipgram", dim=args.dim, epoch=args.epoch, lr=args.lr)
    model.save_model(args.model_path)
    print(f"Saved FastText to {args.model_path}")

if __name__ == "__main__":
    main()
