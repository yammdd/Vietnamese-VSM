# 🇻🇳 Vietnamese Vector Space Model (VSM)

This project builds and visualizes a **Vector Space Model (VSM)** for Vietnamese text using **TF-IDF**, **Word2Vec**, and **FastText**.  
It includes preprocessing, training, similarity search, and t-SNE visualization for both documents and word embeddings.

---

## 🧩 Project Structure
<pre>
  Vietnamese-VSM/
  ├── data/
  │ ├── wikipedia_vi.csv                 # input dataset (id, title, text)
  │ ├── stopwords.txt                    # Vietnamese stopword list
  │ ├── corpus.txt                       # preprocessed & tokenized text (auto-generated)
  │ └── meta.csv                         # metadata: id, title (auto-generated)
  ├── models/
  │ ├── fasttext.bin                     # trained FastText embeddings
  │ ├── tfidf_matrix.svd50.npz           # reduced doc matrix (SVD 50)
  │ ├── tfidf_vectorizer.joblib          # saved TF-IDF model
  │ ├── word2vec.model                   # trained Word2Vec embeddings
  │ ├── word2vec.model.syn1neg.npy       # trained Word2Vec embeddings
  │ └── word2vec.model.wv.vectors.npy    # trained Word2Vec embeddings
  ├── make_data.py                       # text preprocessing & corpus generation
  ├── using_model_demo.py                # TF-IDF model training & similarity search
  ├── visualize.py                       # t-SNE visualization for docs or words
  ├── word2vec.py                        # Word2Vec training
  ├── fastText.py                        # FastText training
  └── requirements.txt                   # Python dependencies
</pre>

---

## ⚙️ Installation

**_Requires: Python 3.11_**

### (Optional) Create and activate a virtual environment 

```bash
python -m venv venv
source venv/bin/activate          # on macOS/Linux
# or
venv\Scripts\activate             # on Windows
```

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📘 Data Preparation



