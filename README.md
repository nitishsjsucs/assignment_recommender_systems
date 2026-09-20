# Vector Space Model (TF-IDF) exercise

A from-scratch implementation of the classic vector space model for document retrieval:
tokenize, build a term-frequency matrix, compute IDF, multiply into TF-IDF, normalize, and
rank documents against a query by cosine similarity. The corpus is the four-document
shopping example used in the course slides:

```
d1: User selected Wedding gown
d2: User ordered on-line rose flowers
d3: User searched diamond ring
d4: User selected white wedding gown, online flowers, 3 carat diamond ring
```

Everything is computed with `math` and `collections.Counter`; numpy is only used for the
similarity matrix and matplotlib for the heatmap. scikit-learn appears only as a
cross-check, not as the implementation.

## What's here

- `vector_space_model.py` — the `VectorSpaceModel` class (preprocessing, TF, IDF, TF-IDF,
  cosine similarity, query ranking, similarity matrix, heatmap) plus two module-level
  functions: `demonstrate_vsm()` prints every intermediate matrix for the four documents,
  and `compare_with_sklearn()` diffs the results against `TfidfVectorizer`.
- `demo.py` — runs both of the above end to end.
- `interactive_demo.py` — a REPL loop; type a query to rank the four documents, or
  `vocab` / `matrix` / `similarity` / `quit`.
- `similarity_matrix.png` — the heatmap, regenerated on every run of the demo.
- `requirements.txt` — numpy, matplotlib, scikit-learn, pandas.

## Running it

```bash
pip install -r requirements.txt
python demo.py              # full walkthrough, writes similarity_matrix.png
python interactive_demo.py  # interactive query loop
```

No API keys, data files, or network access needed.

## A note on the scikit-learn comparison

`compare_with_sklearn()` reports a maximum difference of about 0.19 between the two cosine
similarity matrices — the implementations genuinely disagree, and that is the interesting
part of the exercise. This code uses the textbook `idf = log(N / df)`, which drives the
weight of any term appearing in all four documents (`user`) to exactly zero. scikit-learn
uses smoothed IDF, `log((1 + N) / (1 + df)) + 1`, so those terms keep a non-zero weight and
every document ends up slightly similar to every other. It also applies its own tokenizer,
which drops one-character tokens such as the `3` in d4.

## Course context

SJSU graduate coursework — a recommender systems assignment. The repository carries no
course number or syllabus reference, so none is asserted here.
